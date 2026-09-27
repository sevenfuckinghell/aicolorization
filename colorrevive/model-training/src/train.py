"""ColorRevive training entry point.

Usage:
    python -m model_training.src.train --config model-training/configs/config.yaml
    (run from the repo root, or: python src/train.py --config configs/config.yaml)

Features: reproducible seeds, train/val loops, checkpointing (last + best),
resume, early stopping, TensorBoard logging, optional W&B, periodic sample
grids, and L1 loss on normalized ab channels.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

try:  # TensorBoard is optional at runtime; training works without it.
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]

try:  # package-style and flat-style imports both supported
    from .dataset import ColorizationDataset
    from .losses import ab_l1_loss
    from .model import ModelConfig, build_model
except ImportError:  # pragma: no cover
    from src.dataset import ColorizationDataset  # type: ignore
    from src.losses import ab_l1_loss  # type: ignore
    from src.model import ModelConfig, build_model  # type: ignore


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_ckpt(model: torch.nn.Module, cfg: dict, epoch: int, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "config": {
            "base_channels": cfg["model"]["base_channels"],
            "image_size": cfg["data"]["image_size"],
        },
        "model_name": cfg["model"].get("model_name", "unet-lab-v1"),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def run_epoch(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for batch in loader:
        l_ch = batch["l"].to(device)
        ab = batch["ab"].to(device)
        if training:
            with torch.enable_grad():
                pred = model(l_ch)
                loss = ab_l1_loss(pred, ab)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        else:
            with torch.no_grad():
                pred = model(l_ch)
                loss = ab_l1_loss(pred, ab)
        bs = l_ch.size(0)
        total += float(loss.item()) * bs
        count += bs
    return total / max(count, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg, dcfg, mcfg = cfg["train"], cfg["data"], cfg["model"]
    set_seed(int(tcfg.get("seed", 42)))
    device = resolve_device(tcfg.get("device", "auto"))
    print(f"[train] device={device} seed={tcfg.get('seed', 42)}")

    full = ColorizationDataset(dcfg["root"], image_size=int(dcfg["image_size"]), augment=False)
    val_frac = float(dcfg.get("val_split", 0.1))
    n_val = max(1, int(len(full) * val_frac))
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(int(tcfg.get("seed", 42)))
    )
    train_loader = DataLoader(
        ColorizationDataset(dcfg["root"], image_size=int(dcfg["image_size"]),
                            augment=bool(dcfg.get("augment", False)), max_images=len(train_set)),
        batch_size=int(tcfg["batch_size"]), shuffle=True, num_workers=int(tcfg.get("workers", 2)),
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=int(tcfg["batch_size"]), shuffle=False,
                            num_workers=int(tcfg.get("workers", 2)))

    model = build_model(ModelConfig.from_dict(mcfg)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                                  weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(tcfg["epochs"])
    ) if tcfg.get("scheduler", "cosine") == "cosine" else None

    start_epoch = 0
    best_val = float("inf")
    ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints"))
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val", float("inf")))
        print(f"[train] resumed from {args.resume} at epoch {start_epoch}")

    use_wandb = bool(tcfg.get("wandb", False))
    if use_wandb:
        try:
            import wandb
            wandb.init(project="colorrevive", config=cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"[train] W&B unavailable ({exc}); continuing without it")
            use_wandb = False

    writer = SummaryWriter(str(ckpt_dir / "runs")) if SummaryWriter else None
    if writer is None:
        print("[train] tensorboard not installed; skipping TB logging")
    patience, bad_epochs = int(tcfg.get("early_stopping_patience", 0)), 0

    for epoch in range(start_epoch, int(tcfg["epochs"])):
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device)
        if scheduler:
            scheduler.step()
        if writer:
            writer.add_scalars("loss", {"train": tr, "val": va}, epoch)
        if use_wandb:
            import wandb
            wandb.log({"train_l1": tr, "val_l1": va, "lr": optimizer.param_groups[0]["lr"]})
        print(f"[train] epoch {epoch:3d} | train L1 {tr:.4f} | val L1 {va:.4f}")

        save_ckpt(model, cfg, epoch, ckpt_dir / "last.pt", {"best_val": best_val})
        if va < best_val - 1e-5:
            best_val, bad_epochs = va, 0
            save_ckpt(model, cfg, epoch, ckpt_dir / "best.pt")
            print(f"[train]   new best val L1 {va:.4f} -> best.pt")
        elif patience > 0:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[train] early stopping after {patience} non-improving epochs")
                break

    if writer:
        writer.close()
    print(f"[train] done. best val L1 {best_val:.4f}. Checkpoints in {ckpt_dir}/")
    print("[train] serve with: MODEL_CHECKPOINT_PATH=<path>/best.pt (backend must use matching base_channels)")


if __name__ == "__main__":
    main()
