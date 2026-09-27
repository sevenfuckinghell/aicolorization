"""Evaluation utilities: MAE/MSE, PSNR, SSIM, Delta E, and comparison grids.

Usage:
    python -m model_training.src.evaluate --checkpoint checkpoints/best.pt \
        --config configs/config.yaml [--split val] [--grid out/grid.png]

NOTE: pixel metrics do not fully measure color plausibility — many different
colors are reasonable for the same grayscale input. Use them to track
convergence, not to certify realism.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.color import lab2rgb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    from .dataset import ColorizationDataset
    from .model import ModelConfig, build_model
except ImportError:  # pragma: no cover
    from src.dataset import ColorizationDataset  # type: ignore
    from src.model import ModelConfig, build_model  # type: ignore


def denorm_lab_to_rgb(l_norm: np.ndarray, ab_chw: np.ndarray) -> np.ndarray:
    """Normalized L (H,W) + ab in CHW (2,H,W) -> uint8 RGB."""
    lab = np.empty((*l_norm.shape, 3), dtype=np.float64)
    lab[..., 0] = (l_norm + 1.0) * 50.0
    lab[..., 1] = ab_chw[0] * 128.0
    lab[..., 2] = ab_chw[1] * 128.0
    rgb = lab2rgb(np.clip(lab, np.array([0, -128, -128]), np.array([100, 127, 127])))
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def delta_e_lab(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Mean Euclidean distance in Lab space (CIE76)."""
    return float(np.sqrt(((lab1 - lab2) ** 2).sum(axis=-1)).mean())


@torch.no_grad()
def evaluate(checkpoint: str, config_path: str, split: str = "val", grid_out: str | None = None) -> dict:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    mcfg = dict(cfg["model"])
    if "config" in state:  # trust checkpoint dims over yaml
        mcfg["base_channels"] = state["config"].get("base_channels", mcfg["base_channels"])
    model = build_model(ModelConfig.from_dict(mcfg)).to(device).eval()
    model.load_state_dict(state["state_dict"])

    ds = ColorizationDataset(cfg["data"]["root"], image_size=int(cfg["data"]["image_size"]))
    n_val = max(1, int(len(ds) * float(cfg["data"].get("val_split", 0.1))))
    indices = range(len(ds) - n_val, len(ds)) if split == "val" else range(min(n_val, len(ds)))

    maes, mses, psnrs, ssims, des = [], [], [], [], []
    grid_rows = []
    for i in list(indices)[: int(cfg["eval"].get("max_samples", 32)) if "eval" in cfg else 32]:
        s = ds[i]
        l_np = s["l"].numpy()[0]
        ab_true = s["ab"].numpy()
        pred = model(s["l"].unsqueeze(0).to(device))[0].float().cpu().numpy()
        lab_t = np.stack([(l_np + 1) * 50, ab_true[0] * 128, ab_true[1] * 128], axis=-1)
        lab_p = np.stack([(l_np + 1) * 50, pred[0] * 128, pred[1] * 128], axis=-1)
        rgb_t = denorm_lab_to_rgb(l_np, ab_true).astype(np.float64)
        rgb_p = denorm_lab_to_rgb(l_np, pred).astype(np.float64)
        diff = rgb_p - rgb_t
        maes.append(float(np.abs(diff).mean()))
        mses.append(float((diff ** 2).mean()))
        psnrs.append(float(peak_signal_noise_ratio(rgb_t, rgb_p, data_range=255)))
        ssims.append(float(structural_similarity(rgb_t, rgb_p, channel_axis=2, data_range=255)))
        des.append(delta_e_lab(lab_t, lab_p))
        if len(grid_rows) < 8:
            gray = np.repeat(((l_np + 1) * 127.5).clip(0, 255)[..., None], 3, axis=2).astype(np.uint8)
            grid_rows.append((rgb_t.astype(np.uint8), gray, rgb_p.astype(np.uint8),
                              np.abs(diff).astype(np.uint8)))

    result = {
        "samples": len(maes),
        "mae_rgb": float(np.mean(maes)),
        "mse_rgb": float(np.mean(mses)),
        "psnr_db": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "delta_e_lab": float(np.mean(des)),
    }

    if grid_out and grid_rows:
        from PIL import Image, ImageDraw
        tile = grid_rows[0][0].shape[0]
        canvas = Image.new("RGB", (tile * 4 + 30, (tile + 24) * len(grid_rows)), "white")
        draw = ImageDraw.Draw(canvas)
        labels = ["Original RGB", "Grayscale input", "Colorized output", "Abs difference"]
        for r, row in enumerate(grid_rows):
            for c, img in enumerate(row):
                x = c * (tile + 10)
                y = r * (tile + 24)
                if r == 0:
                    draw.text((x + 2, y), labels[c], fill="black")
                canvas.paste(Image.fromarray(img), (x, y + 20 if r == 0 else y))
        Path(grid_out).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(grid_out)
        print(f"[evaluate] comparison grid written to {grid_out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--grid", default=None, help="optional path for comparison grid PNG")
    args = ap.parse_args()
    res = evaluate(args.checkpoint, args.config, args.split, args.grid)
    print("[evaluate]", ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in res.items()))
    print("[evaluate] reminder: low error != plausible color; inspect the grid visually.")


if __name__ == "__main__":
    main()
