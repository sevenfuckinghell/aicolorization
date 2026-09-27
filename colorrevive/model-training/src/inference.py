"""Inference CLI: colorize a folder/file of images with a trained checkpoint.

Usage:
    python -m model_training.src.inference --checkpoint checkpoints/best.pt \
        --input path/to/image_or_folder --output out/ [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from skimage.color import rgb2lab

try:
    from .evaluate import denorm_lab_to_rgb
    from .model import ModelConfig, build_model
except ImportError:  # pragma: no cover
    from src.evaluate import denorm_lab_to_rgb  # type: ignore
    from src.model import ModelConfig, build_model  # type: ignore

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@torch.no_grad()
def colorize_file(model: torch.nn.Module, image_size: int, src: Path) -> np.ndarray:
    with Image.open(src) as im:
        original = im.convert("RGB")
    ow, oh = original.size
    side = image_size
    lab = rgb2lab(np.asarray(original.resize((side, side), Image.BILINEAR)))
    l_norm = (lab[..., 0] / 50.0 - 1.0).astype(np.float32)
    x = torch.from_numpy(l_norm)[None, None].to(next(model.parameters()).device)
    pred_ab = model(x)[0].float().cpu().numpy()
    rgb_small = denorm_lab_to_rgb(l_norm, pred_ab)
    # Restore original dimensions; keep the true L channel for sharpness.
    out = Image.fromarray(rgb_small).resize((ow, oh), Image.LANCZOS)
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="out/colorized")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8")) if Path(args.config).exists() else {}
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mcfg = dict(cfg.get("model", {}))
    if "config" in state:
        mcfg["base_channels"] = state["config"].get("base_channels", mcfg.get("base_channels", 64))
    model = build_model(ModelConfig.from_dict(mcfg))
    model.load_state_dict(state["state_dict"])
    model.eval()
    image_size = int(state.get("config", {}).get("image_size", cfg.get("data", {}).get("image_size", 256)))

    src = Path(args.input)
    files = sorted([src]) if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        rgb = colorize_file(model, image_size, f)
        dest = out_dir / (f.stem + "-colorized.png")
        Image.fromarray(rgb).save(dest)
        print(f"[inference] {f.name} -> {dest}")
    print(f"[inference] done ({len(files)} images). NOTE: quality depends entirely on the checkpoint.")


if __name__ == "__main__":
    main()
