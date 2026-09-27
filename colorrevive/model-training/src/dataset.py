"""Dataset for self-supervised colorization training.

Any folder of RGB images works (COCO, ImageNet subsets, Places365, or a
personal photo directory). Each sample:

1. Center-crops/resizes to ``image_size``.
2. Converts RGB -> CIE Lab (skimage, D65 illuminant — same as the API).
3. Returns the normalized L channel as input and normalized ab as target.

Normalization matches ``backend/app/ml/preprocessing.py``:
L in [0,1] -> [-1,1];  a,b in [-128,127] -> /128 -> ~[-1,1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from skimage.color import rgb2lab
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly corrupt scans

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def normalize_lab(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an HxWx3 float Lab image into normalized L and ab arrays."""
    l_norm = lab[..., 0] / 50.0 - 1.0          # [0,100] -> [-1,1]
    ab_norm = lab[..., 1:] / 128.0             # [-128,127] -> ~[-1,1]
    return l_norm.astype(np.float32), ab_norm.astype(np.float32)


class ColorizationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        augment: bool = False,
        max_images: int | None = None,
    ):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.augment = augment
        files = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        if not files:
            raise FileNotFoundError(f"No images found under {self.root}")
        if max_images:
            files = files[: int(max_images)]
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def _load_rgb(self, path: Path) -> np.ndarray:
        with Image.open(path) as im:
            im = im.convert("RGB")
            side = self.image_size
            # Shortest-side resize + center crop (no aspect distortion).
            w, h = im.size
            scale = side / min(w, h)
            im = im.resize((max(side, round(w * scale)), max(side, round(h * scale))), Image.BILINEAR)
            left = (im.width - side) // 2
            top = (im.height - side) // 2
            im = im.crop((left, top, left + side, top + side))
            return np.asarray(im, dtype=np.uint8)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.files[idx]
        try:
            rgb = self._load_rgb(path)
        except Exception:
            # Skip unreadable file by substituting the next one.
            return self.__getitem__((idx + 1) % len(self))
        if self.augment and np.random.rand() < 0.5:
            rgb = rgb[:, ::-1]  # random horizontal flip
        lab = rgb2lab(rgb)      # float64, L in [0,100]
        l_norm, ab_norm = normalize_lab(lab)
        l_t = torch.from_numpy(l_norm).unsqueeze(0)         # 1xHxW
        ab_t = torch.from_numpy(ab_norm).permute(2, 0, 1)   # 2xHxW
        return {"l": l_t, "ab": ab_t, "path": str(path.name)}
