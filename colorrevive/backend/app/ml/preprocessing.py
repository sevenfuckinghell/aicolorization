"""Image preprocessing for the Lab-color-space colorization pipeline.

Conventions used across the whole project:
- L channel float32 in [0, 1] (skimage style) is normalized to [-1, 1]
  for the network with ``l_norm = l * 2 - 1``.
- a and b channels live in [-128, 127] for uint8 Lab; they are normalized
  to [-1, 1] by dividing by 110 (a common empirical chroma range).
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.color import rgb2lab

from ..schemas import ImageValidationError

# Chroma normalization constant (~99th percentile of |a|, |b| in natural images).
AB_SCALE = 110.0

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def decode_upload(
    data: bytes,
    declared_mime: str | None,
    max_bytes: int,
    max_pixels: int,
) -> Image.Image:
    """Safely decode uploaded bytes into an RGB PIL image.

    Validates size, MIME type, extension-independent magic bytes, decodability
    and pixel count (decompression-bomb protection). The *decoded* format —
    not the client's claim — is authoritative.
    """
    if data is None or len(data) == 0:
        raise ImageValidationError("EMPTY_FILE")
    if len(data) > max_bytes:
        raise ImageValidationError("FILE_TOO_LARGE")

    mime = (declared_mime or "").lower().split(";")[0].strip()
    if mime and mime not in ALLOWED_MIME_TYPES:
        raise ImageValidationError("UNSUPPORTED_TYPE")

    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force full decode now, before we trust anything
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("INVALID_IMAGE") from exc

    if img.format not in ALLOWED_FORMATS:
        raise ImageValidationError("UNSUPPORTED_TYPE")

    width, height = img.size
    if width * height > max_pixels:
        raise ImageValidationError(
            "IMAGE_TOO_LARGE",
            f"The image has {width}x{height} pixels which exceeds the "
            f"processing limit of {max_pixels} pixels.",
        )

    # Apply EXIF orientation then convert to plain RGB.
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def resize_for_model(rgb: np.ndarray, target_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize an HxWx3 uint8 RGB array so its longer side equals ``target_size``.

    Returns the resized float array in [0,1] plus the original (h, w) so the
    output can be mapped back to the exact source dimensions later.
    """
    h, w = rgb.shape[:2]
    scale = target_size / max(h, w)
    new_h = max(16, int(round(h * scale)))
    new_w = max(16, int(round(w * scale)))
    img = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr, (h, w)


def rgb_to_lab(rgb_float: np.ndarray) -> np.ndarray:
    """Convert float RGB [0,1] HxWx3 to Lab HxWx3 (L in [0,100], ab ~[-128,127])."""
    return rgb2lab(rgb_float)


def extract_normalized_l(lab: np.ndarray) -> np.ndarray:
    """Return the L channel normalized to [-1, 1] as float32."""
    l_channel = lab[..., 0].astype(np.float32) / 100.0  # -> [0, 1]
    return l_channel * 2.0 - 1.0  # -> [-1, 1]


def normalize_ab(ab: np.ndarray) -> np.ndarray:
    """Normalize raw a,b channels (range approx [-128,127]) to [-1, 1]."""
    return np.clip(ab / AB_SCALE, -1.0, 1.0).astype(np.float32)


def denormalize_ab(ab_norm: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normalize_ab`."""
    return np.clip(ab_norm, -1.0, 1.0) * AB_SCALE


def l_to_tensor(l_norm: np.ndarray) -> torch.Tensor:
    """HxW float32 [-1,1] -> 1x1xHxW tensor for the network."""
    return torch.from_numpy(l_norm).unsqueeze(0).unsqueeze(0).float()
