"""Post-processing: recombine channels, restore size, enhance, encode."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from .preprocessing import AB_SCALE


def combine_lab_rgb(l_0_100: np.ndarray, ab_norm: np.ndarray) -> np.ndarray:
    """Combine an L channel in [0,100] with normalized ab into uint8 RGB.

    Uses OpenCV's float32 LAB conversion (L in [0,100], a/b already offset to
    [0,255] like skimage uint8 Lab) so colors round-trip consistently.
    """
    h, w = l_0_100.shape
    ab = np.clip(ab_norm, -1.0, 1.0) * AB_SCALE
    lab_u8 = np.zeros((h, w, 3), dtype=np.uint8)
    lab_u8[..., 0] = np.clip(l_0_100 * 255.0 / 100.0, 0, 255).astype(np.uint8)
    lab_u8[..., 1] = np.clip(ab[..., 0] + 128.0, 0, 255).astype(np.uint8)
    lab_u8[..., 2] = np.clip(ab[..., 1] + 128.0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)
    return rgb


def restore_size(rgb_u8: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    """Resize model-resolution RGB back to the original image dimensions."""
    oh, ow = original_hw
    img = Image.fromarray(rgb_u8).resize((ow, oh), Image.LANCZOS)
    return np.asarray(img)


def denoise(rgb_u8: np.ndarray, strength: float = 5.0) -> np.ndarray:
    """Light non-local-means denoising on the color image."""
    return cv2.fastNlMeansDenoisingColored(rgb_u8, None, strength, strength, 7, 21)


def preserve_source_contrast(orig_gray_u8: np.ndarray, out_rgb_u8: np.ndarray) -> np.ndarray:
    """Reinforce the luminance of the source photo inside the colorized output.

    Blends 70% of the original grayscale luminance back over the colorized
    result via a low-frequency transfer so fine predicted colors survive while
    the contrast/tone of the photograph is preserved.
    """
    out_lab = cv2.cvtColor(out_rgb_u8, cv2.COLOR_RGB2LAB)
    orig_lab = cv2.cvtColor(
        cv2.cvtColor(orig_gray_u8, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2LAB
    )
    blended_l = cv2.addWeighted(out_lab[..., 0], 0.3, orig_lab[..., 0], 0.7, 0)
    out_lab[..., 0] = blended_l
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)


def face_enhance(rgb_u8: np.ndarray, cascade_path: str | None = None) -> np.ndarray:
    """Experimental chroma smoothing inside Haar-detected faces.

    Falls back to returning the image unchanged when no cascade is available
    or no faces are detected. Never raises.
    """
    try:
        path = cascade_path or cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(path)
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return rgb_u8
        lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        mask = np.zeros(lab.shape[:2], dtype=np.float32)
        for x, y, fw, fh in faces:
            cv2.rectangle(mask, (x, y), (x + fw, y + fh), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2, fw // 8))
        smoothed = lab.copy()
        smoothed[..., 1:] = cv2.GaussianBlur(lab[..., 1:], (0, 0), sigmaX=3)
        blended = lab * (1 - mask[..., None] * 0.6) + smoothed * (mask[..., None] * 0.6)
        return cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    except Exception:
        return rgb_u8


def clamp_uint8(rgb_u8: np.ndarray) -> np.ndarray:
    return np.clip(rgb_u8, 0, 255).astype(np.uint8)


def encode_image(rgb_u8: np.ndarray, output_format: str, jpeg_quality: int = 92) -> tuple[bytes, str]:
    """Encode RGB uint8 array as PNG or JPEG bytes. Returns (bytes, mime_type)."""
    img = Image.fromarray(clamp_uint8(rgb_u8))
    buf = io.BytesIO()
    if output_format == "jpeg":
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return buf.getvalue(), mime
