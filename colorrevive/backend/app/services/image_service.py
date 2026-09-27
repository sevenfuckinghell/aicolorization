"""Image handling service: safe decoding, metadata, filename sanitization."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from PIL import Image

from ..config import Settings
from ..ml.preprocessing import decode_upload
from ..schemas import ImageValidationError

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str | None) -> str:
    """Never trust the uploaded filename — keep only a safe stem."""
    if not name:
        return "image"
    stem = Path(name).stem.lower()
    stem = _UNSAFE_CHARS.sub("-", stem).strip("-.") or "image"
    return stem[:64]


def load_image(data: bytes, declared_mime: str | None, settings: Settings) -> Image.Image:
    """Validate and decode upload bytes into an RGB PIL image."""
    return decode_upload(
        data,
        declared_mime,
        max_bytes=settings.max_upload_mb * 1024 * 1024,
        max_pixels=settings.max_image_pixels,
    )


def describe_image(img: Image.Image) -> dict:
    return {
        "width": img.width,
        "height": img.height,
        "format": img.format or "",
        "mode": img.mode,
    }


def new_request_id() -> str:
    return str(uuid.uuid4())
