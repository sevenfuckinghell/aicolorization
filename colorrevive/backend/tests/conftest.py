"""Shared pytest fixtures for the ColorRevive backend test-suite."""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point all writable paths and model config at a clean tmp directory."""
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "tmp_uploads"))
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("ENABLE_PERSISTENT_STORAGE", "false")
    monkeypatch.setenv("ENABLE_FALLBACK_MODE", "false")
    monkeypatch.setenv("MAX_UPLOAD_MB", "10")
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "25000000")
    monkeypatch.setenv("DEVICE", "cpu")
    from app.config import get_settings

    get_settings.cache_clear()
    from app.ml.inference import reset_model_singleton

    reset_model_singleton()
    yield
    get_settings.cache_clear()
    reset_model_singleton()


def make_image_bytes(
    width: int = 64,
    height: int = 48,
    fmt: str = "PNG",
    gradient: bool = True,
) -> bytes:
    """Generate deterministic grayscale-ish test images."""
    if gradient:
        row = np.linspace(30, 220, width, dtype=np.uint8)
        arr = np.tile(row[None, :], (height, 1))
        img = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        img = Image.new("RGB", (width, height), (120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def png_bytes() -> bytes:
    return make_image_bytes(fmt="PNG")


@pytest.fixture()
def jpeg_bytes() -> bytes:
    return make_image_bytes(fmt="JPEG")


@pytest.fixture()
def client(isolated_env):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
