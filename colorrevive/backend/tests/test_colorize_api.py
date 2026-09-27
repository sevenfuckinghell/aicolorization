"""Tests for POST /api/v1/colorize: happy paths, rejections, cleanup."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from .conftest import make_image_bytes


def _post(client, data: bytes, filename="photo.png", content_type="image/png", **form):
    fields = {
        "quality": form.get("quality", "standard"),
        "preserve_contrast": form.get("preserve_contrast", "true"),
        "face_enhancement": form.get("face_enhancement", "false"),
        "denoise": form.get("denoise", "false"),
        "output_format": form.get("output_format", "png"),
    }
    return client.post(
        "/api/v1/colorize",
        files={"image": (filename, data, content_type)},
        data=fields,
    )


def test_valid_png_upload_succeeds_in_fallback_mode(client):
    response = _post(client, make_image_bytes(80, 60))
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["fallback_mode"] is True  # honestly labeled
    assert body["model"] == "fallback-heuristic"
    assert body["width"] == 80 and body["height"] == 60
    assert body["mime_type"] == "image/png"
    assert body["request_id"]
    assert body["filename"].startswith("colorrevive-photo-colorized")
    pixels = base64.b64decode(body["image_base64"])
    img = Image.open(io.BytesIO(pixels))
    assert img.size == (80, 60)
    arr = np.asarray(img.convert("RGB"))
    # Output must contain real chrominance variation — not three identical channels.
    assert arr[..., 0].std() > 0
    diff_rg = np.abs(arr[..., 0].astype(int) - arr[..., 1].astype(int)).mean()
    diff_rb = np.abs(arr[..., 0].astype(int) - arr[..., 2].astype(int)).mean()
    assert diff_rg + diff_rb > 1.0, "output looks like a plain grayscale copy"


def test_jpeg_output_and_metadata(client):
    response = _post(
        client, make_image_bytes(64, 64, fmt="JPEG"),
        filename="scan.jpg", content_type="image/jpeg", output_format="jpeg",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mime_type"] == "image/jpeg"
    assert body["filename"] == "colorrevive-scan-colorized.jpg"


def test_aspect_ratio_preserved_for_wide_image(client):
    response = _post(client, make_image_bytes(300, 50))
    body = response.json()
    assert (body["width"], body["height"]) == (300, 50)


def test_invalid_extension_rejected(client):
    response = _post(client, b"GIF89a....", filename="anim.gif", content_type="image/gif")
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "UNSUPPORTED_TYPE"
    assert body["error"]["request_id"]


def test_spoofed_mime_rejected(client):
    """A text file declared as image/png must fail on decode, not pass silently."""
    response = _post(client, b"just some text, definitely not a png", filename="x.png")
    assert response.status_code == 400
    assert response.json()["error"]["code"] in {"INVALID_IMAGE", "UNSUPPORTED_TYPE"}


def test_oversized_file_rejected(client):
    big = b"\x00" * (11 * 1024 * 1024)
    response = _post(client, big, filename="big.png")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_empty_file_rejected(client):
    response = _post(client, b"", filename="empty.png")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_FILE"


def test_corrupt_image_rejected(client):
    good = bytearray(make_image_bytes(32, 32))
    good[24:60] = b"\xff" * 36  # destroy IDAT chunk header area
    response = _post(client, bytes(good), filename="broken.png")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_pixel_limit_rejects_decompression_bomb_style_input(monkeypatch):
    """Settings are read at app creation, so build a fresh app with a low limit."""
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "1000")
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    from app.ml.inference import reset_model_singleton

    reset_model_singleton()
    with TestClient(create_app()) as limited_client:
        response = _post(limited_client, make_image_bytes(200, 200))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_temp_files_cleaned_up(client):
    from app.config import get_settings

    settings = get_settings()
    before = list(Path(settings.temp_dir).glob("*")) if Path(settings.temp_dir).exists() else []
    _post(client, make_image_bytes(48, 48))
    after = list(Path(settings.temp_dir).glob("*"))
    assert len(after) == len(before), "temporary request directories must be removed"


def test_unsupported_boolean_setting_rejected(client):
    response = _post(client, make_image_bytes(), preserve_contrast="maybe")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_SETTINGS"


def test_validate_image_endpoint(client):
    response = client.post(
        "/api/v1/validate-image",
        files={"image": ("ok.png", make_image_bytes(40, 30), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["width"] == 40


def test_cors_preflight_allowed_for_configured_origin(client):
    response = client.options(
        "/api/v1/colorize",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        assert (
            response.headers.get("access-control-allow-origin")
            == "http://localhost:3000"
        )


def test_real_checkpoint_produces_nonzero_chroma(tmp_path, monkeypatch):
    """When a valid checkpoint exists, health must report model_loaded=true."""
    import torch

    from app.ml.model import UNetConfig, build_model

    config = UNetConfig()
    model = build_model(config)
    ckpt = tmp_path / "colorization.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt)

    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(ckpt))
    from app.config import get_settings

    get_settings.cache_clear()
    from app.ml.inference import ColorizationModel, reset_model_singleton

    reset_model_singleton()
    instance = ColorizationModel(get_settings())
    instance.load()
    assert instance.is_loaded() is True
    assert instance.fallback_active() is False
    meta = instance.metadata()
    assert meta["model_name"] == "unet-lab-v1"
    ab = instance.predict(np.zeros((32, 32), dtype=np.float32))
    assert ab.shape == (32, 32, 2)
    assert np.isfinite(ab).all()
    reset_model_singleton()
