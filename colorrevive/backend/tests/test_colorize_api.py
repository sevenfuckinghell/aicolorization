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
    for k, v in form.items():
        if k not in fields and v is not None:
            fields[k] = str(v)
    return client.post(
        "/api/v1/colorize",
        files={"image": (filename, data, content_type)},
        data=fields,
    )


def test_valid_png_upload_succeeds_with_ddcolor(client):
    response = _post(client, make_image_bytes(80, 60))
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["fallback_mode"] is False
    assert body["model"] == "DDColor"
    assert body["width"] == 80 and body["height"] == 60
    assert body["mime_type"] == "image/png"
    assert body["request_id"]
    assert body["filename"].startswith("colorrevive-photo-colorized")
    pixels = base64.b64decode(body["image_base64"])
    img = Image.open(io.BytesIO(pixels))
    assert img.size == (80, 60)
    arr = np.asarray(img.convert("RGB"))
    assert arr.shape == (60, 80, 3)


def test_jpeg_output_and_metadata(client):
    response = _post(
        client,
        make_image_bytes(64, 64, fmt="JPEG"),
        filename="scan.jpg",
        content_type="image/jpeg",
        output_format="jpeg",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mime_type"] == "image/jpeg"
    assert body["filename"] == "colorrevive-scan-colorized.jpg"


def test_corrupt_file_rejected_as_invalid_image(client):
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
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_original_dimensions_preserved_for_arbitrary_sizes(client):
    # 75x113 odd dimension test
    response = _post(client, make_image_bytes(75, 113))
    assert response.status_code == 200
    body = response.json()
    assert body["width"] == 75
    assert body["height"] == 113


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
        files={"image": ("test.png", make_image_bytes(64, 48), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["width"] == 64
    assert body["height"] == 48


def test_model_unavailable_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_FALLBACK_MODE", "false")
    monkeypatch.setenv("DDCOLOR_MODEL", str(tmp_path / "missing.pt"))
    from fastapi.testclient import TestClient
    from app.config import get_settings
    from app.main import create_app
    from app.ml.inference import reset_model_singleton

    get_settings.cache_clear()
    reset_model_singleton()
    with TestClient(create_app()) as broken_client:
        response = _post(broken_client, make_image_bytes(32, 32))
        assert response.status_code == 503
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == "MODEL_UNAVAILABLE"


def test_high_and_maximum_quality_api_requests(client):
    """Verify that quality=high and quality=maximum return valid responses with timing metadata."""
    for q in ["high", "maximum"]:
        response = _post(
            client,
            make_image_bytes(50, 50),
            quality=q,
            edge_refinement="true",
            chroma_strength="1.1",
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["quality_preset"] == q
        assert body["edge_refinement_applied"] is True
        assert "timing_breakdown" in body
        assert body["width"] == 50 and body["height"] == 50


def test_edge_refinement_toggle(client):
    """Verify edge_refinement flag is accepted and reported in output."""
    response = _post(
        client,
        make_image_bytes(40, 40),
        quality="standard",
        edge_refinement="false",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["edge_refinement_applied"] is False

