"""Model architecture and inference-service tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.config import Settings
from app.ml.inference import ColorizationModel
from app.ml.model import UNetConfig, build_model
from app.schemas import ModelUnavailableError


def test_unet_output_shape_and_range():
    config = UNetConfig(base_channels=16)
    model = build_model(config).eval()
    x = torch.zeros(1, 1, 64, 96)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2, 64, 96)
    assert out.min() >= -1.0 and out.max() <= 1.0  # tanh-bounded


def test_unet_handles_odd_dimensions():
    model = build_model(UNetConfig(base_channels=8)).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 33, 47))
    assert out.shape == (1, 2, 33, 47)


def test_checkpoint_rejects_garbage(tmp_path):
    from app.ml.checkpoint_loader import CheckpointError, load_checkpoint

    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch archive")
    try:
        load_checkpoint(bad, UNetConfig())
        raise AssertionError("expected CheckpointError")
    except CheckpointError:
        pass


def test_ddcolor_model_metadata():
    settings = Settings(
        colorization_engine="ddcolor",
        ddcolor_model="piddnad/ddcolor_modelscope",
        enable_fallback_mode=False,
    )
    model = ColorizationModel(settings)
    model.load()
    assert model.is_loaded() is True
    assert model.fallback_active() is False
    meta = model.metadata()
    assert meta["model_name"] == "DDColor"
    assert meta["model_variant"] == "piddnad/ddcolor_modelscope"
    assert meta["loaded"] is True
    assert meta["fallback_mode"] is False


def test_fallback_when_explicitly_enabled(tmp_path):
    settings = Settings(
        colorization_engine="ddcolor",
        ddcolor_model=str(tmp_path / "missing.pt"),
        enable_fallback_mode=True,
    )
    model = ColorizationModel(settings)
    model.load()
    assert model.fallback_active() is True
    meta = model.metadata()
    assert meta["model_name"] == "fallback-heuristic"
    assert meta["fallback_mode"] is True

    img = np.full((40, 50, 3), 128, dtype=np.uint8)
    first = model.predict(img)
    second = model.predict(img)
    assert first.shape == (40, 50, 3)
    assert np.array_equal(first, second), "fallback must be deterministic"


def test_model_unavailable_when_fallback_disabled(tmp_path):
    settings = Settings(
        colorization_engine="ddcolor",
        ddcolor_model=str(tmp_path / "missing.pt"),
        enable_fallback_mode=False,
    )
    model = ColorizationModel(settings)
    with pytest.raises(ModelUnavailableError):
        model.load()


def test_accidental_monochrome_detection():
    """Verify that the model output is not accidentally monochrome/grayscale."""
    settings = Settings(
        colorization_engine="ddcolor",
        ddcolor_model="piddnad/ddcolor_modelscope",
        enable_fallback_mode=False,
    )
    model = ColorizationModel(settings)
    model.load()

    # Create a gradient grayscale input image
    row = np.linspace(30, 220, 64, dtype=np.uint8)
    arr = np.tile(row[None, :], (64, 1))
    rgb_input = np.stack([arr, arr, arr], axis=-1)

    colored = model.predict(rgb_input, quality="standard")
    assert colored.shape == (64, 64, 3)
    assert colored.dtype == np.uint8

    # Channel differences must be significant (not grayscale where R == G == B)
    r = colored[:, :, 0].astype(int)
    g = colored[:, :, 1].astype(int)
    b = colored[:, :, 2].astype(int)
    mean_diff = (np.abs(r - g).mean() + np.abs(r - b).mean() + np.abs(g - b).mean()) / 3.0
    assert mean_diff > 2.0, f"Accidental monochrome output detected: mean channel diff={mean_diff}"
