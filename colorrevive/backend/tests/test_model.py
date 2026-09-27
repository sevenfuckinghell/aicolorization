"""Model architecture and inference-service tests."""

from __future__ import annotations

import numpy as np
import torch

from app.ml.inference import ColorizationModel
from app.ml.model import UNetConfig, build_model


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


def test_fallback_is_deterministic_and_labels_itself():
    from app.config import get_settings

    model = ColorizationModel(get_settings())
    model.load()
    assert model.fallback_active() is True
    assert model.metadata()["model_name"] == "fallback-heuristic"

    l_channel = np.linspace(-1, 1, 40 * 50, dtype=np.float32).reshape(40, 50)
    first = model.predict(l_channel)
    second = model.predict(l_channel)
    assert first.shape == (40, 50, 2)
    assert np.array_equal(first, second), "fallback must be deterministic"
    # The fallback must actually predict chroma, not return zeros/gray.
    assert np.abs(first).max() > 0.01
