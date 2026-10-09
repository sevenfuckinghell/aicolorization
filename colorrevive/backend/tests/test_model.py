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


def test_color_space_roundtrip():
    """Verify that OpenCV RGB to Lab and Lab to RGB conversions preserve colors accurately."""
    import cv2
    colors = [
        np.array([[[255, 0, 0]]], dtype=np.uint8),      # Pure Red
        np.array([[[0, 255, 0]]], dtype=np.uint8),      # Pure Green
        np.array([[[0, 0, 255]]], dtype=np.uint8),      # Pure Blue
        np.array([[[128, 128, 128]]], dtype=np.uint8),  # Neutral Gray
        np.array([[[210, 160, 130]]], dtype=np.uint8),  # Skin tone
    ]
    for c in colors:
        # 1. Float32 pipeline precision test (matching DDColor pipeline)
        f_bgr = cv2.cvtColor(c, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
        f_lab = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2Lab)
        f_rec_bgr = cv2.cvtColor(f_lab, cv2.COLOR_Lab2BGR)
        f_rec_rgb = (cv2.cvtColor(f_rec_bgr, cv2.COLOR_BGR2RGB) * 255.0).round().clip(0, 255).astype(np.uint8)
        f_diff = np.abs(c.astype(int) - f_rec_rgb.astype(int))
        assert f_diff.max() <= 1, f"Float32 pipeline roundtrip deviation {f_diff.max()} exceeds tolerance for {c}"

        # 2. 8-bit integer quantization check (OpenCV uint8 Lab maps [-128, 127] -> [0, 255])
        u_bgr = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
        u_lab = cv2.cvtColor(u_bgr, cv2.COLOR_BGR2Lab)
        u_rec_bgr = cv2.cvtColor(u_lab, cv2.COLOR_Lab2BGR)
        u_rec_rgb = cv2.cvtColor(u_rec_bgr, cv2.COLOR_BGR2RGB)
        u_diff = np.abs(c.astype(int) - u_rec_rgb.astype(int))
        assert u_diff.max() <= 8, f"Uint8 roundtrip deviation {u_diff.max()} exceeds integer quantization bound for {c}"


def test_guided_filter_refinement():
    """Verify that guided filter sharpens chrominance transitions along guidance edges."""
    from app.ml.ddcolor.refinement import guided_filter_channel

    # Create a step edge guidance image (0.0 on left, 1.0 on right)
    guidance = np.zeros((32, 32), dtype=np.float32)
    guidance[:, 16:] = 1.0

    # Create a blurred source chrominance ramp across the edge
    src = np.tile(np.linspace(-20, 20, 32, dtype=np.float32)[None, :], (32, 1))

    # Apply guided filter
    filtered = guided_filter_channel(guidance, src, radius=3, eps=1e-3)
    assert filtered.shape == (32, 32)
    assert filtered.dtype == np.float32

    # The step transition at column 16 should be steeper in filtered than src
    src_gradient = np.abs(src[:, 16] - src[:, 15]).mean()
    filt_gradient = np.abs(filtered[:, 16] - filtered[:, 15]).mean()
    assert filt_gradient > src_gradient, "Guided filter must sharpen chrominance along guidance edges"


def test_quality_presets_and_metadata():
    """Verify all three quality presets (standard, high, maximum) execute and return timing metadata."""
    settings = Settings(
        colorization_engine="ddcolor",
        ddcolor_model="piddnad/ddcolor_modelscope",
        enable_fallback_mode=False,
    )
    model = ColorizationModel(settings)
    model.load()

    input_img = np.full((64, 48, 3), 120, dtype=np.uint8)

    for q in ["standard", "high", "maximum"]:
        res = model.predict(input_img, quality=q, return_metadata=True)
        assert isinstance(res, tuple)
        out_img, meta = res
        assert out_img.shape == (64, 48, 3), f"Quality {q} must preserve original dimensions"
        assert meta["quality_preset"] == q
        assert "inference_ms" in meta
        assert "refinement_ms" in meta
        assert "total_ms" in meta


def test_shadow_highlight_gamut_protection():
    """Verify that deep shadows and specular highlights attenuate chrominance."""
    from app.ml.ddcolor.refinement import refine_chrominance

    # Construct an L image with deep black (L=5), midtone (L=50), and specular white (L=98)
    L = np.zeros((3, 3, 1), dtype=np.float32)
    L[0, :, 0] = 5.0   # Deep black
    L[1, :, 0] = 50.0  # Midtone
    L[2, :, 0] = 98.0  # Specular white

    ab = np.full((3, 3, 2), 30.0, dtype=np.float32)

    refined, _ = refine_chrominance(L, ab, black_preserve=True, shadow_knee=16.0, highlight_knee=94.0)

    # Chrominance in deep shadow (row 0) and specular highlight (row 2) must be significantly attenuated
    assert np.all(refined[0] < refined[1] * 0.4), "Deep shadow must attenuate chrominance"
    assert np.all(refined[2] < refined[1] * 0.4), "Specular highlight must attenuate chrominance"
