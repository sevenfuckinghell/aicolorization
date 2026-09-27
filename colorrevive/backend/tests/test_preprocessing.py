"""Unit tests for Lab preprocessing / postprocessing utilities."""

from __future__ import annotations

import numpy as np
import pytest
from skimage.color import lab2rgb

from app.ml.postprocessing import combine_lab_rgb, encode_image, restore_size
from app.ml.preprocessing import (
    AB_SCALE,
    decode_upload,
    denormalize_ab,
    extract_normalized_l,
    normalize_ab,
    rgb_to_lab,
    resize_for_model,
)

from .conftest import make_image_bytes


def test_decode_upload_accepts_png():
    img = decode_upload(make_image_bytes(50, 40), "image/png", 10 * 1024 * 1024, 25_000_000)
    assert img.size == (50, 40)
    assert img.mode == "RGB"


def test_resize_for_model_keeps_aspect_and_returns_original_hw():
    rgb = np.random.default_rng(0).integers(0, 256, (100, 400, 3), dtype=np.uint8)
    resized, original_hw = resize_for_model(rgb, 256)
    assert original_hw == (100, 400)
    assert max(resized.shape[:2]) <= 257
    assert resized.dtype == np.float32
    assert 0.0 <= resized.min() and resized.max() <= 1.0


def test_l_normalization_range_and_shape():
    rgb = np.random.default_rng(1).random((32, 48, 3), dtype=np.float32)
    lab = rgb_to_lab(rgb)
    l_norm = extract_normalized_l(lab)
    assert l_norm.shape == (32, 48)
    assert l_norm.dtype == np.float32
    assert l_norm.min() >= -1.0 - 1e-6
    assert l_norm.max() <= 1.0 + 1e-6


def test_ab_normalize_roundtrip():
    rng = np.random.default_rng(2)
    ab = rng.uniform(-AB_SCALE, AB_SCALE, (10, 10, 2)).astype(np.float32)
    norm = normalize_ab(ab)
    assert norm.shape == (10, 10, 2)
    assert np.abs(norm).max() <= 1.0
    restored = denormalize_ab(norm)
    assert np.allclose(restored, ab, atol=1e-4)


def test_combine_lab_rgb_matches_skimage_pipeline():
    """Our uint8 Lab recombination must match the reference skimage round-trip."""
    rng = np.random.default_rng(3)
    rgb_u8 = rng.integers(0, 256, (16, 20, 3), dtype=np.uint8)
    lab = rgb_to_lab(rgb_u8.astype(np.float32) / 255.0)
    ab_norm = normalize_ab(lab[..., 1:])
    ours = combine_lab_rgb(lab[..., 0], ab_norm)
    reference = (np.clip(lab2rgb(lab), 0, 1) * 255).astype(np.uint8)
    # Allow small quantization differences between OpenCV and skimage.
    assert np.abs(ours.astype(int) - reference.astype(int)).mean() < 4.0


def test_restore_size_returns_exact_original_dimensions():
    arr = np.zeros((64, 128, 3), dtype=np.uint8)
    # original_hw is (height, width): restoring must yield 300 rows x 500 cols.
    out = restore_size(arr, (300, 500))
    assert out.shape == (300, 500, 3)


def test_encode_png_and_jpeg():
    arr = np.full((20, 30, 3), 128, dtype=np.uint8)
    data, mime = encode_image(arr, "png")
    assert mime == "image/png" and data[:8] == b"\x89PNG\r\n\x1a\n"
    data, mime = encode_image(arr, "jpeg")
    assert mime == "image/jpeg" and data[:2] == b"\xff\xd8"
