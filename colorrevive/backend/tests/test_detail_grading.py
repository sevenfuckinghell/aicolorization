"""Unit tests for detail enhancement and photographic color grading modules."""

import numpy as np
import pytest

from app.ml.color_grading import apply_color_grading
from app.ml.detail_enhancement import apply_detail_sharpening


class TestDetailSharpening:
    """Test suite for luminance-only detail sharpening."""

    def test_sharpening_preserves_chrominance_exactly(self):
        """Sharpening MUST only modify L and leave a, b completely untouched."""
        # Create a synthetic Lab image
        H, W = 64, 64
        rng = np.random.RandomState(42)
        L = rng.uniform(20.0, 80.0, (H, W)).astype(np.float32)
        a = rng.uniform(-40.0, 40.0, (H, W)).astype(np.float32)
        b = rng.uniform(-40.0, 40.0, (H, W)).astype(np.float32)
        img_lab = np.stack([L, a, b], axis=-1)

        sharpened, metrics = apply_detail_sharpening(
            img_lab, enabled=True, strength=0.5, radius=1.0, threshold=1.0
        )

        assert metrics["sharpening_applied"] == 1.0
        # Chrominance channels must be strictly identical
        np.testing.assert_array_equal(sharpened[..., 1], img_lab[..., 1])
        np.testing.assert_array_equal(sharpened[..., 2], img_lab[..., 2])
        # Luminance channel should be modified due to noise/gradient
        assert not np.array_equal(sharpened[..., 0], img_lab[..., 0])

    def test_sharpening_disabled_or_zero_strength(self):
        """If disabled or strength is 0, array is returned untouched."""
        img_lab = np.ones((32, 32, 3), dtype=np.float32) * 50.0

        out_disabled, m1 = apply_detail_sharpening(img_lab, enabled=False)
        assert m1["sharpening_applied"] == 0.0
        np.testing.assert_array_equal(out_disabled, img_lab)

        out_zero, m2 = apply_detail_sharpening(img_lab, enabled=True, strength=0.0)
        assert m2["sharpening_applied"] == 0.0
        np.testing.assert_array_equal(out_zero, img_lab)

    def test_sharpening_bounds_luminance_to_valid_range(self):
        """Sharpening must never produce L values outside [0, 100]."""
        # Create extreme contrast edges near boundaries
        img_lab = np.zeros((32, 32, 3), dtype=np.float32)
        img_lab[:16, :, 0] = 98.0
        img_lab[16:, :, 0] = 2.0

        sharpened, _ = apply_detail_sharpening(
            img_lab, enabled=True, strength=1.5, radius=1.0, threshold=0.0
        )
        assert np.all(sharpened[..., 0] >= 0.0)
        assert np.all(sharpened[..., 0] <= 100.0)

    def test_soft_coring_suppresses_flat_regions(self):
        """Variations smaller than threshold must not be sharpened."""
        # Flat image with minuscule variation (0.5 delta, well below threshold 3.0)
        img_lab = np.full((32, 32, 3), 50.0, dtype=np.float32)
        img_lab[10, 10, 0] = 50.5  # delta is ~0.5

        sharpened, _ = apply_detail_sharpening(
            img_lab, enabled=True, strength=0.5, radius=1.0, threshold=3.0
        )
        # Difference was below threshold 3.0, so no sharpening applied
        np.testing.assert_allclose(sharpened[..., 0], img_lab[..., 0], atol=1e-4)


class TestColorGrading:
    """Test suite for CIELAB color grading presets."""

    def test_raw_ai_bypasses_all_modifications(self):
        """'raw_ai' and 'original_ai' return input array completely unchanged."""
        img_lab = np.full((32, 32, 3), 50.0, dtype=np.float32)
        img_lab[..., 1] = 20.0
        img_lab[..., 2] = -15.0

        for raw_alias in ["raw_ai", "original_ai", "raw"]:
            graded, metrics = apply_color_grading(img_lab, preset=raw_alias)
            assert metrics["color_grading_preset"] == "raw_ai"
            np.testing.assert_array_equal(graded, img_lab)

    def test_all_presets_produce_valid_outputs(self):
        """All supported presets execute cleanly without NaN or infinite values."""
        H, W = 32, 32
        rng = np.random.RandomState(42)
        img_lab = np.zeros((H, W, 3), dtype=np.float32)
        img_lab[..., 0] = rng.uniform(0.0, 100.0, (H, W))
        img_lab[..., 1] = rng.uniform(-50.0, 50.0, (H, W))
        img_lab[..., 2] = rng.uniform(-50.0, 50.0, (H, W))

        for preset in ["natural", "historical", "vivid", "cinematic", "raw_ai"]:
            graded, metrics = apply_color_grading(img_lab, preset=preset, chroma_strength=1.0)
            assert not np.isnan(graded).any()
            assert not np.isinf(graded).any()
            # Luminance must stay within [0, 100]
            assert np.all(graded[..., 0] >= 0.0)
            assert np.all(graded[..., 0] <= 100.0)

    def test_historical_preset_preserves_hue_without_sepia(self):
        """Historical preset must NOT convert blues or greens into brown/sepia."""
        # Blue pixel: negative b
        img_lab = np.zeros((1, 1, 3), dtype=np.float32)
        img_lab[0, 0, 0] = 50.0
        img_lab[0, 0, 1] = -10.0
        img_lab[0, 0, 2] = -35.0  # Cool blue

        graded, _ = apply_color_grading(img_lab, preset="historical", chroma_strength=1.0)
        # b channel must remain negative (still blue, NOT positive sepia yellow/brown)
        assert graded[0, 0, 2] < -5.0

    def test_shadow_and_highlight_neutrality(self):
        """Deep blacks (L=0) and blown highlights (L=100) must roll off chroma to 0."""
        img_lab = np.zeros((2, 2, 3), dtype=np.float32)
        # Pixel 0,0: pure black with high stray chroma
        img_lab[0, 0, 0] = 0.0
        img_lab[0, 0, 1] = 30.0
        img_lab[0, 0, 2] = -30.0
        # Pixel 1,1: pure white with high stray chroma
        img_lab[1, 1, 0] = 100.0
        img_lab[1, 1, 1] = 30.0
        img_lab[1, 1, 2] = -30.0

        graded, _ = apply_color_grading(img_lab, preset="natural", black_preserve=True)
        # Deep shadow L=0 -> chroma must be 0
        assert np.isclose(graded[0, 0, 1], 0.0, atol=1e-3)
        assert np.isclose(graded[0, 0, 2], 0.0, atol=1e-3)
        # Blown highlight L=100 -> chroma must be 0
        assert np.isclose(graded[1, 1, 1], 0.0, atol=1e-3)
        assert np.isclose(graded[1, 1, 2], 0.0, atol=1e-3)

    def test_chroma_strength_scaling(self):
        """Higher chroma strength should increase chroma magnitude."""
        img_lab = np.zeros((10, 10, 3), dtype=np.float32)
        img_lab[..., 0] = 50.0  # midtone
        img_lab[..., 1] = 20.0
        img_lab[..., 2] = 20.0

        low, _ = apply_color_grading(img_lab, preset="natural", chroma_strength=0.6)
        high, _ = apply_color_grading(img_lab, preset="natural", chroma_strength=1.2)

        chroma_low = np.linalg.norm(low[5, 5, 1:])
        chroma_high = np.linalg.norm(high[5, 5, 1:])

        assert chroma_high > chroma_low
