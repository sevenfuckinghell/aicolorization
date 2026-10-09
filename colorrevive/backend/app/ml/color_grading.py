"""Photographic color grading and chrominance modulation for ColorRevive AI.

Provides genuine, color-science-based color grading presets:
- 'natural' (default): Balanced saturation, neutral whites, realistic highlights,
                       restrained shadows, clean photographic contrast.
- 'vivid': Moderately stronger color separation without neon or oversaturated clipping.
- 'cinematic': Gentle filmic toe/shoulder curves, natural color density, soft highlights.
- 'original_ai': Unmodified raw neural DDColor prediction.

Operates purely on continuous CIELAB chrominance (a, b) and luminance transfer curves.
NEVER applies fake hue shifts, hardcoded lookup tables, or arbitrary color washes.
"""

from __future__ import annotations

import numpy as np


def apply_color_grading(
    img_lab: np.ndarray,
    preset: str = "natural",
    chroma_strength: float = 1.0,
    black_preserve: bool = True,
    shadow_knee: float = 15.0,
    highlight_knee: float = 92.0,
) -> tuple[np.ndarray, dict[str, str | float]]:
    """Apply photographic color grading to a CIELAB image.

    Args:
        img_lab: H x W x 3 float32 array in CIELAB space.
                 L in [0, 100], a in [-128, 127], b in [-128, 127].
        preset: 'natural' | 'vivid' | 'cinematic' | 'original_ai'.
        chroma_strength: User saturation multiplier (default 1.0, range 0.5 - 1.2).
        black_preserve: Whether to apply shadow/highlight neutrality protection.
        shadow_knee: Luminance below which stray chroma tapers smoothly to neutral.
        highlight_knee: Luminance above which stray chroma tapers smoothly to neutral.

    Returns:
        Tuple of (graded_lab_float32, metadata_dict).
    """
    preset_clean = preset.strip().lower() if preset else "natural"
    if preset_clean not in ("natural", "vivid", "cinematic", "original_ai"):
        preset_clean = "natural"

    # Fast bypass for raw neural prediction
    if preset_clean == "original_ai":
        return img_lab.copy(), {
            "color_grading_preset": "original_ai",
            "effective_chroma_strength": 1.0,
        }

    out_lab = img_lab.copy()
    L = out_lab[..., 0]  # [0, 100]
    ab = out_lab[..., 1:]  # [..., 2]

    # 1. Base saturation scaling from user slider
    cs = max(0.2, min(2.0, float(chroma_strength)))

    if preset_clean == "natural":
        # Neutral natural scaling
        ab_scaled = ab * cs

        # Subtle photographic S-curve on luminance for subject-background separation
        # Enhances midtone definition while keeping black at 0 and white at 100
        L_norm = L / 100.0
        # Gentle Hermite contrast curve with blend factor 0.15
        L_curve = L_norm * L_norm * (3.0 - 2.0 * L_norm)
        L_graded = (L_norm * 0.85 + L_curve * 0.15) * 100.0
        out_lab[..., 0] = np.clip(L_graded, 0.0, 100.0)

    elif preset_clean == "vivid":
        # Moderate vibrancy boost (12% extra in midtones, well below neon clipping)
        vivid_boost = 1.12 * cs
        # Midtone mask: prioritize boosting midtones where color is natural
        mid_weight = np.sin(np.clip(L / 100.0, 0.0, 1.0) * np.pi) ** 1.5
        effective_mult = cs * (1.0 - mid_weight * 0.3) + (vivid_boost * mid_weight * 1.3)
        ab_scaled = ab * effective_mult[..., None]

        # Soft-knee gamut limiter to prevent digital color clipping
        chroma = np.sqrt(ab_scaled[..., 0] ** 2 + ab_scaled[..., 1] ** 2)
        knee = 75.0
        excess = chroma > knee
        if np.any(excess):
            scale = np.ones_like(chroma)
            scale[excess] = (knee + (chroma[excess] - knee) * 0.4) / chroma[excess]
            ab_scaled = ab_scaled * scale[..., None]

    elif preset_clean == "cinematic":
        # Filmic toe and shoulder: rich black depth and soft highlight roll-off
        L_norm = L / 100.0
        # Photographic film toe/shoulder
        toe_shoulder = (
            L_norm ** 1.08 * (1.0 - np.exp(-3.5 * L_norm))
            / (1.0 - np.exp(-3.5))
        )
        out_lab[..., 0] = np.clip(toe_shoulder * 100.0, 0.0, 100.0)

        # Soft color compression mimicking film emulsion saturation density
        chroma = np.sqrt(ab[..., 0] ** 2 + ab[..., 1] ** 2)
        film_mult = cs * 0.96
        ab_scaled = ab * film_mult
        film_knee = 52.0
        excess = chroma > film_knee
        if np.any(excess):
            scale = np.ones_like(chroma)
            scale[excess] = (film_knee + (chroma[excess] - film_knee) * 0.35) / chroma[excess]
            ab_scaled = ab_scaled * scale[..., None]

    # 2. Shadow & Highlight neutrality protection
    if black_preserve:
        # Smooth Hermite curve for deep shadow roll-off (prevents noise tint in blacks)
        sw = np.clip(L / max(shadow_knee, 1e-4), 0.0, 1.0)
        sf = sw * sw * (3.0 - 2.0 * sw)

        # Smooth Hermite curve for specular highlight roll-off (prevents tinted whites)
        hw = np.clip((100.0 - L) / max(100.0 - highlight_knee, 1e-4), 0.0, 1.0)
        hf = hw * hw * (3.0 - 2.0 * hw)

        ab_scaled = ab_scaled * (sf * hf)[..., None]

    out_lab[..., 1:] = ab_scaled

    metrics = {
        "color_grading_preset": preset_clean,
        "effective_chroma_strength": round(cs, 2),
    }

    return out_lab, metrics
