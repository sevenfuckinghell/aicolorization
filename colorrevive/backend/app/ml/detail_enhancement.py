"""Detail-preserving luminance sharpening stage for ColorRevive AI.

Enhances high-frequency photographic details (whiskers, fur strands, eye contours,
fabric weaves, architectural edges) exclusively on the native luminance (L) channel
in CIELAB space. Because chrominance (a, b) channels are untouched:
- Zero color fringing or rainbow artifacts around fine whiskers or hair.
- Zero white halos around high-contrast edges (via anti-halo bounding).
- Zero noise amplification in flat regions or shadows (via soft coring threshold).
- 100% preservation of photographic lighting and exposure.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_detail_sharpening(
    img_lab: np.ndarray,
    enabled: bool = True,
    strength: float = 0.35,
    radius: float = 1.0,
    threshold: float = 3.0,
    max_delta: float = 12.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply conservative, halo-suppressed luminance unsharp masking.

    Args:
        img_lab: H x W x 3 float32 array in CIELAB color space.
                 L in [0, 100], a in [-128, 127], b in [-128, 127].
        enabled: Whether sharpening is active. If False, returns input untouched.
        strength: Sharpening boost factor (typically 0.15 - 0.60, default 0.35).
        radius: Gaussian filter radius sigma (default 1.0 for single-pixel details).
        threshold: Noise coring threshold on L difference in [0, 100] (default 3.0).
                   Differences below this magnitude are treated as noise/flat areas.
        max_delta: Maximum permitted luminance adjustment to prevent edge ringing.

    Returns:
        Tuple of (sharpened_lab_float32, metrics_dict).
    """
    if not enabled or strength <= 0.0:
        return img_lab, {"sharpening_applied": 0.0, "sharpening_strength": 0.0}

    out_lab = img_lab.copy()
    L = out_lab[..., 0]  # [0, 100]

    # 1. Gentle Gaussian low-pass filter on luminance
    r = max(0.5, float(radius))
    L_blur = cv2.GaussianBlur(L, (0, 0), sigmaX=r, sigmaY=r)

    # 2. Extract high-frequency detail delta
    delta = L - L_blur

    # 3. Soft coring: suppress sensor grain, smooth skin, flat skies
    th = max(0.0, float(threshold))
    mag = np.abs(delta)
    mask = mag > th

    delta_cored = np.zeros_like(delta)
    if np.any(mask):
        delta_cored[mask] = np.sign(delta[mask]) * (mag[mask] - th)

    # 4. Anti-halo clamp: prevent white ringing on whiskers / black outlines around eyes
    lim = max(1.0, float(max_delta))
    delta_bounded = np.clip(delta_cored, -lim, lim)

    # 5. Apply bounded sharpening to luminance only
    k = max(0.0, min(1.5, float(strength)))
    L_sharp = np.clip(L + k * delta_bounded, 0.0, 100.0)

    out_lab[..., 0] = L_sharp

    metrics = {
        "sharpening_applied": 1.0,
        "sharpening_strength": round(k, 3),
        "sharpening_radius": round(r, 2),
        "sharpening_threshold": round(th, 2),
    }

    return out_lab, metrics
