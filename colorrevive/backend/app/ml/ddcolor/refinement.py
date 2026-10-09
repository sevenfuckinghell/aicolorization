"""Edge-aware chrominance refinement for neural image colorization.

Applies fast guided filtering using the source image's high-resolution luminance
channel (L) to guide the upsampled low-frequency predicted chrominance (ab).
This eliminates color bleeding across sharp boundaries (e.g., fur silhouettes,
hair strands, whiskers, pupil/iris contours, clothing folds) while preserving
100% of the original spatial and luminance details.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np


def guided_filter_channel(
    guidance: np.ndarray,
    src: np.ndarray,
    radius: int = 4,
    eps: float = 1e-3,
) -> np.ndarray:
    """Fast edge-preserving guided filter (He et al., ECCV 2010 / TPAMI 2013).

    Args:
        guidance: 2D float32 array in [0, 1] (normalized luminance channel).
        src: 2D or 3D float32 array (chrominance a or b channel).
        radius: Local filter window radius (window size = 2 * radius + 1).
        eps: Regularization parameter penalizing large variance in guidance.

    Returns:
        Refined float32 array with boundary alignment matching the guidance image.
    """
    r = max(1, int(radius))
    ksize = (2 * r + 1, 2 * r + 1)

    mean_I = cv2.boxFilter(guidance, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize)

    if src.ndim == 3:
        mean_Ip = cv2.boxFilter(guidance[..., None] * src, cv2.CV_32F, ksize)
        cov_Ip = mean_Ip - (mean_I[..., None] * mean_p)
        mean_II = cv2.boxFilter(guidance * guidance, cv2.CV_32F, ksize)
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I[..., None] + eps)
        b = mean_p - (a * mean_I[..., None])
        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
        q = (mean_a * guidance[..., None]) + mean_b
    else:
        mean_Ip = cv2.boxFilter(guidance * src, cv2.CV_32F, ksize)
        cov_Ip = mean_Ip - (mean_I * mean_p)
        mean_II = cv2.boxFilter(guidance * guidance, cv2.CV_32F, ksize)
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - (a * mean_I)
        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
        q = (mean_a * guidance) + mean_b

    return q.astype(np.float32)


def refine_chrominance(
    orig_l: np.ndarray,
    ab: np.ndarray,
    quality: str = "standard",
    edge_refinement: bool = True,
    chroma_strength: float = 1.0,
    black_preserve: bool = True,
    shadow_knee: float = 16.0,
    highlight_knee: float = 94.0,
    max_chroma: float = 80.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Refine and scale predicted chrominance channels with edge and gamut preservation.

    Args:
        orig_l: H x W x 1 float32 array in [0, 100] (original luminance).
        ab: H x W x 2 float32 array (predicted chrominance).
        quality: 'standard', 'high', or 'maximum'.
        edge_refinement: Whether to apply edge-aware guided filtering.
        chroma_strength: Multiplier for chrominance magnitude (1.0 = raw prediction).
        black_preserve: Whether to smoothly attenuate chrominance in deep shadows and highlights.
        shadow_knee: Luminance threshold below which chrominance tapers to neutral.
        highlight_knee: Luminance threshold above which chrominance tapers to neutral.
        max_chroma: Maximum allowable chrominance magnitude before soft compression.

    Returns:
        Tuple of (refined_ab, timing_dict).
    """
    t0 = time.perf_counter()
    refined_ab = ab.copy().astype(np.float32)
    guidance = (orig_l[..., 0] / 100.0).clip(0.0, 1.0).astype(np.float32)

    # 1. Edge-aware guided filtering
    q_lower = quality.lower()
    if edge_refinement:
        if q_lower == "maximum":
            # Two-stage multi-scale guided filtering: broad field smoothing followed by fine edge snapping
            refined_ab = guided_filter_channel(guidance, refined_ab, radius=6, eps=2e-3)
            refined_ab = guided_filter_channel(guidance, refined_ab, radius=3, eps=5e-4)
        elif q_lower == "high":
            refined_ab = guided_filter_channel(guidance, refined_ab, radius=4, eps=1e-3)
        else:
            # standard
            refined_ab = guided_filter_channel(guidance, refined_ab, radius=3, eps=1e-3)

    # 2. Configurable chrominance strength scaling
    if chroma_strength != 1.0:
        refined_ab *= max(0.0, float(chroma_strength))

    # 3. Luminance-aware gamut & neutral preservation
    if black_preserve:
        L = orig_l[..., 0]  # [0, 100]

        # Smooth cubic Hermite curve for shadow taper (L -> 0)
        sw = np.clip(L / max(shadow_knee, 1e-4), 0.0, 1.0)
        sf = sw * sw * (3.0 - 2.0 * sw)

        # Smooth cubic Hermite curve for highlight taper (L -> 100)
        hw = np.clip((100.0 - L) / max(100.0 - highlight_knee, 1e-4), 0.0, 1.0)
        hf = hw * hw * (3.0 - 2.0 * hw)

        refined_ab = refined_ab * (sf * hf)[..., None]

        # Soft chroma cap to prevent extreme out-of-gamut Lab clipping
        chroma = np.sqrt(refined_ab[..., 0] ** 2 + refined_ab[..., 1] ** 2)
        excess = chroma > max_chroma
        if np.any(excess):
            scale = np.ones_like(chroma)
            scale[excess] = max_chroma / chroma[excess]
            refined_ab = refined_ab * scale[..., None]

    duration_ms = (time.perf_counter() - t0) * 1000.0
    return refined_ab.astype(np.float32), {"refinement_ms": round(duration_ms, 2)}
