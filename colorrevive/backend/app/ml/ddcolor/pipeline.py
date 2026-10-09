"""Shared image colorization pipeline used by DDColor engine.

- input: BGR uint8 image (OpenCV format)
- output: BGR uint8 image (OpenCV format) at original resolution
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..color_grading import apply_color_grading
from ..detail_enhancement import apply_detail_sharpening
from .refinement import refine_chrominance


def load_checkpoint_state_dict(model_path: str, map_location="cpu"):
    """Load a checkpoint and return a state_dict.

    Supports both:
    - {'params': state_dict, ...} (common in this repo)
    - raw state_dict
    """
    ckpt = torch.load(model_path, map_location=map_location)
    if isinstance(ckpt, dict) and "params" in ckpt:
        return ckpt["params"]
    return ckpt


def build_ddcolor_model(
    model_cls,
    *,
    model_path: str,
    input_size: int = 512,
    model_size: str = "large",
    decoder_type: str = "MultiScaleColorDecoder",
    device=None,
    **kwargs,
):
    """Build a DDColor model and load weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_size not in ("tiny", "large"):
        raise ValueError(f"model_size must be 'tiny' or 'large', got: {model_size}")
    encoder_name = "convnext-t" if model_size == "tiny" else "convnext-l"

    if decoder_type == "MultiScaleColorDecoder":
        kwargs.setdefault("num_queries", 100)
        kwargs.setdefault("num_scales", 3)
        kwargs.setdefault("dec_layers", 9)
    elif decoder_type == "SingleColorDecoder":
        kwargs.setdefault("num_queries", 256)
    else:
        raise NotImplementedError(f"decoder_type not implemented: {decoder_type}")

    model = model_cls(
        encoder_name=encoder_name,
        decoder_name=decoder_type,
        input_size=[input_size, input_size],
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        **kwargs,
    )

    state_dict = load_checkpoint_state_dict(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


class ColorizationPipeline:
    """Production image colorization pipeline with edge-guided chrominance upsampling,
    natural photographic color grading, and detail-preserving luminance sharpening."""

    def __init__(self, model, *, input_size: int = 512, device=None):
        self.input_size = int(input_size)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

    def process(
        self,
        img_bgr: np.ndarray,
        quality: str = "standard",
        chroma_strength: float = 1.0,
        black_preserve: bool = True,
        edge_refinement: bool = True,
        shadow_knee: float = 15.0,
        highlight_knee: float = 92.0,
        max_chroma: float = 80.0,
        color_grading: str = "natural",
        sharpening: bool | None = None,
        sharpening_strength: float | None = None,
        sharpening_radius: float = 1.0,
        sharpening_threshold: float = 3.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run colorization pipeline on a BGR image.

        Processing Order:
        1. Preprocessing (native L extraction & model tensor construction)
        2. Neural Inference (predict low-resolution ab)
        3. High-quality bilinear upsampling of ab
        4. Optional edge-aware chrominance boundary refinement
        5. Natural color grading preset & saturation control
        6. Detail-preserving luminance sharpening (whiskers, fur, eyes)
        7. Safe clipping and BGR reconstruction

        Returns:
            Tuple of (output_bgr_uint8, metadata_dict).
        """
        if img_bgr is None:
            raise ValueError("img_bgr is None (cv2.imread failed?)")

        total_t0 = time.perf_counter()
        height, width = img_bgr.shape[:2]
        q_norm = quality.lower().strip()

        # Resolve sharpening default based on quality preset if not explicitly specified
        if sharpening is None:
            if q_norm == "standard":
                effective_sharpening = False
            else:
                effective_sharpening = True
        else:
            effective_sharpening = bool(sharpening)

        if sharpening_strength is None:
            if q_norm == "maximum":
                effective_sharp_strength = 0.45
            elif q_norm == "high":
                effective_sharp_strength = 0.35
            else:
                effective_sharp_strength = 0.25
        else:
            effective_sharp_strength = float(sharpening_strength)

        # Resolve grading default based on quality preset if not explicitly specified
        effective_grading = color_grading.strip().lower() if color_grading else "natural"
        if q_norm == "standard" and not color_grading:
            effective_grading = "original_ai"

        # -------------------------------------------------------------
        # 1. Preprocessing (Preserve original L at native resolution)
        # -------------------------------------------------------------
        pre_t0 = time.perf_counter()
        img = (img_bgr / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (H, W, 1), [0, 100]

        # Resize to model input dimensions
        img_resized = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        tensor_gray_rgb = (
            torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        preprocess_ms = (time.perf_counter() - pre_t0) * 1000.0

        # -------------------------------------------------------------
        # 2. Neural Inference
        # -------------------------------------------------------------
        inf_t0 = time.perf_counter()
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            output_ab = self.model(tensor_gray_rgb)
            if output_ab.is_cuda:
                torch.cuda.synchronize()
            output_ab = output_ab.cpu()
        inference_ms = (time.perf_counter() - inf_t0) * 1000.0

        # -------------------------------------------------------------
        # 3. High-Quality Bilinear Resizing & Edge Refinement
        # -------------------------------------------------------------
        refine_t0 = time.perf_counter()
        output_ab_resized = (
            F.interpolate(output_ab, size=(height, width), mode="bilinear", align_corners=False)[0]
            .float()
            .numpy()
            .transpose(1, 2, 0)
        )

        # Refine boundaries without double-scaling chroma (handled in grading stage)
        refined_ab, refine_timing = refine_chrominance(
            orig_l=orig_l,
            ab=output_ab_resized,
            quality=q_norm,
            edge_refinement=edge_refinement,
            chroma_strength=1.0,
            black_preserve=black_preserve,
            shadow_knee=shadow_knee,
            highlight_knee=highlight_knee,
            max_chroma=max_chroma,
        )
        refinement_ms = (time.perf_counter() - refine_t0) * 1000.0

        # -------------------------------------------------------------
        # 4. Color Grading & Saturation Modulation in CIELAB
        # -------------------------------------------------------------
        grade_t0 = time.perf_counter()
        combined_lab = np.concatenate((orig_l, refined_ab), axis=-1)
        graded_lab, grade_meta = apply_color_grading(
            combined_lab,
            preset=effective_grading,
            chroma_strength=chroma_strength,
            black_preserve=black_preserve,
            shadow_knee=shadow_knee,
            highlight_knee=highlight_knee,
        )
        grading_ms = (time.perf_counter() - grade_t0) * 1000.0

        # -------------------------------------------------------------
        # 5. Detail-Preserving Luminance Sharpening (L-only)
        # -------------------------------------------------------------
        sharp_t0 = time.perf_counter()
        sharpened_lab, sharp_meta = apply_detail_sharpening(
            graded_lab,
            enabled=effective_sharpening,
            strength=effective_sharp_strength,
            radius=sharpening_radius,
            threshold=sharpening_threshold,
        )
        sharpening_ms = (time.perf_counter() - sharp_t0) * 1000.0

        # -------------------------------------------------------------
        # 6. Safe Reconstruction to BGR uint8
        # -------------------------------------------------------------
        post_t0 = time.perf_counter()
        output_bgr = cv2.cvtColor(sharpened_lab, cv2.COLOR_LAB2BGR)
        output_img = (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)
        postprocess_ms = (time.perf_counter() - post_t0) * 1000.0

        total_ms = (time.perf_counter() - total_t0) * 1000.0

        timing_metadata = {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "refinement_ms": round(refinement_ms, 2),
            "grading_ms": round(grading_ms, 2),
            "sharpening_ms": round(sharpening_ms, 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "total_ms": round(total_ms, 2),
            "input_resolution": f"{self.input_size}x{self.input_size}",
            "output_resolution": f"{width}x{height}",
            "quality_preset": q_norm,
            "edge_refinement_applied": bool(edge_refinement),
            "color_grading_preset": effective_grading,
            "chroma_strength": float(chroma_strength),
            "shadow_protection_applied": bool(black_preserve),
            "sharpening_applied": bool(effective_sharpening),
            "sharpening_strength": round(effective_sharp_strength, 3) if effective_sharpening else 0.0,
        }

        return output_img, timing_metadata
