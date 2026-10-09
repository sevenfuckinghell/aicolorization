"""End-to-end colorization pipeline service.

Pipeline (DDColor AI inference):
    uploaded bytes -> validate & decode -> RGB uint8
    -> DDColor neural network inference (dual-decoder chrominance prediction)
    -> edge-aware guided chrominance refinement using native source luminance
    -> combine with original luminance at native resolution
    -> optional conservative enhancements (denoise / face enhance / contrast preservation)
    -> clamp -> encode PNG/JPEG
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Settings
from ..ml.inference import ColorizationModel
from ..ml.postprocessing import (
    clamp_uint8,
    denoise,
    encode_image,
    face_enhance,
    preserve_source_contrast,
    restore_size,
)
from ..schemas import InferenceTimeoutError


@dataclass
class ColorizeOutput:
    data: bytes
    mime_type: str
    width: int
    height: int
    processing_time_ms: int
    model_name: str
    model_variant: str
    device: str
    fallback_mode: bool
    quality_preset: str = "standard"
    edge_refinement_applied: bool = True
    shadow_protection_applied: bool = True
    color_grading_preset: str = "natural"
    sharpening_applied: bool = True
    sharpening_strength: float = 0.35
    chroma_strength: float = 1.0
    timing_breakdown: dict[str, float] = field(default_factory=dict)


# ThreadPoolExecutor to serialize inference and protect device memory + enforce timeout.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")


def run_colorization(
    rgb_u8: np.ndarray,
    settings: Settings,
    model: ColorizationModel,
    quality: str = "standard",
    preserve_contrast: bool = True,
    face_enhancement: bool = False,
    denoise_on: bool = False,
    output_format: str = "png",
    chroma_strength: float | None = None,
    black_preserve: bool | None = None,
    edge_refinement: bool | None = None,
    model_variant: str | None = None,
    color_grading: str = "natural",
    sharpening: bool | None = None,
    sharpening_strength: float | None = None,
    sharpening_radius: float = 1.0,
    sharpening_threshold: float = 3.0,
) -> ColorizeOutput:
    """Run the DDColor neural colorization pipeline with an inference timeout guard."""
    import cv2
    start = time.perf_counter()
    original_hw = (int(rgb_u8.shape[0]), int(rgb_u8.shape[1]))

    def _pipeline() -> tuple[np.ndarray, bool, dict[str, Any]]:
        # 1. Optional pre-inference noise filtering (conservative bilateral filter)
        if denoise_on:
            input_for_model = cv2.bilateralFilter(rgb_u8, d=5, sigmaColor=15, sigmaSpace=15)
        else:
            input_for_model = rgb_u8

        # 2. Run deep-learning colorization with edge-guided refinement, grading & sharpening
        res = model.predict(
            input_for_model,
            quality=quality,
            chroma_strength=chroma_strength,
            black_preserve=black_preserve,
            edge_refinement=edge_refinement,
            model_variant=model_variant,
            color_grading=color_grading,
            sharpening=sharpening,
            sharpening_strength=sharpening_strength,
            sharpening_radius=sharpening_radius,
            sharpening_threshold=sharpening_threshold,
            return_metadata=True,
        )

        if isinstance(res, tuple):
            rgb_colored, meta = res
        else:
            rgb_colored, meta = res, {}

        # 3. Guarantee exact original dimensions
        if rgb_colored.shape[:2] != original_hw:
            rgb_colored = restore_size(rgb_colored, original_hw)

        # 4. Optional conservative enhancements
        if face_enhancement:
            rgb_colored = face_enhance(rgb_colored)
        if preserve_contrast and model.fallback_active():
            gray = (
                rgb_u8[..., 0] * 0.299
                + rgb_u8[..., 1] * 0.587
                + rgb_u8[..., 2] * 0.114
            ).astype(np.uint8)
            rgb_colored = preserve_source_contrast(gray, rgb_colored)

        return clamp_uint8(rgb_colored), model.fallback_active(), meta

    try:
        future = _executor.submit(_pipeline)
        rgb_final, fallback, pipe_meta = future.result(timeout=settings.inference_timeout_seconds)
    except TimeoutError as exc:
        raise InferenceTimeoutError() from exc

    # 5. Encode result to PNG / JPEG
    data, mime = encode_image(rgb_final, output_format)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    model_meta = model.metadata()
    effective_variant = pipe_meta.get("model_variant") or model_meta.get("model_variant") or settings.ddcolor_model
    effective_device = pipe_meta.get("device") or model_meta.get("device", "cpu")

    timing_breakdown = {
        "preprocess_ms": pipe_meta.get("preprocess_ms", 0.0),
        "inference_ms": pipe_meta.get("inference_ms", 0.0),
        "refinement_ms": pipe_meta.get("refinement_ms", 0.0),
        "grading_ms": pipe_meta.get("grading_ms", 0.0),
        "sharpening_ms": pipe_meta.get("sharpening_ms", 0.0),
        "postprocess_ms": pipe_meta.get("postprocess_ms", 0.0),
        "total_ms": round(elapsed_ms, 2),
    }

    return ColorizeOutput(
        data=data,
        mime_type=mime,
        width=int(rgb_final.shape[1]),
        height=int(rgb_final.shape[0]),
        processing_time_ms=elapsed_ms,
        model_name=str(model_meta.get("model_name", "DDColor")),
        model_variant=str(effective_variant),
        device=str(effective_device),
        fallback_mode=bool(fallback),
        quality_preset=quality,
        edge_refinement_applied=bool(pipe_meta.get("edge_refinement_applied", True)),
        shadow_protection_applied=bool(pipe_meta.get("shadow_protection_applied", True)),
        color_grading_preset=str(pipe_meta.get("color_grading_preset", color_grading or "natural")),
        sharpening_applied=bool(pipe_meta.get("sharpening_applied", True)),
        sharpening_strength=float(pipe_meta.get("sharpening_strength", 0.35)),
        chroma_strength=float(pipe_meta.get("chroma_strength", 1.0)),
        timing_breakdown=timing_breakdown,
    )
