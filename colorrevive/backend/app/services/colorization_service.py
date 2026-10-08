"""End-to-end colorization pipeline service.

Pipeline (DDColor AI inference):
    uploaded bytes -> validate & decode -> RGB uint8
    -> DDColor neural network inference (dual-decoder chrominance prediction)
    -> combine with original luminance at native resolution
    -> optional conservative enhancements (denoise / contrast preservation)
    -> clamp -> encode PNG/JPEG
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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


# ThreadPoolExecutor to serialize inference and protect device memory + enforce timeout.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")


def run_colorization(
    rgb_u8: np.ndarray,
    settings: Settings,
    model: ColorizationModel,
    quality: str,
    preserve_contrast: bool,
    face_enhancement: bool,
    denoise_on: bool,
    output_format: str,
    chroma_strength: float | None = None,
    black_preserve: bool | None = None,
) -> ColorizeOutput:
    """Run the DDColor neural colorization pipeline with an inference timeout guard."""
    start = time.perf_counter()
    original_hw = (int(rgb_u8.shape[0]), int(rgb_u8.shape[1]))

    def _pipeline() -> tuple[np.ndarray, bool]:
        # 1. Run deep-learning colorization
        rgb_colored = model.predict(
            rgb_u8,
            quality=quality,
            chroma_strength=chroma_strength,
            black_preserve=black_preserve,
        )

        # 2. Guarantee exact original dimensions
        if rgb_colored.shape[:2] != original_hw:
            rgb_colored = restore_size(rgb_colored, original_hw)

        # 3. Optional conservative enhancements
        if denoise_on:
            rgb_colored = denoise(rgb_colored)
        if face_enhancement:
            rgb_colored = face_enhance(rgb_colored)
        if preserve_contrast and model.fallback_active():
            gray = (
                rgb_u8[..., 0] * 0.299
                + rgb_u8[..., 1] * 0.587
                + rgb_u8[..., 2] * 0.114
            ).astype(np.uint8)
            rgb_colored = preserve_source_contrast(gray, rgb_colored)

        return clamp_uint8(rgb_colored), model.fallback_active()

    try:
        future = _executor.submit(_pipeline)
        rgb_final, fallback = future.result(timeout=settings.inference_timeout_seconds)
    except TimeoutError as exc:
        raise InferenceTimeoutError() from exc

    # 4. Encode result to PNG / JPEG
    data, mime = encode_image(rgb_final, output_format)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    meta = model.metadata()
    return ColorizeOutput(
        data=data,
        mime_type=mime,
        width=int(rgb_final.shape[1]),
        height=int(rgb_final.shape[0]),
        processing_time_ms=elapsed_ms,
        model_name=str(meta.get("model_name", "DDColor")),
        model_variant=str(meta.get("model_variant", settings.ddcolor_model)),
        device=str(meta.get("device", "cpu")),
        fallback_mode=bool(fallback),
    )
