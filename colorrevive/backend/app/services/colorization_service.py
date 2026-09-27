"""End-to-end colorization pipeline service.

Pipeline (conceptual flow required by the spec):

    uploaded bytes -> validate & decode -> RGB uint8
    -> resize for model -> RGB->Lab -> normalized L
    -> model.predict (neural UNet or labeled fallback) -> normalized ab
    -> combine L + ab -> Lab->RGB -> restore original size
    -> optional denoise / contrast preservation / face enhancement
    -> clamp -> encode PNG/JPEG
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2lab

from ..config import Settings
from ..ml.inference import ColorizationModel
from ..ml.postprocessing import (
    clamp_uint8,
    combine_lab_rgb,
    denoise,
    encode_image,
    face_enhance,
    preserve_source_contrast,
    restore_size,
)
from ..ml.preprocessing import extract_normalized_l, resize_for_model
from ..schemas import InferenceTimeoutError


@dataclass
class ColorizeOutput:
    data: bytes
    mime_type: str
    width: int
    height: int
    processing_time_ms: int
    model_name: str
    fallback_mode: bool


# Single-worker executor => serialized CPU inference queue + hard timeout.
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
) -> ColorizeOutput:
    """Run the full pipeline with an inference timeout guard."""
    target_size = settings.high_input_size if quality == "high" else settings.standard_input_size
    start = time.perf_counter()

    def _pipeline() -> tuple[np.ndarray, bool]:
        # 1. Resize to model input while remembering original dimensions.
        resized, original_hw = resize_for_model(rgb_u8, target_size)

        # 2. RGB -> Lab, extract normalized L.
        lab = rgb2lab(resized)
        l_norm = extract_normalized_l(lab)

        # 3. Predict chrominance (neural when a checkpoint is loaded).
        ab_norm = model.predict(l_norm, input_size=target_size)

        # 4. Recombine using the *source* luminance so the photograph's
        #    tonal structure is never overwritten, then Lab -> RGB.
        rgb_out = combine_lab_rgb(lab[..., 0], ab_norm)

        # 5. Restore original dimensions exactly.
        rgb_out = restore_size(rgb_out, original_hw)

        # 6. Optional enhancements.
        if denoise_on:
            rgb_out = denoise(rgb_out)
        if face_enhancement:
            rgb_out = face_enhance(rgb_out)
        if preserve_contrast:
            gray = (
                rgb_u8[..., 0] * 0.299
                + rgb_u8[..., 1] * 0.587
                + rgb_u8[..., 2] * 0.114
            ).astype(np.uint8)
            rgb_out = preserve_source_contrast(gray, rgb_out)

        return clamp_uint8(rgb_out), model.fallback_active()

    try:
        future = _executor.submit(_pipeline)
        rgb_final, fallback = future.result(timeout=settings.inference_timeout_seconds)
    except TimeoutError as exc:
        raise InferenceTimeoutError() from exc

    # 7. Encode.
    data, mime = encode_image(rgb_final, output_format)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    meta = model.metadata()
    return ColorizeOutput(
        data=data,
        mime_type=mime,
        width=int(rgb_final.shape[1]),
        height=int(rgb_final.shape[0]),
        processing_time_ms=elapsed_ms,
        model_name=str(meta["model_name"]),
        fallback_mode=bool(fallback),
    )
