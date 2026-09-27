"""POST /api/v1/colorize and related v1 endpoints."""

from __future__ import annotations

import base64
import time

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from ..schemas import (
    ApiError,
    ColorizeResponse,
    InfoResponse,
    ModelStatusResponse,
    OutputFormat,
    Quality,
    SettingsError,
    ValidateImageResponse,
)
from ..services import image_service
from ..services.colorization_service import run_colorization
from ..services.storage_service import TempWorkspace, maybe_persist_result, sweep_expired
from ..utils.logging import get_logger, log_fields, request_id_var

router = APIRouter(prefix="/api/v1", tags=["colorize"])
logger = get_logger("colorrevive.api")

_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}


def _parse_bool(raw: str | bool, field_name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise SettingsError(f"Field '{field_name}' must be a boolean.")


@router.post("/colorize")
async def colorize(
    request: Request,
    image: UploadFile = File(...),
    quality: Quality = Form("standard"),
    preserve_contrast: str = Form("true"),
    face_enhancement: str = Form("false"),
    denoise: str = Form("false"),
    output_format: OutputFormat = Form("png"),
):
    settings = request.app.state.settings
    model = request.app.state.model
    request_id = image_service.new_request_id()
    request_id_var.set(request_id)
    start = time.perf_counter()

    try:
        preserve = _parse_bool(preserve_contrast, "preserve_contrast")
        faces = _parse_bool(face_enhancement, "face_enhancement")
        denoise_flag = _parse_bool(denoise, "denoise")

        data = await image.read()
        declared_mime = image.content_type

        with TempWorkspace(settings).session(request_id) as directory:
            img = image_service.load_image(data, declared_mime, settings)
            rgb_u8 = np.asarray(img)
            log_fields(
                logger, "info", "image decoded",
                endpoint="/api/v1/colorize",
                width=img.width, height=img.height, format=img.format,
            )

            output = run_colorization(
                rgb_u8,
                settings,
                model,
                quality=quality,
                preserve_contrast=preserve,
                face_enhancement=faces,
                denoise_on=denoise_flag,
                output_format=output_format,
            )

            safe_stem = image_service.sanitize_filename(image.filename)
            ext = "jpg" if output_format == "jpeg" else "png"
            filename = f"colorrevive-{safe_stem}-colorized.{ext}"

            # Write result inside the temp workspace so persistence &
            # cleanup share one code path.
            out_path = directory / filename
            out_path.write_bytes(output.data)
            maybe_persist_result(settings, directory, filename)

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            log_fields(
                logger, "info", "colorization completed",
                endpoint="/api/v1/colorize",
                processing_time_ms=elapsed_ms,
                width=output.width, height=output.height,
                model=output.model_name, device=model.metadata()["device"],
                fallback_mode=output.fallback_mode,
            )

            if settings.response_mode == "binary":
                from fastapi.responses import Response

                return Response(
                    content=output.data,
                    media_type=output.mime_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"',
                        "X-Request-Id": request_id,
                        "X-Fallback-Mode": str(output.fallback_mode),
                    },
                )

            # Base64 mode: bounded by MAX_UPLOAD_MB on input, so responses
            # stay within ~1.4x of that limit.
            b64 = base64.b64encode(output.data).decode("ascii")
            return ColorizeResponse(
                success=True,
                request_id=request_id,
                filename=filename,
                mime_type=output.mime_type,
                width=output.width,
                height=output.height,
                processing_time_ms=output.processing_time_ms,
                model=output.model_name,
                fallback_mode=output.fallback_mode,
                image_base64=b64,
            )
    except ApiError as exc:
        log_fields(
            logger, "warning", "colorization rejected",
            error_category=exc.code, endpoint="/api/v1/colorize",
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": request_id,
                },
            },
        )
    finally:
        sweep_expired(settings)


@router.get("/info", response_model=InfoResponse)
async def info(request: Request) -> InfoResponse:
    settings = request.app.state.settings
    meta = request.app.state.model.metadata()
    return InfoResponse(
        name=settings.app_name,
        api_version=settings.api_version,
        supported_formats=["image/jpeg", "image/png", "image/webp"],
        max_upload_mb=settings.max_upload_mb,
        model_name=str(meta["model_name"]),
        fallback_mode=bool(meta["fallback_mode"]),
    )


@router.get("/model-status", response_model=ModelStatusResponse)
async def model_status(request: Request) -> ModelStatusResponse:
    meta = request.app.state.model.metadata()
    return ModelStatusResponse(
        model_name=str(meta["model_name"]),
        loaded=bool(meta["loaded"]),
        device=str(meta["device"]),
        precision=str(meta["precision"]),
        version=str(meta["version"]),
        fallback_mode=bool(meta["fallback_mode"]),
    )


@router.post("/validate-image", response_model=ValidateImageResponse)
async def validate_image(
    request: Request, image: UploadFile = File(...)
) -> ValidateImageResponse:
    """Check file type, decodability, and dimensions without running inference."""
    settings = request.app.state.settings
    request_id_var.set(image_service.new_request_id())
    data = await image.read()
    try:
        img = image_service.load_image(data, image.content_type, settings)
        desc = image_service.describe_image(img)
        return ValidateImageResponse(success=True, valid=True, **desc)
    except ApiError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": request_id_var.get(),
                },
            },
        )
