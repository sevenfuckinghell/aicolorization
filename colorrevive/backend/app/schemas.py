"""Pydantic schemas and error taxonomy shared across the API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Quality = Literal["standard", "high"]
OutputFormat = Literal["png", "jpeg"]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorBody


class ColorizeResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    filename: str
    mime_type: str
    width: int
    height: int
    processing_time_ms: int
    model: str
    model_variant: Optional[str] = None
    device: Optional[str] = None
    fallback_mode: bool
    image_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    model_name: Optional[str] = None
    model_variant: Optional[str] = None
    model_loaded: bool
    device: str
    fallback_mode: bool


class InfoResponse(BaseModel):
    name: str
    api_version: str
    supported_formats: list[str]
    max_upload_mb: int
    model_name: str
    model_variant: Optional[str] = None
    device: Optional[str] = None
    fallback_mode: bool


class ModelStatusResponse(BaseModel):
    model_name: str
    model_variant: Optional[str] = None
    loaded: bool
    device: str
    precision: str
    version: str
    fallback_mode: bool


class ValidateImageResponse(BaseModel):
    success: Literal[True] = True
    valid: bool
    width: int = 0
    height: int = 0
    format: str = ""
    mode: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

PUBLIC_ERROR_MESSAGES: dict[str, str] = {
    "INVALID_IMAGE": "The uploaded file could not be decoded as a supported image.",
    "UNSUPPORTED_TYPE": "Unsupported file type. Only JPG, JPEG, PNG, and WEBP images are accepted.",
    "FILE_TOO_LARGE": "The uploaded file exceeds the maximum allowed size.",
    "EMPTY_FILE": "The uploaded file is empty.",
    "IMAGE_TOO_LARGE": "The image contains too many pixels to process safely.",
    "MISSING_FIELDS": "Required form fields (image, quality, output_format) are missing.",
    "INVALID_SETTINGS": "One or more settings have invalid values.",
    "MODEL_UNAVAILABLE": "The DDColor model could not be loaded. Check model configuration and network/model cache.",
    "TIMEOUT": "Inference exceeded the configured time limit. Try a smaller image or Standard quality.",
    "SERVER_ERROR": "An unexpected internal error occurred. Please try again.",
}


class ApiError(Exception):
    """Domain error mapped to a safe public JSON response."""

    def __init__(self, code: str, message: str | None = None, status_code: int = 400):
        self.code = code
        self.status_code = status_code
        self.message = message or PUBLIC_ERROR_MESSAGES.get(code, PUBLIC_ERROR_MESSAGES["SERVER_ERROR"])
        super().__init__(f"{code}: {self.message}")


class ImageValidationError(ApiError):
    def __init__(self, code: str, message: str | None = None):
        super().__init__(code=code, message=message, status_code=400)


class SettingsError(ApiError):
    def __init__(self, message: str):
        super().__init__(code="INVALID_SETTINGS", message=message, status_code=400)


class ModelUnavailableError(ApiError):
    def __init__(self, message: str | None = None):
        super().__init__(code="MODEL_UNAVAILABLE", message=message, status_code=503)


class InferenceTimeoutError(ApiError):
    def __init__(self, message: str | None = None):
        super().__init__(code="TIMEOUT", message=message, status_code=504)
