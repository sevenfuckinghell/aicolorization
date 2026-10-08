"""Application configuration loaded from environment variables (.env supported)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the ColorRevive API."""

    model_config = SettingsConfigDict(
        env_file=os.environ.get("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "colorrevive-api"
    api_version: str = "1.0.0"

    # CORS — comma-separated list of allowed origins.
    cors_origins: str = Field(default="http://localhost:3000")

    # Primary ML Engine Configuration
    colorization_engine: str = "ddcolor"  # "ddcolor" | "unet"
    ddcolor_model: str = "piddnad/ddcolor_modelscope"
    ddcolor_input_size: int = 512
    ddcolor_high_size: int = 768
    ddcolor_model_dir: str = "./models"
    color_chroma_strength: float = 1.0  # 1.0 = raw DDColor prediction
    color_black_preserve: bool = True   # Attenuates chroma in deep shadows and specular highlights

    # Legacy/swappable model options
    model_checkpoint_path: str = "./checkpoints/colorization.pt"
    model_name: str = "DDColor"
    device: str = "auto"  # auto | cpu | cuda
    enable_fallback_mode: bool = False

    # Upload / safety limits
    max_upload_mb: int = 10
    max_image_pixels: int = 25_000_000
    inference_timeout_seconds: int = 120

    # Response strategy: "base64" (JSON with embedded image) or "binary".
    response_mode: str = "base64"

    # Storage / privacy
    enable_persistent_storage: bool = False
    storage_dir: str = "./storage"
    temp_dir: str = "./tmp_uploads"
    retention_seconds: int = 0  # 0 => delete immediately after response

    # Inference input sizes per quality setting
    standard_input_size: int = 512
    high_input_size: int = 768

    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def checkpoint_file(self) -> Path:
        return Path(self.model_checkpoint_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
