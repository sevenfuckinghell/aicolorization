"""Colorization model service: clean abstraction over deep-learning engines.

Exposes the unified production interface:

    class ColorizationModel:
        def load(self) -> None
        def predict(self, image: np.ndarray, quality: str = "standard", **kwargs) -> np.ndarray
        def is_loaded(self) -> bool
        def metadata(self) -> dict[str, Any]
        def fallback_active(self) -> bool

By default, loads and serves the official pre-trained DDColor neural network
(ICCV 2023). Deterministic fallback is disabled by default in production.
If the AI model cannot be loaded, a clear MODEL_UNAVAILABLE error is raised.
"""

from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np
import torch

from ..config import Settings
from ..schemas import ModelUnavailableError
from ..utils.logging import get_logger, log_fields
from .ddcolor_engine import DDColorEngine

logger = get_logger("colorrevive.model")


class ColorizationModel:
    """Thread-safe facade managing the active colorization neural network."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine_type = settings.colorization_engine.lower()
        self._lock = threading.Lock()
        self._fallback = False
        self._load_error: str | None = None

        if self.engine_type == "ddcolor":
            self._engine: DDColorEngine | None = DDColorEngine(
                model_name=settings.ddcolor_model,
                model_dir=settings.ddcolor_model_dir,
                device_name=settings.device,
                input_size=settings.standard_input_size,
                high_input_size=settings.high_input_size,
                chroma_strength=settings.color_chroma_strength,
                black_preserve=settings.color_black_preserve,
            )
        else:
            self._engine = None

    # ------------------------------------------------------------------ Lifecycle

    def load(self) -> None:
        """Load the model into memory once.

        Called at application startup. If model loading fails and fallback mode
        is disabled (the default), raises ModelUnavailableError immediately.
        """
        with self._lock:
            if self.is_loaded():
                return

            if self._engine is not None:
                try:
                    self._engine.load()
                    self._fallback = False
                    self._load_error = None
                    return
                except Exception as exc:
                    self._load_error = str(exc)
                    log_fields(
                        logger,
                        "error",
                        "failed to load DDColor model",
                        reason=str(exc),
                    )

            # If fallback is explicitly enabled, fall back safely with clear labeling
            if self.settings.enable_fallback_mode:
                self._fallback = True
                log_fields(
                    logger,
                    "warning",
                    "running in deterministic fallback mode (explicitly enabled)",
                )
                return

            # Normal path: Fail clearly when AI model is unavailable
            self._fallback = False
            raise ModelUnavailableError(
                f"The {self.settings.ddcolor_model} model could not be loaded. "
                "Check model configuration and network/model cache."
            )

    def is_loaded(self) -> bool:
        if self._engine is not None:
            return self._engine.is_loaded()
        return self._fallback

    def fallback_active(self) -> bool:
        return self._fallback or (not self.is_loaded() and self.settings.enable_fallback_mode)

    def metadata(self) -> dict[str, Any]:
        if self._engine is not None and self._engine.is_loaded():
            meta = self._engine.metadata()
            meta["fallback_mode"] = False
            return meta

        if self.fallback_active():
            return {
                "model_name": "fallback-heuristic",
                "model_variant": "deterministic-lut",
                "loaded": False,
                "device": "cpu",
                "precision": "float32",
                "version": "fallback",
                "fallback_mode": True,
            }

        return {
            "model_name": "DDColor",
            "model_variant": self.settings.ddcolor_model,
            "loaded": False,
            "device": self.settings.device,
            "precision": "float32",
            "version": "unavailable",
            "fallback_mode": False,
            "error": self._load_error,
        }

    # ------------------------------------------------------------------ Inference

    def predict(
        self,
        image: np.ndarray,
        quality: str = "standard",
        chroma_strength: float | None = None,
        black_preserve: bool | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run colorization inference.

        Args:
            image: H x W x 3 uint8 RGB array (or 2D grayscale array).
            quality: 'standard' or 'high'.
            chroma_strength: Multiplier for chrominance intensity.
            black_preserve: Smoothly attenuates chrominance in deep shadows and specular highlights.

        Returns:
            H x W x 3 uint8 RGB array with predicted colors at original dimensions.
        """
        # Ensure image is in RGB uint8 format (H, W, 3)
        rgb_input = self._ensure_rgb(image)

        if self._engine is not None and self._engine.is_loaded():
            return self._engine.predict(
                rgb_input,
                quality=quality,
                chroma_strength=chroma_strength,
                black_preserve=black_preserve,
            )

        if self.fallback_active():
            return self._predict_fallback(rgb_input)

        raise ModelUnavailableError(
            "The DDColor model could not be loaded. Check model configuration and network/model cache."
        )

    def _ensure_rgb(self, img: np.ndarray) -> np.ndarray:
        """Ensure input array is a 3-channel uint8 RGB image."""
        if img.ndim == 2:
            # Grayscale 2D
            if img.dtype != np.uint8:
                if img.min() >= -1.01 and img.max() <= 1.01:
                    img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if img.ndim == 3:
            if img.shape[2] == 1:
                return cv2.cvtColor(img.squeeze(2), cv2.COLOR_GRAY2RGB)
            if img.dtype != np.uint8:
                img = img.clip(0, 255).astype(np.uint8)
            return img

        raise ValueError(f"Invalid image array shape: {img.shape}")

    def _predict_fallback(self, rgb_u8: np.ndarray) -> np.ndarray:
        """DETERMINISTIC HEURISTIC (used ONLY when explicitly enabled)."""
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        bins = 12
        idx = np.clip((gray * bins).astype(np.int32), 0, bins - 1)
        lut_a = np.array([-0.05, -0.04, -0.03, -0.02, 0.00, 0.02,
                          0.04, 0.05, 0.05, 0.04, 0.03, 0.02], dtype=np.float32)
        lut_b = np.array([0.10, 0.16, 0.24, 0.30, 0.32, 0.30,
                          0.24, 0.16, 0.08, 0.02, -0.02, -0.05], dtype=np.float32)
        a = (lut_a[idx] * 128.0 + 128.0).astype(np.uint8)
        b = (lut_b[idx] * 128.0 + 128.0).astype(np.uint8)
        l_ch = (gray * 255.0).astype(np.uint8)
        lab = np.stack([l_ch, a, b], axis=-1)
        return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)


# Module-level singleton
_model_singleton: ColorizationModel | None = None
_singleton_lock = threading.Lock()


def get_colorization_model(settings: Settings | None = None) -> ColorizationModel:
    """Return the shared ColorizationModel instance, creating it if needed."""
    global _model_singleton
    with _singleton_lock:
        if _model_singleton is None:
            if settings is None:
                from ..config import get_settings

                settings = get_settings()
            _model_singleton = ColorizationModel(settings)
            try:
                _model_singleton.load()
            except Exception as exc:
                logger.warning("ColorizationModel startup load deferred/failed: %s", exc)
        return _model_singleton


def reset_model_singleton() -> None:
    """Reset the model singleton (used in test fixtures)."""
    global _model_singleton
    with _singleton_lock:
        _model_singleton = None
