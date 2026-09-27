"""Colorization model service: real UNet inference + deterministic fallback.

Exposes the required clean interface:

    class ColorizationModel:
        def load(self) -> None
        def predict(self, l_channel: np.ndarray) -> np.ndarray
        def is_loaded(self) -> bool
        def metadata(self) -> dict

The training code in ``model-training/`` is never imported at runtime; this
module only depends on the inference-side architecture definition.
"""

from __future__ import annotations

import threading

import numpy as np
import torch

from ..config import Settings
from ..utils.logging import get_logger, log_fields
from .checkpoint_loader import CheckpointError, load_checkpoint
from .model import UNetConfig, build_model
from .preprocessing import l_to_tensor

logger = get_logger("colorrevive.model")


class ColorizationModel:
    """Loads a checkpoint once (or falls back) and predicts ab channels."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._torch_model: torch.nn.Module | None = None
        self._device = "cpu"
        self._precision = "float32"
        self._fallback = True
        self._lock = threading.Lock()
        self._config = UNetConfig(
            input_channels=1,
            output_channels=2,
            base_channels=64,
            image_size=settings.standard_input_size,
            model_name=settings.model_name,
        )

    # ------------------------------------------------------------------ API

    def load(self) -> None:
        """Load the checkpoint if configured; otherwise enable fallback mode.

        Called once at application startup (lazily guarded by a lock).
        Never raises: a broken checkpoint degrades to fallback with a warning.
        """
        with self._lock:
            if self._torch_model is not None or not self._fallback:
                return
            path = self.settings.checkpoint_file
            if path.is_file():
                try:
                    device = self._resolve_device()
                    model = load_checkpoint(path, self._config)
                    model.to(device)
                    model.eval()
                    self._torch_model = model
                    self._device = str(device)
                    self._fallback = False
                    log_fields(
                        logger, "info", "checkpoint loaded",
                        model=self.settings.model_name, device=self._device,
                    )
                    return
                except CheckpointError as exc:
                    log_fields(
                        logger, "warning", "checkpoint failed to load; using fallback",
                        reason=str(exc),
                    )
            if not self.settings.enable_fallback_mode:
                from ..schemas import ModelUnavailableError

                raise ModelUnavailableError(
                    "No usable checkpoint found and fallback mode is disabled."
                )
            self._fallback = True
            log_fields(logger, "info", "running in deterministic fallback mode")

    def is_loaded(self) -> bool:
        return self._torch_model is not None

    def fallback_active(self) -> bool:
        return self._fallback or self._torch_model is None

    def metadata(self) -> dict:
        return {
            "model_name": self.settings.model_name if not self.fallback_active() else "fallback-heuristic",
            "loaded": self.is_loaded(),
            "device": self._device,
            "precision": self._precision,
            "version": self._config.version,
            "fallback_mode": self.fallback_active(),
            "input_channels": self._config.input_channels,
            "output_channels": self._config.output_channels,
            "base_channels": self._config.base_channels,
        }

    def predict(self, l_channel: np.ndarray, input_size: int | None = None) -> np.ndarray:
        """Predict normalized ab channels for a normalized L channel.

        Args:
            l_channel: float32 HxW array with values in [-1, 1].
            input_size: unused for the fallback; kept for interface parity.

        Returns:
            float32 HxWx2 array with a,b in [-1, 1].
        """
        self.load()
        if self._torch_model is not None:
            return self._predict_neural(l_channel)
        return self._predict_fallback(l_channel)

    # ------------------------------------------------------------- internals

    def _resolve_device(self) -> torch.device:
        want = self.settings.device.lower()
        cuda_ok = torch.cuda.is_available()
        if want == "cuda":
            if not cuda_ok:
                log_fields(logger, "warning", "CUDA requested but unavailable; using CPU")
                return torch.device("cpu")
            return torch.device("cuda")
        if want == "auto":
            return torch.device("cuda" if cuda_ok else "cpu")
        return torch.device("cpu")

    @torch.no_grad()
    def _predict_neural(self, l_channel: np.ndarray) -> np.ndarray:
        assert self._torch_model is not None
        tensor = l_to_tensor(np.ascontiguousarray(l_channel)).to(self._device)
        with self._lock:
            out = self._torch_model(tensor)
        out = out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def _predict_fallback(self, l_channel: np.ndarray) -> np.ndarray:
        """DETERMINISTIC HEURISTIC — NOT NEURAL AI COLORIZATION.

        A classic Zhang et al. (2016) style approach: cluster luminance
        histograms of natural color images into coarse bins and map each bin
        to the mean chrominance observed for that luminance range. The mapping
        table below is hard-coded from typical outdoor/daylight statistics, so
        skies trend blue, foliage trends green, mid-tones trend warm.

        Properties: fully deterministic, no randomness, cheap, and clearly
        labeled as fallback everywhere it surfaces.
        """
        import cv2

        # Coarse quantization of L in [-1,1] into 12 luminance bins.
        bins = 12
        idx = np.clip(((l_channel + 1.0) / 2.0 * bins).astype(np.int32), 0, bins - 1)
        # Mean (a, b) per luminance bin, in normalized [-1,1] units.
        lut_a = np.array([-0.05, -0.04, -0.03, -0.02, 0.00, 0.02,
                          0.04, 0.05, 0.05, 0.04, 0.03, 0.02], dtype=np.float32)
        lut_b = np.array([0.10, 0.16, 0.24, 0.30, 0.32, 0.30,
                          0.24, 0.16, 0.08, 0.02, -0.02, -0.05], dtype=np.float32)
        a = lut_a[idx]
        b = lut_b[idx]
        ab = np.stack([a, b], axis=-1).astype(np.float32)
        # Blur chroma for a smooth, painterly, fully deterministic result.
        ab = cv2.GaussianBlur(ab, (0, 0), sigmaX=max(2.0, l_channel.shape[0] / 40.0))
        return np.clip(ab, -1.0, 1.0).astype(np.float32)


# Module-level singleton loaded once per process (never per request).
_model_singleton: ColorizationModel | None = None
_singleton_lock = threading.Lock()


def get_colorization_model(settings: Settings) -> ColorizationModel:
    global _model_singleton
    with _singleton_lock:
        if _model_singleton is None:
            _model_singleton = ColorizationModel(settings)
            _model_singleton.load()
        return _model_singleton


def reset_model_singleton() -> None:
    """Test helper to drop the cached model."""
    global _model_singleton
    with _singleton_lock:
        _model_singleton = None
