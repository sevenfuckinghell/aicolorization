"""DDColor deep-learning inference engine.

Wraps the official ICCV 2023 DDColor architecture (Alibaba DAMO Academy)
with Hugging Face Hub pre-trained weight loading, multi-variant model caching,
device management (CUDA/CPU), edge-aware guided chrominance refinement, and quality presets.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin

from ..utils.logging import get_logger, log_fields
from .ddcolor import ColorizationPipeline, DDColor, build_ddcolor_model

logger = get_logger("colorrevive.ddcolor")


class DDColorHF(DDColor, PyTorchModelHubMixin):
    """Hugging Face Hub wrapper for the official DDColor model."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if isinstance(config, dict):
            kwargs = {**config, **kwargs}
        super().__init__(**kwargs)


class DDColorEngine:
    """Thread-safe engine managing pretrained DDColor neural networks and quality presets."""

    def __init__(
        self,
        model_name: str = "piddnad/ddcolor_modelscope",
        model_dir: str = "./models",
        device_name: str = "auto",
        input_size: int = 512,
        high_input_size: int = 768,
        chroma_strength: float = 1.0,
        black_preserve: bool = True,
        edge_refinement: bool = True,
    ) -> None:
        self.model_name = self._normalize_repo(model_name)
        self.model_dir = model_dir
        self.device_name = device_name
        self.standard_input_size = input_size
        self.high_input_size = high_input_size
        self.chroma_strength = float(chroma_strength)
        self.black_preserve = bool(black_preserve)
        self.edge_refinement = bool(edge_refinement)

        self.device = self._resolve_device(device_name)
        self._models: dict[str, DDColor] = {}
        self.model: DDColor | None = None
        self._lock = threading.Lock()
        self._precision = "float32"
        self._loaded = False
        self._load_error: str | None = None

    @staticmethod
    def _normalize_repo(name: str) -> str:
        s = name.strip()
        if not s.startswith("piddnad/") and "/" not in s and not os.path.exists(s):
            return f"piddnad/{s}"
        return s

    def _resolve_device(self, name: str) -> torch.device:
        want = name.strip().lower()
        cuda_ok = torch.cuda.is_available()
        if want == "cuda":
            if not cuda_ok:
                logger.warning("CUDA requested but not available; falling back to CPU")
                return torch.device("cpu")
            return torch.device("cuda")
        if want == "auto":
            return torch.device("cuda" if cuda_ok else "cpu")
        return torch.device("cpu")

    def _load_single_model(self, repo_or_path: str, target_input_size: int) -> DDColor:
        """Load a specific DDColor model into memory."""
        local_path = Path(repo_or_path)
        cache_dir = Path(self.model_dir) if self.model_dir else None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        if local_path.is_file():
            logger.info("Loading DDColor from local checkpoint: %s", local_path)
            model_size = "tiny" if "tiny" in local_path.name.lower() else "large"
            model = build_ddcolor_model(
                DDColor,
                model_path=str(local_path),
                input_size=target_input_size,
                model_size=model_size,
                device=self.device,
            )
        else:
            repo_id = self._normalize_repo(repo_or_path)
            logger.info("Loading DDColor from Hugging Face: %s", repo_id)
            try:
                model = DDColorHF.from_pretrained(
                    repo_id,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    local_files_only=True,
                )
            except Exception:
                model = DDColorHF.from_pretrained(
                    repo_id,
                    cache_dir=str(cache_dir) if cache_dir else None,
                )
            model = model.to(self.device)

        model.eval()
        return model

    def load(self, variant: str | None = None) -> None:
        """Load the pretrained DDColor model weights into memory once."""
        target_variant = self._normalize_repo(variant or self.model_name)
        with self._lock:
            if target_variant in self._models:
                self.model = self._models[target_variant]
                self._loaded = True
                return

            logger.info("==================================================")
            logger.info("ColorRevive AI Colorization")
            logger.info("==================================================")
            logger.info("Engine: DDColor")
            logger.info("Loading model variant: %s", target_variant)
            logger.info("Device: %s", str(self.device).upper())
            logger.info("Input size: %d (standard) / %d (high)", self.standard_input_size, self.high_input_size)
            logger.info("Fallback: disabled")
            logger.info("==================================================")

            t0 = time.perf_counter()
            try:
                model = self._load_single_model(target_variant, self.standard_input_size)
                self._models[target_variant] = model
                self.model = model
                self.model_name = target_variant
                self._loaded = True
                self._load_error = None
                load_time = time.perf_counter() - t0
                log_fields(
                    logger,
                    "info",
                    "DDColor model loaded successfully",
                    model=target_variant,
                    device=str(self.device),
                    load_time_sec=round(load_time, 2),
                )
                self._warmup(model)

            except Exception as exc:
                self._loaded = False
                self._load_error = str(exc)
                logger.error("Failed to load DDColor model: %s", exc, exc_info=True)
                raise

    def get_or_load_model(self, variant: str | None = None) -> tuple[DDColor, str]:
        """Retrieve an already loaded model or load it on demand."""
        target_variant = self._normalize_repo(variant or self.model_name)
        with self._lock:
            if target_variant in self._models:
                return self._models[target_variant], target_variant
        self.load(target_variant)
        return self._models[target_variant], target_variant

    def _warmup(self, model: DDColor) -> None:
        """Run a lightweight warmup inference pass to initialize GPU/CPU kernels."""
        try:
            logger.info("Warming up DDColor model...")
            dummy_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
            pipe = ColorizationPipeline(model, input_size=256, device=self.device)
            _ = pipe.process(dummy_bgr)
            logger.info("DDColor warmup completed.")
        except Exception as exc:
            logger.warning("Warmup pass skipped due to: %s", exc)

    def is_loaded(self) -> bool:
        return self._loaded and len(self._models) > 0

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": "DDColor",
            "model_variant": self.model_name,
            "loaded": self.is_loaded(),
            "device": str(self.device),
            "precision": self._precision,
            "version": "ICCV 2023",
            "fallback_mode": False,
        }

    def predict(
        self,
        img_rgb: np.ndarray,
        quality: str = "standard",
        chroma_strength: float | None = None,
        black_preserve: bool | None = None,
        edge_refinement: bool | None = None,
        model_variant: str | None = None,
        return_metadata: bool = False,
        color_grading: str = "natural",
        sharpening: bool | None = None,
        sharpening_strength: float | None = None,
        sharpening_radius: float = 1.0,
        sharpening_threshold: float = 3.0,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Run DDColor neural colorization on an RGB image.

        Args:
            img_rgb: H x W x 3 uint8 RGB array.
            quality: 'standard' (512x512), 'high' (768x768), or 'maximum' (768x768 dual-stage).
            chroma_strength: Multiplier for chrominance intensity (default 1.0).
            black_preserve: Smoothly attenuates chrominance in deep shadows and specular highlights.
            edge_refinement: Edge-aware guided chrominance refinement using original luminance.
            model_variant: Pretrained model repository ID or checkpoint path.
            return_metadata: If True, returns (output_rgb, metadata_dict).
            color_grading: Preset name ('natural', 'vivid', 'cinematic', 'original_ai').
            sharpening: Whether to apply detail-preserving luminance sharpening.
            sharpening_strength: Sharpening boost factor (default 0.35).
            sharpening_radius: Gaussian sigma for detail extraction (default 1.0).
            sharpening_threshold: Coring threshold to suppress noise amplification (default 3.0).

        Returns:
            H x W x 3 uint8 RGB array at original image dimensions, or (array, metadata).
        """
        q_norm = quality.lower().strip()
        if q_norm in ("high", "maximum"):
            target_input_size = self.high_input_size
        else:
            target_input_size = self.standard_input_size

        # Resolve variant: if user explicitly requested a variant, use it;
        # otherwise default to loaded model or config
        effective_variant = model_variant or self.model_name
        model, used_variant = self.get_or_load_model(effective_variant)

        cs = self.chroma_strength if chroma_strength is None else float(chroma_strength)
        bp = self.black_preserve if black_preserve is None else bool(black_preserve)
        er = self.edge_refinement if edge_refinement is None else bool(edge_refinement)

        # Convert RGB to BGR for DDColor pipeline
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        with self._lock:
            pipe = ColorizationPipeline(
                model,
                input_size=target_input_size,
                device=self.device,
            )
            out_bgr, timing_meta = pipe.process(
                img_bgr,
                quality=q_norm,
                chroma_strength=cs,
                black_preserve=bp,
                edge_refinement=er,
                color_grading=color_grading,
                sharpening=sharpening,
                sharpening_strength=sharpening_strength,
                sharpening_radius=sharpening_radius,
                sharpening_threshold=sharpening_threshold,
            )

        # Convert back to RGB
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

        # Safety check: ensure output matches input dimensions
        if out_rgb.shape[:2] != img_rgb.shape[:2]:
            out_rgb = cv2.resize(
                out_rgb,
                (img_rgb.shape[1], img_rgb.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
        timing_meta["model_variant"] = used_variant
        timing_meta["device"] = str(self.device)

        if return_metadata:
            return out_rgb, timing_meta
        return out_rgb
