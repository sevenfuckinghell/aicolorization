"""DDColor deep-learning inference engine.

Wraps the official ICCV 2023 DDColor architecture (Alibaba DAMO Academy)
with Hugging Face Hub pre-trained weight loading, device management (CUDA/CPU),
precision handling, and quality-aware inference pipelines.
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
    """Thread-safe engine managing the pretrained DDColor neural network."""

    def __init__(
        self,
        model_name: str = "piddnad/ddcolor_modelscope",
        model_dir: str = "./models",
        device_name: str = "auto",
        input_size: int = 512,
        high_input_size: int = 768,
        chroma_strength: float = 1.0,
        black_preserve: bool = True,
    ) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.device_name = device_name
        self.standard_input_size = input_size
        self.high_input_size = high_input_size
        self.chroma_strength = float(chroma_strength)
        self.black_preserve = bool(black_preserve)

        self.device = self._resolve_device(device_name)
        self.model: DDColor | None = None
        self._lock = threading.Lock()
        self._precision = "float32"
        self._loaded = False
        self._load_error: str | None = None

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

    def load(self) -> None:
        """Load the pretrained DDColor model weights into memory once."""
        with self._lock:
            if self._loaded and self.model is not None:
                return

            logger.info("==================================================")
            logger.info("ColorRevive AI Colorization")
            logger.info("==================================================")
            logger.info("Engine: DDColor")
            logger.info("Model: %s", self.model_name)
            logger.info("Device: %s", str(self.device).upper())
            logger.info("Input size: %d (standard) / %d (high)", self.standard_input_size, self.high_input_size)
            logger.info("Fallback: disabled")
            logger.info("==================================================")

            t0 = time.perf_counter()
            try:
                # Check if model_name is a local path or a Hugging Face repo
                local_path = Path(self.model_name)
                cache_dir = Path(self.model_dir) if self.model_dir else None
                if cache_dir:
                    cache_dir.mkdir(parents=True, exist_ok=True)

                if local_path.is_file():
                    logger.info("Loading DDColor from local checkpoint: %s", local_path)
                    model_size = "tiny" if "tiny" in local_path.name.lower() else "large"
                    model = build_ddcolor_model(
                        DDColor,
                        model_path=str(local_path),
                        input_size=self.standard_input_size,
                        model_size=model_size,
                        device=self.device,
                    )
                else:
                    repo_id = self.model_name
                    if not repo_id.startswith("piddnad/") and "/" not in repo_id:
                        repo_id = f"piddnad/{repo_id}"

                    logger.info("Loading DDColor from Hugging Face: %s", repo_id)
                    model = DDColorHF.from_pretrained(
                        repo_id,
                        cache_dir=str(cache_dir) if cache_dir else None,
                    )
                    model = model.to(self.device)

                model.eval()
                self.model = model
                self._loaded = True
                self._load_error = None
                load_time = time.perf_counter() - t0
                log_fields(
                    logger,
                    "info",
                    "DDColor model loaded successfully",
                    model=self.model_name,
                    device=str(self.device),
                    load_time_sec=round(load_time, 2),
                )

                # Warmup inference
                self._warmup()

            except Exception as exc:
                self._loaded = False
                self._load_error = str(exc)
                logger.error("Failed to load DDColor model: %s", exc, exc_info=True)
                raise

    def _warmup(self) -> None:
        """Run a lightweight warmup inference pass to initialize GPU/CPU kernels."""
        if self.model is None:
            return
        try:
            logger.info("Warming up DDColor model...")
            dummy_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
            pipe = ColorizationPipeline(self.model, input_size=256, device=self.device)
            _ = pipe.process(dummy_bgr)
            logger.info("DDColor warmup completed.")
        except Exception as exc:
            logger.warning("Warmup pass skipped due to: %s", exc)

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

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
    ) -> np.ndarray:
        """Run DDColor neural colorization on an RGB image.

        Args:
            img_rgb: H x W x 3 uint8 RGB array.
            quality: 'standard' (512x512 inference) or 'high' (768x768 inference).
            chroma_strength: Multiplier for chrominance intensity (default from config).
            black_preserve: Whether to smoothly attenuate chrominance in deep shadows and specular highlights.

        Returns:
            H x W x 3 uint8 RGB array at original image dimensions with predicted colors.
        """
        if not self.is_loaded() or self.model is None:
            raise RuntimeError("DDColor model is not loaded.")

        target_input_size = self.high_input_size if quality.lower() == "high" else self.standard_input_size
        cs = self.chroma_strength if chroma_strength is None else float(chroma_strength)
        bp = self.black_preserve if black_preserve is None else bool(black_preserve)

        # Convert RGB to BGR for DDColor pipeline
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        with self._lock:
            pipe = ColorizationPipeline(
                self.model,
                input_size=target_input_size,
                device=self.device,
            )
            out_bgr = pipe.process(
                img_bgr,
                chroma_strength=cs,
                black_preserve=bp,
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

        return np.clip(out_rgb, 0, 255).astype(np.uint8)
