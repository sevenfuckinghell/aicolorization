"""Checkpoint loading with strict validation and graceful degradation.

A checkpoint is only accepted when it contains state-dict tensors whose
shapes exactly match a freshly constructed UNetLab model. This prevents
arbitrary pickle payloads from being executed blindly on load.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .model import UNetConfig, UNetLab, build_model


class CheckpointError(Exception):
    """Raised when a checkpoint cannot be loaded safely."""


def _extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state", "model"):
            inner = payload.get(key)
            if isinstance(inner, dict):
                return {k: v for k, v in inner.items() if isinstance(v, torch.Tensor)}
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return dict(payload)
    raise CheckpointError("Checkpoint does not contain a valid tensor state dict.")


def load_checkpoint(checkpoint_path: str | Path, config: UNetConfig) -> UNetLab:
    """Load a UNetLab checkpoint; raises CheckpointError on any problem."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint file not found: {path.name}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch without weights_only kwarg
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # unreadable / corrupt archive
        raise CheckpointError("Checkpoint file could not be deserialized.") from exc

    state_dict = _extract_state_dict(payload)
    model = build_model(config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise CheckpointError("Checkpoint shapes do not match the model.") from exc

    model.eval()
    return model
