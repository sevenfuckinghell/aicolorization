"""Loss functions for colorization training.

Starts with plain L1 on the ab channels (the recommended baseline: it avoids
the gray-collapse that L2 causes). A perceptual-loss hook is provided so VGG
feature loss can be added later without touching the training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ab_l1_loss(pred_ab: torch.Tensor, target_ab: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between predicted and target normalized ab."""
    return F.l1_loss(pred_ab, target_ab)


class CombinedLoss(nn.Module):
    """Weighted L1 (+ optional perceptual term reserved for future use)."""

    def __init__(self, l1_weight: float = 1.0, perceptual_weight: float = 0.0):
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.perceptual_weight = float(perceptual_weight)
        self._perceptual: nn.Module | None = None
        if self.perceptual_weight > 0:
            from torchvision.models import vgg16, VGG16_Weights  # lazy import

            features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
            for p in features.parameters():
                p.requires_grad = False
            self._perceptual = features.eval()

    def forward(self, pred_ab: torch.Tensor, target_ab: torch.Tensor) -> torch.Tensor:
        loss = self.l1_weight * ab_l1_loss(pred_ab, target_ab)
        if self.perceptual_weight > 0 and self._perceptual is not None:
            # NOTE: requires decoding lab->rgb first; documented extension point.
            pass
        return loss
