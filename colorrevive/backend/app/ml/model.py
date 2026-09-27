"""Baseline U-Net colorization model (Lab color space).

Input : 1 channel  — normalized L channel in [-1, 1]
Output: 2 channels — normalized a and b chrominance in [-1, 1]

The architecture is intentionally simple and CPU-friendly. A ResNet-backed
encoder can be added later by implementing another ``Encoder`` subclass
without touching the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class UNetConfig:
    input_channels: int = 1
    output_channels: int = 2
    base_channels: int = 64
    image_size: int = 256
    model_name: str = "unet-lab-v1"
    version: str = "1.0.0"
    extra: dict = field(default_factory=dict)


class DoubleConv(nn.Module):
    """Two 3x3 convolutions each followed by GroupNorm + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad/crop skip to match upsampled size for odd dimensions.
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x,
            [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetLab(nn.Module):
    """U-Net encoder-decoder mapping L -> (a, b)."""

    def __init__(self, config: UNetConfig | None = None):
        super().__init__()
        self.config = config or UNetConfig()
        bc = self.config.base_channels

        self.enc1 = EncoderBlock(self.config.input_channels, bc)
        self.enc2 = EncoderBlock(bc, bc * 2)
        self.enc3 = EncoderBlock(bc * 2, bc * 4)
        self.enc4 = EncoderBlock(bc * 4, bc * 8)

        self.bottleneck = DoubleConv(bc * 8, bc * 16)

        self.dec4 = DecoderBlock(bc * 16, bc * 8)
        self.dec3 = DecoderBlock(bc * 8, bc * 4)
        self.dec2 = DecoderBlock(bc * 4, bc * 2)
        self.dec1 = DecoderBlock(bc * 2, bc)

        self.out_conv = nn.Conv2d(bc, self.config.output_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, l_channel: torch.Tensor) -> torch.Tensor:
        s1, x = self.enc1(l_channel)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # Tanh keeps outputs bounded in [-1, 1] which matches our
        # normalized ab range and stabilizes inference.
        return self.tanh(self.out_conv(x))


def build_model(config: UNetConfig | None = None) -> UNetLab:
    return UNetLab(config)
