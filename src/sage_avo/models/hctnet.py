"""Compact HCTNet-style comparison model.

Comparisons are valid only when this model and SAGE-AVO receive the same
low-frequency priors, inputs, splits, and evaluation masks.
"""

from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.block(inputs)


class HCTNetBaseline(nn.Module):
    """Dilated-CNN and pooled-attention residual baseline."""

    def __init__(self, input_channels: int = 3, hidden_channels: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(input_channels + 3, hidden_channels, 3, padding=1)
        self.local = nn.Sequential(*(ResidualDilatedBlock(hidden_channels, value) for value in (1, 2, 4)))
        self.attention = nn.MultiheadAttention(hidden_channels, heads, batch_first=True)
        self.output = nn.Conv2d(hidden_channels, 3, 1)

    def forward(self, avo: Tensor, low: Tensor) -> Tensor:
        features = self.local(F.relu(self.stem(torch.cat((avo, low), dim=1))))
        pooled = F.avg_pool2d(features, 4)
        batch, channels, height, width = pooled.shape
        tokens = pooled.flatten(2).transpose(1, 2)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        attended = F.interpolate(attended, size=features.shape[-2:], mode="bilinear", align_corners=False)
        return low + self.output(features + attended)
