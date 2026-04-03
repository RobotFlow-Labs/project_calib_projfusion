from __future__ import annotations

from torch import nn


class MiniResAggregation(nn.Module):
    def __init__(self, in_channels: int = 384, planes: int = 96, output_dim: int = 768) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, planes, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.residual = nn.Conv2d(in_channels, planes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(planes, output_dim)

    def forward(self, feature_map):
        fused = self.stem(feature_map) + self.residual(feature_map)
        pooled = self.pool(fused).flatten(1)
        return self.proj(pooled)
