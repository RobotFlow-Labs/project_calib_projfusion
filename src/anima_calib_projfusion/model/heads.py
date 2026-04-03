from __future__ import annotations

from torch import nn


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dims: tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.GELU()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)
