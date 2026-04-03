from __future__ import annotations

import math

import torch
from torch import nn


class HarmonicEmbedding(nn.Module):
    def __init__(
        self,
        num_harmonic_functions: int = 6,
        omega_0: float | None = None,
        append_input: bool = True,
    ) -> None:
        super().__init__()
        self.num_harmonic_functions = num_harmonic_functions
        self.omega_0 = omega_0 if omega_0 is not None else 1.0 / 3.0
        self.append_input = append_input
        frequencies = self.omega_0 * (2.0 ** torch.arange(num_harmonic_functions, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def output_dim(self, input_dim: int) -> int:
        multiplier = 2 * self.num_harmonic_functions + (1 if self.append_input else 0)
        return input_dim * multiplier

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        embeds = []
        if self.append_input:
            embeds.append(xy)
        scaled = xy.unsqueeze(-1) * self.frequencies
        embeds.append(torch.sin(2.0 * math.pi * scaled).flatten(start_dim=-2))
        embeds.append(torch.cos(2.0 * math.pi * scaled).flatten(start_dim=-2))
        return torch.cat(embeds, dim=-1)
