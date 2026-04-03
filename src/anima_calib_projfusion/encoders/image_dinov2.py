from __future__ import annotations

import torch
from torch import nn


class DINOv2ImageEncoder(nn.Module):
    """A thin wrapper that preserves the DINOv2 token contract.

    The default implementation uses a local patch embedding backbone for offline development.
    A real DINOv2 backbone can be injected later without changing downstream shapes.
    """

    def __init__(
        self,
        image_hw: tuple[int, int] = (224, 448),
        patch_size: int = 14,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.image_hw = image_hw
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def token_hw(self) -> tuple[int, int]:
        return (self.image_hw[0] // self.patch_size, self.image_hw[1] // self.patch_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[-2:]) != self.image_hw:
            raise ValueError(f"Expected image size {self.image_hw}, got {tuple(image.shape[-2:])}")
        tokens = self.patch_embed(image)
        tokens = tokens.flatten(2).transpose(1, 2)
        return self.norm(tokens)
