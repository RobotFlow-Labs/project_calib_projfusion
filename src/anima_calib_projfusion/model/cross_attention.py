from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ScaleFreeCrossAttention(nn.Module):
    def __init__(
        self, embed_dim: int = 384, num_heads: int = 6, pos_dim: int | None = None
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.pos_dim = pos_dim or embed_dim
        self.image_pos_proj = nn.Linear(self.pos_dim, embed_dim)
        self.point_pos_proj = nn.Linear(self.pos_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        tensor = tensor.view(batch, length, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def forward(
        self,
        image_tokens: torch.Tensor,
        point_tokens: torch.Tensor,
        image_pos: torch.Tensor,
        point_pos: torch.Tensor,
    ) -> torch.Tensor:
        query = self.q_proj(image_tokens + self.image_pos_proj(image_pos))
        key = self.k_proj(point_tokens + self.point_pos_proj(point_pos))
        value = self.v_proj(point_tokens)
        query = F.normalize(self._reshape_heads(query), dim=-1)
        key = F.normalize(self._reshape_heads(key), dim=-1)
        value = self._reshape_heads(value)
        attention = torch.softmax(query @ key.transpose(-1, -2), dim=-1)
        fused = attention @ value
        fused = fused.transpose(1, 2).reshape(
            image_tokens.shape[0], image_tokens.shape[1], self.embed_dim
        )
        fused = self.norm_1(image_tokens + self.out_proj(fused))
        return self.norm_2(fused + self.ff(fused))
