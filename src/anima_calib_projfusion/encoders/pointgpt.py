from __future__ import annotations

import torch
from torch import nn


def _farthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    batch, num_points, _ = points.shape
    centroids = torch.zeros(batch, num_samples, dtype=torch.long, device=points.device)
    distance = torch.full((batch, num_points), float("inf"), device=points.device)
    farthest = torch.randint(num_points, (batch,), device=points.device)
    batch_indices = torch.arange(batch, device=points.device)
    for index in range(num_samples):
        centroids[:, index] = farthest
        centroid = points[batch_indices, farthest].unsqueeze(1)
        dist = ((points - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=-1).indices
    return centroids


class PointGPTEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        num_groups: int = 128,
        group_size: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.group_size = group_size
        self.local_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        self.post_norm = nn.LayerNorm(embed_dim)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_points, _ = points.shape
        if num_points < self.num_groups:
            raise ValueError(f"Need at least {self.num_groups} points, got {num_points}")
        centroid_indices = _farthest_point_sample(points, self.num_groups)
        centroids = points.gather(1, centroid_indices.unsqueeze(-1).expand(-1, -1, 3))
        distances = torch.cdist(centroids, points)
        neighbor_indices = distances.topk(k=self.group_size, largest=False).indices
        grouped = points.unsqueeze(1).expand(batch, self.num_groups, num_points, 3)
        grouped = grouped.gather(2, neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, 3))
        local_coords = grouped - centroids.unsqueeze(2)
        tokens = self.local_mlp(local_coords).mean(dim=2)
        return centroids, self.post_norm(tokens)
