"""PointGPT-tiny encoder for ProjFusion.

Pure-PyTorch implementation of the PointGPT grouping + point encoder used
by ProjFusion for LiDAR feature extraction. Only the ``get_point_tokens()``
path is needed for calibration (not the GPT generative head).

Architecture from reference repo: models/pointgpt/PointGPT.py
- FPS (farthest point sampling) to select 128 group centroids
- KNN to find 64 nearest neighbors per group
- Encoder_small: Conv1d(3→128→256) + pool + Conv1d(512→512→384) → [B, 128, 384]
- Morton-like sorting of groups for sequential ordering
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _farthest_point_sample_cuda(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Farthest point sampling on GPU. Returns indices [B, num_samples]."""
    batch, num_points, _ = points.shape
    centroids = torch.zeros(batch, num_samples, dtype=torch.long, device=points.device)
    distance = torch.full((batch, num_points), float("inf"), device=points.device)
    farthest = torch.randint(num_points, (batch,), device=points.device)
    batch_idx = torch.arange(batch, device=points.device)
    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = points[batch_idx, farthest].unsqueeze(1)  # [B, 1, 3]
        dist = ((points - centroid) ** 2).sum(dim=-1)  # [B, N]
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=-1).indices
    return centroids


def _knn_gather(points: torch.Tensor, centroids: torch.Tensor, k: int) -> torch.Tensor:
    """KNN gather using cdist. Returns neighbor indices [B, G, K]."""
    # centroids: [B, G, 3], points: [B, N, 3]
    dists = torch.cdist(centroids, points)  # [B, G, N]
    _, indices = dists.topk(k, dim=-1, largest=False)  # [B, G, K]
    return indices


def _simplified_morton_sort(center: torch.Tensor, num_group: int) -> torch.Tensor:
    """Greedy nearest-neighbor ordering of group centroids (matches reference repo)."""
    batch_size = center.shape[0]
    device = center.device
    idx_base = torch.arange(0, batch_size, device=device) * num_group

    # Distance matrix between centroids
    dist = torch.cdist(center, center)  # [B, G, G]
    dist[:, torch.arange(num_group), torch.arange(num_group)] = float("inf")

    # Flatten for batch-indexed operations
    dist_flat = dist.reshape(batch_size * num_group, num_group).clone()
    sorted_indices = [idx_base.clone()]

    for _ in range(num_group - 1):
        # Pick closest centroid to last selected
        row = dist_flat[sorted_indices[-1]]  # [B, G]
        closest = row.argmin(dim=-1) + idx_base  # [B]
        sorted_indices.append(closest)
        # Mark selected as visited
        dist_flat[closest] = float("inf")

    return torch.stack(sorted_indices, dim=-1).reshape(-1)  # [B*G]


class PointEncoderSmall(nn.Module):
    """Encoder_small from PointGPT: local point cloud → group features.

    Input: [B*G, K, 3] local neighborhoods
    Output: [B*G, C] group features
    """

    def __init__(self, encoder_channel: int = 384) -> None:
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        """point_groups: [B, G, K, 3] -> [B, G, C]"""
        bs, g, k, _ = point_groups.shape
        x = point_groups.reshape(bs * g, k, 3)
        # First conv: [BG, 3, K] -> [BG, 256, K]
        feat = self.first_conv(x.transpose(2, 1))
        feat_global = feat.max(dim=2, keepdim=True)[0]  # [BG, 256, 1]
        feat = torch.cat([feat_global.expand(-1, -1, k), feat], dim=1)  # [BG, 512, K]
        feat = self.second_conv(feat)  # [BG, C, K]
        feat_global = feat.max(dim=2)[0]  # [BG, C]
        return feat_global.reshape(bs, g, self.encoder_channel)


class PointGPTEncoder(nn.Module):
    """PointGPT-tiny encoder for ProjFusion calibration.

    Groups input point cloud into 128 groups of 64 points via FPS+KNN,
    encodes each group with PointEncoderSmall, returns group centroids and features.

    Supports optional pre-trained checkpoint loading. If no checkpoint, trains end-to-end.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_groups: int = 128,
        group_size: int = 64,
        checkpoint_path: str | Path | None = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.group_size = group_size
        self.encoder = PointEncoderSmall(encoder_channel=embed_dim)

        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(Path(checkpoint_path))
            if freeze:
                self._freeze()

    def _load_checkpoint(self, path: Path) -> None:
        """Load pre-trained PointGPT encoder weights."""
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        # Handle full PointGPT checkpoint (extract encoder weights)
        encoder_state = {}
        for k, v in state.items():
            if k.startswith("GPT_Transformer.encoder."):
                new_key = k.replace("GPT_Transformer.encoder.", "")
                encoder_state[new_key] = v
            elif k.startswith("encoder."):
                new_key = k.replace("encoder.", "")
                encoder_state[new_key] = v
        if encoder_state:
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
            logger.info("Loaded PointGPT encoder from %s (missing=%d, unexpected=%d)",
                        path, len(missing), len(unexpected))
        else:
            logger.warning("No encoder keys found in %s, using random init", path)

    def _freeze(self) -> None:
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract point group tokens.

        Args:
            points: [B, N, 3] input point cloud (N >= num_groups).

        Returns:
            centroids: [B, num_groups, 3] group center coordinates.
            tokens: [B, num_groups, embed_dim] group feature tokens.
        """
        batch, num_points, _ = points.shape
        if num_points < self.num_groups:
            raise ValueError(f"Need >= {self.num_groups} points, got {num_points}")

        # FPS: select group centers
        centroid_idx = _farthest_point_sample_cuda(points, self.num_groups)  # [B, G]
        centroids = points.gather(
            1, centroid_idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # [B, G, 3]

        # KNN: find neighbors for each group
        neighbor_idx = _knn_gather(points, centroids, self.group_size)  # [B, G, K]

        # Gather neighbor points
        idx_flat = neighbor_idx.reshape(batch, -1)  # [B, G*K]
        neighbors = points.gather(
            1, idx_flat.unsqueeze(-1).expand(-1, -1, 3)
        ).reshape(batch, self.num_groups, self.group_size, 3)

        # Normalize to local coordinates
        local_coords = neighbors - centroids.unsqueeze(2)  # [B, G, K, 3]

        # Morton sorting (group ordering for sequential patterns)
        sorted_idx = _simplified_morton_sort(centroids, self.num_groups)
        local_coords = local_coords.reshape(
            batch * self.num_groups, self.group_size, 3
        )[sorted_idx].reshape(batch, self.num_groups, self.group_size, 3)
        centroids = centroids.reshape(
            batch * self.num_groups, 3
        )[sorted_idx].reshape(batch, self.num_groups, 3)

        # Encode groups
        tokens = self.encoder(local_coords)  # [B, G, C]
        return centroids, tokens
