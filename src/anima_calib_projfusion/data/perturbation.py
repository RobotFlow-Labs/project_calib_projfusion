"""Random SE(3) perturbation sampling for calibration training.

Generates random rotation + translation perturbations within specified
magnitude ranges, following ProjFusion paper §IV-A.
"""
from __future__ import annotations

import torch
import numpy as np

from anima_calib_projfusion.geometry.se3 import se3_exp


def sample_perturbation(
    batch_size: int,
    max_deg: float = 10.0,
    max_tran: float = 0.5,
    min_deg: float = 0.0,
    min_tran: float = 0.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample random SE(3) perturbation matrices.

    Args:
        batch_size: Number of perturbations to sample.
        max_deg: Maximum rotation magnitude in degrees.
        max_tran: Maximum translation magnitude in meters.
        min_deg: Minimum rotation magnitude.
        min_tran: Minimum translation magnitude.
        device: Target device.

    Returns:
        [B, 4, 4] SE(3) perturbation matrices.
    """
    max_rad = np.deg2rad(max_deg)
    min_rad = np.deg2rad(min_deg)

    # Sample rotation axis (unit sphere) and magnitude
    axis = torch.randn(batch_size, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    rot_mag = torch.rand(batch_size, 1, device=device) * (max_rad - min_rad) + min_rad
    rot_vec = axis * rot_mag  # [B, 3]

    # Sample translation direction and magnitude
    tsl_dir = torch.randn(batch_size, 3, device=device)
    tsl_dir = tsl_dir / tsl_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tsl_mag = torch.rand(batch_size, 1, device=device) * (max_tran - min_tran) + min_tran
    tsl_vec = tsl_dir * tsl_mag  # [B, 3]

    # Compose as se(3) twist and exponentiate
    xi = torch.cat([rot_vec, tsl_vec], dim=-1)  # [B, 6]
    return se3_exp(xi)
