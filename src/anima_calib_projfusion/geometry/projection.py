"""Camera-LiDAR projection utilities for ProjFusion.

Projects 3D point cloud coordinates into 2D image feature space
using camera intrinsics and the current extrinsic estimate.
"""

from __future__ import annotations

import torch


def project_points(
    xyz_cam: torch.Tensor,
    camera_info: dict,
    feature_hw: tuple[int, int],
) -> torch.Tensor:
    """Project 3D camera-frame points to 2D feature grid coordinates.

    Args:
        xyz_cam: [B, N, 3] points in camera frame.
        camera_info: dict with fx, fy, cx, cy, sensor_h, sensor_w (batched tensors or scalars).
        feature_hw: (H, W) of the feature map.

    Returns:
        uv: [B, N, 2] coordinates in feature grid space.
    """
    fx = camera_info["fx"]
    fy = camera_info["fy"]
    cx = camera_info["cx"]
    cy = camera_info["cy"]
    sensor_h = camera_info["sensor_h"]
    sensor_w = camera_info["sensor_w"]

    # Broadcast scalars to match batch dim
    if isinstance(fx, torch.Tensor) and fx.dim() > 0:
        fx = fx.view(-1, 1)
        fy = fy.view(-1, 1)
        cx = cx.view(-1, 1)
        cy = cy.view(-1, 1)

    z = xyz_cam[..., 2].clamp_min(1e-6)
    u_px = fx * xyz_cam[..., 0] / z + cx
    v_px = fy * xyz_cam[..., 1] / z + cy

    feat_h, feat_w = feature_hw
    sh = sensor_h if isinstance(sensor_h, int) else sensor_h
    sw = sensor_w if isinstance(sensor_w, int) else sensor_w
    u_feat = u_px * (feat_w - 1) / max(int(sw) - 1, 1)
    v_feat = v_px * (feat_h - 1) / max(int(sh) - 1, 1)
    return torch.stack((u_feat, v_feat), dim=-1)


def normalize_grid(uv: torch.Tensor, feature_hw: tuple[int, int]) -> torch.Tensor:
    """Normalize grid coordinates to [-1, 1]."""
    feat_h, feat_w = feature_hw
    u_norm = 2.0 * uv[..., 0] / max(feat_w - 1, 1) - 1.0
    v_norm = 2.0 * uv[..., 1] / max(feat_h - 1, 1) - 1.0
    return torch.stack((u_norm, v_norm), dim=-1)


def clamp_normalized_grid(grid: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    """Clamp normalized grid to [-margin, margin]."""
    return grid.clamp(min=-margin, max=margin)


def align_point_groups(
    xyz_groups: torch.Tensor,
    extrinsic: torch.Tensor,
    camera_info: dict,
    feature_hw: tuple[int, int],
    margin: float = 2.0,
) -> torch.Tensor:
    """Transform point groups by extrinsic and project to clamped feature coordinates.

    Args:
        xyz_groups: [B, G, 3] group centroids in velodyne frame.
        extrinsic: [B, 4, 4] current extrinsic estimate.
        camera_info: dict with intrinsics.
        feature_hw: (H, W) of image feature map.
        margin: clamping margin for projection (paper r_p).

    Returns:
        uv: [B, G, 2] clamped normalized projected coordinates.
    """
    # Transform to camera frame: R @ points + t
    R = extrinsic[:, :3, :3]  # [B, 3, 3]
    t = extrinsic[:, :3, 3:]  # [B, 3, 1]
    xyz_cam = (R @ xyz_groups.transpose(1, 2) + t).transpose(1, 2)  # [B, G, 3]

    uv = project_points(xyz_cam, camera_info, feature_hw)
    return clamp_normalized_grid(normalize_grid(uv, feature_hw), margin=margin)
