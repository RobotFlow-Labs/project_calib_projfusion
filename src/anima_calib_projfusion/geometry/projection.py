from __future__ import annotations

import torch

from anima_calib_projfusion.data.camera_info import CameraInfo
from anima_calib_projfusion.geometry.se3 import apply_transform


def _as_camera_info(camera_info: CameraInfo | dict[str, float | int]) -> CameraInfo:
    if isinstance(camera_info, CameraInfo):
        return camera_info
    return CameraInfo.from_mapping(camera_info)


def project_points(
    xyz_cam: torch.Tensor,
    camera_info: CameraInfo | dict[str, float | int],
    feature_hw: tuple[int, int],
) -> torch.Tensor:
    camera = _as_camera_info(camera_info)
    z = xyz_cam[..., 2].clamp_min(1e-6)
    u_px = camera.fx * xyz_cam[..., 0] / z + camera.cx
    v_px = camera.fy * xyz_cam[..., 1] / z + camera.cy
    feat_h, feat_w = feature_hw
    u_feat = u_px * (feat_w - 1) / max(camera.sensor_w - 1, 1)
    v_feat = v_px * (feat_h - 1) / max(camera.sensor_h - 1, 1)
    return torch.stack((u_feat, v_feat), dim=-1)


def normalize_grid(uv: torch.Tensor, feature_hw: tuple[int, int]) -> torch.Tensor:
    feat_h, feat_w = feature_hw
    u_norm = 2.0 * uv[..., 0] / max(feat_w - 1, 1) - 1.0
    v_norm = 2.0 * uv[..., 1] / max(feat_h - 1, 1) - 1.0
    return torch.stack((u_norm, v_norm), dim=-1)


def clamp_normalized_grid(grid: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    return grid.clamp(min=-margin, max=margin)


def align_point_groups(
    xyz_groups: torch.Tensor,
    extrinsic: torch.Tensor,
    camera_info: CameraInfo | dict[str, float | int],
    feature_hw: tuple[int, int],
    margin: float = 2.0,
) -> torch.Tensor:
    xyz_cam = apply_transform(extrinsic, xyz_groups)
    uv = project_points(xyz_cam, camera_info, feature_hw)
    return clamp_normalized_grid(normalize_grid(uv, feature_hw), margin=margin)
