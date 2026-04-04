from __future__ import annotations

import numpy as np
import torch

from anima_calib_projfusion.geometry.projection import project_points


def _to_hwc_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("Expected an image shaped [H,W,C] or [C,H,W]")
    if array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0) * 255.0
        array = array.astype(np.uint8)
    return array.copy()


def render_projection_overlay(
    image: torch.Tensor | np.ndarray,
    xyz: torch.Tensor,
    extrinsic: torch.Tensor,
    camera_info: dict,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Render point cloud projection overlay on image."""
    canvas = _to_hwc_uint8(image)
    # Transform to camera frame
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3:]
    xyz_cam = (R @ xyz.T + t).T.unsqueeze(0)  # [1, N, 3]
    sensor_h = camera_info.get("sensor_h", canvas.shape[0])
    sensor_w = camera_info.get("sensor_w", canvas.shape[1])
    uv = project_points(xyz_cam, camera_info, feature_hw=(sensor_h, sensor_w))
    uv = uv.squeeze(0).round().to(torch.int64)
    height, width = canvas.shape[:2]
    for u, v in uv.reshape(-1, 2).tolist():
        if 0 <= u < width and 0 <= v < height:
            canvas[max(v - 1, 0) : min(v + 2, height), max(u - 1, 0) : min(u + 2, width)] = color
    return canvas
