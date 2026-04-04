from __future__ import annotations

import numpy as np
import torch

from anima_calib_projfusion.data.camera_info import CameraInfo
from anima_calib_projfusion.geometry.projection import project_points
from anima_calib_projfusion.geometry.se3 import apply_transform


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
    camera_info: CameraInfo,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    canvas = _to_hwc_uint8(image)
    xyz_cam = apply_transform(extrinsic, xyz)
    uv = project_points(
        xyz_cam, camera_info, feature_hw=(camera_info.sensor_h, camera_info.sensor_w)
    )
    uv = uv.round().to(torch.int64)
    height, width = canvas.shape[:2]
    for u, v in uv.reshape(-1, 2).tolist():
        if 0 <= u < width and 0 <= v < height:
            canvas[max(v - 1, 0) : min(v + 2, height), max(u - 1, 0) : min(u + 2, width)] = color
    return canvas
