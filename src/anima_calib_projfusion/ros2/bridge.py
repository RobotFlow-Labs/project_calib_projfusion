from __future__ import annotations

from typing import Any

import numpy as np
import torch

from anima_calib_projfusion.data.camera_info import CameraInfo
from anima_calib_projfusion.inference.pipeline import CalibrationBatch


def _value_from_message(message: Any, key: str):
    if isinstance(message, dict):
        return message[key]
    return getattr(message, key)


def _image_to_tensor(image_msg: Any) -> torch.Tensor:
    data = np.asarray(_value_from_message(image_msg, "data"), dtype=np.float32)
    height = int(_value_from_message(image_msg, "height"))
    width = int(_value_from_message(image_msg, "width"))
    if data.ndim == 1:
        data = data.reshape(height, width, 3)
    if data.shape[-1] == 3:
        data = np.moveaxis(data, -1, 0)
    return torch.from_numpy(data)


def _points_to_tensor(pointcloud_msg: Any) -> torch.Tensor:
    points = np.asarray(_value_from_message(pointcloud_msg, "points"), dtype=np.float32)
    return torch.from_numpy(points)


def ros_to_batch(
    image_msg: Any,
    pointcloud_msg: Any,
    camera_info_msg: Any,
    init_extrinsic_msg: Any | None = None,
) -> CalibrationBatch:
    image = _image_to_tensor(image_msg).unsqueeze(0)
    point_cloud = _points_to_tensor(pointcloud_msg).unsqueeze(0)
    camera = CameraInfo.from_mapping(
        {
            "fx": _value_from_message(camera_info_msg, "fx"),
            "fy": _value_from_message(camera_info_msg, "fy"),
            "cx": _value_from_message(camera_info_msg, "cx"),
            "cy": _value_from_message(camera_info_msg, "cy"),
            "sensor_h": _value_from_message(camera_info_msg, "sensor_h"),
            "sensor_w": _value_from_message(camera_info_msg, "sensor_w"),
        }
    )
    if init_extrinsic_msg is None:
        init_extrinsic = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    else:
        init_extrinsic = torch.tensor(_value_from_message(init_extrinsic_msg, "matrix"), dtype=torch.float32).unsqueeze(0)
    return CalibrationBatch(
        image=image,
        point_cloud=point_cloud,
        init_extrinsic=init_extrinsic,
        camera_info=camera,
    )
