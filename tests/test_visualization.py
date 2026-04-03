import numpy as np
import torch

from anima_calib_projfusion.data.camera_info import CameraInfo
from anima_calib_projfusion.inference.visualize import render_projection_overlay


def test_overlay_renderer_returns_image():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    xyz = torch.tensor([[[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]]], dtype=torch.float32)
    extrinsic = torch.eye(4).unsqueeze(0)
    camera = CameraInfo(fx=20.0, fy=20.0, cx=32.0, cy=32.0, sensor_h=64, sensor_w=64)
    overlay = render_projection_overlay(image, xyz, extrinsic, camera)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
