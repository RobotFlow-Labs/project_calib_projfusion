import torch

from anima_calib_projfusion.data.camera_info import CameraInfo
from anima_calib_projfusion.model.projfusion import ProjDualFusion


def test_projfusion_forward_contract():
    model = ProjDualFusion()
    image = torch.randn(2, 3, 224, 448)
    points = torch.randn(2, 8192, 3)
    extrinsic = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    camera = CameraInfo(fx=700.0, fy=700.0, cx=600.0, cy=180.0, sensor_h=376, sensor_w=1241)
    rot_log, tsl_log = model(image, points, extrinsic, camera)
    assert rot_log.shape == (2, 3)
    assert tsl_log.shape == (2, 3)
