import torch

from anima_calib_projfusion.model.projfusion import ProjDualFusion


def test_projfusion_forward_contract():
    model = ProjDualFusion(dinov2_pretrained=False)
    image = torch.randn(2, 3, 224, 448)
    points = torch.randn(2, 8192, 3)
    extrinsic = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    camera_info = {
        "fx": torch.tensor([700.0, 700.0]),
        "fy": torch.tensor([700.0, 700.0]),
        "cx": torch.tensor([600.0, 600.0]),
        "cy": torch.tensor([180.0, 180.0]),
        "sensor_h": 376,
        "sensor_w": 1241,
    }
    rot_log, tsl_log = model(image, points, extrinsic, camera_info)
    assert rot_log.shape == (2, 3)
    assert tsl_log.shape == (2, 3)
