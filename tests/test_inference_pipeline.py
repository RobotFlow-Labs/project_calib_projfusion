import torch

from anima_calib_projfusion.inference.pipeline import CalibrationBatch, iterative_calibrate


class DummyModel:
    def __call__(self, image, point_cloud, pred_extrinsic, camera_info):
        batch = image.shape[0]
        return torch.zeros(batch, 3), torch.full((batch, 3), 0.1)


def test_iterative_refinement_records_four_steps():
    camera_info = {
        "fx": torch.tensor([700.0, 700.0]),
        "fy": torch.tensor([700.0, 700.0]),
        "cx": torch.tensor([600.0, 600.0]),
        "cy": torch.tensor([180.0, 180.0]),
        "sensor_h": 376,
        "sensor_w": 1241,
    }
    batch = CalibrationBatch(
        image=torch.randn(2, 3, 224, 448),
        point_cloud=torch.randn(2, 8192, 3),
        init_extrinsic=torch.eye(4).unsqueeze(0).repeat(2, 1, 1),
        camera_info=camera_info,
    )
    pred_extrinsic, trajectory = iterative_calibrate(DummyModel(), batch, run_iter=3)
    assert pred_extrinsic.shape == (2, 4, 4)
    assert trajectory.shape == (2, 4, 6)
