import torch

from anima_calib_projfusion.data.camera_info import CameraInfo
from anima_calib_projfusion.geometry.projection import align_point_groups, normalize_grid, project_points


def test_project_points_maps_to_feature_plane():
    camera = CameraInfo(fx=100.0, fy=100.0, cx=50.0, cy=25.0, sensor_h=100, sensor_w=200)
    xyz = torch.tensor([[[0.0, 0.0, 1.0], [0.5, 0.25, 1.0]]])
    uv = project_points(xyz, camera, feature_hw=(10, 20))
    assert uv.shape == (1, 2, 2)
    assert torch.all(uv >= 0.0)


def test_align_point_groups_outputs_clamped_normalized_grid():
    camera = CameraInfo(fx=100.0, fy=100.0, cx=50.0, cy=25.0, sensor_h=100, sensor_w=200)
    xyz = torch.tensor([[[0.0, 0.0, 1.0], [0.5, 0.25, 1.0]]])
    extrinsic = torch.eye(4).unsqueeze(0)
    grid = align_point_groups(xyz, extrinsic, camera, feature_hw=(10, 20), margin=2.0)
    normalized = normalize_grid(project_points(xyz, camera, feature_hw=(10, 20)), feature_hw=(10, 20))
    assert grid.shape == (1, 2, 2)
    assert torch.all(grid.abs() <= 2.0)
    assert torch.allclose(grid, normalized.clamp(-2.0, 2.0))
