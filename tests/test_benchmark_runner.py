import torch

from anima_calib_projfusion.eval.benchmark import BenchmarkRunner


def test_runner_returns_metric_bundle():
    runner = BenchmarkRunner()
    pred = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    gt = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    result = runner.run_range("kitti", (10.0, 0.5), pred, gt, checkpoint_name="smoke")
    assert result["dataset_id"] == "kitti"
    assert result["perturbation_range"] == "10deg/50cm"
    assert result["sample_count"] == 2
