import math

import torch

from anima_calib_projfusion.eval.metrics import calibration_metrics


def test_l1_l2_thresholds():
    pred = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    gt = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    pred[1, 0, 3] = 0.1
    metrics = calibration_metrics(pred, gt)
    assert metrics.sample_count == 2
    assert math.isclose(metrics.l1_success_rate, 0.5)
    assert math.isclose(metrics.l2_success_rate, 0.5)
