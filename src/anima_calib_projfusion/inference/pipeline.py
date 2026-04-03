from __future__ import annotations

from dataclasses import dataclass

import torch

from anima_calib_projfusion.geometry.se3 import se3_exp, se3_log


@dataclass
class CalibrationBatch:
    image: torch.Tensor
    point_cloud: torch.Tensor
    init_extrinsic: torch.Tensor
    camera_info: object


def iterative_calibrate(
    model,
    batch: CalibrationBatch,
    run_iter: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_extrinsic = batch.init_extrinsic
    steps = [se3_log(pred_extrinsic)]
    for _ in range(run_iter):
        rot_log, tsl_log = model(batch.image, batch.point_cloud, pred_extrinsic, batch.camera_info)
        update = torch.cat((rot_log, tsl_log), dim=-1)
        pred_extrinsic = se3_exp(update) @ pred_extrinsic
        steps.append(se3_log(pred_extrinsic))
    return pred_extrinsic, torch.stack(steps, dim=1)
