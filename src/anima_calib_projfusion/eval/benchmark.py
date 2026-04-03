from __future__ import annotations

from dataclasses import asdict

import torch

from anima_calib_projfusion.eval.metrics import calibration_metrics


class BenchmarkRunner:
    def run_range(
        self,
        dataset_id: str,
        perturbation_range: tuple[float, float],
        pred_extrinsic: torch.Tensor,
        gt_extrinsic: torch.Tensor,
        checkpoint_name: str = "anonymous",
    ) -> dict[str, float | int | str]:
        metrics = calibration_metrics(pred_extrinsic, gt_extrinsic)
        result = asdict(metrics)
        result.update(
            {
                "dataset_id": dataset_id,
                "perturbation_range": f"{perturbation_range[0]:.0f}deg/{perturbation_range[1] * 100:.0f}cm",
                "checkpoint_name": checkpoint_name,
            }
        )
        return result

    def run_all(
        self,
        dataset_id: str,
        batches: list[tuple[tuple[float, float], torch.Tensor, torch.Tensor]],
        checkpoint_name: str = "anonymous",
    ) -> list[dict[str, float | int | str]]:
        return [
            self.run_range(dataset_id, perturbation_range, pred_extrinsic, gt_extrinsic, checkpoint_name)
            for perturbation_range, pred_extrinsic, gt_extrinsic in batches
        ]
