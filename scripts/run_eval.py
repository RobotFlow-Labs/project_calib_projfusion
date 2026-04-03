from __future__ import annotations

import argparse
from pathlib import Path

import torch

from anima_calib_projfusion.eval.benchmark import BenchmarkRunner


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CALIB-PROJFUSION evaluation on saved tensors.")
    parser.add_argument("--pred", required=True, help="Path to predicted extrinsic tensor (.pt)")
    parser.add_argument("--gt", required=True, help="Path to ground-truth extrinsic tensor (.pt)")
    parser.add_argument("--dataset", required=True, choices=["kitti", "nuscenes"])
    parser.add_argument("--rot-deg", type=float, required=True)
    parser.add_argument("--tsl-m", type=float, required=True)
    parser.add_argument("--checkpoint-name", default="anonymous")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    pred_extrinsic = torch.load(Path(args.pred), map_location="cpu")
    gt_extrinsic = torch.load(Path(args.gt), map_location="cpu")
    runner = BenchmarkRunner()
    result = runner.run_range(
        dataset_id=args.dataset,
        perturbation_range=(args.rot_deg, args.tsl_m),
        pred_extrinsic=pred_extrinsic,
        gt_extrinsic=gt_extrinsic,
        checkpoint_name=args.checkpoint_name,
    )
    print(result)


if __name__ == "__main__":
    main()
