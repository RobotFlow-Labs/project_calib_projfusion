from __future__ import annotations

import argparse

import torch

from anima_calib_projfusion.config import ProjFusionSettings
from anima_calib_projfusion.data.camera_info import CameraInfo
from anima_calib_projfusion.model.projfusion import ProjDualFusion


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CALIB-PROJFUSION training entrypoint scaffold")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def run_smoke_test(settings: ProjFusionSettings) -> None:
    model = ProjDualFusion(
        image_hw=settings.model.image_hw,
        feature_dim=settings.model.feature_dim,
        num_groups=settings.model.num_groups,
        group_size=settings.model.group_size,
        harmonic_functions=settings.model.harmonic_functions,
        projection_margin=settings.model.projection_margin,
        attention_heads=settings.model.attention_heads,
        aggregation_planes=settings.model.aggregation_planes,
        mlp_hidden_dims=settings.model.mlp_hidden_dims,
    )
    image = torch.randn(2, 3, *settings.model.image_hw)
    points = torch.randn(2, settings.model.point_count, 3)
    extrinsic = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    camera = CameraInfo(fx=700.0, fy=700.0, cx=600.0, cy=180.0, sensor_h=376, sensor_w=1241)
    rot_log, tsl_log = model(image, points, extrinsic, camera)
    print({"rot_shape": tuple(rot_log.shape), "tsl_shape": tuple(tsl_log.shape)})


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = ProjFusionSettings.from_toml(args.config)
    if args.smoke_test:
        run_smoke_test(settings)
        return
    raise NotImplementedError(
        "Training data integration is not wired yet. Use --smoke-test while building the module."
    )


if __name__ == "__main__":
    main()
