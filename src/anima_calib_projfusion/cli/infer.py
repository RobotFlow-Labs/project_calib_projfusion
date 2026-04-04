from __future__ import annotations

import torch
import typer

from anima_calib_projfusion.config import ProjFusionSettings
from anima_calib_projfusion.inference.pipeline import CalibrationBatch, iterative_calibrate
from anima_calib_projfusion.model.projfusion import ProjDualFusion

app = typer.Typer(help="CALIB-PROJFUSION inference CLI")


@app.command()
def smoke(config: str = "configs/default.toml", run_iter: int = 3) -> None:
    settings = ProjFusionSettings.from_toml(config)
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
    camera_info = {
        "fx": torch.tensor([700.0]),
        "fy": torch.tensor([700.0]),
        "cx": torch.tensor([600.0]),
        "cy": torch.tensor([180.0]),
        "sensor_h": 376,
        "sensor_w": 1241,
    }
    batch = CalibrationBatch(
        image=torch.randn(1, 3, *settings.model.image_hw),
        point_cloud=torch.randn(1, settings.model.point_count, 3),
        init_extrinsic=torch.eye(4).unsqueeze(0),
        camera_info=camera_info,
    )
    _, trajectory = iterative_calibrate(model, batch, run_iter=run_iter)
    typer.echo({"trajectory_shape": tuple(trajectory.shape)})


def main() -> None:
    app()


if __name__ == "__main__":
    main()
