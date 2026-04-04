from __future__ import annotations

import numpy as np
import torch

from anima_calib_projfusion.api.schemas import CalibrationRequest, CalibrationResponse
from anima_calib_projfusion.config import ProjFusionSettings
from anima_calib_projfusion.inference.pipeline import CalibrationBatch, iterative_calibrate
from anima_calib_projfusion.inference.visualize import render_projection_overlay
from anima_calib_projfusion.model.projfusion import ProjDualFusion


class CalibrationService:
    def __init__(
        self,
        model: ProjDualFusion | None = None,
        settings: ProjFusionSettings | None = None,
    ) -> None:
        self.settings = settings or ProjFusionSettings()
        self.model = model or ProjDualFusion(
            image_hw=self.settings.model.image_hw,
            feature_dim=self.settings.model.feature_dim,
            num_groups=self.settings.model.num_groups,
            group_size=self.settings.model.group_size,
            harmonic_functions=self.settings.model.harmonic_functions,
            projection_margin=self.settings.model.projection_margin,
            attention_heads=self.settings.model.attention_heads,
            aggregation_planes=self.settings.model.aggregation_planes,
            mlp_hidden_dims=self.settings.model.mlp_hidden_dims,
        )

    def is_ready(self) -> bool:
        return self.model is not None

    def calibrate(
        self, request: CalibrationRequest, debug_overlay: bool = False
    ) -> CalibrationResponse:
        image = torch.tensor(request.image, dtype=torch.float32).unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("Expected image shaped [C,H,W]")
        point_cloud = torch.tensor(
            request.point_cloud, dtype=torch.float32
        ).unsqueeze(0)
        init_extrinsic = torch.tensor(
            request.init_extrinsic, dtype=torch.float32
        ).unsqueeze(0)
        camera = request.camera_info.model_dump()
        batch = CalibrationBatch(
            image=image,
            point_cloud=point_cloud,
            init_extrinsic=init_extrinsic,
            camera_info=camera,
        )
        pred_extrinsic, trajectory = iterative_calibrate(
            self.model, batch, run_iter=request.run_iter
        )
        overlay_shape = None
        if debug_overlay:
            overlay = render_projection_overlay(
                image=image[0],
                xyz=point_cloud.squeeze(0),
                extrinsic=pred_extrinsic.squeeze(0),
                camera_info=camera,
            )
            overlay_shape = list(np.asarray(overlay).shape)
        return CalibrationResponse(
            pred_extrinsic=pred_extrinsic[0].detach().cpu().tolist(),
            trajectory=trajectory[0].detach().cpu().tolist(),
            ready=self.is_ready(),
            overlay_shape=overlay_shape,
        )
