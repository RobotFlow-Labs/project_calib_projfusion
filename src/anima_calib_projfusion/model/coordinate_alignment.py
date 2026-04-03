from __future__ import annotations

import torch

from anima_calib_projfusion.data.camera_info import CameraInfo
from anima_calib_projfusion.geometry.projection import align_point_groups


def build_image_grid(feature_hw: tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    feat_h, feat_w = feature_hw
    ys = torch.linspace(-1.0, 1.0, steps=feat_h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=feat_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, feat_h * feat_w, 2)


class ExtrinsicAwareAligner:
    def __init__(self, feature_hw: tuple[int, int], margin: float = 2.0) -> None:
        self.feature_hw = feature_hw
        self.margin = margin

    def __call__(
        self,
        xyz_groups: torch.Tensor,
        extrinsic: torch.Tensor,
        camera_info: CameraInfo | dict[str, float | int],
    ) -> torch.Tensor:
        return align_point_groups(
            xyz_groups=xyz_groups,
            extrinsic=extrinsic,
            camera_info=camera_info,
            feature_hw=self.feature_hw,
            margin=self.margin,
        )
