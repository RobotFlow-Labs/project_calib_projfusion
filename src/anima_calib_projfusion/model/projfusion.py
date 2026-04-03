from __future__ import annotations

import torch
from torch import nn

from anima_calib_projfusion.encoders.image_dinov2 import DINOv2ImageEncoder
from anima_calib_projfusion.encoders.pointgpt import PointGPTEncoder
from anima_calib_projfusion.model.aggregation import MiniResAggregation
from anima_calib_projfusion.model.coordinate_alignment import ExtrinsicAwareAligner, build_image_grid
from anima_calib_projfusion.model.cross_attention import ScaleFreeCrossAttention
from anima_calib_projfusion.model.heads import RegressionHead
from anima_calib_projfusion.model.positional_encoding import HarmonicEmbedding


class ProjDualFusion(nn.Module):
    def __init__(
        self,
        image_hw: tuple[int, int] = (224, 448),
        feature_dim: int = 384,
        num_groups: int = 128,
        group_size: int = 64,
        harmonic_functions: int = 6,
        projection_margin: float = 2.0,
        attention_heads: int = 6,
        aggregation_planes: int = 96,
        mlp_hidden_dims: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.image_encoder = DINOv2ImageEncoder(image_hw=image_hw, embed_dim=feature_dim)
        self.point_encoder = PointGPTEncoder(
            embed_dim=feature_dim,
            num_groups=num_groups,
            group_size=group_size,
        )
        self.feature_hw = self.image_encoder.token_hw
        omega_0 = 1.0 / (1.0 + projection_margin)
        self.positional_encoding = HarmonicEmbedding(
            num_harmonic_functions=harmonic_functions,
            omega_0=omega_0,
        )
        pos_dim = self.positional_encoding.output_dim(2)
        self.aligner = ExtrinsicAwareAligner(feature_hw=self.feature_hw, margin=projection_margin)
        self.rotation_attention = ScaleFreeCrossAttention(
            embed_dim=feature_dim,
            num_heads=attention_heads,
            pos_dim=pos_dim,
        )
        self.translation_attention = ScaleFreeCrossAttention(
            embed_dim=feature_dim,
            num_heads=attention_heads,
            pos_dim=pos_dim,
        )
        self.rotation_aggregation = MiniResAggregation(
            in_channels=feature_dim,
            planes=aggregation_planes,
            output_dim=768,
        )
        self.translation_aggregation = MiniResAggregation(
            in_channels=feature_dim,
            planes=aggregation_planes,
            output_dim=768,
        )
        self.rotation_head = RegressionHead(input_dim=768, hidden_dims=mlp_hidden_dims)
        self.translation_head = RegressionHead(input_dim=768, hidden_dims=mlp_hidden_dims)

    def forward(
        self,
        img: torch.Tensor,
        pcd: torch.Tensor,
        init_extrinsic: torch.Tensor,
        camera_info,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tokens = self.image_encoder(img)
        xyz_groups, point_tokens = self.point_encoder(pcd)
        point_uv = self.aligner(xyz_groups, init_extrinsic, camera_info)
        image_grid = build_image_grid(self.feature_hw, img.device, img.dtype).expand(img.shape[0], -1, -1)
        image_pos = self.positional_encoding(image_grid)
        point_pos = self.positional_encoding(point_uv)
        rot_tokens = self.rotation_attention(image_tokens, point_tokens, image_pos, point_pos)
        tsl_tokens = self.translation_attention(image_tokens, point_tokens, image_pos, point_pos)
        batch = img.shape[0]
        feat_h, feat_w = self.feature_hw
        rot_map = rot_tokens.transpose(1, 2).reshape(batch, -1, feat_h, feat_w)
        tsl_map = tsl_tokens.transpose(1, 2).reshape(batch, -1, feat_h, feat_w)
        rot_features = self.rotation_aggregation(rot_map)
        tsl_features = self.translation_aggregation(tsl_map)
        return self.rotation_head(rot_features), self.translation_head(tsl_features)
