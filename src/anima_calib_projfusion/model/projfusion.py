"""ProjDualFusion: Dual-branch calibration model from ProjFusion paper.

Paper: Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration
Architecture: DINOv2 image encoder + PointGPT point encoder → dual cross-attention
→ dual aggregation → rotation/translation MLP heads → se(3) Lie algebra output.
"""
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
    """ProjFusion dual-branch calibration model.

    Args:
        image_hw: Input image size (H, W).
        feature_dim: Encoder embedding dimension (384 for ViT-S/14).
        num_groups: PointGPT group count.
        group_size: Points per group.
        harmonic_functions: Number of harmonic frequency bands.
        projection_margin: Clamping margin r_p for projected coordinates.
        attention_heads: Number of attention heads.
        aggregation_planes: Intermediate channel count for aggregation.
        mlp_hidden_dims: Hidden layer sizes for regression heads.
        dinov2_pretrained: Load pretrained DINOv2 weights.
        freeze_encoders: Freeze both encoders (paper default).
    """

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
        dinov2_pretrained: bool = True,
        freeze_encoders: bool = True,
    ) -> None:
        super().__init__()

        # Frozen backbone encoders
        self.image_encoder = DINOv2ImageEncoder(
            image_hw=image_hw,
            embed_dim=feature_dim,
            pretrained=dinov2_pretrained,
            freeze=freeze_encoders,
        )
        self.point_encoder = PointGPTEncoder(
            embed_dim=feature_dim,
            num_groups=num_groups,
            group_size=group_size,
            freeze=False,  # PointGPT trains end-to-end (no pretrained checkpoint)
        )

        self.feature_hw = self.image_encoder.token_hw  # (16, 32)

        # Positional encoding
        omega_0 = 1.0 / (1.0 + projection_margin)
        self.positional_encoding = HarmonicEmbedding(
            num_harmonic_functions=harmonic_functions,
            omega_0=omega_0,
        )
        pos_dim = self.positional_encoding.output_dim(2)

        # Coordinate alignment
        self.aligner = ExtrinsicAwareAligner(
            feature_hw=self.feature_hw,
            margin=projection_margin,
        )

        # Dual cross-attention branches
        self.rotation_attention = ScaleFreeCrossAttention(
            embed_dim=feature_dim, num_heads=attention_heads, pos_dim=pos_dim,
        )
        self.translation_attention = ScaleFreeCrossAttention(
            embed_dim=feature_dim, num_heads=attention_heads, pos_dim=pos_dim,
        )

        # Dual aggregation + regression heads
        self.rotation_aggregation = MiniResAggregation(
            in_channels=feature_dim, planes=aggregation_planes, output_dim=768,
        )
        self.translation_aggregation = MiniResAggregation(
            in_channels=feature_dim, planes=aggregation_planes, output_dim=768,
        )
        self.rotation_head = RegressionHead(input_dim=768, hidden_dims=mlp_hidden_dims)
        self.translation_head = RegressionHead(input_dim=768, hidden_dims=mlp_hidden_dims)

    def forward(
        self,
        img: torch.Tensor,
        pcd: torch.Tensor,
        init_extrinsic: torch.Tensor,
        camera_info: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            img: [B, 3, 224, 448] normalized RGB image.
            pcd: [B, N, 3] point cloud.
            init_extrinsic: [B, 4, 4] current extrinsic estimate.
            camera_info: dict with fx, fy, cx, cy, sensor_h, sensor_w.

        Returns:
            rot_log: [B, 3] rotation Lie algebra vector.
            tsl_log: [B, 3] translation vector.
        """
        # Extract features
        image_tokens = self.image_encoder(img)  # [B, 512, 384]
        xyz_groups, point_tokens = self.point_encoder(pcd)  # [B, 128, 3], [B, 128, 384]

        # Project point groups into image feature space
        point_uv = self.aligner(xyz_groups, init_extrinsic, camera_info)  # [B, 128, 2]

        # Build image grid coordinates
        image_grid = build_image_grid(
            self.feature_hw, img.device, img.dtype
        ).expand(img.shape[0], -1, -1)  # [B, 512, 2]

        # Harmonic positional encoding
        image_pos = self.positional_encoding(image_grid)  # [B, 512, pos_dim]
        point_pos = self.positional_encoding(point_uv)  # [B, 128, pos_dim]

        # Dual cross-attention
        rot_tokens = self.rotation_attention(
            image_tokens, point_tokens, image_pos, point_pos
        )
        tsl_tokens = self.translation_attention(
            image_tokens, point_tokens, image_pos, point_pos
        )

        # Unflatten to 2D feature maps
        batch = img.shape[0]
        feat_h, feat_w = self.feature_hw
        rot_map = rot_tokens.transpose(1, 2).reshape(batch, -1, feat_h, feat_w)
        tsl_map = tsl_tokens.transpose(1, 2).reshape(batch, -1, feat_h, feat_w)

        # Aggregate and regress
        rot_features = self.rotation_aggregation(rot_map)
        tsl_features = self.translation_aggregation(tsl_map)
        return self.rotation_head(rot_features), self.translation_head(tsl_features)
