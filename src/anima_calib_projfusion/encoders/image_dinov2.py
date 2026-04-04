"""DINOv2 ViT-S/14 image encoder for ProjFusion.

Loads the real DINOv2-small backbone via timm and extracts patch tokens
matching the paper's [B, 512, 384] output for 224×448 inputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import timm
import torch
from torch import nn

logger = logging.getLogger(__name__)

_DINOV2_SMALL_TIMM = "vit_small_patch14_dinov2"
_LOCAL_WEIGHTS = Path("/mnt/forge-data/models/calib_projfusion/dinov2_vits14_timm.pth")


class DINOv2ImageEncoder(nn.Module):
    """Real DINOv2 ViT-S/14 encoder using timm.

    Frozen by default (paper §IV-B: encoders are frozen during calibration training).
    Produces [B, H*W, 384] patch tokens where H=16, W=32 for 224×448 input.
    """

    def __init__(
        self,
        image_hw: tuple[int, int] = (224, 448),
        patch_size: int = 14,
        embed_dim: int = 384,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.image_hw = image_hw
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        wp = Path(weights_path) if weights_path else _LOCAL_WEIGHTS

        if pretrained and wp.exists():
            # Load from local timm-format checkpoint
            self.backbone = timm.create_model(
                _DINOV2_SMALL_TIMM,
                pretrained=False,
                img_size=image_hw,
                num_classes=0,
            )
            state = torch.load(str(wp), map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state, strict=False)
            logger.info("Loaded DINOv2-small from %s", wp)
        elif pretrained:
            # Download from timm hub
            self.backbone = timm.create_model(
                _DINOV2_SMALL_TIMM,
                pretrained=True,
                img_size=image_hw,
                num_classes=0,
            )
            logger.info("Loaded DINOv2-small from timm hub")
        else:
            self.backbone = timm.create_model(
                _DINOV2_SMALL_TIMM,
                pretrained=False,
                img_size=image_hw,
                num_classes=0,
            )

        if freeze:
            self._freeze()

    def _freeze(self) -> None:
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> "DINOv2ImageEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def token_hw(self) -> tuple[int, int]:
        return (self.image_hw[0] // self.patch_size, self.image_hw[1] // self.patch_size)

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens: [B, 3, H, W] -> [B, H*W/P^2, 384]."""
        if tuple(image.shape[-2:]) != self.image_hw:
            raise ValueError(f"Expected image size {self.image_hw}, got {tuple(image.shape[-2:])}")
        return self.backbone.get_intermediate_layers(image, n=1)[0]
