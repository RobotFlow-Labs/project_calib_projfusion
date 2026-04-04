"""ANIMA Docker Serving — CALIB-PROJFUSION module node.

Provides camera-LiDAR extrinsic calibration as a service via
FastAPI (REST) and optionally ROS2 topics.

Inference: accepts RGB image + point cloud + initial extrinsic guess,
returns corrected extrinsic via three-step iterative refinement.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CalibProjFusionNode:
    """ProjFusion calibration serving node.

    Implements setup_inference + process for Docker-based serving.
    Compatible with anima_serve.node.AnimaNode when available.
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.img_transform = None
        self._ready = False

    def setup_inference(self, weights_path: str | Path | None = None, device: str = "auto"):
        """Load model weights and configure inference backend."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        from anima_calib_projfusion.model.projfusion import ProjDualFusion

        self.model = ProjDualFusion(dinov2_pretrained=True, freeze_encoders=True).to(self.device)
        self.model.eval()

        if weights_path and Path(weights_path).exists():
            state = torch.load(str(weights_path), map_location=self.device, weights_only=True)
            if "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)
            logger.info("Loaded weights from %s", weights_path)
        else:
            logger.warning("No weights loaded — using random init for trainable params")

        self.img_transform = T.Compose(
            [
                T.Resize((224, 448)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        self._ready = True
        logger.info("CalibProjFusion node ready on %s", self.device)

    @torch.no_grad()
    def process(self, input_data: dict) -> dict:
        """Run calibration inference.

        Args:
            input_data: dict with:
                image: PIL Image or [3, H, W] tensor
                pointcloud: [N, 3] numpy array or tensor
                init_extrinsic: [4, 4] initial extrinsic guess
                camera_info: dict with fx, fy, cx, cy, sensor_h, sensor_w
                refinement_steps: int (default 3)

        Returns:
            dict with:
                extrinsic: [4, 4] corrected extrinsic matrix
                rot_log: [3] rotation Lie algebra
                tsl_log: [3] translation vector
                refinement_steps: int
        """
        from anima_calib_projfusion.geometry.se3 import se3_exp

        # Parse inputs
        image = input_data["image"]
        if isinstance(image, Image.Image):
            image = self.img_transform(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        pcd = input_data["pointcloud"]
        if isinstance(pcd, np.ndarray):
            pcd = torch.from_numpy(pcd).float()
        if pcd.dim() == 2:
            pcd = pcd.unsqueeze(0)
        pcd = pcd.to(self.device)

        T_current = input_data["init_extrinsic"]
        if isinstance(T_current, np.ndarray):
            T_current = torch.from_numpy(T_current).float()
        if T_current.dim() == 2:
            T_current = T_current.unsqueeze(0)
        T_current = T_current.to(self.device)

        ci = input_data["camera_info"]
        ci_gpu = {}
        for k, v in ci.items():
            if isinstance(v, (int, float)):
                ci_gpu[k] = (
                    torch.tensor([v], device=self.device) if k in ("fx", "fy", "cx", "cy") else v
                )
            elif isinstance(v, torch.Tensor):
                ci_gpu[k] = v.to(self.device)
            else:
                ci_gpu[k] = v

        n_steps = input_data.get("refinement_steps", 3)

        # Three-step iterative refinement (paper §IV-B)
        for step in range(n_steps):
            rot_log, tsl_log = self.model(image, pcd, T_current, ci_gpu)
            xi = torch.cat([rot_log, tsl_log], dim=-1)  # [1, 6]
            delta_T = se3_exp(xi)  # [1, 4, 4]
            T_current = delta_T @ T_current

        return {
            "extrinsic": T_current.squeeze(0).cpu().numpy(),
            "rot_log": rot_log.squeeze(0).cpu().numpy(),
            "tsl_log": tsl_log.squeeze(0).cpu().numpy(),
            "refinement_steps": n_steps,
        }

    def get_status(self) -> dict:
        return {
            "model_loaded": self.model is not None,
            "device": str(self.device) if self.device else "none",
            "ready": self._ready,
        }

    @property
    def ready(self) -> bool:
        return self._ready


# Try to register as AnimaNode if anima_serve is available
try:
    from anima_serve.node import AnimaNode

    class CalibProjFusionAnimaNode(AnimaNode):
        """AnimaNode-compatible wrapper for CalibProjFusion."""

        def __init__(self):
            super().__init__()
            self._node = CalibProjFusionNode()

        def setup_inference(self):
            weights_path = self.weight_manager.download_weights()
            self._node.setup_inference(weights_path=weights_path)

        def process(self, input_data):
            return self._node.process(input_data)

        def get_status(self) -> dict:
            return self._node.get_status()

except ImportError:
    pass
