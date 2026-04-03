from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class CalibrationMetrics:
    rotation_rmse_deg: float
    translation_rmse_cm: float
    l1_success_rate: float
    l2_success_rate: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _inverse_transform(transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    inverse = torch.eye(4, dtype=transform.dtype, device=transform.device).expand(transform.shape).clone()
    inverse[..., :3, :3] = rotation.transpose(-1, -2)
    inverse[..., :3, 3] = -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)
    return inverse


def _matrix_to_euler_xyz_deg(rotation: torch.Tensor) -> torch.Tensor:
    sy = torch.sqrt(rotation[..., 0, 0] ** 2 + rotation[..., 1, 0] ** 2)
    singular = sy < 1e-6
    x = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    y = torch.atan2(-rotation[..., 2, 0], sy)
    z = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    xs = torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1])
    ys = torch.atan2(-rotation[..., 2, 0], sy)
    zs = torch.zeros_like(z)
    x = torch.where(singular, xs, x)
    y = torch.where(singular, ys, y)
    z = torch.where(singular, zs, z)
    return torch.rad2deg(torch.stack((x, y, z), dim=-1)).abs()


def per_sample_errors(pred_extrinsic: torch.Tensor, gt_extrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    delta = pred_extrinsic @ _inverse_transform(gt_extrinsic)
    rotation_error_deg = _matrix_to_euler_xyz_deg(delta[..., :3, :3]).norm(dim=-1)
    translation_error_cm = delta[..., :3, 3].abs().norm(dim=-1) * 100.0
    return rotation_error_deg, translation_error_cm


def calibration_metrics(
    pred_extrinsic: torch.Tensor,
    gt_extrinsic: torch.Tensor,
    *,
    l1_deg: float = 1.0,
    l1_cm: float = 2.5,
    l2_deg: float = 2.0,
    l2_cm: float = 5.0,
) -> CalibrationMetrics:
    rotation_error_deg, translation_error_cm = per_sample_errors(pred_extrinsic, gt_extrinsic)
    l1 = ((rotation_error_deg < l1_deg) & (translation_error_cm < l1_cm)).float().mean().item()
    l2 = ((rotation_error_deg < l2_deg) & (translation_error_cm < l2_cm)).float().mean().item()
    return CalibrationMetrics(
        rotation_rmse_deg=float(torch.sqrt((rotation_error_deg.square()).mean()).item()),
        translation_rmse_cm=float(torch.sqrt((translation_error_cm.square()).mean()).item()),
        l1_success_rate=float(l1),
        l2_success_rate=float(l2),
        sample_count=int(rotation_error_deg.numel()),
    )
