from __future__ import annotations

import torch


def _skew(omega: torch.Tensor) -> torch.Tensor:
    ox, oy, oz = omega.unbind(dim=-1)
    zeros = torch.zeros_like(ox)
    return torch.stack(
        (
            torch.stack((zeros, -oz, oy), dim=-1),
            torch.stack((oz, zeros, -ox), dim=-1),
            torch.stack((-oy, ox, zeros), dim=-1),
        ),
        dim=-2,
    )


def _vee(mat: torch.Tensor) -> torch.Tensor:
    return torch.stack((mat[..., 2, 1], mat[..., 0, 2], mat[..., 1, 0]), dim=-1)


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta.square()
    A = torch.where(theta > 1e-6, torch.sin(theta) / theta, 1.0 - theta2 / 6.0 + theta2.square() / 120.0)
    B = torch.where(
        theta > 1e-6,
        (1.0 - torch.cos(theta)) / theta2,
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
    )
    W = _skew(omega)
    identity = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(W.shape)
    return identity + A[..., None] * W + B[..., None] * (W @ W)


def _so3_log(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.arccos(cos_theta)
    skew_part = rotation - rotation.transpose(-1, -2)
    scale = torch.where(theta.abs() > 1e-6, theta / (2.0 * torch.sin(theta)), 0.5 + theta.square() / 12.0)
    return scale[..., None] * _vee(skew_part)


def _left_jacobian(omega: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta.square()
    W = _skew(omega)
    identity = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(W.shape)
    A = torch.where(
        theta > 1e-6,
        (1.0 - torch.cos(theta)) / theta2,
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
    )
    B = torch.where(
        theta > 1e-6,
        (theta - torch.sin(theta)) / (theta2 * theta),
        1.0 / 6.0 - theta2 / 120.0 + theta2.square() / 5040.0,
    )
    return identity + A[..., None] * W + B[..., None] * (W @ W)


def _inv_left_jacobian(omega: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    half_theta = 0.5 * theta
    theta2 = theta.square()
    W = _skew(omega)
    identity = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(W.shape)
    coeff = torch.where(
        theta > 1e-6,
        (1.0 - theta * torch.cos(half_theta) / (2.0 * torch.sin(half_theta))) / theta2,
        1.0 / 12.0 + theta2 / 720.0 + theta2.square() / 30240.0,
    )
    return identity - 0.5 * W + coeff[..., None] * (W @ W)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    omega = xi[..., :3]
    upsilon = xi[..., 3:]
    rotation = _so3_exp(omega)
    translation = (_left_jacobian(omega) @ upsilon.unsqueeze(-1)).squeeze(-1)
    transform = torch.eye(4, dtype=xi.dtype, device=xi.device).expand(*xi.shape[:-1], 4, 4).clone()
    transform[..., :3, :3] = rotation
    transform[..., :3, 3] = translation
    return transform


def se3_log(transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    omega = _so3_log(rotation)
    upsilon = (_inv_left_jacobian(omega) @ translation.unsqueeze(-1)).squeeze(-1)
    return torch.cat((omega, upsilon), dim=-1)


def apply_transform(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points shaped (..., 3), got {points.shape}")
    return torch.matmul(points, rotation.transpose(-1, -2)) + translation.unsqueeze(-2)


def compose_transform(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    return delta @ base
