"""SE(3) Lie group and Lie algebra operations for ProjFusion.

Ported from reference repo: models/util/se3.py + so3.py + sinc.py
All operations are differentiable and batched.

Convention: twist ξ = [rot_x, rot_y, rot_z, tx, ty, tz] ∈ R^6.
"""

from __future__ import annotations

import torch
from torch import cos, sin

# ─── sinc helpers (Taylor-safe near zero) ────────────────────────


def _sinc1(t: torch.Tensor) -> torch.Tensor:
    """sin(t)/t, Taylor-safe."""
    r = torch.zeros_like(t)
    a = torch.abs(t)
    s, c = a < 0.01, a >= 0.01
    t2 = t[s] ** 2
    r[s] = 1 - t2 / 6 * (1 - t2 / 20 * (1 - t2 / 42))
    r[c] = sin(t[c]) / t[c]
    return r


def _sinc2(t: torch.Tensor) -> torch.Tensor:
    """(1 - cos(t)) / t^2, Taylor-safe."""
    r = torch.zeros_like(t)
    a = torch.abs(t)
    s, c = a < 0.01, a >= 0.01
    t2 = t**2
    r[s] = 0.5 * (1 - t2[s] / 12 * (1 - t2[s] / 30 * (1 - t2[s] / 56)))
    r[c] = (1 - cos(t[c])) / t2[c]
    return r


def _sinc3(t: torch.Tensor) -> torch.Tensor:
    """(t - sin(t)) / t^3, Taylor-safe."""
    r = torch.zeros_like(t)
    a = torch.abs(t)
    s, c = a < 0.01, a >= 0.01
    t2 = t[s] ** 2
    r[s] = 1 / 6 * (1 - t2 / 20 * (1 - t2 / 42 * (1 - t2 / 72)))
    r[c] = (t[c] - sin(t[c])) / (t[c] ** 3)
    return r


# ─── SO(3) operations ─────────────────────────────────────────


def _skew(w: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix: [*, 3] -> [*, 3, 3]."""
    w_ = w.view(-1, 3)
    x1, x2, x3 = w_[:, 0], w_[:, 1], w_[:, 2]
    zero = torch.zeros_like(x1)
    X = torch.stack(
        [
            torch.stack([zero, -x3, x2], dim=1),
            torch.stack([x3, zero, -x1], dim=1),
            torch.stack([-x2, x1, zero], dim=1),
        ],
        dim=1,
    )
    return X.view(*w.shape[:-1], 3, 3)


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Exponential map SO(3): [*, 3] -> [*, 3, 3] (Rodrigues)."""
    w_ = w.view(-1, 3)
    t = w_.norm(p=2, dim=1).view(-1, 1, 1)
    W = _skew(w_)
    S = W.bmm(W)
    eye3 = torch.eye(3, device=w.device, dtype=w.dtype)
    R = eye3 + _sinc1(t) * W + _sinc2(t) * S
    return R.view(*w.shape[:-1], 3, 3)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Logarithmic map SO(3): [*, 3, 3] -> [*, 3]."""
    R_ = R.view(-1, 3, 3)
    tr = R_[:, 0, 0] + R_[:, 1, 1] + R_[:, 2, 2]
    cos_angle = ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos_angle)
    w = torch.stack(
        [
            R_[:, 2, 1] - R_[:, 1, 2],
            R_[:, 0, 2] - R_[:, 2, 0],
            R_[:, 1, 0] - R_[:, 0, 1],
        ],
        dim=1,
    )
    s = angle.unsqueeze(-1)
    scale = torch.where(s.abs() < 1e-7, torch.ones_like(s) * 0.5, s / (2 * torch.sin(s)))
    return (w * scale).view(*R.shape[:-2], 3)


def _so3_inv_V(w: torch.Tensor) -> torch.Tensor:
    """Inverse of V matrix for SE(3) log map."""
    w_ = w.view(-1, 3)
    t = w_.norm(p=2, dim=1).view(-1, 1, 1)
    W = _skew(w_)
    eye3 = torch.eye(3, device=w.device, dtype=w.dtype)
    t2 = t**2
    coeff = torch.zeros_like(t)
    small = t.squeeze() < 0.01
    big = ~small
    if small.any():
        coeff[small] = 1.0 / 12 * (1 + t2[small] / 60 * (1 + t2[small] / 42))
    if big.any():
        coeff[big] = (1 - t[big] * cos(t[big] / 2) / (2 * sin(t[big] / 2))) / t2[big]
    return eye3 - 0.5 * W + coeff * W.bmm(W)


# ─── SE(3) operations ─────────────────────────────────────────


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Exponential map se(3) → SE(3): [*, 6] -> [*, 4, 4]."""
    xi_ = xi.view(-1, 6)
    w, v = xi_[:, :3], xi_[:, 3:]
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = _skew(w)
    S = W.bmm(W)
    eye3 = torch.eye(3, device=xi.device, dtype=xi.dtype)
    R = eye3 + _sinc1(t) * W + _sinc2(t) * S
    V = eye3 + _sinc2(t) * W + _sinc3(t) * S
    p = V.bmm(v.view(-1, 3, 1))
    z = (
        torch.tensor([0, 0, 0, 1], device=xi.device, dtype=xi.dtype)
        .view(1, 1, 4)
        .expand(xi_.size(0), -1, -1)
    )
    g = torch.cat([torch.cat([R, p], dim=2), z], dim=1)
    return g.view(*xi.shape[:-1], 4, 4)


def se3_log(g: torch.Tensor) -> torch.Tensor:
    """Logarithmic map SE(3) → se(3): [*, 4, 4] -> [*, 6]."""
    g_ = g.view(-1, 4, 4)
    R, p = g_[:, :3, :3], g_[:, :3, 3]
    w = so3_log(R)
    H = _so3_inv_V(w)
    v = H.bmm(p.view(-1, 3, 1)).view(-1, 3)
    return torch.cat([w, v], dim=1).view(*g.shape[:-2], 6)


def se3_inv(g: torch.Tensor) -> torch.Tensor:
    """Inverse of SE(3) matrix: [*, 4, 4] -> [*, 4, 4]."""
    g_ = g.view(-1, 4, 4)
    R, p = g_[:, :3, :3], g_[:, :3, 3]
    Q = R.transpose(1, 2)
    q = -Q.matmul(p.unsqueeze(-1))
    z = (
        torch.tensor([0, 0, 0, 1], device=g.device, dtype=g.dtype)
        .view(1, 1, 4)
        .expand(g_.size(0), -1, -1)
    )
    return torch.cat([torch.cat([Q, q], dim=2), z], dim=1).view(*g.shape[:-2], 4, 4)


def se3_transform(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply SE(3) transform to points.

    Args:
        T: [*, 4, 4] transformation.
        points: [*, 3, N] points in channel-first format.

    Returns:
        Transformed points [*, 3, N].
    """
    return T[..., :3, :3] @ points + T[..., :3, [3]]
