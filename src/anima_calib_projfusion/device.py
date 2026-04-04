"""Runtime backend resolution for local macOS dev and future CUDA training."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendInfo:
    backend: str
    device: str


def resolve_backend(preferred: str = "auto") -> BackendInfo:
    backend = os.environ.get("ANIMA_BACKEND", preferred)
    if backend == "mlx":
        return BackendInfo(backend="mlx", device="mlx")
    if backend == "cuda":
        return BackendInfo(backend="cuda", device="cuda")
    if backend == "cpu":
        return BackendInfo(backend="cpu", device="cpu")

    try:
        import mlx.core as mx  # type: ignore

        _ = mx.default_device()
        return BackendInfo(backend="mlx", device="mlx")
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return BackendInfo(backend="cuda", device="cuda")
    except Exception:
        pass

    return BackendInfo(backend="cpu", device="cpu")
