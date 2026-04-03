from __future__ import annotations

from pydantic import BaseModel, Field


class CameraInfoPayload(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    sensor_h: int
    sensor_w: int


class CalibrationRequest(BaseModel):
    image: list[list[list[float]]]
    point_cloud: list[list[float]]
    init_extrinsic: list[list[float]]
    camera_info: CameraInfoPayload
    run_iter: int = Field(default=3, ge=1, le=10)


class CalibrationResponse(BaseModel):
    pred_extrinsic: list[list[float]]
    trajectory: list[list[float]]
    ready: bool
    overlay_shape: list[int] | None = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
