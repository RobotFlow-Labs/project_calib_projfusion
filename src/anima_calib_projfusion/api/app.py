from __future__ import annotations

from fastapi import FastAPI

from anima_calib_projfusion.api.schemas import CalibrationRequest, CalibrationResponse, HealthResponse
from anima_calib_projfusion.inference.service import CalibrationService

app = FastAPI(title="CALIB-PROJFUSION")
service = CalibrationService()


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok", ready=service.is_ready())


@app.get("/readyz", response_model=HealthResponse)
def readyz() -> HealthResponse:
    return HealthResponse(status="ready" if service.is_ready() else "not-ready", ready=service.is_ready())


@app.post("/calibrate", response_model=CalibrationResponse)
def calibrate(request: CalibrationRequest) -> CalibrationResponse:
    return service.calibrate(request, debug_overlay=False)
