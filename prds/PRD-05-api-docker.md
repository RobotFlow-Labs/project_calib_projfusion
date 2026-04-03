# PRD-05: API & Docker

> Module: CALIB-PROJFUSION | Priority: P1
> Depends on: PRD-03, PRD-04
> Status: ⬜ Not started

## Objective
Package the paper-faithful inference path behind a small FastAPI service and ship a Docker image that can run inference, health checks, and benchmark smoke tests.

## Context (from paper)
The paper itself is model-focused, but ANIMA requires the calibration module to be callable as a service without altering the calibration math. The API layer must therefore wrap the exact three-step inference loop defined in PRD-03.

**Paper reference**: §IV-B  
**Key paper cues**: "final prediction after three iterations is taken as the output"

## Acceptance Criteria
- [ ] FastAPI endpoint accepts image, point cloud, intrinsics, and initial extrinsics.
- [ ] Service returns final extrinsic, intermediate updates, and optional debug overlays.
- [ ] Docker image installs required CUDA/PyTorch dependencies or provides a CPU-safe fallback mode.
- [ ] `/healthz` and `/readyz` endpoints validate checkpoint readiness.
- [ ] Test: `uv run pytest tests/test_api_app.py tests/test_service_contract.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_calib_projfusion/api/schemas.py` | request/response schemas | §IV-B | ~120 |
| `src/anima_calib_projfusion/api/app.py` | FastAPI app | — | ~160 |
| `src/anima_calib_projfusion/inference/service.py` | service orchestrator | — | ~120 |
| `docker/Dockerfile` | inference image | — | ~120 |
| `docker/compose.yaml` | local service stack | — | ~60 |
| `tests/test_api_app.py` | API endpoint tests | — | ~120 |
| `tests/test_service_contract.py` | service-level tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- Encoded service request:
  - image bytes or path
  - point-cloud file or array
  - intrinsics
  - initial extrinsic hypothesis
  - `run_iter` default `3`

### Outputs
- `pred_extrinsic`: `4 x 4`
- `updates`: list of 6-DoF updates
- `overlay_uri`: optional

### Algorithm
```python
@app.post("/calibrate")
def calibrate(request: CalibrationRequest) -> CalibrationResponse:
    batch = decode_request(request)
    pred, steps = iterative_calibrate(model, batch, run_iter=request.run_iter or 3)
    return CalibrationResponse.from_prediction(pred, steps)
```

## Dependencies
```toml
fastapi = ">=0.115"
uvicorn = ">=0.30"
python-multipart = ">=0.0.9"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| Inference checkpoint | model | `/mnt/forge-data/models/calib_projfusion/...` | from PRD-03 / PRD-04 |

## Test Plan
```bash
uv run pytest tests/test_api_app.py tests/test_service_contract.py -v
```

## References
- Depends on: PRD-03, PRD-04
- Feeds into: PRD-06, PRD-07
