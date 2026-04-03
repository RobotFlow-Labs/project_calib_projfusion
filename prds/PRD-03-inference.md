# PRD-03: Inference

> Module: CALIB-PROJFUSION | Priority: P0
> Depends on: PRD-01, PRD-02
> Status: ⬜ Not started

## Objective
Expose the paper-faithful inference pipeline, including reference checkpoint loading, three-step iterative refinement, CLI entrypoints, and projection overlays for debugging.

## Context (from paper)
The paper states that inference uses a three-step iterative refinement strategy, where the output extrinsic of each step becomes the input to the next step. The released repo implements this update loop in `test.py`.

**Paper reference**: §IV-B  
**Key paper cues**: "three-step iterative refinement strategy"

## Acceptance Criteria
- [ ] Inference pipeline performs exactly three calibration updates by default.
- [ ] Checkpoint adapter loads reference ProjFusion/ProjDualFusion weight keys.
- [ ] CLI accepts image, point cloud, camera info, and initial extrinsic inputs.
- [ ] Intermediate poses can be persisted in the same "initial + 3 updates" format as the released repo.
- [ ] Overlay utility projects calibrated points back into image space for visual inspection.
- [ ] Test: `uv run pytest tests/test_inference_pipeline.py tests/test_checkpoint_adapter.py tests/test_visualization.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_calib_projfusion/inference/pipeline.py` | three-step iterative inference | §IV-B | ~180 |
| `src/anima_calib_projfusion/encoders/checkpoint_adapters.py` | released-checkpoint compatibility | repo `test.py` | ~160 |
| `src/anima_calib_projfusion/inference/visualize.py` | projected overlay generation | Fig. 3 | ~140 |
| `src/anima_calib_projfusion/cli/infer.py` | command-line inference | §IV-B | ~140 |
| `tests/test_inference_pipeline.py` | iterative refinement tests | — | ~120 |
| `tests/test_checkpoint_adapter.py` | checkpoint loading tests | — | ~100 |
| `tests/test_visualization.py` | overlay tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- `batch.image`: `Tensor[B, 3, 224, 448]`
- `batch.point_cloud`: `Tensor[B, N, 3]`
- `batch.init_extrinsic`: `Tensor[B, 4, 4]`
- `batch.camera_info`
- `run_iter`: `int = 3`

### Outputs
- `pred_extrinsic`: `Tensor[B, 4, 4]`
- `trajectory_log`: `Tensor[B, 4, 6]` for `initial + 3 refined poses`
- `overlay_image`: optional RGB preview

### Algorithm
```python
# Paper §IV-B and released repo `test.py`

def iterative_calibrate(model, batch, run_iter: int = 3):
    pred = batch.init_extrinsic
    steps = [se3_log(pred)]
    for _ in range(run_iter):
        rot_log, tsl_log = model(
            batch.image,
            batch.point_cloud,
            pred,
            batch.camera_info,
        )
        xi = torch.cat([rot_log, tsl_log], dim=-1)
        pred = se3_exp(xi) @ pred
        steps.append(se3_log(pred))
    return pred, torch.stack(steps, dim=1)
```

## Dependencies
```toml
opencv-python = ">=4.10"
typer = ">=0.12"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| Reproduced checkpoint | per dataset | `/mnt/forge-data/models/calib_projfusion/...` | train with PRD-02 + PRD-04 |

## Test Plan
```bash
uv run pytest tests/test_inference_pipeline.py tests/test_checkpoint_adapter.py tests/test_visualization.py -v
```

## References
- Paper: §IV-B
- Reference impl: `repositories/ProjFusion/test.py`
- Depends on: PRD-01, PRD-02
- Feeds into: PRD-04, PRD-05, PRD-06
