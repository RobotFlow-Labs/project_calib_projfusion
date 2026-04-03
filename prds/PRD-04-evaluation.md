# PRD-04: Evaluation

> Module: CALIB-PROJFUSION | Priority: P0
> Depends on: PRD-01, PRD-02, PRD-03
> Status: ⬜ Not started

## Objective
Implement paper-matched calibration metrics, benchmark runners, and report generation for KITTI and nuScenes so reproduced results can be compared directly with Table I and Table II.

## Context (from paper)
The paper evaluates RMSE for rotation and translation plus two success-rate metrics: L1 `(1°, 2.5 cm)` and L2 `(2°, 5 cm)`. It reports headline robustness on KITTI and nuScenes at large perturbation ranges.

**Paper reference**: §IV-C, §IV-D, Table I, Table II  
**Key paper cues**: "L1", "L2", "10° 50cm", "41.04%", "87.68%", "99%"

## Acceptance Criteria
- [ ] Metric implementation computes rotation/translation RMSE from predicted vs. ground-truth extrinsics.
- [ ] L1 and L2 success rates match the paper definitions exactly.
- [ ] Benchmark configs cover the three perturbation ranges `(15°,15 cm)`, `(10°,25 cm)`, `(10°,50 cm)`.
- [ ] Report builder emits a Markdown/CSV summary keyed by dataset, range, and checkpoint.
- [ ] A reproduction target document is generated for Table I and Table II comparisons.
- [ ] Test: `uv run pytest tests/test_metrics.py tests/test_benchmark_runner.py tests/test_report_builder.py -v` passes.
- [ ] Benchmark target: KITTI `10° / 50 cm` reaches at least `L1=41.0%`, `L2=87.5%`.
- [ ] Benchmark target: nuScenes `10° / 50 cm` reaches at least `L1=90.0%`, `L2=99.0%`.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_calib_projfusion/eval/metrics.py` | RMSE, L1, L2 metrics | §IV-C | ~180 |
| `src/anima_calib_projfusion/eval/benchmark.py` | evaluation loop per dataset/range | §IV-D | ~220 |
| `src/anima_calib_projfusion/eval/report.py` | Table I / II style summaries | Table I / II | ~160 |
| `scripts/run_eval.py` | evaluation entrypoint | §IV-D | ~120 |
| `scripts/build_report.py` | produce markdown / csv results | Table I / II | ~120 |
| `tests/test_metrics.py` | metric correctness tests | — | ~120 |
| `tests/test_benchmark_runner.py` | runner tests | — | ~120 |
| `tests/test_report_builder.py` | reporting tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- `pred_extrinsic`: `Tensor[B, 4, 4]`
- `gt_extrinsic`: `Tensor[B, 4, 4]`
- `dataset_id`: `Literal["kitti", "nuscenes"]`
- `perturb_range`: `tuple[float, float]`

### Outputs
- `rotation_rmse_deg`
- `translation_rmse_cm`
- `l1_success_rate`
- `l2_success_rate`

### Algorithm
```python
# Paper §IV-C

def calibration_metrics(pred_extrinsic, gt_extrinsic):
    delta = pred_extrinsic @ inverse(gt_extrinsic)
    euler_deg = rotation_error_euler_deg(delta)
    translation_cm = translation_error_cm(delta)
    rot_rmse = rmse(euler_deg)
    tsl_rmse = rmse(translation_cm)
    l1 = success_rate(rot_rmse < 1.0 and tsl_rmse < 2.5)
    l2 = success_rate(rot_rmse < 2.0 and tsl_rmse < 5.0)
    return rot_rmse, tsl_rmse, l1, l2
```

## Dependencies
```toml
pandas = ">=2.0"
tabulate = ">=0.9"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| Evaluation split definitions | text metadata | `src/anima_calib_projfusion/data/splits.py` | created in PRD-01 |
| Generated prediction logs | per run | `/mnt/forge-data/results/calib_projfusion/...` | produced by PRD-03 |

## Test Plan
```bash
uv run pytest tests/test_metrics.py tests/test_benchmark_runner.py tests/test_report_builder.py -v
```

## References
- Paper: §IV-C, §IV-D, Table I, Table II
- Reference impl: `repositories/ProjFusion/metrics.py`
- Reference impl: `repositories/ProjFusion/table.py`
- Depends on: PRD-01, PRD-02, PRD-03
- Feeds into: PRD-05, PRD-07
