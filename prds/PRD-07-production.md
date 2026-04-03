# PRD-07: Production

> Module: CALIB-PROJFUSION | Priority: P2
> Depends on: PRD-04, PRD-05, PRD-06
> Status: ⬜ Not started

## Objective
Harden the reproduced ProjFusion module for repeatable release, deployment, checkpoint export, and operational validation without regressing paper-matched calibration behavior.

## Context (from paper)
The paper demonstrates robust calibration under large perturbations, but productionizing the method requires checkpoint provenance, reproducible reports, degraded-mode behavior, and explicit release packaging. These are ANIMA requirements layered on top of the paper-faithful core.

**Paper reference**: §IV-D, §V  
**Key paper cues**: "consistently outperforms", "remains robust"

## Acceptance Criteria
- [ ] Model card / release notes capture datasets, splits, checkpoints, and benchmark results.
- [ ] Export workflow packages checkpoints, configs, metrics summary, and API contract.
- [ ] Smoke validation covers CLI, API, and ROS2 entrypoints with the same checkpoint.
- [ ] Graceful degradation path is documented for CPU-only or missing-checkpoint modes.
- [ ] Test: `uv run pytest tests/test_release_bundle.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `scripts/export_release.py` | package release bundle | — | ~160 |
| `scripts/validate_prod.py` | end-to-end smoke validation | — | ~140 |
| `docs/model_card.md` | release metadata | §IV / §V | ~120 |
| `tests/test_release_bundle.py` | bundle tests | — | ~100 |

## Architecture Detail (from paper)

### Inputs
- checkpoint
- config bundle
- evaluation summary

### Outputs
- versioned release artifact
- model card
- production validation report

### Algorithm
```python
def export_release(checkpoint_path, config_path, eval_report_path):
    bundle = collect_release_files(checkpoint_path, config_path, eval_report_path)
    validate_bundle(bundle)
    write_manifest(bundle)
    return bundle
```

## Dependencies
```toml
rich = ">=13.0"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| Final checkpoint | model | `/mnt/forge-data/models/calib_projfusion/release/` | produced after evaluation parity |
| Benchmark summary | report | `/mnt/forge-data/results/calib_projfusion/reports/` | produced by PRD-04 |

## Test Plan
```bash
uv run pytest tests/test_release_bundle.py -v
```

## References
- Paper: §IV-D, §V
- Depends on: PRD-04, PRD-05, PRD-06
