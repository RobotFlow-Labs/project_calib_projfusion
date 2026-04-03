# CALIB-PROJFUSION — Execution Ledger

Resume rule: Read this file completely before writing code.
This project covers exactly one paper: ProjFusion / Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration Under Large Initial Perturbations.

## 1. Working Rules
- Work only inside `project_calib_projfusion/`
- Prefix every commit with `[CALIB-PROJFUSION]`
- Use `uv` for environment and dependency operations
- Target Python `3.11`
- Keep macOS development paths working while preserving CUDA-ready Linux paths for later training

## 2. Current Status
- **Date**: 2026-04-03
- **Phase**: Structural build complete, data/checkpoint integration pending
- **MVP Readiness**: 75%
- **Completed PRDs**: PRD-01, PRD-02, PRD-03, PRD-04, PRD-05, PRD-06, PRD-07
- **Pending PRDs**: none at scaffold level
- **Validated locally**:
  - `uv sync --python 3.11`
  - `uv run ruff check src tests scripts`
  - targeted pytest suite for config, geometry, projection, data, encoders, forward pass, inference, evaluation, API, and ROS bridge
- **Main blocker**: no real datasets or pretrained checkpoints are staged on the local/shared volume yet

## 3. What Exists Now
- Correct package namespace: `src/anima_calib_projfusion/`
- Typed TOML/Pydantic settings with paper/repo defaults
- SE(3), projection, split metadata, perturbation sampling
- Paper-aligned model scaffold:
  - DINOv2-style image token wrapper
  - PointGPT-style grouped point encoder
  - harmonic positional encoding
  - dual cross-attention branches
  - dual aggregation + regression heads
- Three-step iterative inference pipeline
- Checkpoint key translation adapter
- Evaluation metrics, benchmark runner, markdown/csv report builders
- FastAPI service + Docker runtime scaffold
- Host-safe ROS2 bridge/node/launch scaffold
- CUDA helper: `scripts/sync_cuda.sh`

## 4. Immediate Next Steps
1. Materialize datasets at the configured roots:
   - KITTI: `/mnt/forge-data/datasets/kitti_odometry`
   - nuScenes: `/mnt/forge-data/datasets/nuscenes`
2. Materialize pretrained checkpoints:
   - DINOv2 tiny
   - PointGPT tiny (KITTI / nuScenes)
3. Replace local stub encoder implementations with real checkpoint-backed wrappers.
4. Add readiness logic that reports false when required checkpoints are absent.
5. Run real export/validation flow once benchmarked checkpoints exist.

## 5. Environment Notes
- macOS local build works with base `torch` and the current smoke-test stack
- Linux/CUDA path is prepared via:
  - optional dependency group `cuda`
  - `scripts/sync_cuda.sh`
- Full GPU training has not started

## 6. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | PRD suite and task bundle created from paper + reference repo |
| 2026-04-03 | Codex | Replaced stale scaffold, implemented PRD-01 through PRD-07, validated full local test suite, left real data/checkpoint integration as the primary blocker |
