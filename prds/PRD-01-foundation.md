# PRD-01: Foundation & Config

> Module: CALIB-PROJFUSION | Priority: P0
> Depends on: None
> Status: ⬜ Not started

## Objective
The repo exposes a clean `anima_calib_projfusion` package with correct calibration config, dataset split metadata, and SE(3) / projection utilities required by the paper.

## Context (from paper)
ProjFusion predicts Lie-algebra calibration updates from RGB images, point clouds, intrinsics, and an initial extrinsic estimate. The paper’s core requirement is preserving the image and point inputs in their "native domains" before fusion and applying the calibration update with Eq. (1).

**Paper reference**: §III-A and §IV-A  
**Key paper cues**: "native domains", "Lie algebra representation", "KITTI sequences 00, 02–08, 10, 12, and 21"

## Acceptance Criteria
- [ ] Stale package identity is removed; the codebase targets `src/anima_calib_projfusion/`.
- [ ] Pydantic/TOML settings encode paper and repo defaults for model, inference, and datasets.
- [ ] KITTI and nuScenes split definitions match the paper and released configs.
- [ ] SE(3) exponential/logarithm, pose composition, and camera projection utilities are unit tested.
- [ ] Perturbation sampling supports `(15°, 15 cm)`, `(10°, 25 cm)`, and `(10°, 50 cm)`.
- [ ] Test: `uv run pytest tests/test_config.py tests/test_geometry_se3.py tests/test_projection.py tests/test_data_splits.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_calib_projfusion/__init__.py` | package identity | — | ~20 |
| `src/anima_calib_projfusion/config.py` | typed settings + defaults | §IV-A / §IV-B | ~180 |
| `src/anima_calib_projfusion/geometry/se3.py` | Lie algebra helpers | §III-A, Eq. (1) | ~220 |
| `src/anima_calib_projfusion/geometry/projection.py` | camera projection + normalization | §III-C.2, Eq. (2)-(4) | ~180 |
| `src/anima_calib_projfusion/data/camera_info.py` | intrinsics schema | §III-C | ~80 |
| `src/anima_calib_projfusion/data/splits.py` | paper split metadata | §IV-A | ~100 |
| `src/anima_calib_projfusion/data/perturbation.py` | perturbation samplers | §IV-A | ~140 |
| `configs/default.toml` | module defaults | §IV-B / repo configs | ~120 |
| `tests/test_config.py` | config tests | — | ~80 |
| `tests/test_geometry_se3.py` | SE(3) tests | — | ~120 |
| `tests/test_projection.py` | projection tests | — | ~120 |
| `tests/test_data_splits.py` | dataset split tests | — | ~80 |

## Architecture Detail (from paper)

### Inputs
- `image_rgb`: `Tensor[B, 3, 224, 448]`
- `point_cloud_xyz`: `Tensor[B, N, 3]`
- `init_extrinsic`: `Tensor[B, 4, 4]`
- `camera_info`: mapping with `fx`, `fy`, `cx`, `cy`, `sensor_h`, `sensor_w`

### Outputs
- `pose_update_log`: `Tensor[B, 6]`
- `projected_uv`: `Tensor[B, G, 2]` where `G=128` for the released PointGPT setup

### Algorithm
```python
# Paper §III-A / §III-C.2

def compose_calibration_update(xi_log: Tensor, init_extrinsic: Tensor) -> Tensor:
    delta = se3_exp(xi_log)          # [B, 4, 4]
    return delta @ init_extrinsic    # [B, 4, 4]


def align_point_groups(
    xyz_groups: Tensor,              # [B, G, 3]
    extrinsic: Tensor,               # [B, 4, 4]
    camera: CameraInfo,
    feature_hw: tuple[int, int],     # (16, 32)
    margin: float = 2.0,
) -> Tensor:
    xyz_cam = apply_transform(extrinsic, xyz_groups)
    uv = project_points(xyz_cam, camera, feature_hw)
    return clamp_normalized_grid(uv, margin=margin)
```

## Dependencies
```toml
pydantic = ">=2.0"
pydantic-settings = ">=2.0"
torch = ">=2.0"
numpy = ">=1.26"
einops = ">=0.8"
pykitti = ">=0.3"
nuscenes-devkit = ">=1.1"
open3d = ">=0.18"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| KITTI odometry | multi-sequence | `/mnt/forge-data/datasets/kitti_odometry` | `wget` from KITTI website, then stage under shared volume |
| nuScenes trainval/test | full benchmark | `/mnt/forge-data/datasets/nuscenes` | download from nuScenes official site |

## Test Plan
```bash
uv run pytest tests/test_config.py tests/test_geometry_se3.py tests/test_projection.py tests/test_data_splits.py -v
```

## References
- Paper: §III-A, §III-C.2, §IV-A
- Reference impl: `repositories/ProjFusion/dataset.py`
- Reference impl: `repositories/ProjFusion/models/util/se3.py`
- Feeds into: PRD-02, PRD-03, PRD-04
