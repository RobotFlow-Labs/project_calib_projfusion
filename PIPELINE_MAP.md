# CALIB-PROJFUSION — Pipeline Map

## Goal
Map the paper components and released ProjFusion implementation onto the ANIMA build plan for this module.

## Paper-to-Code Map
| Paper Area | Paper Reference | Reference Repo | ANIMA Target |
|-----------|-----------------|----------------|--------------|
| Problem setup and SE(3) update | §III-A, Eq. (1) | `models/util/se3.py`, `train.py`, `test.py` | `src/anima_calib_projfusion/geometry/se3.py`, `src/anima_calib_projfusion/calibration/update.py` |
| Image encoder: DINOv2 patch tokens | §III-B | `models/tools/attention.py` (`ViTEncoder`) | `src/anima_calib_projfusion/encoders/image_dinov2.py` |
| Point encoder: PointGPT groups + transformer | §III-B | `models/pointgpt/PointGPT.py`, `models/tools/core.py` | `src/anima_calib_projfusion/encoders/pointgpt.py` |
| Coordinate alignment and projection margin | §III-C.2, Eq. (2)-(4) | `models/tools/core.py` | `src/anima_calib_projfusion/model/coordinate_alignment.py` |
| Harmonic positional embedding | §III-C.3, Eq. (5) | `models/tools/embedding.py` | `src/anima_calib_projfusion/model/positional_encoding.py` |
| Scale-free multi-head cross-attention | §III-C.4, Eq. (6)-(10) | `models/tools/attention.py` | `src/anima_calib_projfusion/model/cross_attention.py` |
| Dual branches for rotation and translation | §III-C / §III-D | `models/model.py`, `models/tools/core.py` | `src/anima_calib_projfusion/model/projfusion.py` |
| 2D aggregation blocks + MLP heads | §III-D | `models/tools/aggregation.py`, `models/model.py` | `src/anima_calib_projfusion/model/aggregation.py`, `src/anima_calib_projfusion/model/heads.py` |
| Dataset splits and perturbation ranges | §IV-A | `cfg/dataset/*.yml`, `dataset.py` | `src/anima_calib_projfusion/data/*.py`, `configs/datasets/*.toml` |
| Iterative three-step inference | §IV-B | `test.py` | `src/anima_calib_projfusion/inference/pipeline.py` |
| Metrics: RMSE, L1, L2 | §IV-C | `models/loss.py`, `metrics.py` | `src/anima_calib_projfusion/eval/metrics.py` |
| Table I / Table II evaluation | §IV-D / §IV-E | `bash_metric.py`, `table.py` | `scripts/run_eval.py`, `scripts/build_report.py` |

## Execution Flow
1. Load RGB image, filtered point cloud, initial extrinsic hypothesis, and camera intrinsics.
2. Encode RGB into DINOv2 patch tokens with shape `[B, 16 x 32, 384]` for `224 x 448` inputs.
3. Encode point cloud into PointGPT group tokens with shape `[B, 128, 384]` plus centroids `[B, 128, 3]`.
4. Transform centroids by the current extrinsic estimate and project them into the image-feature plane.
5. Clamp normalized projected coordinates into `[-(1+r_p), +(1+r_p)]` with `r_p = 2`.
6. Concatenate harmonic position codes with image and point tokens.
7. Apply two parallel cross-attention branches, one for rotation and one for translation.
8. Unflatten outputs into `[B, 384, 16, 32]`, aggregate through residual blocks to `[B, 768]`.
9. Regress `rot_log` and `tsl_log` with two MLP heads, concatenate to `ξ ∈ R^6`.
10. Update the current estimate with `T_pred = exp(ξ) @ T_init`; repeat for three inference iterations.

## Planned ANIMA File Layout
```text
src/anima_calib_projfusion/
├── __init__.py
├── config.py
├── geometry/
│   ├── se3.py
│   └── projection.py
├── data/
│   ├── camera_info.py
│   ├── kitti.py
│   ├── nuscenes.py
│   ├── perturbation.py
│   └── samplers.py
├── encoders/
│   ├── image_dinov2.py
│   ├── pointgpt.py
│   └── checkpoint_adapters.py
├── model/
│   ├── positional_encoding.py
│   ├── coordinate_alignment.py
│   ├── cross_attention.py
│   ├── aggregation.py
│   ├── heads.py
│   └── projfusion.py
├── inference/
│   ├── pipeline.py
│   ├── service.py
│   └── visualize.py
├── eval/
│   ├── metrics.py
│   ├── benchmark.py
│   └── report.py
├── api/
│   ├── schemas.py
│   └── app.py
└── ros2/
    ├── node.py
    └── bridge.py
```

## Paper-to-Implementation Decisions
- First milestone: reproduce the released PyTorch/CUDA path, including DINOv2 and PointGPT checkpoint handling, before any MLX or ONNX adaptation.
- The ANIMA package will use `src/anima_calib_projfusion/` even though the current scaffold still contains `src/anima_ebisu/`.
- Public repo truth wins when the paper and release diverge. The 40k/20k vs 8,192 point-count discrepancy is preserved as an explicit config toggle.
- API, Docker, and ROS2 integration are downstream ANIMA deliverables and are intentionally separated from the paper-faithful core-model PRDs.

## Risks and Validation Hooks
- CUDA/C++ ops in `models/tools/csrc` and PointGPT dependencies are likely the first reproduction blocker.
- Torch Hub DINOv2 loading may depend on a cached local checkout; the ANIMA implementation should vendor or explicitly pin this.
- Dataset coordinate conventions must be verified against the repo’s `camera_info` and `se3` utilities before training parity claims.
