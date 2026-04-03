# CALIB-PROJFUSION: ProjFusion — Implementation PRD
## ANIMA Wave-7 Calibration Module

**Status:** Active implementation through PRD-06 scaffold  
**Version:** 0.2  
**Date:** 2026-04-03  
**Paper:** Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration Under Large Initial Perturbations  
**Paper Link:** https://arxiv.org/abs/2603.29414  
**Repo:** https://github.com/gitouni/ProjFusion  
**Local Reference Repo:** `repositories/ProjFusion`  
**Functional Name:** `CALIB-PROJFUSION`  
**Target Package:** `src/anima_calib_projfusion/`

## 1. Executive Summary
CALIB-PROJFUSION will reproduce the released ProjFusion calibration pipeline as a paper-faithful ANIMA module. The implementation target is a dual-branch extrinsic-aware cross-attention model that fuses DINOv2 image patches and PointGPT point groups in their native domains, predicts Lie-algebra extrinsic updates, and applies a three-step iterative refinement loop during inference.

## 2. Paper Verification Status
- [x] ArXiv ID verified
- [x] Local PDF present in `papers/2603.29414_ProjFusion.pdf`
- [x] Upstream repo identified and local reference repo available in `repositories/ProjFusion`
- [x] Paper read for architecture, datasets, metrics, and inference path
- [ ] Datasets confirmed on shared volume
- [ ] Backbone checkpoints materialized in module-local asset manifest
- [ ] Released repo reproduced end-to-end in this module
- **Verdict:** READY FOR PRD/TASK MANUFACTURING

## 3. What We Take From The Paper
- Native-domain feature extraction: DINOv2 image patch tokens and PointGPT point-group tokens remain in their original 2D/3D domains until cross-modal fusion.
- Extrinsic-aware coordinate alignment: transform point centroids by the current extrinsic estimate and project them into the image-feature plane before attention.
- Harmonic positional encoding with `n_h = 6` and projection margin `r_p = 2`.
- Dual cross-attention and dual aggregation branches for rotation and translation.
- Three-step iterative refinement at inference time.
- Paper metrics and dataset splits as the reproduction target, with repo-backed configs as the executable baseline.

## 4. What We Skip
- Re-implementing every baseline model from the paper inside this module.
- Reinforcement-learning codepaths and non-paper experimental branches in the reference repo.
- Nonessential visualization scripts that do not contribute to core calibration, evaluation, API, or ROS2 deployment.
- MLX-first training; parity on the released PyTorch/CUDA path comes first.

## 5. What We Adapt
- Rename stale scaffold identifiers (`EBISU`, `anima_ebisu`) to `CALIB-PROJFUSION`.
- Replace the repo’s YAML-include config layering with ANIMA-friendly TOML + Pydantic settings while preserving the same default values.
- Add explicit checkpoint adapters, API service layers, ROS2 nodes, and Docker packaging that are not part of the paper but are required by ANIMA.
- Preserve the public-repo point-count behavior (`8192`) as the first-class reproducible path, with a documented option for paper-faithful higher-count preprocessing.

## 6. Architecture
### Inputs
- RGB image: `Tensor[B, 3, 224, 448]`
- Filtered point cloud: `Tensor[B, N, 3]`, default `N=8192` for released-repo parity
- Initial extrinsic matrix: `Tensor[B, 4, 4]`
- Camera intrinsics / sensor info:
  - `fx`, `fy`, `cx`, `cy`
  - `sensor_h`, `sensor_w`

### Core Internal Shapes
- DINOv2 patch features: `Tensor[B, 384, 16, 32]` -> flattened to `Tensor[B, 512, 384]`
- PointGPT group features: `Tensor[B, 128, 384]`
- PointGPT centroids: `Tensor[B, 128, 3]`
- Projected aligned coordinates: `Tensor[B, 128, 2]`
- Cross-attention outputs: `Tensor[B, 512, 384]`
- Unflattened fusion map: `Tensor[B, 384, 16, 32]`
- Aggregated branch embedding: `Tensor[B, 768]`
- Predicted Lie update: `Tensor[B, 6]` split into `rot_log: Tensor[B, 3]` and `tsl_log: Tensor[B, 3]`

### Inference Update Rule
`T_pred = exp(ξ) @ T_init`, repeated for three iterations.

## 7. Datasets
| Dataset | Splits | Notes |
|---------|--------|-------|
| KITTI Odometry | Train `00,02-08,10,12,21`; Val `11,17,20`; Test `13,14,15,16,18` | Paper §IV-A |
| nuScenes | Official split, reserve 20% of train for validation | Paper §IV-A |

## 8. Success Criteria
- The ANIMA core model matches the released ProjFusion architecture and configuration defaults.
- Evaluation code reproduces paper metrics and success-rate definitions exactly.
- KITTI `10° / 50 cm`: `L1 >= 41.0%`, `L2 >= 87.5%`.
- nuScenes `10° / 50 cm`: `L1 >= 90.0%`, `L2 >= 99.0%`.
- The inference path performs three-step iterative refinement and can emit intermediate poses and projection overlays.
- The module exposes API, Docker, and ROS2 interfaces without changing the core calibration math.

## 9. Risk Assessment
- The public repo relies on custom CUDA ops and PointGPT dependencies that may block portable reproduction.
- The paper and released config disagree on point-count preprocessing; this can affect claims of strict paper parity.
- Backbone checkpoint provenance is partially implicit in the repo and must be made explicit in ANIMA assets.
- The current repo scaffold is stale and could cause namespace drift if not corrected in the first implementation phase.

## 10. Build Plan
| PRD# | Task | Status |
|------|------|--------|
| [PRD-01](prds/PRD-01-foundation.md) | Foundation, package rename, configs, geometry, datasets | ✅ |
| [PRD-02](prds/PRD-02-core-model.md) | Core ProjFusion model, encoders, cross-attention, dual heads | ✅ |
| [PRD-03](prds/PRD-03-inference.md) | Checkpoint loading, three-step iterative inference, CLI, visualization | ✅ |
| [PRD-04](prds/PRD-04-evaluation.md) | Metrics, benchmark runners, Table I/II style evaluation | ✅ |
| [PRD-05](prds/PRD-05-api-docker.md) | FastAPI serving, Docker image, health checks | ✅ |
| [PRD-06](prds/PRD-06-ros2-integration.md) | ROS2 node, topic bridge, launch configuration | ✅ |
| [PRD-07](prds/PRD-07-production.md) | Production hardening, export, ops validation, release packaging | ✅ |

## 11. Supporting Artifacts
- Asset manifest: `ASSETS.md`
- Pipeline mapping: `PIPELINE_MAP.md`
- PRD index: `prds/README.md`
- Build tasks: `tasks/`

## 12. Immediate Next Implementation Order
1. Materialize real datasets and pretrained checkpoints on the shared volume.
2. Replace stub encoder wrappers with released-backbone checkpoint loading.
3. Wire checkpoint-aware readiness into API and ROS runtime.
4. Run real benchmark reproduction against KITTI and nuScenes.
