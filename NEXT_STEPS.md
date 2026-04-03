# CALIB-PROJFUSION — Execution Ledger

Resume rule: Read this file completely before writing code.
This project covers exactly one paper: ProjFusion / Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration Under Large Initial Perturbations.

## 1. Working Rules
- Work only inside `project_calib_projfusion/`
- Prefix every commit with `[CALIB-PROJFUSION]`
- Use `uv` for environment and dependency operations
- Target Python `3.11`, torch 2.10.0+cu128

## 2. Current Status
- **Date**: 2026-04-03
- **Phase**: Training COMPLETE, exports DONE
- **MVP Readiness**: 90%

## 3. Training Results
- **Dataset**: KITTI Detection (6732 train / 374 val / 375 test)
- **Best epoch**: 18/28 (early stopped at 28)
- **Best val_loss**: 0.0139
- **Rotation error**: ~4.8°
- **Translation error**: ~0.23m
- **Batch size**: 256 (61% VRAM on L4)
- **Total training time**: 35 minutes on 1x NVIDIA L4

## 4. Exports Complete
- [x] best.pth (97MB)
- [x] model.safetensors (97MB)
- [x] model.onnx + data (101MB)
- [x] TRT FP16 (57.9MB)
- [x] TRT FP32 (106.1MB)
- Path: `/mnt/artifacts-datai/exports/project_calib_projfusion/`

## 5. Inference Performance
- Latency: 38.5ms avg, 41.1ms p95 (batch=1, L4 GPU)
- Throughput: ~26 FPS

## 6. Shared Infrastructure Created
- [x] KITTI point cloud cache: 7481 frames, 381MB fp16
      → `/mnt/forge-data/shared_infra/datasets/kitti_pointcloud_cache/`
- [x] KITTI DINOv2-S/14 features: 7481 frames, 2.8GB fp16
      → `/mnt/forge-data/shared_infra/datasets/kitti_dinov2_features/`
- [x] Triton batched 3D→2D projection kernel: 18.3B pts/s
      → `/mnt/forge-data/shared_infra/cuda_extensions/batched_projection/`
- [x] Updated `/mnt/forge-data/shared_infra/MAP.md`
- [x] Pipeline benchmark: `/mnt/artifacts-datai/logs/project_calib_projfusion/pipeline_benchmark.json`

## 7. Real Implementations Built
- [x] DINOv2 ViT-S/14 encoder via timm (pretrained, frozen, 21.7M params)
- [x] PointGPT-tiny encoder (pure PyTorch, no KNN_CUDA dep, trainable)
- [x] SE(3) Lie algebra (exp/log/inv) ported from reference repo
- [x] KITTI Detection data loader with random perturbation sampling
- [x] Camera-LiDAR projection (align_point_groups)
- [x] Full ProjDualFusion model (25.3M params, 3.6M trainable)
- [x] Config-driven training (AMP fp16, cosine-warmup, early stopping)
- [x] Checkpoint manager (keep top-2 by val_loss)
- [x] Export pipeline (pth + safetensors + ONNX + TRT FP16/FP32)

## 8. TODO
- [x] Push to HuggingFace: https://huggingface.co/ilessio-aiflowlab/project_calib_projfusion
- [ ] nuScenes data loader + nuScenes training run
- [ ] Dockerfile.serve + docker-compose.serve.yml + serve.py
- [ ] Registry entry in anima-infra-main
- [ ] Three-step iterative refinement inference (paper §IV-B)
- [ ] Full paper metrics evaluation (Table I/II style)

## 9. Blocking
- None

## 10. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | PRD suite and task bundle created from paper + reference repo |
| 2026-04-03 | Codex | Replaced stale scaffold, implemented PRD-01 through PRD-07 |
| 2026-04-03 | Opus 4.6 | CUDA pipeline: real encoders, data loader, SE3, training loop. Shared caches (PCD 381MB + DINOv2 2.8GB). Triton kernel (18.3B pts/s). Training on GPU 4: 28 epochs, val_loss=0.0139, rot=4.8°, tsl=0.23m. Full export: pth+safetensors+ONNX+TRT. Pushed to HF. |
