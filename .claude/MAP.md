# ANIMA Shared Infrastructure Map
# Updated: 2026-04-03
# Location: /mnt/forge-data/shared_infra/

## Quick Start
```bash
# Bootstrap any module in 1 command:
bash /mnt/forge-data/shared_infra/bootstrap_module.sh /path/to/your/module
```

---

## CUDA Extensions (pre-compiled, py3.11, cu128, L4 arch 8.9)

| Extension | Path | Used By |
|-----------|------|---------|
| Gaussian semantic rasterizer | `/mnt/forge-data/shared_infra/cuda_extensions/gaussian_semantic_rasterization/` | All 3DGS SLAM (GS3LAM, CokO, MipSLAM) |
| Deformable attention (DETR) | `/mnt/forge-data/shared_infra/cuda_extensions/deformable_attention/` | LOKI, any DETR/DINO module |
| Deformable attention wheel | `.../deformable_attention/dist/ms_deform_attn-1.0-cp311-cp311-linux_x86_64.whl` | pip installable |
| EAA renderer (anti-alias) | `/mnt/forge-data/shared_infra/cuda_extensions/eaa_renderer/` | MipSLAM, anti-aliased SLAM |
| diff-gaussian-rasterization | `/mnt/forge-data/shared_infra/cuda_extensions/wheels_py311_cu128/` | All 3DGS modules |
| simple-knn | `/mnt/forge-data/shared_infra/cuda_extensions/wheels_py311_cu128/` | Gaussian splatting |
| BUILD_CU128.sh | `/mnt/forge-data/shared_infra/cuda_extensions/BUILD_CU128.sh` | Force cu128 for any CUDA build |

## Pre-Processed Datasets

| Dataset | Path | Size | Used By |
|---------|------|------|---------|
| Replica RGB-D (rendered) | `/mnt/forge-data/datasets/replica_rgbd/` | 17GB | All SLAM |
| Replica SLAM 2-agent | `/mnt/forge-data/datasets/replica_slam/` | 895MB | CokO multi-agent SLAM |
| COCO HDINO cache | `/mnt/forge-data/shared_infra/datasets/coco_hdino_cache/` | ~60GB | LOKI, DETR modules |
| Replica source meshes | `/mnt/forge-data/datasets/replica/` | — | DO NOT re-render |

## Raw Datasets (already on disk — DO NOT download)

| Dataset | Path |
|---------|------|
| COCO val+train | `/mnt/forge-data/datasets/coco/` + `/mnt/train-data/datasets/coco/` |
| nuScenes | `/mnt/forge-data/datasets/nuscenes/` |
| KITTI | `/mnt/forge-data/datasets/kitti/` |
| TUM RGB-D / TUM-VI | `/mnt/forge-data/datasets/tum/` |
| Replica | `/mnt/forge-data/datasets/replica/` |
| COD10K | `/mnt/forge-data/datasets/cod10k/` |
| MCOD | `/mnt/forge-data/datasets/mcod/` |
| NUAA-SIRST | `/mnt/forge-data/datasets/nuaa_sirst_yolo/` |

## Models (already on disk — DO NOT download)

| Model | Path |
|-------|------|
| DINOv2 ViT-B/14 | `/mnt/forge-data/models/dinov2_vitb14_pretrain.pth` |
| DINOv2 ViT-G/14 | `/mnt/forge-data/models/dinov2_vitg14_reg4_pretrain.pth` |
| DINOv2-Small | `/mnt/forge-data/models/facebook--dinov2-small/` |
| SAM ViT-B | `/mnt/forge-data/models/sam_vit_b_01ec64.pth` |
| SAM ViT-H | `/mnt/forge-data/models/sam_vit_h_4b8939.pth` |
| SAM 2.1 | `/mnt/forge-data/models/sam2.1_hiera_base_plus.pt` |
| GroundingDINO | `/mnt/forge-data/models/groundingdino_swint_ogc.pth` |
| YOLOv5l6 | `/mnt/forge-data/models/yolov5l6.pt` |
| YOLOv12n | `/mnt/forge-data/models/yolov12n.pt` |
| YOLO11n | `/mnt/forge-data/models/yolo11n.pt` |
| OccAny checkpoints | `/mnt/train-data/models/occany/` |

## Tools

| Tool | Path | Usage |
|------|------|-------|
| TRT export toolkit | `/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py` | `python export_to_trt.py --onnx model.onnx --output-dir exports/` |
| Bootstrap script | `/mnt/forge-data/shared_infra/bootstrap_module.sh` | `bash bootstrap_module.sh /path/to/module` |
| CUDA 12 build script | `/mnt/forge-data/shared_infra/cuda_extensions/BUILD_CU128.sh` | Force cu128 compilation |

## Output Paths

| Type | Path |
|------|------|
| Checkpoints | `/mnt/artifacts-datai/checkpoints/{module_name}/` |
| Logs | `/mnt/artifacts-datai/logs/{module_name}/` |
| Exports | `/mnt/artifacts-datai/exports/{module_name}/` |

## Rules
- ALWAYS install torch with cu128: `--index-url https://download.pytorch.org/whl/cu128`
- NEVER download datasets that already exist on disk
- ALWAYS use nohup+disown for training
- ALWAYS export TRT FP16 + TRT FP32 (use shared toolkit)
- Save ANY new CUDA kernels to `/mnt/forge-data/shared_infra/cuda_extensions/`
- Save ANY new pre-processed data to `/mnt/forge-data/shared_infra/datasets/`
