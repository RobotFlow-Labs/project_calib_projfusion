# CALIB-PROJFUSION

## Paper
**ProjFusion: Camera-LiDAR Calibration (RA-L 2026)**
arXiv: https://arxiv.org/abs/2603.29414

## Module Identity
- Codename: CALIB-PROJFUSION
- Domain: Calibration
- Part of ANIMA Intelligence Compiler Suite

## Structure
```
project_calib_projfusion/
├── pyproject.toml
├── configs/
├── src/anima_calib_projfusion/
├── tests/
├── scripts/
├── papers/          # Paper PDF
├── CLAUDE.md        # This file
├── NEXT_STEPS.md
├── ASSETS.md
└── PRD.md
```

## Commands
```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Conventions
- Package manager: uv (never pip)
- Build backend: hatchling
- Python: >=3.10
- Config: TOML + Pydantic BaseSettings
- Lint: ruff
- Git commit prefix: [CALIB-PROJFUSION]

## Server Data Paths (GPU Server — datai_srv7)

### Datasets (already on disk — DO NOT download)
- Replica: /mnt/forge-data/datasets/replica/
- Replica CAD: /mnt/forge-data/datasets/replica_cad/
- TUM RGB-D: /mnt/forge-data/datasets/tum/
- TUM Dynamic: /mnt/forge-data/datasets/tum-rgbd-dynamic/
- nuScenes: /mnt/forge-data/datasets/nuscenes/
- KITTI: /mnt/forge-data/datasets/kitti/
- COCO: /mnt/forge-data/datasets/coco/
- COD10K: /mnt/forge-data/datasets/cod10k/
- NUAA-SIRST: /mnt/forge-data/datasets/nuaa_sirst_yolo/
- MCOD: /mnt/forge-data/datasets/mcod/
- OccAny checkpoints: /mnt/train-data/models/occany/

### Models (already on disk)
- DINOv2 ViT-B/14: /mnt/forge-data/models/dinov2_vitb14_pretrain.pth
- DINOv2 ViT-G/14: /mnt/forge-data/models/dinov2_vitg14_reg4_pretrain.pth
- SAM ViT-B: /mnt/forge-data/models/sam_vit_b_01ec64.pth
- SAM ViT-H: /mnt/forge-data/models/sam_vit_h_4b8939.pth
- SAM 2.1: /mnt/forge-data/models/sam2.1_hiera_base_plus.pt
- GroundingDINO: /mnt/forge-data/models/groundingdino_swint_ogc.pth
- YOLOv5l6: /mnt/forge-data/models/yolov5l6.pt
- YOLOv12n: /mnt/forge-data/models/yolov12n.pt
- YOLO11n: /mnt/forge-data/models/yolo11n.pt

### Output Paths
- Checkpoints: /mnt/artifacts-datai/checkpoints/$MODULE_NAME/
- Logs: /mnt/artifacts-datai/logs/$MODULE_NAME/
- Exports: /mnt/artifacts-datai/exports/$MODULE_NAME/

### Rules
- NEVER download datasets that already exist on disk
- ALWAYS use nohup+disown for training
- ALWAYS export TRT FP16 + TRT FP32 (MANDATORY)
- Use /anima-hf-strategy when training is done
