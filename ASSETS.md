# CALIB-PROJFUSION — Asset Manifest

## Paper
- Title: Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration Under Large Initial Perturbations
- ArXiv: 2603.29414
- Authors: Ni Ou, Zhuo Chen, Xinru Zhang, Junzheng Wang
- Local PDF: `papers/2603.29414_ProjFusion.pdf`
- Reference repo: `repositories/ProjFusion`
- Upstream repo: `https://github.com/gitouni/ProjFusion`

## Status: ALMOST

The paper PDF and reference code are present locally. Public dataset links and pretrained backbone references are known, but the project-local ANIMA package, reproducible checkpoints, and shared-volume asset mapping are not yet finalized.

## Reproducibility Notes
- Paper Section IV-B reports point-cloud downsampling of 40,000 points for KITTI and 20,000 points for nuScenes.
- The released repo config and PointGPT grouping resolve to `num_group=128`, `group_size=64`, which implies an effective model input of 8,192 points.
- PRDs in this repo treat the released code path as the first implementation target and preserve a documented hook for a stricter paper-faithful variant.

## Pretrained Weights
| Model | Size | Source | Path on Server | Status |
|-------|------|--------|---------------|--------|
| DINOv2-tiny (`dinov2_vits14`) | ViT-S/14, 384-dim | Torch Hub / DINOv2 | `/mnt/forge-data/models/vision/dinov2_vits14` | MISSING |
| PointGPT tiny, KITTI | 384-dim, 128 groups | Reference repo config `cfg/pointgpt/finetune_kitti_tiny.yaml` | `/mnt/forge-data/models/pointgpt/kitti_pointgpt_tiny.pth` | MISSING |
| PointGPT tiny, nuScenes | 384-dim, 128 groups | Reference repo config `cfg/pointgpt/finetune_nuscenes_tiny.yaml` | `/mnt/forge-data/models/pointgpt/nuscenes_pointgpt_tiny.ckpt` | MISSING |
| ProjFusion reproduced checkpoint, KITTI | calibration head | Trained from this module | `/mnt/forge-data/models/calib_projfusion/kitti/best_model.pth` | MISSING |
| ProjFusion reproduced checkpoint, nuScenes | calibration head | Trained from this module | `/mnt/forge-data/models/calib_projfusion/nuscenes/best_model.pth` | MISSING |

## Datasets
| Dataset | Size | Split | Source | Path | Status |
|---------|------|-------|--------|------|--------|
| KITTI Odometry RGB + Velodyne + calibration | seqs 00-21 subset | Train: `00,02-08,10,12,21`; Val: `11,17,20`; Test: `13,14,15,16,18` | `https://www.cvlibs.net/datasets/kitti/eval_odometry.php` | `/mnt/forge-data/datasets/kitti_odometry` | MISSING |
| nuScenes v1.0-trainval | 850 scenes | Official train split, reserve 20% train for val | `https://www.nuscenes.org/nuscenes#download` | `/mnt/forge-data/datasets/nuscenes` | MISSING |
| nuScenes v1.0-test | official test scenes | Test | `https://www.nuscenes.org/nuscenes#download` | `/mnt/forge-data/datasets/nuscenes` | MISSING |

## Hyperparameters (from paper and released repo)
| Param | Value | Paper / Repo Section |
|-------|-------|----------------------|
| image_encoder | `DINOv2-tiny (dinov2_vits14)` | Paper §IV-B |
| point_encoder | `PointGPT-tiny` | Paper §IV-B |
| feature_dim | `384` | Paper §IV-B |
| image_size | `224 x 448` | Paper §IV-B |
| harmonic_functions | `6` | Paper §III-C / §IV-B |
| projection_margin `r_p` | `2.0` | Paper §III-C / §IV-B |
| attention_heads | `6` | Paper §IV-B |
| attention_dim_head | `64` | Paper §IV-B |
| aggregation_planes | `96` | Paper §IV-B / repo `cfg/model/module/aggregation/miniaggregation.yml` |
| mlp_hidden_dims | `[128, 128]` | Paper §IV-B / repo `cfg/model/projdualfusion_harmonic.yml` |
| optimizer | `AdamW` | Released repo `cfg/model/module/adamw.yml` |
| lr | `5e-4` | Released repo `cfg/model/module/adamw.yml` |
| weight_decay | `1e-2` | Released repo `cfg/model/module/adamw.yml` |
| scheduler | `cosine-warmup` | Released repo `cfg/model/module/cosine-warmup.yml` |
| epochs | `30` | Released repo `cfg/model/module/run.yml` |
| warmup_steps | `2` | Released repo `cfg/model/module/cosine-warmup.yml` |
| iterative_refinement_steps | `3` | Paper §IV-B / repo `test.py` |

## Expected Metrics (from paper)
| Benchmark | Metric | Paper Value | Our Target |
|-----------|--------|-------------|-----------|
| KITTI, `10° / 50 cm` | L1 success rate | `41.04%` | `>= 41.0%` |
| KITTI, `10° / 50 cm` | L2 success rate | `87.68%` | `>= 87.5%` |
| KITTI, `10° / 50 cm` | paper summary headline | `88%` accurate cases | `match Table I L2` |
| nuScenes, `10° / 50 cm` | L1 success rate | `90%` | `>= 90.0%` |
| nuScenes, `10° / 50 cm` | L2 success rate | `99%` | `>= 99.0%` |

## Reference File Anchors
- Training entry: `repositories/ProjFusion/train.py`
- Evaluation / iterative refinement: `repositories/ProjFusion/test.py`
- Dataset logic: `repositories/ProjFusion/dataset.py`
- Main model wrapper: `repositories/ProjFusion/models/model.py`
- Core fusion encoder: `repositories/ProjFusion/models/tools/core.py`
- Attention implementation: `repositories/ProjFusion/models/tools/attention.py`

## Local Gaps To Close
- Rename stale scaffold identifiers from `EBISU` / `anima_ebisu` to `CALIB-PROJFUSION` / `anima_calib_projfusion`.
- Decide whether the first milestone reproduces the released 8,192-point setup or a stricter paper-faithful 40k/20k pre-group pipeline.
- Materialize shared-volume paths for KITTI, nuScenes, and pretrained backbones before starting reproduction work.
