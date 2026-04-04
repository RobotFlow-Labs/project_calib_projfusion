# TRAINING_REPORT — CALIB-PROJFUSION

## Paper
**Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration Under Large Initial Perturbations**
arXiv: 2603.29414 | RA-L 2026

## Model
| Parameter | Value |
|-----------|-------|
| Architecture | ProjDualFusion |
| Image Encoder | DINOv2 ViT-S/14 (frozen) |
| Point Encoder | PointGPT-tiny (trainable) |
| Total Params | 25.3M |
| Trainable Params | 3.6M |
| Inference Latency | 38.5ms avg (L4) |

## Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 5e-4 |
| Weight Decay | 1e-2 |
| Scheduler | Cosine warmup (2 epochs) |
| Precision | FP16 mixed |
| Batch Size | 256 |
| GPU | NVIDIA L4 (23GB) |

## KITTI Results

### Training
| Metric | Value |
|--------|-------|
| Train Samples | 6,732 |
| Val Samples | 374 |
| Epochs | 18 (early stopped) |
| Best Val Loss | 0.0139 |

### Test Evaluation (3-step iterative refinement)
| Metric | Value |
|--------|-------|
| Test Samples | 375 |
| Perturbation | 10deg / 0.5m |
| Rotation RMSE | 5.77° |
| Translation RMSE | 27.88 cm |
| Success (1°/2.5cm) | 1.1% |
| Success (2°/5cm) | 3.5% |
| Success (10°/50cm) | 99.7% |

## nuScenes Results

### Training
| Metric | Value |
|--------|-------|
| Train Samples | 24,653 |
| Val Samples | 2,978 |
| Epochs | 14 (early stopped) |
| Best Val Loss | 0.0151 |

## Export Formats
| Format | KITTI | nuScenes |
|--------|-------|----------|
| PyTorch (.pth) | ✅ | ✅ |
| SafeTensors | ✅ | ✅ |
| ONNX | ✅ | ✅ |
| TensorRT FP16 | ✅ | ✅ |
| TensorRT FP32 | ✅ | ✅ |

## HuggingFace
Repository: [ilessio-aiflowlab/project_calib_projfusion](https://huggingface.co/ilessio-aiflowlab/project_calib_projfusion)

## Shared Infrastructure Produced
- KITTI point cloud cache: 7,481 frames, 381MB (fp16)
- KITTI DINOv2 features: 7,481 frames, 2.8GB (fp16)
- Triton batched 3D→2D projection kernel: 18.3B pts/s
