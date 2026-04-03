# CALIB-PROJFUSION Model Card

## Overview
CALIB-PROJFUSION is the ANIMA Wave-7 calibration module derived from the ProjFusion paper on camera-LiDAR extrinsic calibration under large initial perturbations.

## Current Release State
- Foundation, model scaffold, inference, evaluation, API, and ROS bridge are implemented and locally validated.
- Real dataset wiring and pretrained checkpoint integration are still pending.
- Current macOS smoke tests use offline-safe encoder wrappers with the same tensor contracts as the target paper implementation.

## Intended Datasets
- KITTI Odometry
- nuScenes

## Intended Runtime Modes
- macOS local smoke development
- Linux CUDA training/inference with `uv sync --extra cuda`

## Graceful Degradation
- If no real checkpoint is available, use the smoke-test path only.
- Production release bundles should not be treated as benchmark-valid until checkpoint provenance and evaluation reports are attached.
