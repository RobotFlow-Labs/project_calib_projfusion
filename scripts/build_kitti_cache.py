"""Build shared KITTI point cloud + DINOv2 feature caches.

Outputs:
  /mnt/forge-data/shared_infra/datasets/kitti_pointcloud_cache/  — per-frame .pt tensors
  /mnt/forge-data/shared_infra/datasets/kitti_dinov2_features/   — per-frame DINOv2 features
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cache_builder")

KITTI_ROOT = Path("/mnt/forge-data/datasets/kitti/training")
PCD_CACHE_DIR = Path("/mnt/forge-data/shared_infra/datasets/kitti_pointcloud_cache")
DINOV2_CACHE_DIR = Path("/mnt/forge-data/shared_infra/datasets/kitti_dinov2_features")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def filter_points(pts: np.ndarray, min_dist: float = 0.1, max_depth: float = 50.0) -> np.ndarray:
    dist = np.linalg.norm(pts, axis=1)
    mask = (dist > min_dist) & (dist < max_depth) & (pts[:, 0] > 0)
    return pts[mask]


def build_pointcloud_cache(num_points: int = 8192):
    """Cache filtered + resampled velodyne scans as GPU-ready tensors."""
    PCD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    velodyne_dir = KITTI_ROOT / "velodyne"
    files = sorted(velodyne_dir.glob("*.bin"))
    logger.info("Building point cloud cache: %d files → %s", len(files), PCD_CACHE_DIR)

    for f in tqdm(files, desc="PCD cache"):
        out_path = PCD_CACHE_DIR / f"{f.stem}.pt"
        if out_path.exists():
            continue
        scan = np.fromfile(str(f), dtype=np.float32).reshape(-1, 4)[:, :3]
        pts = filter_points(scan)
        # Resample
        n = pts.shape[0]
        idx = np.random.permutation(n)
        if n >= num_points:
            pts = pts[idx[:num_points]]
        else:
            extra = np.random.choice(n, num_points - n, replace=True)
            pts = pts[np.concatenate([idx, extra])]
        torch.save(torch.from_numpy(pts).half(), out_path)

    logger.info("Point cloud cache complete: %d files", len(files))


def build_dinov2_cache(batch_size: int = 64, device: str = "cuda"):
    """Extract DINOv2 features for all KITTI images."""
    DINOV2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    image_dir = KITTI_ROOT / "image_2"
    files = sorted(image_dir.glob("*.png"))
    logger.info("Building DINOv2 feature cache: %d images → %s", len(files), DINOV2_CACHE_DIR)

    from anima_calib_projfusion.encoders.image_dinov2 import DINOv2ImageEncoder
    encoder = DINOv2ImageEncoder(pretrained=True, freeze=True).to(device)

    transform = T.Compose([T.Resize((224, 448)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    # Process in batches
    uncached = [(f, DINOV2_CACHE_DIR / f"{f.stem}.pt") for f in files
                if not (DINOV2_CACHE_DIR / f"{f.stem}.pt").exists()]
    logger.info("  %d already cached, %d to process", len(files) - len(uncached), len(uncached))

    for i in tqdm(range(0, len(uncached), batch_size), desc="DINOv2 cache"):
        batch_files = uncached[i:i + batch_size]
        imgs = []
        for f, _ in batch_files:
            img = Image.open(f).convert("RGB")
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            features = encoder(batch)  # [B, 512, 384]

        for j, (_, out_path) in enumerate(batch_files):
            torch.save(features[j].cpu().half(), out_path)

    logger.info("DINOv2 feature cache complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd-only", action="store_true")
    parser.add_argument("--dinov2-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.dinov2_only:
        build_pointcloud_cache(args.num_points)
    if not args.pcd_only:
        build_dinov2_cache(args.batch_size, args.device)


if __name__ == "__main__":
    main()
