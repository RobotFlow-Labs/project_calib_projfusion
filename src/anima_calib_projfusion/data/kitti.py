"""KITTI Detection dataset adapter for camera-LiDAR calibration training.

Uses KITTI 3D Object Detection format on disk:
  /mnt/forge-data/datasets/kitti/training/{calib,velodyne,image_2}/

Each sample provides:
  - RGB image resized to 224×448
  - Velodyne point cloud filtered + resampled to 8192 points
  - Ground-truth extrinsic Tr_velo_to_cam (4×4)
  - Perturbed initial extrinsic
  - Camera intrinsics (fx, fy, cx, cy, sensor_h, sensor_w)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from anima_calib_projfusion.data.perturbation import sample_perturbation

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_DEFAULT_KITTI_ROOT = Path("/mnt/forge-data/datasets/kitti")


def _parse_calib(calib_path: Path) -> dict:
    """Parse KITTI calibration file → dict of numpy arrays."""
    data = {}
    with open(calib_path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, vals = line.split(":", 1)
            data[key.strip()] = np.array([float(x) for x in vals.split()], dtype=np.float32)
    return data


def _load_velodyne(path: Path) -> np.ndarray:
    """Load velodyne .bin → [N, 3] xyz points."""
    scan = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    return scan[:, :3]


def _filter_points(
    points: np.ndarray,
    min_dist: float = 0.1,
    max_depth: float = 50.0,
    positive_x: bool = True,
) -> np.ndarray:
    """Filter point cloud: remove close/far points, keep forward-facing."""
    dist = np.linalg.norm(points, axis=1)
    mask = dist > min_dist
    if positive_x:
        mask &= points[:, 0] > 0
    mask &= dist < max_depth
    return points[mask]


def _resample(points: np.ndarray, num: int) -> np.ndarray:
    """Randomly resample to exactly `num` points."""
    n = points.shape[0]
    idx = np.random.permutation(n)
    if n >= num:
        return points[idx[:num]]
    # Pad with random repeats
    extra = np.random.choice(n, num - n, replace=True)
    return points[np.concatenate([idx, extra])]


class KITTICalibDataset(Dataset):
    """KITTI Detection dataset for camera-LiDAR extrinsic calibration training.

    Each item returns a dict with:
        img: [3, 224, 448] normalized RGB
        pcd: [8192, 3] filtered point cloud
        gt_extrinsic: [4, 4] ground-truth Tr_velo_to_cam
        init_extrinsic: [4, 4] perturbed extrinsic (gt @ perturbation)
        pose_target: [4, 4] inverse perturbation (what model should predict)
        camera_info: dict with fx, fy, cx, cy, sensor_h, sensor_w
    """

    def __init__(
        self,
        root: str | Path = _DEFAULT_KITTI_ROOT,
        split: str = "train",
        image_hw: tuple[int, int] = (224, 448),
        pcd_sample_num: int = 8192,
        max_deg: float = 10.0,
        max_tran: float = 0.5,
        min_deg: float = 0.0,
        min_tran: float = 0.0,
        indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_hw = image_hw
        self.pcd_sample_num = pcd_sample_num
        self.max_deg = max_deg
        self.max_tran = max_tran
        self.min_deg = min_deg
        self.min_tran = min_tran

        # Determine split directory
        if split in ("train", "val"):
            base = self.root / "training"
        else:
            base = self.root / "testing"

        self.calib_dir = base / "calib"
        self.velodyne_dir = base / "velodyne"
        self.image_dir = base / "image_2"

        # Build frame list
        all_frames = sorted([p.stem for p in self.calib_dir.glob("*.txt")])
        if indices is not None:
            self.frames = [all_frames[i] for i in indices]
        else:
            self.frames = all_frames

        logger.info("KITTICalibDataset: %d frames from %s", len(self.frames), base)

        # Image transforms
        self.img_transform = T.Compose([
            T.Resize(image_hw),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        frame = self.frames[idx]

        # Load calibration
        calib = _parse_calib(self.calib_dir / f"{frame}.txt")
        P2 = calib["P2"].reshape(3, 4)
        R0 = np.eye(4, dtype=np.float32)
        R0[:3, :3] = calib["R0_rect"].reshape(3, 3)
        Tr_velo = np.eye(4, dtype=np.float32)
        Tr_velo[:3, :4] = calib["Tr_velo_to_cam"].reshape(3, 4)

        # Ground truth extrinsic: R0_rect @ Tr_velo_to_cam
        gt_extrinsic = R0 @ Tr_velo

        # Camera intrinsics from P2
        fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]

        # Load image
        img = Image.open(self.image_dir / f"{frame}.png").convert("RGB")
        orig_w, orig_h = img.size
        img_tensor = self.img_transform(img)

        # Scale intrinsics to match resized image
        scale_x = self.image_hw[1] / orig_w
        scale_y = self.image_hw[0] / orig_h
        fx_s, fy_s = fx * scale_x, fy * scale_y
        cx_s, cy_s = cx * scale_x, cy * scale_y

        # Load and filter point cloud
        pts = _load_velodyne(self.velodyne_dir / f"{frame}.bin")
        pts = _filter_points(pts)
        pts = _resample(pts, self.pcd_sample_num)
        pcd_tensor = torch.from_numpy(pts).float()

        # Sample random perturbation
        perturbation = sample_perturbation(
            1, self.max_deg, self.max_tran, self.min_deg, self.min_tran
        ).squeeze(0)  # [4, 4]

        gt_T = torch.from_numpy(gt_extrinsic).float()
        # Perturbed initial = perturbation @ gt
        init_T = perturbation @ gt_T

        # Target: the inverse perturbation (what model should regress as Lie algebra)
        pose_target = perturbation  # model predicts xi such that exp(xi) @ init ≈ gt

        camera_info = {
            "fx": torch.tensor(fx_s, dtype=torch.float32),
            "fy": torch.tensor(fy_s, dtype=torch.float32),
            "cx": torch.tensor(cx_s, dtype=torch.float32),
            "cy": torch.tensor(cy_s, dtype=torch.float32),
            "sensor_h": self.image_hw[0],
            "sensor_w": self.image_hw[1],
        }

        return {
            "img": img_tensor,
            "pcd": pcd_tensor,
            "gt_extrinsic": gt_T,
            "init_extrinsic": init_T,
            "pose_target": pose_target,
            "camera_info": camera_info,
        }


def collate_calib(batch: list[dict]) -> dict:
    """Custom collate that handles camera_info dict."""
    result = {}
    for key in batch[0]:
        if key == "camera_info":
            ci = {}
            for k in batch[0]["camera_info"]:
                vals = [b["camera_info"][k] for b in batch]
                if isinstance(vals[0], torch.Tensor):
                    ci[k] = torch.stack(vals)
                else:
                    ci[k] = vals[0]  # scalar, same for all
            result["camera_info"] = ci
        else:
            result[key] = torch.stack([b[key] for b in batch])
    return result


def make_kitti_splits(
    root: str | Path = _DEFAULT_KITTI_ROOT,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
    **kwargs,
) -> tuple[KITTICalibDataset, KITTICalibDataset, KITTICalibDataset]:
    """Create train/val/test splits from KITTI Detection."""
    root = Path(root)
    n_total = len(list((root / "training" / "calib").glob("*.txt")))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_idx = sorted(perm[:n_train].tolist())
    val_idx = sorted(perm[n_train:n_train + n_val].tolist())
    test_idx = sorted(perm[n_train + n_val:].tolist())

    return (
        KITTICalibDataset(root, "train", indices=train_idx, **kwargs),
        KITTICalibDataset(root, "val", indices=val_idx, **kwargs),
        KITTICalibDataset(root, "val", indices=test_idx, **kwargs),
    )
