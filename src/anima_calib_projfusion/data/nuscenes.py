"""nuScenes dataset for camera-LiDAR calibration training.

Uses nuscenes-devkit to load CAM_FRONT + LIDAR_TOP pairs with
calibrated extrinsic transforms. Follows reference repo dataset.py.

Data path: /mnt/forge-data/datasets/nuscenes/
"""
from __future__ import annotations

import logging
import os
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

_DEFAULT_NUSCENES_ROOT = Path("/mnt/forge-data/datasets/nuscenes")


def _inv_pose(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 pose matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=T.dtype)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def _filter_points(pts: np.ndarray, min_dist: float = 0.15, max_depth: float = 50.0) -> np.ndarray:
    dist = np.linalg.norm(pts, axis=1)
    mask = (dist > min_dist) & (dist < max_depth)
    return pts[mask]


def _resample(pts: np.ndarray, num: int) -> np.ndarray:
    n = pts.shape[0]
    idx = np.random.permutation(n)
    if n >= num:
        return pts[idx[:num]]
    extra = np.random.choice(n, num - n, replace=True)
    return pts[np.concatenate([idx, extra])]


class NuScenesCalibDataset(Dataset):
    """nuScenes dataset for camera-LiDAR extrinsic calibration.

    Each item returns a dict with:
        img: [3, 224, 448] normalized RGB
        pcd: [8192, 3] filtered LiDAR point cloud
        gt_extrinsic: [4, 4] ground-truth extrinsic (lidar→camera)
        init_extrinsic: [4, 4] perturbed extrinsic
        pose_target: [4, 4] perturbation matrix
        camera_info: dict with fx, fy, cx, cy, sensor_h, sensor_w
    """

    def __init__(
        self,
        root: str | Path = _DEFAULT_NUSCENES_ROOT,
        version: str = "v1.0-trainval",
        image_hw: tuple[int, int] = (224, 448),
        pcd_sample_num: int = 8192,
        max_deg: float = 10.0,
        max_tran: float = 0.5,
        min_deg: float = 0.0,
        min_tran: float = 0.0,
        cam_channel: str = "CAM_FRONT",
        lidar_channel: str = "LIDAR_TOP",
        scene_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_hw = image_hw
        self.pcd_sample_num = pcd_sample_num
        self.max_deg = max_deg
        self.max_tran = max_tran
        self.min_deg = min_deg
        self.min_tran = min_tran
        self.cam_channel = cam_channel
        self.lidar_channel = lidar_channel

        # Load nuScenes metadata
        from nuscenes.nuscenes import NuScenes
        logger.info("Loading nuScenes %s from %s ...", version, root)
        self.nusc = NuScenes(version=version, dataroot=str(root), verbose=False)

        # Build sample list: (sample_token, scene_name)
        scenes = sorted(self.nusc.scene, key=lambda s: s["name"])
        if scene_indices is not None:
            scenes = [scenes[i] for i in scene_indices]

        all_tokens = []
        for scene in scenes:
            token = scene["first_sample_token"]
            for _ in range(scene["nbr_samples"]):
                all_tokens.append(token)
                sample = self.nusc.get("sample", token)
                token = sample["next"]
                if token == "":
                    break

        # Filter to only samples where both image and lidar files exist on disk
        self.samples = []
        skipped = 0
        for token in all_tokens:
            sample = self.nusc.get("sample", token)
            cam_sd = self.nusc.get("sample_data", sample["data"][cam_channel])
            lidar_sd = self.nusc.get("sample_data", sample["data"][lidar_channel])
            img_path = os.path.join(str(root), cam_sd["filename"])
            pcl_path = os.path.join(str(root), lidar_sd["filename"])
            if os.path.exists(img_path) and os.path.exists(pcl_path):
                self.samples.append(token)
            else:
                skipped += 1

        logger.info(
            "NuScenesCalibDataset: %d samples available (%d skipped due to missing files), %d scenes",
            len(self.samples), skipped, len(scenes),
        )

        # Image transforms
        self.img_transform = T.Compose([
            T.Resize(image_hw),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _get_extrinsic_and_intrinsic(self, sample_record: dict):
        """Compute lidar→camera extrinsic and camera intrinsic."""
        # Get sensor data tokens
        cam_token = sample_record["data"][self.cam_channel]
        lidar_token = sample_record["data"][self.lidar_channel]

        cam_sd = self.nusc.get("sample_data", cam_token)
        lidar_sd = self.nusc.get("sample_data", lidar_token)

        # Calibrated sensor records
        cam_cs = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])

        # Build 4x4 poses (sensor → ego)
        from pyquaternion import Quaternion

        pose_lidar = np.eye(4, dtype=np.float32)
        pose_lidar[:3, :3] = Quaternion(lidar_cs["rotation"]).rotation_matrix.astype(np.float32)
        pose_lidar[:3, 3] = np.array(lidar_cs["translation"], dtype=np.float32)

        pose_cam = np.eye(4, dtype=np.float32)
        pose_cam[:3, :3] = Quaternion(cam_cs["rotation"]).rotation_matrix.astype(np.float32)
        pose_cam[:3, 3] = np.array(cam_cs["translation"], dtype=np.float32)

        # Extrinsic: lidar → camera = inv(pose_cam) @ pose_lidar
        extrinsic = _inv_pose(pose_cam) @ pose_lidar

        # Intrinsic: 3x3
        intrinsic = np.array(cam_cs["camera_intrinsic"], dtype=np.float32)

        return extrinsic, intrinsic, cam_sd, lidar_sd

    def __getitem__(self, idx: int) -> dict:
        sample_token = self.samples[idx]
        sample = self.nusc.get("sample", sample_token)

        extrinsic, intrinsic, cam_sd, lidar_sd = self._get_extrinsic_and_intrinsic(sample)

        # Load image
        img_path = os.path.join(self.nusc.dataroot, cam_sd["filename"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Scale intrinsics
        scale_x = self.image_hw[1] / orig_w
        scale_y = self.image_hw[0] / orig_h
        fx = intrinsic[0, 0] * scale_x
        fy = intrinsic[1, 1] * scale_y
        cx = intrinsic[0, 2] * scale_x
        cy = intrinsic[1, 2] * scale_y

        img_tensor = self.img_transform(img)

        # Load point cloud
        pcl_path = os.path.join(self.nusc.dataroot, lidar_sd["filename"])
        from nuscenes.utils.data_classes import LidarPointCloud
        pc = LidarPointCloud.from_file(pcl_path)
        pts = pc.points.T[:, :3].astype(np.float32)  # [N, 3]

        pts = _filter_points(pts)
        pts = _resample(pts, self.pcd_sample_num)
        pcd_tensor = torch.from_numpy(pts).float()

        # Perturbation
        perturbation = sample_perturbation(
            1, self.max_deg, self.max_tran, self.min_deg, self.min_tran
        ).squeeze(0)

        gt_T = torch.from_numpy(extrinsic).float()
        init_T = perturbation @ gt_T

        camera_info = {
            "fx": torch.tensor(fx, dtype=torch.float32),
            "fy": torch.tensor(fy, dtype=torch.float32),
            "cx": torch.tensor(cx, dtype=torch.float32),
            "cy": torch.tensor(cy, dtype=torch.float32),
            "sensor_h": self.image_hw[0],
            "sensor_w": self.image_hw[1],
        }

        return {
            "img": img_tensor,
            "pcd": pcd_tensor,
            "gt_extrinsic": gt_T,
            "init_extrinsic": init_T,
            "pose_target": perturbation,
            "camera_info": camera_info,
        }


def make_nuscenes_splits(
    root: str | Path = _DEFAULT_NUSCENES_ROOT,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> tuple[NuScenesCalibDataset, NuScenesCalibDataset, NuScenesCalibDataset]:
    """Create train/val/test splits from nuScenes scenes."""
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(root), verbose=False)
    n_scenes = len(nusc.scene)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_scenes)

    n_train = int(n_scenes * train_ratio)
    n_val = int(n_scenes * val_ratio)
    train_idx = sorted(perm[:n_train].tolist())
    val_idx = sorted(perm[n_train:n_train + n_val].tolist())
    test_idx = sorted(perm[n_train + n_val:].tolist())

    return (
        NuScenesCalibDataset(root, scene_indices=train_idx, **kwargs),
        NuScenesCalibDataset(root, scene_indices=val_idx, **kwargs),
        NuScenesCalibDataset(root, scene_indices=test_idx, **kwargs),
    )
