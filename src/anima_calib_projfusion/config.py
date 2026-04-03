"""Typed settings for the CALIB-PROJFUSION module."""

from __future__ import annotations

from pathlib import Path
import tomllib

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    name: str = "anima-calib-projfusion"
    codename: str = "CALIB-PROJFUSION"
    functional_name: str = "CALIB-PROJFUSION"
    wave: int = 7
    python_version: str = "3.11"
    paper_arxiv: str = "2603.29414"
    paper_title: str = (
        "Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration Under Large "
        "Initial Perturbations"
    )


class RuntimeConfig(BaseModel):
    backend: str = "auto"
    device: str = "auto"
    precision: str = "fp32"
    torch_variant: str = "cpu"


class ModelConfig(BaseModel):
    image_hw: tuple[int, int] = (224, 448)
    patch_size: int = 14
    feature_dim: int = 384
    feature_hw: tuple[int, int] = (16, 32)
    point_count: int = 8192
    num_groups: int = 128
    group_size: int = 64
    harmonic_functions: int = 6
    projection_margin: float = 2.0
    attention_heads: int = 6
    attention_dim_head: int = 64
    aggregation_planes: int = 96
    mlp_hidden_dims: tuple[int, int] = (128, 128)
    iterative_refinement_steps: int = 3


class TrainingConfig(BaseModel):
    optimizer: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 1e-2
    scheduler: str = "cosine-warmup"
    epochs: int = 30
    warmup_steps: int = 2


class DataConfig(BaseModel):
    dataset_root: Path = Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets/datasets")
    artifact_root: Path = Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets/artifacts")
    model_root: Path = Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets/models")
    kitti_root: Path = Path("/mnt/forge-data/datasets/kitti_odometry")
    nuscenes_root: Path = Path("/mnt/forge-data/datasets/nuscenes")


class PretrainedConfig(BaseModel):
    dinov2: Path = Path("/mnt/forge-data/models/vision/dinov2_vits14")
    pointgpt_kitti: Path = Path("/mnt/forge-data/models/pointgpt/kitti_pointgpt_tiny.pth")
    pointgpt_nuscenes: Path = Path("/mnt/forge-data/models/pointgpt/nuscenes_pointgpt_tiny.ckpt")


class HardwareConfig(BaseModel):
    zed2i: bool = True
    unitree_l2_lidar: bool = True
    cobot_xarm6: bool = False


class ProjFusionSettings(BaseSettings):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    pretrained: PretrainedConfig = Field(default_factory=PretrainedConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    model_config = SettingsConfigDict(
        env_prefix="ANIMA_CALIB_PROJFUSION_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def from_toml(cls, path: str | Path) -> "ProjFusionSettings":
        with Path(path).open("rb") as handle:
            payload = tomllib.load(handle)
        return cls.model_validate(payload)

    @property
    def package_name(self) -> str:
        return "anima_calib_projfusion"

    @property
    def image_hw(self) -> tuple[int, int]:
        return self.model.image_hw

    @property
    def feature_hw(self) -> tuple[int, int]:
        return self.model.feature_hw
