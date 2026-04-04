"""Typed settings for the CALIB-PROJFUSION module."""

from __future__ import annotations

import tomllib
from pathlib import Path

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
        "Native-Domain Cross-Attention for Camera-LiDAR Extrinsic"
        " Calibration Under Large Initial Perturbations"
    )


class RuntimeConfig(BaseModel):
    backend: str = "auto"
    device: str = "auto"
    precision: str = "fp16"
    torch_variant: str = "cu128"


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
    warmup_epochs: int = 2
    batch_size: int = 256
    precision: str = "fp16"
    gradient_clip: float = 1.0
    seed: int = 42


class DataConfig(BaseModel):
    kitti_root: Path = Path("/mnt/forge-data/datasets/kitti")
    nuscenes_root: Path = Path("/mnt/forge-data/datasets/nuscenes")
    pcd_sample_num: int = 8192
    max_deg: float = 10.0
    max_tran: float = 0.5
    num_workers: int = 4
    train_ratio: float = 0.90
    val_ratio: float = 0.05


class PretrainedConfig(BaseModel):
    dinov2: Path = Path(
        "/mnt/forge-data/models/calib_projfusion/dinov2_vits14_timm.pth"
    )


class CheckpointConfig(BaseModel):
    output_dir: Path = Path(
        "/mnt/artifacts-datai/checkpoints/project_calib_projfusion"
    )
    keep_top_k: int = 2
    metric: str = "val_loss"
    mode: str = "min"


class LoggingConfig(BaseModel):
    log_dir: Path = Path(
        "/mnt/artifacts-datai/logs/project_calib_projfusion"
    )


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
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
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
