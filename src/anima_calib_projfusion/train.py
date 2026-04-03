"""ProjFusion CUDA-optimized training loop.

Config-driven, torch.compile, AMP mixed precision, cosine-warmup LR,
early stopping, checkpointing to /mnt/artifacts-datai/.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from anima_calib_projfusion.data.kitti import (
    KITTICalibDataset,
    collate_calib,
    make_kitti_splits,
)
from anima_calib_projfusion.geometry.se3 import se3_exp, se3_log, se3_inv
from anima_calib_projfusion.model.projfusion import ProjDualFusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("projfusion.train")

# ─── Paths ────────────────────────────────────────────────────
PROJECT = "project_calib_projfusion"
ARTIFACTS = Path("/mnt/artifacts-datai")
CHECKPOINT_DIR = ARTIFACTS / "checkpoints" / PROJECT
LOG_DIR = ARTIFACTS / "logs" / PROJECT


# ─── Loss ─────────────────────────────────────────────────────

def calibration_loss(
    rot_log: torch.Tensor,
    tsl_log: torch.Tensor,
    gt_log: torch.Tensor,
) -> torch.Tensor:
    """Smooth L1 loss on predicted vs ground-truth Lie algebra vectors."""
    pred = torch.cat([rot_log, tsl_log], dim=-1)  # [B, 6]
    return nn.functional.smooth_l1_loss(pred, gt_log, beta=0.5)


def se3_error(pred_T: torch.Tensor, gt_T: torch.Tensor):
    """Compute rotation (degrees) and translation (meters) errors."""
    delta = pred_T @ se3_inv(gt_T)
    # Rotation error from trace
    R = delta[:, :3, :3]
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_angle = ((tr - 1) / 2).clamp(-1, 1)
    rot_err_deg = torch.acos(cos_angle).abs() * 180 / math.pi  # [B]
    # Translation error
    tsl_err_m = delta[:, :3, 3].norm(dim=-1)  # [B]
    return rot_err_deg, tsl_err_m


# ─── LR Scheduler ────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state):
        self.current_step = state["current_step"]


# ─── Checkpoint Manager ──────────────────────────────────────

class CheckpointManager:
    def __init__(self, save_dir: Path, keep_top_k: int = 2, mode: str = "min"):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        self.mode = mode
        self.history: list[tuple[float, Path]] = []
        save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict, metric: float, step: int) -> None:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        while len(self.history) > self.keep_top_k:
            _, old = self.history.pop()
            old.unlink(missing_ok=True)
        # Always keep best
        best_val, best_path = self.history[0]
        best_dst = self.save_dir / "best.pth"
        if best_path != best_dst:
            import shutil
            shutil.copy2(best_path, best_dst)


# ─── Training ─────────────────────────────────────────────────

def train(
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    warmup_epochs: int = 2,
    num_workers: int = 4,
    max_deg: float = 10.0,
    max_tran: float = 0.5,
    compile_model: bool = True,
    resume: str | None = None,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    # ─── Data ──────────────────────────────────────
    logger.info("Loading KITTI dataset...")
    train_ds, val_ds, _ = make_kitti_splits(
        max_deg=max_deg, max_tran=max_tran, pcd_sample_num=8192,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_calib, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_calib, num_workers=num_workers,
        pin_memory=True,
    )

    # ─── Model ─────────────────────────────────────
    model = ProjDualFusion(dinov2_pretrained=True, freeze_encoders=True).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[MODEL] {n_total/1e6:.1f}M params, {n_train/1e6:.1f}M trainable")

    if compile_model:
        logger.info("[COMPILE] torch.compile on calibration head...")
        model.rotation_attention = torch.compile(model.rotation_attention)
        model.translation_attention = torch.compile(model.translation_attention)

    # ─── Optimizer ─────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler = GradScaler()

    # ─── Checkpointing ─────────────────────────────
    ckpt_mgr = CheckpointManager(CHECKPOINT_DIR, keep_top_k=2)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"train_{time.strftime('%Y%m%d_%H%M')}.log"
    metrics_file = LOG_DIR / "metrics.jsonl"

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10

    # ─── Resume ────────────────────────────────────
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"[RESUME] from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ─── Config log ────────────────────────────────
    logger.info(f"[CONFIG] epochs={epochs}, batch_size={batch_size}, lr={lr}")
    logger.info(f"[DATA] train={len(train_ds)}, val={len(val_ds)}")
    logger.info(f"[GPU] {torch.cuda.get_device_name()}")
    logger.info(f"[CKPT] save to {CHECKPOINT_DIR}")

    # ─── Train loop ────────────────────────────────
    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, epochs):
        model.train()
        model.image_encoder.eval()  # Keep frozen
        epoch_loss = 0.0
        t0 = time.time()

        for i, batch in enumerate(train_loader):
            img = batch["img"].to(device, non_blocking=True)
            pcd = batch["pcd"].to(device, non_blocking=True)
            init_ext = batch["init_extrinsic"].to(device, non_blocking=True)
            target = batch["pose_target"].to(device, non_blocking=True)
            ci = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch["camera_info"].items()
            }

            # Ground truth Lie algebra
            gt_log = se3_log(target)  # [B, 6]

            optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16):
                rot_log, tsl_log = model(img, pcd, init_ext, ci)
                loss = calibration_loss(rot_log, tsl_log, gt_log)

            # NaN check
            if torch.isnan(loss):
                logger.error("[FATAL] Loss is NaN at step %d — stopping", global_step)
                return

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if (i + 1) % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[Epoch {epoch+1}/{epochs}] Step {i+1}/{len(train_loader)} "
                    f"loss={loss.item():.4f} lr={lr_now:.2e}"
                )

        train_loss = epoch_loss / len(train_loader)
        dt = time.time() - t0

        # ─── Validation ────────────────────────────
        model.eval()
        val_loss = 0.0
        val_rot_err = 0.0
        val_tsl_err = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                img = batch["img"].to(device)
                pcd = batch["pcd"].to(device)
                init_ext = batch["init_extrinsic"].to(device)
                target = batch["pose_target"].to(device)
                ci = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch["camera_info"].items()
                }
                gt_log = se3_log(target)
                rot_log, tsl_log = model(img, pcd, init_ext, ci)
                loss = calibration_loss(rot_log, tsl_log, gt_log)
                val_loss += loss.item() * img.size(0)

                # SE3 error
                pred_xi = torch.cat([rot_log, tsl_log], dim=-1)
                pred_T = se3_exp(pred_xi)
                rot_err, tsl_err = se3_error(pred_T, target)
                val_rot_err += rot_err.sum().item()
                val_tsl_err += tsl_err.sum().item()
                n_val += img.size(0)

        val_loss /= max(n_val, 1)
        val_rot_err /= max(n_val, 1)
        val_tsl_err /= max(n_val, 1)

        logger.info(
            f"[Epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} rot_err={val_rot_err:.2f}° "
            f"tsl_err={val_tsl_err:.4f}m time={dt:.1f}s"
        )

        # Log metrics
        with open(metrics_file, "a") as f:
            json.dump({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rot_err_deg": val_rot_err,
                "val_tsl_err_m": val_tsl_err,
                "lr": optimizer.param_groups[0]["lr"],
                "time_s": dt,
            }, f)
            f.write("\n")

        # ─── Checkpoint ───────────────────────────
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": min(best_val_loss, val_loss),
        }
        ckpt_mgr.save(state, val_loss, global_step)

        # ─── Early stopping ───────────────────────
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[EARLY STOP] No improvement for {patience} epochs")
                break

    logger.info(f"[DONE] Best val_loss={best_val_loss:.4f}")
    logger.info(f"[CKPT] Best model at {CHECKPOINT_DIR / 'best.pth'}")


def main():
    parser = argparse.ArgumentParser(description="ProjFusion Training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-deg", type=float, default=10.0)
    parser.add_argument("--max-tran", type=float, default=0.5)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        max_deg=args.max_deg,
        max_tran=args.max_tran,
        compile_model=not args.no_compile,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
