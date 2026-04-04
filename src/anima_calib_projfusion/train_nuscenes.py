"""ProjFusion nuScenes training — reuses KITTI training infrastructure.

Differences from KITTI training:
- Uses nuScenes data loader (850 scenes, ~34k samples)
- Larger dataset → more steps per epoch
- Same model, optimizer, scheduler, loss
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from anima_calib_projfusion.data.kitti import collate_calib
from anima_calib_projfusion.data.nuscenes import make_nuscenes_splits
from anima_calib_projfusion.geometry.se3 import se3_exp, se3_log
from anima_calib_projfusion.model.projfusion import ProjDualFusion
from anima_calib_projfusion.train import (
    CheckpointManager,
    WarmupCosineScheduler,
    calibration_loss,
    se3_error,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("projfusion.train_nuscenes")

PROJECT = "project_calib_projfusion"
ARTIFACTS = Path("/mnt/artifacts-datai")
CHECKPOINT_DIR = ARTIFACTS / "checkpoints" / PROJECT / "nuscenes"
LOG_DIR = ARTIFACTS / "logs" / PROJECT


def train(
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    warmup_epochs: int = 2,
    num_workers: int = 4,
    max_deg: float = 10.0,
    max_tran: float = 0.5,
    resume: str | None = None,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    # ─── Data ──────────────────────────────────────
    logger.info("Loading nuScenes dataset...")
    train_ds, val_ds, _ = make_nuscenes_splits(
        max_deg=max_deg,
        max_tran=max_tran,
        pcd_sample_num=8192,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_calib,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_calib,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ─── Model ─────────────────────────────────────
    model = ProjDualFusion(dinov2_pretrained=True, freeze_encoders=True).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[MODEL] {n_total / 1e6:.1f}M params, {n_train / 1e6:.1f}M trainable")

    # ─── Optimizer ─────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    # ─── Checkpointing ─────────────────────────────
    ckpt_mgr = CheckpointManager(CHECKPOINT_DIR, keep_top_k=2)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = LOG_DIR / "nuscenes_metrics.jsonl"

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
        logger.info(f"[RESUME] from epoch {start_epoch}")

    logger.info(f"[CONFIG] epochs={epochs}, batch_size={batch_size}, lr={lr}")
    logger.info(f"[DATA] train={len(train_ds)}, val={len(val_ds)}")
    logger.info(f"[GPU] {torch.cuda.get_device_name()}")
    logger.info(f"[CKPT] save to {CHECKPOINT_DIR}")

    # ─── Train loop ────────────────────────────────
    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, epochs):
        model.train()
        model.image_encoder.eval()
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
            gt_log = se3_log(target)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                rot_log, tsl_log = model(img, pcd, init_ext, ci)
                loss = calibration_loss(rot_log, tsl_log, gt_log)

            if torch.isnan(loss):
                logger.error("[FATAL] Loss is NaN at step %d", global_step)
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
                    f"[Epoch {epoch + 1}/{epochs}] Step {i + 1}/{len(train_loader)} "
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
            f"[Epoch {epoch + 1}/{epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} rot_err={val_rot_err:.2f}° "
            f"tsl_err={val_tsl_err:.4f}m time={dt:.1f}s"
        )

        with open(metrics_file, "a") as f:
            json.dump(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_rot_err_deg": val_rot_err,
                    "val_tsl_err_m": val_tsl_err,
                    "lr": optimizer.param_groups[0]["lr"],
                    "time_s": dt,
                },
                f,
            )
            f.write("\n")

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
    parser = argparse.ArgumentParser(description="ProjFusion nuScenes Training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-deg", type=float, default=10.0)
    parser.add_argument("--max-tran", type=float, default=0.5)
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
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
