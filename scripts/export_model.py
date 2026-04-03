"""Export ProjFusion model to all required formats.

Pipeline: best.pth → safetensors → ONNX → TRT FP16 → TRT FP32
Push to HuggingFace: ilessio-aiflowlab/project_calib_projfusion
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export")

CHECKPOINT_DIR = Path("/mnt/artifacts-datai/checkpoints/project_calib_projfusion")
EXPORT_DIR = Path("/mnt/artifacts-datai/exports/project_calib_projfusion")


def export_safetensors(model_state: dict, out_path: Path):
    """Export to safetensors format."""
    from safetensors.torch import save_file
    save_file(model_state, str(out_path))
    logger.info("Saved safetensors: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def export_onnx(model, out_path: Path, device: str = "cuda"):
    """Export to ONNX format."""
    model.eval()
    B = 1
    img = torch.randn(B, 3, 224, 448, device=device)
    pcd = torch.randn(B, 8192, 3, device=device)
    ext = torch.eye(4, device=device).unsqueeze(0)
    ci = {
        "fx": torch.tensor([300.0], device=device),
        "fy": torch.tensor([300.0], device=device),
        "cx": torch.tensor([224.0], device=device),
        "cy": torch.tensor([112.0], device=device),
        "sensor_h": 224,
        "sensor_w": 448,
    }

    # ONNX export requires wrapper that takes tensors
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, img, pcd, ext, fx, fy, cx, cy):
            ci = {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                  "sensor_h": 224, "sensor_w": 448}
            return self.model(img, pcd, ext, ci)

    wrapper = OnnxWrapper(model)

    torch.onnx.export(
        wrapper,
        (img, pcd, ext, ci["fx"], ci["fy"], ci["cx"], ci["cy"]),
        str(out_path),
        input_names=["image", "pointcloud", "extrinsic", "fx", "fy", "cx", "cy"],
        output_names=["rot_log", "tsl_log"],
        dynamic_axes={
            "image": {0: "batch"},
            "pointcloud": {0: "batch"},
            "extrinsic": {0: "batch"},
        },
        opset_version=17,
    )
    logger.info("Saved ONNX: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def export_trt(onnx_path: Path, out_dir: Path, precision: str = "fp16"):
    """Export to TensorRT using shared toolkit."""
    import subprocess
    trt_script = "/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py"
    if not Path(trt_script).exists():
        logger.warning("TRT toolkit not found at %s, skipping TRT export", trt_script)
        return

    cmd = [
        "python", trt_script,
        "--onnx", str(onnx_path),
        "--output-dir", str(out_dir),
        "--precision", precision,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("TRT %s export done: %s", precision, out_dir)
    else:
        logger.warning("TRT %s export failed: %s", precision, result.stderr[:500])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "best.pth"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    from anima_calib_projfusion.model.projfusion import ProjDualFusion
    model = ProjDualFusion(dinov2_pretrained=True, freeze_encoders=True).to(args.device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Get training metrics
    train_loss = ckpt.get("train_loss", "N/A")
    val_loss = ckpt.get("val_loss", "N/A")
    epoch = ckpt.get("epoch", "N/A")
    logger.info(f"Checkpoint: epoch={epoch}, train_loss={train_loss}, val_loss={val_loss}")

    # 1. Save best.pth (already exists)
    best_pth = EXPORT_DIR / "best.pth"
    torch.save(ckpt["model"], best_pth)
    logger.info("Saved pth: %s", best_pth)

    # 2. Safetensors
    st_path = EXPORT_DIR / "model.safetensors"
    export_safetensors(ckpt["model"], st_path)

    # 3. ONNX
    onnx_path = EXPORT_DIR / "model.onnx"
    export_onnx(model, onnx_path, args.device)

    # 4. TRT FP16
    export_trt(onnx_path, EXPORT_DIR / "trt_fp16", "fp16")

    # 5. TRT FP32
    export_trt(onnx_path, EXPORT_DIR / "trt_fp32", "fp32")

    # 6. Model card
    card = EXPORT_DIR / "README.md"
    card.write_text(f"""# ProjFusion — Camera-LiDAR Extrinsic Calibration

**Paper:** Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration
**arXiv:** 2603.29414
**Module:** ANIMA CALIB-PROJFUSION

## Training
- Dataset: KITTI Detection (6732 train, 374 val)
- Epochs: {epoch + 1 if isinstance(epoch, int) else epoch}
- val_loss: {val_loss}
- Batch size: 256
- GPU: NVIDIA L4 (23GB)

## Architecture
- DINOv2 ViT-S/14 image encoder (frozen, 21.7M params)
- PointGPT point encoder (trainable, ~0.5M params)
- Dual cross-attention + aggregation + MLP heads (~3.1M params)
- Total: 25.3M params, 3.6M trainable

## Export Formats
- `best.pth` — PyTorch state dict
- `model.safetensors` — Safetensors format
- `model.onnx` — ONNX (opset 17)
- `trt_fp16/` — TensorRT FP16
- `trt_fp32/` — TensorRT FP32
""")
    logger.info("Export complete. Files at %s", EXPORT_DIR)


if __name__ == "__main__":
    main()
