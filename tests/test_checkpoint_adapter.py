from pathlib import Path

import torch

from anima_calib_projfusion.encoders.checkpoint_adapters import load_reference_checkpoint, translate_reference_key
from anima_calib_projfusion.model.projfusion import ProjDualFusion


def test_checkpoint_key_translation(tmp_path: Path):
    model = ProjDualFusion()
    checkpoint_path = tmp_path / "checkpoint.pt"
    state_dict = {
        "img_encoder.patch_embed.weight": model.image_encoder.patch_embed.weight.detach().clone(),
        "rot_head.net.0.weight": model.rotation_head.net[0].weight.detach().clone(),
    }
    torch.save({"state_dict": state_dict}, checkpoint_path)
    missing, unexpected = load_reference_checkpoint(model, checkpoint_path, strict=False)
    assert "image_encoder.patch_embed.weight" == translate_reference_key("img_encoder.patch_embed.weight")
    assert "rotation_head.net.0.weight" == translate_reference_key("rot_head.net.0.weight")
    assert unexpected == []
    assert "image_encoder.patch_embed.bias" in missing
