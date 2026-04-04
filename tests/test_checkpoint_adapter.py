from pathlib import Path

import torch

from anima_calib_projfusion.encoders.checkpoint_adapters import (
    load_reference_checkpoint,
    translate_reference_key,
)
from anima_calib_projfusion.model.projfusion import ProjDualFusion


def test_checkpoint_key_translation(tmp_path: Path):
    model = ProjDualFusion(dinov2_pretrained=False)
    # Use a real key from the model (rotation head)
    rot_weight = model.rotation_head.net[0].weight.detach().clone()
    checkpoint_path = tmp_path / "checkpoint.pt"
    state_dict = {
        "rot_head.net.0.weight": rot_weight,
    }
    torch.save({"state_dict": state_dict}, checkpoint_path)
    missing, unexpected = load_reference_checkpoint(model, checkpoint_path, strict=False)
    assert translate_reference_key("rot_head.net.0.weight") == "rotation_head.net.0.weight"
    assert translate_reference_key("tsl_head.net.0.weight") == "translation_head.net.0.weight"
