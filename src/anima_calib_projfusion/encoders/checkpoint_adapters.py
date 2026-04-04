from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

REFERENCE_KEY_REWRITES = (
    ("img_encoder.", "image_encoder."),
    ("pcd_encoder.", "point_encoder."),
    ("rot_attention.", "rotation_attention."),
    ("tsl_attention.", "translation_attention."),
    ("rot_agg.", "rotation_aggregation."),
    ("tsl_agg.", "translation_aggregation."),
    ("rot_head.", "rotation_head."),
    ("tsl_head.", "translation_head."),
)


def translate_reference_key(key: str) -> str:
    translated = key
    for source, target in REFERENCE_KEY_REWRITES:
        if translated.startswith(source):
            translated = target + translated[len(source) :]
            break
    return translated


def _extract_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    for candidate in ("state_dict", "model", "model_state_dict"):
        payload = checkpoint.get(candidate)
        if isinstance(payload, dict):
            return payload
    return checkpoint


def load_reference_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_state_dict(checkpoint)
    translated = {translate_reference_key(key): value for key, value in state_dict.items()}
    compatible = {key: value for key, value in translated.items() if key in model.state_dict()}
    result = model.load_state_dict(compatible, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
