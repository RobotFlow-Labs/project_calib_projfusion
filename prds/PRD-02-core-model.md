# PRD-02: Core Model

> Module: CALIB-PROJFUSION | Priority: P0
> Depends on: PRD-01
> Status: ⬜ Not started

## Objective
Implement the paper’s dual-branch ProjFusion network with DINOv2 image tokens, PointGPT point tokens, extrinsic-aware coordinate alignment, harmonic positional encoding, cross-attention, aggregation, and MLP heads.

## Context (from paper)
The paper replaces depth-map fusion with an extrinsic-aware cross-attention mechanism that "directly aligns image patches and LiDAR point groups" before regressing SE(3) updates. The released repo implements this as `ProjDualFusion` plus `AttenDualFusionNet`.

**Paper reference**: §III-B, §III-C, §III-D, §IV-B  
**Key paper cues**: "extrinsic-aware cross-attention", "harmonic functions", "rotation and translation estimation benefit from distinct cues"

## Acceptance Criteria
- [ ] DINOv2-tiny image encoder is wrapped with output tokens shaped for `224 x 448` inputs.
- [ ] PointGPT-tiny point encoder returns centroids and 384-dim group tokens.
- [ ] Coordinate alignment uses the current extrinsic hypothesis and image-feature-plane normalization.
- [ ] Harmonic embedding with `n_h = 6`, `omega_0 = 1 / (1 + r_p)` is implemented.
- [ ] Dual cross-attention branches and dual aggregation branches output `rot_log` and `tsl_log`.
- [ ] Forward smoke test verifies shape contract on synthetic tensors.
- [ ] Test: `uv run pytest tests/test_encoders.py tests/test_positional_encoding.py tests/test_projfusion_forward.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_calib_projfusion/encoders/image_dinov2.py` | DINOv2 wrapper | §III-B / §IV-B | ~180 |
| `src/anima_calib_projfusion/encoders/pointgpt.py` | PointGPT wrapper | §III-B / §IV-B | ~220 |
| `src/anima_calib_projfusion/model/positional_encoding.py` | harmonic embedding | §III-C.3, Eq. (5) | ~140 |
| `src/anima_calib_projfusion/model/coordinate_alignment.py` | point-group projection | §III-C.2, Eq. (2)-(4) | ~160 |
| `src/anima_calib_projfusion/model/cross_attention.py` | scale-free multi-head attention | §III-C.4, Eq. (6)-(10) | ~220 |
| `src/anima_calib_projfusion/model/aggregation.py` | 2D residual aggregation | §III-D | ~140 |
| `src/anima_calib_projfusion/model/heads.py` | MLP regression heads | §III-D | ~100 |
| `src/anima_calib_projfusion/model/projfusion.py` | `ProjFusion` / `ProjDualFusion` orchestration | §III-B-§III-D | ~220 |
| `tests/test_encoders.py` | encoder shape tests | — | ~120 |
| `tests/test_positional_encoding.py` | harmonic/RoPE tests | — | ~100 |
| `tests/test_projfusion_forward.py` | full forward-pass tests | — | ~140 |

## Architecture Detail (from paper)

### Inputs
- `img`: `Tensor[B, 3, 224, 448]`
- `pcd`: `Tensor[B, N, 3]`
- `T_cl`: `Tensor[B, 4, 4]`
- `camera_info`: dict

### Outputs
- `rot_log`: `Tensor[B, 3]`
- `tsl_log`: `Tensor[B, 3]`

### Canonical Shapes
- Image tokens: `Tensor[B, 512, 384]`
- Point tokens: `Tensor[B, 128, 384]`
- Projected coords: `Tensor[B, 128, 2]`
- Cross-attended map: `Tensor[B, 384, 16, 32]`
- Aggregated vector: `Tensor[B, 768]`

### Algorithm
```python
# Paper §III-B-§III-D

class ProjDualFusion(nn.Module):
    def forward(self, img, pcd, init_extrinsic, camera_info):
        img_tokens = image_encoder(img)                 # [B, 512, 384]
        xyz_groups, point_tokens = point_encoder(pcd)   # [B, 128, 3], [B, 128, 384]
        proj_uv = align_point_groups(
            xyz_groups,
            init_extrinsic,
            camera_info,
            feature_hw=(16, 32),
            margin=2.0,
        )                                               # [B, 128, 2]
        img_pos = harmonic_embed(image_grid())          # [512, P]
        pt_pos = harmonic_embed(proj_uv)                # [B, 128, P]
        rot_map = rot_attention(
            concat(img_tokens, img_pos),
            concat(point_tokens, pt_pos),
        )                                               # [B, 512, 384]
        tsl_map = tsl_attention(...)
        rot_feat = rot_aggregation(unflatten(rot_map))  # [B, 768]
        tsl_feat = tsl_aggregation(unflatten(tsl_map))  # [B, 768]
        return rot_head(rot_feat), tsl_head(tsl_feat)
```

## Dependencies
```toml
accelerate = ">=1.0"
einops = ">=0.8"
timm = ">=1.0"
```

## Data Requirements
| Asset | Size | Path | Download |
|-------|------|------|----------|
| DINOv2-tiny weights | model artifact | `/mnt/forge-data/models/vision/dinov2_vits14` | provision via torch hub cache or explicit checkpoint |
| PointGPT-tiny weights | model artifact | `/mnt/forge-data/models/pointgpt/*.pth` | provision from upstream/released checkpoints |

## Test Plan
```bash
uv run pytest tests/test_encoders.py tests/test_positional_encoding.py tests/test_projfusion_forward.py -v
```

## References
- Paper: §III-B, §III-C, §III-D, §IV-B
- Reference impl: `repositories/ProjFusion/models/model.py`
- Reference impl: `repositories/ProjFusion/models/tools/core.py`
- Reference impl: `repositories/ProjFusion/models/tools/attention.py`
- Depends on: PRD-01
- Feeds into: PRD-03, PRD-04
