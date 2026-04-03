# CALIB-PROJFUSION

## Paper
**ProjFusion: Camera-LiDAR Calibration (RA-L 2026)**
arXiv: https://arxiv.org/abs/2603.29414

## Module Identity
- Codename: CALIB-PROJFUSION
- Domain: Calibration
- Part of ANIMA Intelligence Compiler Suite

## Structure
```
project_calib_projfusion/
├── pyproject.toml
├── configs/
├── src/anima_calib_projfusion/
├── tests/
├── scripts/
├── papers/          # Paper PDF
├── CLAUDE.md        # This file
├── NEXT_STEPS.md
├── ASSETS.md
└── PRD.md
```

## Commands
```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Conventions
- Package manager: uv (never pip)
- Build backend: hatchling
- Python: >=3.10
- Config: TOML + Pydantic BaseSettings
- Lint: ruff
- Git commit prefix: [CALIB-PROJFUSION]
