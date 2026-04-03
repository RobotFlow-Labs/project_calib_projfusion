from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


REQUIRED_KEYS = ("checkpoint", "config", "report", "model_card")


def export_release(
    checkpoint_path: str | Path,
    config_path: str | Path,
    report_path: str | Path,
    model_card_path: str | Path,
    output_dir: str | Path,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)
    report_path = Path(report_path)
    model_card_path = Path(model_card_path)
    output_dir = Path(output_dir)
    bundle_dir = output_dir / "calib-projfusion-release"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "checkpoint": checkpoint_path.name,
        "config": config_path.name,
        "report": report_path.name,
        "model_card": model_card_path.name,
    }
    for source in (checkpoint_path, config_path, report_path, model_card_path):
        shutil.copy2(source, bundle_dir / source.name)
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bundle_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a CALIB-PROJFUSION release bundle.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-card", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bundle_dir = export_release(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        report_path=args.report,
        model_card_path=args.model_card,
        output_dir=args.output_dir,
    )
    print({"bundle_dir": str(bundle_dir)})


if __name__ == "__main__":
    main()
