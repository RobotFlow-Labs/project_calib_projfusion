from __future__ import annotations

import argparse
import json
from pathlib import Path
import importlib


def validate_bundle(bundle_dir: str | Path) -> dict[str, object]:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_ok = {key: (bundle_dir / filename).exists() for key, filename in manifest.items()}
    cli_ok = importlib.import_module("anima_calib_projfusion.cli.infer") is not None
    api_ok = importlib.import_module("anima_calib_projfusion.api.app") is not None
    ros_launch_ok = Path("scripts/launch/calib_projfusion.launch.py").exists()
    return {
        "bundle_dir": str(bundle_dir),
        "manifest_path": str(manifest_path),
        "all_present": all(files_ok.values()),
        "runtime_contract_ok": bool(cli_ok and api_ok and ros_launch_ok),
        "files": files_ok,
        "entrypoints": {"cli": cli_ok, "api": api_ok, "ros_launch": ros_launch_ok},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a CALIB-PROJFUSION release bundle.")
    parser.add_argument("--bundle-dir", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(validate_bundle(args.bundle_dir))


if __name__ == "__main__":
    main()
