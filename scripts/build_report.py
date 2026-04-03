from __future__ import annotations

import argparse
import json

from anima_calib_projfusion.eval.report import save_report_bundle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper-style evaluation tables from JSON results.")
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    with open(args.results_json, "r", encoding="utf-8") as handle:
        results = json.load(handle)
    csv_path, markdown_path = save_report_bundle(results, args.output_dir)
    print({"csv": str(csv_path), "markdown": str(markdown_path)})


if __name__ == "__main__":
    main()
