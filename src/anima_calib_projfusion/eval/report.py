from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_results_dataframe(results: list[dict[str, object]]) -> pd.DataFrame:
    return (
        pd.DataFrame(results)
        .sort_values(["dataset_id", "perturbation_range"])
        .reset_index(drop=True)
    )


def _build_table(results: list[dict[str, object]], dataset_id: str, title: str) -> str:
    frame = build_results_dataframe(results)
    frame = frame[frame["dataset_id"] == dataset_id]
    heading = f"## {title}\n"
    if frame.empty:
        return heading + "\n_No rows available._\n"
    columns = [
        "checkpoint_name",
        "perturbation_range",
        "rotation_rmse_deg",
        "translation_rmse_cm",
        "l1_success_rate",
        "l2_success_rate",
    ]
    return heading + "\n" + frame[columns].to_markdown(index=False) + "\n"


def build_table_i(results: list[dict[str, object]]) -> str:
    return _build_table(results, dataset_id="kitti", title="Table I — KITTI")


def build_table_ii(results: list[dict[str, object]]) -> str:
    return _build_table(results, dataset_id="nuscenes", title="Table II — nuScenes")


def save_report_bundle(
    results: list[dict[str, object]], output_dir: str | Path
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = build_results_dataframe(results)
    csv_path = output_dir / "results.csv"
    markdown_path = output_dir / "report.md"
    dataframe.to_csv(csv_path, index=False)
    markdown_path.write_text(
        build_table_i(results) + "\n" + build_table_ii(results), encoding="utf-8"
    )
    return csv_path, markdown_path
