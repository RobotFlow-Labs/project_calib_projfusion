from anima_calib_projfusion.eval.report import build_table_i


def test_table_builder_outputs_markdown():
    markdown = build_table_i(
        [
            {
                "dataset_id": "kitti",
                "checkpoint_name": "smoke",
                "perturbation_range": "10deg/50cm",
                "rotation_rmse_deg": 0.1,
                "translation_rmse_cm": 0.2,
                "l1_success_rate": 1.0,
                "l2_success_rate": 1.0,
            }
        ]
    )
    assert "Table I" in markdown
    assert "smoke" in markdown
