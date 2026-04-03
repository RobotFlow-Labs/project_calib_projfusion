from pathlib import Path

from anima_calib_projfusion.config import ProjFusionSettings


def test_default_config_matches_paper_defaults():
    settings = ProjFusionSettings.from_toml("configs/default.toml")
    assert settings.package_name == "anima_calib_projfusion"
    assert settings.project.codename == "CALIB-PROJFUSION"
    assert settings.project.python_version == "3.11"
    assert settings.model.image_hw == (224, 448)
    assert settings.model.feature_hw == (16, 32)
    assert settings.model.harmonic_functions == 6
    assert settings.model.projection_margin == 2.0
    assert settings.model.point_count == 8192
    assert settings.training.lr == 5e-4
    assert Path(settings.pretrained.dinov2).name == "dinov2_vits14"
