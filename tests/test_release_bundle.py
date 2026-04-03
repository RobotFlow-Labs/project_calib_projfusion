import json
from pathlib import Path

from scripts.export_release import export_release
from scripts.validate_prod import validate_bundle


def test_release_manifest_contains_required_files(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    config = tmp_path / "config.toml"
    report = tmp_path / "report.md"
    model_card = tmp_path / "model_card.md"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    config.write_text("config", encoding="utf-8")
    report.write_text("report", encoding="utf-8")
    model_card.write_text("card", encoding="utf-8")

    bundle_dir = export_release(checkpoint, config, report, model_card, tmp_path / "dist")
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"checkpoint", "config", "report", "model_card"}

    validation = validate_bundle(bundle_dir)
    assert validation["all_present"] is True
    assert validation["runtime_contract_ok"] is True
