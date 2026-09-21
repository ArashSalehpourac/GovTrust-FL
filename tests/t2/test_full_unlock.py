import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    path = ROOT / "scripts/t2/run_full_loco.py"
    spec = importlib.util.spec_from_file_location("t2_run_full_loco_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(module) -> dict[str, object]:
    return {
        "protocol": "t2_full_data_experiment_design_audit_v1",
        "gate": "PASS",
        "blockers": [],
        "ready_to_unlock_full_loco_execution": True,
        "training_authorized_by_this_report": False,
        "audit_execution_git_sha": module.VALIDATED_DESIGN_AUDIT_SHA,
        "harmonized_execution_git_sha": module.VALIDATED_HARMONIZATION_SHA,
        "analysis_audit_execution_git_sha": module.VALIDATED_ANALYSIS_AUDIT_SHA,
        "model_input_dim": 583,
        "matrix": {
            "primary_runs": 36,
            "total_with_ablation": 48,
        },
        "heldout_isolation": {
            "target_bytes_opened_before_source_selection": False,
            "target_2025_loaded_only_after_selected_checkpoint": True,
            "target_used_for_tuning_or_privacy_accounting": False,
        },
    }


def test_full_loco_execution_is_unlocked_only_after_design_gate():
    module = _load_script()
    assert module.FULL_LOCO_EXECUTION_ENABLED is True


def test_validated_design_report_unlocks_execution(tmp_path):
    module = _load_script()
    path = tmp_path / "design.json"
    path.write_text(json.dumps(_report(module)), encoding="utf-8")
    module.VALIDATED_DESIGN_REPORT_SHA256 = module.sha256_file(path)
    verified = module._verify_design_audit(path)
    assert verified["gate"] == "PASS"


def test_mismatched_design_sha_fails_closed(tmp_path):
    module = _load_script()
    report = _report(module)
    report["audit_execution_git_sha"] = "bad"
    path = tmp_path / "design.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    module.VALIDATED_DESIGN_REPORT_SHA256 = module.sha256_file(path)
    with pytest.raises(SystemExit, match="design_sha"):
        module._verify_design_audit(path)


def test_execute_requires_design_audit(tmp_path):
    module = _load_script()
    old_argv = module.sys.argv
    try:
        module.sys.argv = [
            "run_full_loco.py",
            "--harmonized-dir",
            str(tmp_path / "harm"),
            "--harmonized-manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--heldout",
            "boston",
            "--mode",
            "nonprivate",
            "--epsilon",
            "inf",
            "--seed",
            "0",
            "--execute",
        ]
        with pytest.raises(SystemExit, match="--design-audit"):
            module.main()
    finally:
        module.sys.argv = old_argv


def test_modified_report_bytes_fail_closed(tmp_path):
    module = _load_script()
    path = tmp_path / "design.json"
    path.write_text(json.dumps(_report(module)), encoding="utf-8")
    module.VALIDATED_DESIGN_REPORT_SHA256 = "0" * 64
    with pytest.raises(SystemExit, match="report_sha256"):
        module._verify_design_audit(path)
