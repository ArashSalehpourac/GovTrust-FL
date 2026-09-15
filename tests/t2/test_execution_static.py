import json
from pathlib import Path

from src.t2.config import diagnostic_plan
from src.t2.provenance import sha256_file


def test_diagnostic_plan_has_24_jobs():
    plan = diagnostic_plan()
    assert len(plan) == 24
    assert {row["held_out_city"] for row in plan} == {"boston", "los_angeles"}
    assert {row["seed"] for row in plan} == {0, 1, 2}


def test_input_manifest_protocol_constant_matches_builder_and_launcher():
    prepare = Path("scripts/t2/prepare_inputs.py").read_text(encoding="utf-8")
    launcher = Path("scripts/t2/run_diagnostic_matrix.py").read_text(encoding="utf-8")
    assert '"protocol": "t2_diagnostic_input_archive_v2"' in prepare
    assert 'INPUT_PROTOCOL = "t2_diagnostic_input_archive_v2"' in launcher


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("abc", encoding="utf-8")
    first = sha256_file(path)
    second = sha256_file(path)
    assert first == second
    assert len(first) == 64


def test_plan_json_serializes_for_colab_preview():
    text = json.dumps(diagnostic_plan(), default=str)
    assert "boston" in text
    assert "los_angeles" in text
