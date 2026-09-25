from pathlib import Path


def test_legacy_training_entry_points_are_fail_closed():
    cli = Path("src/t2/cli.py").read_text(encoding="utf-8")
    matrix = Path("scripts/t2/run_diagnostic_matrix.py").read_text(encoding="utf-8")
    bounded = Path("scripts/t2/prepare_inputs.py").read_text(encoding="utf-8")

    assert "LEGACY_DIAGNOSTIC_EXECUTION_RETIRED = True" in cli
    assert "LEGACY_DIAGNOSTIC_EXECUTION_RETIRED = True" in matrix
    assert "LEGACY_BOUNDED_INPUT_BUILDER_RETIRED = True" in bounded
    assert "Legacy Boston/LA diagnostic execution is retired" in cli
    assert "Legacy 20k-per-city Boston/LA diagnostic matrix is retired" in matrix
    assert "Legacy bounded 20k-per-city diagnostic input builder is retired" in bounded
