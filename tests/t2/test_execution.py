import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _harmonized_frame(city: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in range(2021, 2026):
        for i in range(3):
            rows.append(
                {
                    "request_id": f"{city}-{year}-{i}",
                    "created_date": f"{year}-01-{i + 1:02d}T00:00:00Z",
                    "closed_date": f"{year}-02-{i + 1:02d}T00:00:00Z",
                    "status": "open" if i == 0 else "closed",
                    "category": "category",
                    "descriptor": "descriptor",
                    "agency": "agency",
                    "latitude": 1.0,
                    "longitude": 1.0,
                    "area": "area",
                    "city": city,
                }
            )
    return pd.DataFrame(rows)


def test_diagnostic_input_selection_is_outcome_independent():
    module = _load_script("t2_prepare_inputs_test", "scripts/t2/prepare_inputs.py")
    module.ROWS_PER_YEAR = 2
    original = _harmonized_frame("nyc")
    changed = original.copy()
    changed["status"] = "completed"
    changed["closed_date"] = "2099-01-01T00:00:00Z"

    first, counts_first = module._select(original, "nyc")
    second, counts_second = module._select(changed, "nyc")

    assert list(first["request_id"]) == list(second["request_id"])
    assert counts_first == counts_second
    assert len(first) == 10
    assert counts_first == {str(year): 2 for year in range(2021, 2026)}


def test_matrix_launcher_is_nonexecuting_without_safety_switch(tmp_path, capsys):
    module = _load_script(
        "t2_run_matrix_test",
        "scripts/t2/run_diagnostic_matrix.py",
    )
    old_argv = module.sys.argv
    try:
        module.sys.argv = [
            "run_diagnostic_matrix.py",
            "--input-dir",
            str(tmp_path / "inputs"),
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
        assert module.main() == 0
    finally:
        module.sys.argv = old_argv
    output = capsys.readouterr().out
    assert "No training started" in output
    assert '"held_out_city": "boston"' in output
