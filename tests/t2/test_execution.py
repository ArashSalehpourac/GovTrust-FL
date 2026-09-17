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


def test_los_angeles_2025_cases_schema_harmonizes():
    module = _load_script("t2_prepare_inputs_la2025", "scripts/t2/prepare_inputs.py")
    raw = pd.DataFrame(
        {
            "casenumber": ["01329606"],
            "createddate": ["2025-03-01T12:00:00.000"],
            "closeddate": ["2025-03-02T12:00:00.000"],
            "status": ["Closed"],
            "type": ["Bulky Items"],
            "action_taken__c": ["Completed"],
            "department_name__c": ["LASAN"],
            "latitude": ["34.0"],
            "longitude": ["-118.2"],
        }
    )
    out = module._harmonize(raw, "los_angeles")
    assert out.loc[0, "request_id"] == "01329606"
    assert out.loc[0, "created_date"] == "2025-03-01T12:00:00.000"
    assert out.loc[0, "closed_date"] == "2025-03-02T12:00:00.000"
    assert out.loc[0, "category"] == "Bulky Items"
    assert out.loc[0, "descriptor"] == "Completed"
    assert out.loc[0, "agency"] == "LASAN"
    assert out.loc[0, "city"] == "los_angeles"


def _boston_record(request_id: str, opened: str, **outcomes: object) -> dict[str, object]:
    return {
        "case_enquiry_id": request_id,
        "open_dt": opened,
        "case_status": outcomes.get("case_status", "open"),
        "closed_dt": outcomes.get("closed_dt"),
        "resolution_hours": outcomes.get("resolution_hours"),
    }


def test_boston_pagination_continues_when_first_page_is_ineligible(monkeypatch):
    module = _load_script("t2_prepare_inputs_boston_pagination", "scripts/t2/prepare_inputs.py")
    module.ROWS_PER_YEAR = 4_000
    module.FETCH_PER_YEAR = 6_000
    first_page = [
        _boston_record(f"boston-{i}", "2021-01-01T00:00:00Z")
        for i in range(3_037)
    ] + [
        _boston_record(f"other-{i}", "2022-01-01T00:00:00Z")
        for i in range(2_963)
    ]
    second_page = [
        _boston_record(f"boston-second-{i}", "2021-02-01T00:00:00Z")
        for i in range(1_000)
    ]
    calls: list[dict[str, object]] = []

    def fake_request(_url: str, params: dict[str, object]) -> dict[str, object]:
        calls.append(params)
        return {"success": True, "result": {"records": first_page if params["offset"] == 0 else second_page}}

    monkeypatch.setattr(module, "_request_json", fake_request)
    raw, meta = module._boston(2021)
    assert len(raw) == 7_000
    assert [call["offset"] for call in calls] == [0, 6_000]
    assert meta["eligible_unique_identity_date_rows"] == 4_037


def test_boston_pagination_stops_after_sufficient_candidates(monkeypatch):
    module = _load_script("t2_prepare_inputs_boston_stop", "scripts/t2/prepare_inputs.py")
    module.ROWS_PER_YEAR = 4_000
    module.FETCH_PER_YEAR = 6_000
    page = [_boston_record(f"boston-{i}", "2021-01-01T00:00:00Z") for i in range(4_000)]
    page.extend(_boston_record(f"other-{i}", "2022-01-01T00:00:00Z") for i in range(2_000))
    calls: list[dict[str, object]] = []

    def fake_request(_url: str, params: dict[str, object]) -> dict[str, object]:
        calls.append(params)
        return {"success": True, "result": {"records": page}}

    monkeypatch.setattr(module, "_request_json", fake_request)
    _raw, meta = module._boston(2021)
    assert len(calls) == 1
    assert meta["pages"][0]["eligible_unique_ids_after_page"] == 4_000


def test_boston_candidate_stopping_uses_only_identity_date_year(monkeypatch):
    module = _load_script("t2_prepare_inputs_boston_fields", "scripts/t2/prepare_inputs.py")
    module.ROWS_PER_YEAR = 2
    module.FETCH_PER_YEAR = 2
    first = [
        _boston_record("a", "2021-01-01T00:00:00Z", case_status="closed", closed_dt="2021-02-01"),
        _boston_record("b", "2021-01-02T00:00:00Z", case_status="open", closed_dt=None),
    ]
    calls: list[dict[str, object]] = []

    def fake_request(_url: str, params: dict[str, object]) -> dict[str, object]:
        calls.append(params)
        return {"success": True, "result": {"records": first}}

    monkeypatch.setattr(module, "_request_json", fake_request)
    raw, _meta = module._boston(2021)
    raw_changed = raw.copy()
    raw_changed["case_status"] = ["completed", "cancelled"]
    raw_changed["closed_dt"] = ["1900-01-01", "2099-01-01"]
    raw_changed["resolution_hours"] = [0, 999999]
    assert len(module._boston_candidate_ids(raw.to_dict("records"), 2021)) == 2
    assert module._boston_candidate_ids(raw.to_dict("records"), 2021) == module._boston_candidate_ids(
        raw_changed.to_dict("records"), 2021
    )
    assert calls[0]["sort"] == "open_dt asc,case_enquiry_id asc"


def test_boston_candidate_order_is_deterministic_and_deduplicated():
    module = _load_script("t2_prepare_inputs_boston_order", "scripts/t2/prepare_inputs.py")
    module.ROWS_PER_YEAR = 2
    module.YEARS = (2021,)
    frame = pd.DataFrame(
        [
            {"request_id": "b", "created_date": "2021-01-01T00:00:00Z", "city": "boston"},
            {"request_id": "a", "created_date": "2021-01-01T00:00:00Z", "city": "boston"},
            {"request_id": "a", "created_date": "2021-01-01T00:00:00Z", "city": "boston"},
        ]
    )
    for column in module.HARMONIZED_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    selected, counts = module._select(frame, "boston")
    assert list(selected["request_id"]) == ["a", "b"]
    assert counts["2021"] == 2


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
