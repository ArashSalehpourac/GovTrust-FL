import pandas as pd

from src.t2.audit_analysis import audit_analysis_file
from src.t2.provenance import sha256_file


def test_analysis_audit_decomposes_exclusions_and_boundary_purge(tmp_path):
    path = tmp_path / "CITY_2023_harmonized.parquet"
    frame = pd.DataFrame(
        {
            "request_id": ["ok", "open", "missing", "negative", "cross"],
            "created_date": [
                "2023-01-01T00:00:00Z",
                "2023-02-01T00:00:00Z",
                "2023-03-01T00:00:00Z",
                "2023-04-02T00:00:00Z",
                "2023-12-31T00:00:00Z",
            ],
            "closed_date": [
                "2023-01-02T00:00:00Z",
                None,
                None,
                "2023-04-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
            ],
            "status": ["Closed", "Open", "Resolved", "Closed", "Completed"],
            "category": ["a", "a", "a", "a", "b"],
            "descriptor": ["x", "x", "x", "x", None],
            "agency": ["z"] * 5,
            "latitude": [1.0] * 5,
            "longitude": [1.0] * 5,
            "area": ["q"] * 5,
            "city": ["test_city"] * 5,
        }
    )
    frame.to_parquet(path, index=False)

    output = {
        "city": "test_city",
        "year": 2023,
        "rows": 5,
        "harmonized_sha256": sha256_file(path),
        "harmonized_size_bytes": path.stat().st_size,
    }
    structural = {
        "city": "test_city",
        "year": 2023,
        "rows": 5,
        "final_status_rows": 4,
        "eligible_resolved_rows": 2,
        "closed_before_created": 1,
    }

    result = audit_analysis_file(
        path=path,
        output_entry=output,
        structural_entry=structural,
        batch_size=2,
    )

    assert result["gate"] == "PASS"
    assert result["excluded_nonfinal_status"] == 1
    assert result["excluded_final_missing_or_invalid_outcome"] == 1
    assert result["excluded_final_negative_duration"] == 1
    assert result["eligible_resolved_rows"] == 2
    assert result["boundary_purge_rows"] == 1
    assert result["descriptor_missing_among_eligible"] == 1
    assert result["status_counts_normalized"] == {
        "closed": 2,
        "completed": 1,
        "open": 1,
        "resolved": 1,
    }
