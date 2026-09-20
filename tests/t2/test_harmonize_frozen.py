from __future__ import annotations

import pandas as pd
import pytest

from src.t2.harmonize_frozen import HARMONIZED_COLUMNS, harmonize_chunk


def test_nyc_harmonization_is_row_preserving() -> None:
    raw = pd.DataFrame(
        {
            "unique_key": ["1", "2"],
            "created_date": ["2025-01-01 00:00:12", "2025-01-01 01:00:00"],
            "closed_date": ["2025-01-02", "2025-01-03"],
            "status": ["Closed", "Open"],
            "complaint_type": ["Noise", "Heat"],
            "descriptor": ["Loud", "Cold"],
            "agency": ["NYPD", "HPD"],
            "latitude": ["40.1", "40.2"],
            "longitude": ["-73.9", "-73.8"],
            "borough": ["MANHATTAN", "BRONX"],
        }
    )
    out = harmonize_chunk(raw, "nyc")
    assert list(out.columns) == HARMONIZED_COLUMNS
    assert len(out) == len(raw)
    assert out["request_id"].tolist() == ["1", "2"]
    assert out["area"].tolist() == ["MANHATTAN", "BRONX"]


def test_boston_rejects_datastore_only_id() -> None:
    columns = {
        "case_enquiry_id": ["1"],
        "open_dt": ["2025-01-01"],
        "closed_dt": ["2025-01-02"],
        "case_status": ["Closed"],
        "reason": ["Reason"],
        "type": ["Type"],
        "subject": ["Subject"],
        "latitude": ["42.3"],
        "longitude": ["-71.0"],
        "neighborhood": ["Back Bay"],
    }
    for i in range(20):
        columns[f"extra_{i}"] = ["x"]
    raw = pd.DataFrame(columns)
    raw["_id"] = ["1"]
    with pytest.raises(ValueError, match="datastore-only _id"):
        harmonize_chunk(raw, "boston")


def test_la_2025_display_schema_maps_case_number_without_filtering() -> None:
    raw = pd.DataFrame(
        {
            "CaseNumber": ["A", "B"],
            "CreatedDate": ["2025-03-28 06:10:07", "2025-12-31 12:59:54"],
            "ClosedDate": ["2025-03-29", pd.NA],
            "Status": ["Closed", "Open"],
            "RequestType": ["Bulky Items", "Graffiti"],
            "ActionTaken": ["Completed", "Assigned"],
            "Owner": ["LASAN", "LASAN"],
            "Latitude": ["34.0", "34.1"],
            "Longitude": ["-118.2", "-118.3"],
            "NCName": ["Downtown", "Hollywood"],
        }
    )
    out = harmonize_chunk(raw, "los_angeles")
    assert len(out) == 2
    assert out["request_id"].tolist() == ["A", "B"]
    assert out["category"].tolist() == ["Bulky Items", "Graffiti"]
    assert out["closed_date"].isna().sum() == 1


def test_chicago_owner_department_and_area_mapping() -> None:
    raw = pd.DataFrame(
        {
            "sr_number": ["SR1"],
            "created_date": ["2024-01-01 00:00:12"],
            "closed_date": ["2024-01-02"],
            "status": ["Completed"],
            "sr_type": ["Pothole"],
            "sr_short_code": ["PH"],
            "owner_department": ["CDOT"],
            "latitude": ["41.8"],
            "longitude": ["-87.6"],
            "community_area": ["32"],
            "ward": ["42"],
        }
    )
    out = harmonize_chunk(raw, "chicago")
    assert out.loc[0, "agency"] == "CDOT"
    assert out.loc[0, "area"] == "32"
