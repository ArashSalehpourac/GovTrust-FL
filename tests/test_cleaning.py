import pandas as pd

from src.cleaning import clean_frame


def test_cleaning_removes_invalid_records_and_computes_resolution_hours():
    frame = pd.DataFrame(
        {
            "request_id": ["ok", "missing_created", "negative", "bad_status", "bad_coords"],
            "created_date": ["2024-01-01", None, "2024-01-03", "2024-01-04", "2024-01-05"],
            "closed_date": ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-05", "2024-01-06"],
            "status": ["Closed", "Closed", "Closed", "Cancelled", "Closed"],
            "category": ["noise"] * 5,
            "descriptor": ["desc"] * 5,
            "agency": ["311"] * 5,
            "latitude": [40.0, 40.0, 40.0, 40.0, 0.0],
            "longitude": [-73.0, -73.0, -73.0, -73.0, 0.0],
            "area": ["a"] * 5,
            "city": ["nyc"] * 5,
        }
    )

    cleaned, counts = clean_frame(frame, min_category_samples=1)

    assert cleaned["request_id"].tolist() == ["ok"]
    assert cleaned["resolution_hours"].iloc[0] == 24.0
    assert counts["missing_created_date"] == 1
    assert counts["closed_before_created"] == 1
    assert counts["cancelled_invalid_incomplete_status"] == 1
    assert counts["impossible_coordinates"] == 1
