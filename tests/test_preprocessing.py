import pandas as pd

from src.cleaning import clean_frame
from src.labeling import add_delayed_resolution_label


def test_clean_frame_and_q3_delay_label():
    frame = pd.DataFrame(
        {
            "request_id": ["a", "b", "c", "d", "e"],
            "created_date": [
                "2024-01-01 00:00:00",
                "2024-01-01 01:00:00",
                "2024-01-01 02:00:00",
                "2024-01-01 03:00:00",
                None,
            ],
            "closed_date": [
                "2024-01-01 01:00:00",
                "2024-01-01 03:00:00",
                "2024-01-01 06:00:00",
                "2024-01-01 04:00:00",
                "2024-01-01 05:00:00",
            ],
            "status": ["Closed", "Completed", "Closed", "Cancelled", "Closed"],
            "category": ["Noise", "Noise", "Noise", "Noise", "Noise"],
            "descriptor": ["loud", "loud", "party", "bad", "missing"],
            "agency": ["A", "A", "B", "B", "A"],
            "latitude": [40.0, 40.1, 40.2, 40.3, 40.4],
            "longitude": [-73.0, -73.1, -73.2, -73.3, -73.4],
            "area": ["one", "one", "two", "two", "one"],
            "city": ["nyc", "nyc", "nyc", "nyc", "nyc"],
        }
    )

    cleaned, counts = clean_frame(frame, min_category_samples=1)
    assert counts["missing_created_date"] == 1
    assert counts["cancelled_invalid_incomplete_status"] == 1
    assert cleaned["resolution_hours"].tolist() == [1.0, 2.0, 4.0]

    labeled, thresholds = add_delayed_resolution_label(cleaned)
    assert thresholds["delay_threshold_hours"].iloc[0] == 3.0
    assert labeled["delayed"].tolist() == [0, 0, 1]
    assert labeled["service_routing_target"].tolist() == ["A", "A", "B"]
