import pandas as pd

from src.labeling import add_delayed_resolution_label


def test_city_category_q3_label_generation():
    frame = pd.DataFrame(
        {
            "request_id": [f"r{i}" for i in range(8)],
            "created_date": pd.date_range("2024-01-01", periods=8, freq="h"),
            "closed_date": pd.date_range("2024-01-02", periods=8, freq="h"),
            "status": ["Closed"] * 8,
            "category": ["noise"] * 4 + ["road"] * 4,
            "descriptor": ["x"] * 8,
            "agency": ["a"] * 8,
            "latitude": [40.0] * 8,
            "longitude": [-73.0] * 8,
            "area": ["one"] * 8,
            "city": ["nyc"] * 8,
            "resolution_hours": [1, 2, 3, 100, 10, 20, 30, 40],
        }
    )

    labeled, thresholds = add_delayed_resolution_label(frame)

    noise_threshold = thresholds.loc[thresholds["category"] == "noise", "delay_threshold_hours"].iloc[0]
    road_threshold = thresholds.loc[thresholds["category"] == "road", "delay_threshold_hours"].iloc[0]
    assert noise_threshold == 27.25
    assert road_threshold == 32.5
    assert labeled["delayed"].tolist() == [0, 0, 0, 1, 0, 0, 0, 1]


def test_service_routing_target_can_use_category():
    frame = pd.DataFrame(
        {
            "request_id": ["r1"],
            "created_date": [pd.Timestamp("2024-01-01")],
            "closed_date": [pd.Timestamp("2024-01-02")],
            "status": ["Closed"],
            "category": ["noise"],
            "descriptor": ["x"],
            "agency": ["agency"],
            "latitude": [40.0],
            "longitude": [-73.0],
            "area": ["one"],
            "city": ["nyc"],
            "resolution_hours": [1.0],
        }
    )

    labeled, _ = add_delayed_resolution_label(frame, service_routing_source="category")

    assert labeled["service_routing_target"].iloc[0] == "noise"
