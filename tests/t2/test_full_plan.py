from src.t2.config import CITIES, SEEDS, full_loco_plan


def test_full_loco_plan_covers_all_four_folds_and_three_seeds():
    rows = full_loco_plan()
    assert len(rows) == len(CITIES) * len(SEEDS) * 4
    assert {row["held_out_city"] for row in rows} == set(CITIES)
    for city in CITIES:
        city_rows = [row for row in rows if row["held_out_city"] == city]
        assert {row["seed"] for row in city_rows} == set(SEEDS)
        for seed in SEEDS:
            modes = [
                (row["mode"], row["epsilon"])
                for row in city_rows
                if row["seed"] == seed
            ]
            assert modes == [
                ("nonprivate", float("inf")),
                ("private", 5.0),
                ("private", 1.0),
                ("clipped_no_noise", float("inf")),
            ]


def test_primary_matrix_without_clipped_ablation_has_36_runs():
    rows = full_loco_plan(include_clipped_no_noise=False)
    assert len(rows) == 4 * 3 * 3
