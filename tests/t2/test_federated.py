import pandas as pd

from src.t2.config import T2Config
from src.t2.data import prepare_resolved_frame
from src.t2.federated import resolve_device, train_select, training_step_budget
from src.t2.folds import chronological_split
from src.t2.preprocessing import build_fixed_preprocessor


def _raw(city: str, rows_per_year: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in range(2021, 2026):
        created = pd.date_range(
            f"{year}-06-01",
            periods=rows_per_year,
            freq="h",
            tz="UTC",
        )
        for i, created_at in enumerate(created):
            rows.append(
                {
                    "request_id": f"{city}-{year}-{i}",
                    "created_date": created_at,
                    "closed_date": created_at + pd.Timedelta(hours=(i % 6) + 1),
                    "category": f"{city}-cat-{i % 3}",
                    "descriptor": f"diagnostic-only descriptor {i % 5}",
                    "status": "closed",
                }
            )
    return pd.DataFrame(rows)


def test_cpu_device_resolution_is_explicit():
    assert resolve_device("cpu").type == "cpu"


def test_nonprivate_fedavg_point_regression_path_runs():
    prepared = {
        city: prepare_resolved_frame(_raw(city))
        for city in ("a", "b", "c")
    }
    splits = {
        city: chronological_split(frame)
        for city, frame in prepared.items()
    }
    pre = build_fixed_preprocessor(text_hash_dim=32, category_hash_dim=8)
    cfg = T2Config(
        rounds=2,
        batch_size=8,
        hidden_sizes=(8,),
        learning_rate=0.01,
        seed=3,
        device="cpu",
    )
    result = train_select(splits, pre, cfg)
    assert 1 <= result["best_round"] <= 2
    assert result["source_macro_mae_log1p_hours"] >= 0
    assert result["device"] == "cpu"
    assert set(result["internal_by_city"]) == {"a", "b", "c"}


def test_fixed_total_effective_epoch_budget_is_deterministic():
    cfg = T2Config(
        rounds=20,
        training_budget="fixed_total_effective_epochs",
        total_effective_epochs=1.0,
        batch_size=1024,
    )
    steps_per_round, planned = training_step_budget(1003, cfg)
    assert steps_per_round == 51
    assert planned == 1020
    assert planned / 1003 >= 1.0
    assert planned / 1003 < 1.02


def test_legacy_full_epoch_budget_remains_exact_for_small_tests():
    cfg = T2Config(rounds=2, local_epochs=1, batch_size=8)
    steps_per_round, planned = training_step_budget(5, cfg)
    assert steps_per_round == 5
    assert planned == 10
