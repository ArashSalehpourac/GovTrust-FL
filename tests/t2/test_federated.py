import pandas as pd

from src.t2.config import T2Config
from src.t2.data import prepare_resolved_frame
from src.t2.federated import train_select
from src.t2.folds import chronological_split
from src.t2.preprocessing import fit_source_train


def _raw(city: str, n: int = 36) -> pd.DataFrame:
    created = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    closed = created + pd.to_timedelta([(i % 12) + 1 for i in range(n)], unit="h")
    return pd.DataFrame({
        "created_date": created,
        "closed_date": closed,
        "category": [f"{city}-cat-{i % 3}" for i in range(n)],
        "descriptor": [f"common request token {i % 5}" for i in range(n)],
        "status": "closed",
    })


def test_nonprivate_fedavg_point_regression_path_runs():
    prepared = {city: prepare_resolved_frame(_raw(city)) for city in ("a", "b", "c")}
    splits = {city: chronological_split(frame) for city, frame in prepared.items()}
    pre = fit_source_train(splits, max_text_features=20)
    cfg = T2Config(rounds=2, batch_size=8, hidden_sizes=(8,), learning_rate=0.01, seed=3)
    result = train_select(splits, pre, cfg)
    assert 1 <= result["best_round"] <= 2
    assert result["source_macro_mae_log1p_hours"] >= 0
    assert set(result["internal_by_city"]) == {"a", "b", "c"}
