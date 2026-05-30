"""Baseline model factories."""

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE
from .features import make_tabular_preprocessor


def make_estimator(name: str = "logistic_regression", **kwargs: Any):
    """Create a tabular classifier by name."""

    model_name = name.lower()

    if model_name in {"logistic", "logistic_regression", "lr"}:
        return LogisticRegression(max_iter=1_000, class_weight="balanced", **kwargs)

    if model_name in {"random_forest", "rf"}:
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **kwargs,
        )

    if model_name in {"hist_gradient_boosting", "hgb"}:
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **kwargs)

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **kwargs,
        )

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, **kwargs)

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(random_seed=RANDOM_STATE, verbose=False, **kwargs)

    raise ValueError(f"Unknown model name: {name}")


def make_pipeline(model_name: str = "logistic_regression", **model_kwargs: Any) -> Pipeline:
    """Create a preprocessing + model sklearn pipeline."""

    return Pipeline(
        steps=[
            ("preprocess", make_tabular_preprocessor()),
            ("model", make_estimator(model_name, **model_kwargs)),
        ]
    )

