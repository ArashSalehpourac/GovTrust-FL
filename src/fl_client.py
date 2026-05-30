"""Flower client scaffolding for federated experiments."""

from collections.abc import Callable

import numpy as np

try:
    import flwr as fl
except ImportError:  # pragma: no cover
    fl = None


BaseNumPyClient = fl.client.NumPyClient if fl is not None else object


class SklearnFlowerClient(BaseNumPyClient):
    """Minimal Flower client around a sklearn-like estimator."""

    def __init__(
        self,
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        get_parameters_fn: Callable,
        set_parameters_fn: Callable,
        evaluate_fn: Callable,
    ) -> None:
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.get_parameters_fn = get_parameters_fn
        self.set_parameters_fn = set_parameters_fn
        self.evaluate_fn = evaluate_fn

    def get_parameters(self, config):
        return self.get_parameters_fn(self.model)

    def fit(self, parameters, config):
        self.set_parameters_fn(self.model, parameters)
        self.model.fit(self.x_train, self.y_train)
        return self.get_parameters_fn(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters_fn(self.model, parameters)
        loss, metrics = self.evaluate_fn(self.model, self.x_val, self.y_val)
        return float(loss), len(self.x_val), metrics


def require_flower() -> None:
    """Raise a helpful error when Flower is not installed."""

    if fl is None:
        raise ImportError("Flower is not installed. Run `pip install flwr` first.")


def ndarray_parameters(model) -> list[np.ndarray]:
    """Extract simple coef/intercept parameters from linear sklearn models."""

    return [model.coef_.copy(), model.intercept_.copy()]


def set_ndarray_parameters(model, parameters: list[np.ndarray]) -> None:
    """Set simple coef/intercept parameters on linear sklearn models."""

    model.coef_ = parameters[0].copy()
    model.intercept_ = parameters[1].copy()
    if not hasattr(model, "classes_"):
        model.classes_ = np.array([0, 1])

