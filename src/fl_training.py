"""Flower-compatible federated training loop for synthetic/reproducible runs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .dp_training import DPTrainingConfig, epsilon_or_none, make_private_if_enabled
from .features import build_xy
from .fl_model import DelayRiskMLP, get_parameters, model_size_bytes, set_parameters
from .metrics import binary_classification_metrics
from .secure_aggregation import plain_fedavg, simulated_secure_aggregate


@dataclass
class ClientTensors:
    """Prepared tensors for one city client."""

    city: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def make_dense_preprocessor(train_frame: pd.DataFrame) -> ColumnTransformer:
    """Create a dense tabular preprocessor for PyTorch tensors."""

    x_train, _ = build_xy(train_frame)
    categorical = [column for column in x_train.columns if not pd.api.types.is_numeric_dtype(x_train[column])]
    numeric = [column for column in x_train.columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )


def fit_preprocessor(frames: list[pd.DataFrame]) -> ColumnTransformer:
    """Fit preprocessing on training-city data only."""

    train = pd.concat(frames, ignore_index=True)
    x_train, _ = build_xy(train)
    preprocessor = make_dense_preprocessor(train)
    preprocessor.fit(x_train)
    return preprocessor


def frame_to_tensors(frame: pd.DataFrame, preprocessor: ColumnTransformer) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform a processed frame into tensors."""

    features, target = build_xy(frame)
    x = preprocessor.transform(features).astype(np.float32)
    y = target.to_numpy(dtype=np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


class FlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient with local PyTorch training."""

    def __init__(
        self,
        tensors: ClientTensors,
        input_dim: int,
        hidden_dim: int,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        self.tensors = tensors
        self.model = DelayRiskMLP(input_dim, hidden_dim=hidden_dim).to(device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]):
        set_parameters(self.model, parameters)
        global_parameters = [np.copy(parameter) for parameter in parameters]
        dp_config = DPTrainingConfig(
            enabled=bool(config.get("dp_enabled", False)),
            noise_multiplier=float(config.get("noise_multiplier", 1.0)),
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            delta=float(config.get("delta", 1e-5)),
        )
        train_info = train_local(
            self.model,
            self.tensors.x_train,
            self.tensors.y_train,
            batch_size=self.batch_size,
            local_epochs=int(config.get("local_epochs", 1)),
            learning_rate=self.learning_rate,
            device=self.device,
            proximal_mu=float(config.get("proximal_mu", 0.0)),
            global_parameters=global_parameters,
            dp_config=dp_config,
        )
        return get_parameters(self.model), len(self.tensors.y_train), train_info

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]):
        set_parameters(self.model, parameters)
        metrics = evaluate_model(self.model, self.tensors.x_val, self.tensors.y_val, self.device)
        return float(metrics["brier"]), len(self.tensors.y_val), metrics


def train_local(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    device: torch.device,
    proximal_mu: float,
    global_parameters: list[np.ndarray],
    dp_config: DPTrainingConfig,
) -> dict[str, float | None]:
    """Train one local client model."""

    model.train()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer, loader, privacy_engine = make_private_if_enabled(model, optimizer, loader, dp_config)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    global_tensors = [
        torch.tensor(value, dtype=parameter.dtype, device=device)
        for value, parameter in zip(global_parameters, model.parameters(), strict=True)
    ]
    losses = []
    for _ in range(local_epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            if proximal_mu > 0:
                proximal = torch.zeros((), device=device)
                for parameter, global_parameter in zip(model.parameters(), global_tensors, strict=True):
                    proximal = proximal + torch.sum((parameter - global_parameter) ** 2)
                loss = loss + (proximal_mu / 2.0) * proximal
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    return {
        "train_loss": float(np.mean(losses)) if losses else 0.0,
        "epsilon": epsilon_or_none(privacy_engine, dp_config.delta),
    }


@torch.no_grad()
def predict_scores_torch(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    for start in range(0, len(x), 2048):
        batch = x[start : start + 2048].to(device)
        scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(scores)


def evaluate_model(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> dict[str, float]:
    return binary_classification_metrics(y.numpy().astype(int), predict_scores_torch(model, x, device))


def run_federated_experiment(
    clients: dict[str, FlowerClient],
    *,
    input_dim: int,
    hidden_dim: int,
    rounds: int,
    local_epochs: int,
    algorithm: str,
    fedprox_mu: float,
    dp_config: DPTrainingConfig,
    secure_aggregation: bool,
    secure_overhead_factor: float,
    device: torch.device,
) -> tuple[DelayRiskMLP, list[dict[str, Any]], dict[str, Any]]:
    """Run a local Flower-compatible FedAvg/FedProx experiment."""

    model = DelayRiskMLP(input_dim, hidden_dim=hidden_dim).to(device)
    parameters = get_parameters(model)
    history = []
    started = time.perf_counter()
    secagg_overhead = 0.0
    epsilons = []

    for round_number in range(1, rounds + 1):
        client_updates = []
        losses = []
        for city, client in clients.items():
            fit_config = {
                "local_epochs": local_epochs,
                "proximal_mu": fedprox_mu if algorithm.lower() == "fedprox" else 0.0,
                "dp_enabled": dp_config.enabled,
                "noise_multiplier": dp_config.noise_multiplier,
                "max_grad_norm": dp_config.max_grad_norm,
                "delta": dp_config.delta,
            }
            new_parameters, num_examples, metrics = client.fit(parameters, fit_config)
            client_updates.append((new_parameters, num_examples))
            losses.append(float(metrics.get("train_loss", 0.0)))
            if metrics.get("epsilon") is not None:
                epsilons.append(float(metrics["epsilon"]))

        if secure_aggregation:
            parameters, secagg_info = simulated_secure_aggregate(
                client_updates,
                overhead_factor=secure_overhead_factor,
            )
            secagg_overhead += secagg_info["secure_aggregation_overhead_bytes"]
        else:
            parameters = plain_fedavg(client_updates)
        set_parameters(model, parameters)
        val_metrics = {
            city: evaluate_model(model, client.tensors.x_val, client.tensors.y_val, device)
            for city, client in clients.items()
        }
        history.append(
            {
                "round": round_number,
                "algorithm": algorithm,
                "mean_train_loss": float(np.mean(losses)),
                "mean_val_roc_auc": float(np.nanmean([metrics["roc_auc"] for metrics in val_metrics.values()])),
                "mean_val_f1": float(np.nanmean([metrics["f1"] for metrics in val_metrics.values()])),
            }
        )

    elapsed = time.perf_counter() - started
    model_bytes = model_size_bytes(model)
    communication_bytes = model_bytes * len(clients) * rounds * 2
    summary = {
        "algorithm": algorithm,
        "rounds": rounds,
        "training_time_sec": elapsed,
        "model_size_bytes": model_bytes,
        "communication_bytes": communication_bytes + secagg_overhead,
        "secure_aggregation_overhead_bytes": secagg_overhead,
        "epsilon": max(epsilons) if epsilons else None,
    }
    return model, history, summary
