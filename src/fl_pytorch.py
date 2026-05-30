"""Flower-compatible PyTorch FL utilities for Step 9/10."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CATEGORICAL_FEATURES = ["city", "category", "descriptor", "agency", "area"]
FEDERATED_CITIES = ("nyc", "chicago", "boston")
NON_FEATURE_COLUMNS = {"request_id", "created_date", "delayed", "service_routing_target"}


@dataclass
class ClientData:
    """Prepared tensors for one FL client."""

    city: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


class DelayMLP(nn.Module):
    """Small MLP for delayed-resolution risk."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def build_dense_preprocessor(train_frame: pd.DataFrame) -> ColumnTransformer:
    """Fit a dense sklearn preprocessor suitable for PyTorch tensors."""

    feature_columns = infer_feature_columns(train_frame)
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical),
            ("num", numeric_pipeline, numeric),
        ],
        sparse_threshold=0.0,
    )


def infer_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in NON_FEATURE_COLUMNS]


def clean_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable/string columns into sklearn-friendly dtypes."""

    features = frame[infer_feature_columns(frame)].copy()
    for column in CATEGORICAL_FEATURES:
        if column in features.columns:
            features[column] = features[column].astype("object").where(features[column].notna(), np.nan)
    for column in features.columns:
        if column not in CATEGORICAL_FEATURES:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def sample_frame(frame: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    """Deterministically sample rows while preserving all rows when under limit."""

    if max_rows is None or len(frame) <= max_rows:
        return frame.sort_values(["created_date", "request_id"], kind="stable").reset_index(drop=True)
    return (
        frame.sample(n=max_rows, random_state=random_state)
        .sort_values(["created_date", "request_id"], kind="stable")
        .reset_index(drop=True)
    )


def transform_to_tensors(
    frame: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = preprocessor.transform(clean_feature_frame(frame)).astype(np.float32)
    labels = frame["delayed"].to_numpy(dtype=np.float32)
    return torch.from_numpy(features), torch.from_numpy(labels)


def prepare_client_data(
    city_frames: dict[str, dict[str, pd.DataFrame]],
    preprocessor: ColumnTransformer,
) -> dict[str, ClientData]:
    """Transform client train/val/test frames into tensors."""

    clients = {}
    for city, splits in city_frames.items():
        x_train, y_train = transform_to_tensors(splits["train"], preprocessor)
        x_val, y_val = transform_to_tensors(splits["val"], preprocessor)
        x_test, y_test = transform_to_tensors(splits["test"], preprocessor)
        clients[city] = ClientData(
            city=city,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )
    return clients


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Return model parameters as numpy arrays in state_dict order."""

    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Set model parameters from numpy arrays."""

    state_dict = model.state_dict()
    new_state = {
        key: torch.tensor(value, dtype=state_dict[key].dtype)
        for key, value in zip(state_dict.keys(), parameters, strict=True)
    }
    model.load_state_dict(new_state, strict=True)


def aggregate_parameters(
    client_parameters: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    """Weighted FedAvg aggregation."""

    total_examples = sum(num_examples for _, num_examples in client_parameters)
    aggregated = []
    for parameter_values in zip(*(params for params, _ in client_parameters), strict=True):
        weighted = sum(
            value * (num_examples / total_examples)
            for value, (_, num_examples) in zip(parameter_values, client_parameters, strict=True)
        )
        aggregated.append(weighted.astype(np.float32, copy=False))
    return aggregated


def protect_client_update(
    client_parameters: list[np.ndarray],
    global_parameters: list[np.ndarray],
    *,
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
    num_examples: int,
) -> list[np.ndarray]:
    """Clip and noise one client update for an update-level DP simulation."""

    update = [
        (client_value - global_value).astype(np.float32, copy=False)
        for client_value, global_value in zip(client_parameters, global_parameters, strict=True)
    ]
    update_norm = float(np.sqrt(sum(float(np.sum(value * value)) for value in update)))
    scale = min(1.0, clip_norm / max(update_norm, 1e-12))
    # This is an update-level approximation: Opacus calibrates the target noise
    # multiplier, while the injected noise is scaled down by client sample size
    # to keep this first FL privacy experiment numerically stable on CPU.
    noise_std = float(noise_multiplier * clip_norm / max(np.sqrt(num_examples), 1.0))
    protected_update = [
        value * scale + rng.normal(0.0, noise_std, size=value.shape).astype(np.float32)
        for value in update
    ]
    return [
        (global_value + update_value).astype(np.float32, copy=False)
        for global_value, update_value in zip(global_parameters, protected_update, strict=True)
    ]


class TorchFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient wrapper around local PyTorch training."""

    def __init__(
        self,
        client_data: ClientData,
        input_dim: int,
        batch_size: int,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        self.client_data = client_data
        self.model = DelayMLP(input_dim).to(device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        set_parameters(self.model, parameters)
        global_parameters = [np.copy(parameter) for parameter in parameters]
        local_epochs = int(config.get("local_epochs", 1))
        proximal_mu = float(config.get("proximal_mu", 0.0))
        train_loss = train_local_model(
            self.model,
            self.client_data.x_train,
            self.client_data.y_train,
            self.batch_size,
            local_epochs,
            self.learning_rate,
            self.device,
            proximal_mu=proximal_mu,
            global_parameters=global_parameters,
        )
        return get_parameters(self.model), len(self.client_data.y_train), {"train_loss": train_loss}

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        set_parameters(self.model, parameters)
        metrics = evaluate_model(self.model, self.client_data.x_val, self.client_data.y_val, self.device)
        return float(metrics["brier"]), len(self.client_data.y_val), metrics


def train_local_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    device: torch.device,
    *,
    proximal_mu: float = 0.0,
    global_parameters: list[np.ndarray] | None = None,
) -> float:
    """Train one client model locally."""

    model.train()
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    positives = float(y_train.sum().item())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([max(negatives / max(positives, 1.0), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    global_tensors = None
    if proximal_mu > 0 and global_parameters is not None:
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
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            if global_tensors is not None:
                proximal_term = torch.zeros((), device=device)
                for parameter, global_parameter in zip(model.parameters(), global_tensors, strict=True):
                    proximal_term = proximal_term + torch.sum((parameter - global_parameter) ** 2)
                loss = loss + (proximal_mu / 2.0) * proximal_term
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    scores = []
    loader = DataLoader(TensorDataset(features, labels), batch_size=4096, shuffle=False)
    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        scores.append(torch.sigmoid(model(batch_x)).cpu().numpy())
    y_score = np.concatenate(scores)
    y_true = labels.numpy().astype(int)
    return binary_metrics(y_true, y_score)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "ece": float(expected_calibration_error(y_true, y_score)),
    }


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bin_edges[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(ece)


def model_size_bytes(model: nn.Module) -> int:
    return int(sum(parameter.nelement() * parameter.element_size() for parameter in model.parameters()))


def communication_cost(model: nn.Module, clients: int, rounds: int) -> dict[str, int]:
    size = model_size_bytes(model)
    upload = size * clients * rounds
    download = size * clients * rounds
    return {
        "model_size_bytes": size,
        "upload_bytes": upload,
        "download_bytes": download,
        "total_bytes": upload + download,
    }


def run_fl_experiment(
    algorithm: str,
    clients: dict[str, TorchFlowerClient],
    input_dim: int,
    rounds: int,
    local_epochs: int,
    proximal_mu: float,
    patience: int,
    min_delta: float,
    device: torch.device,
    dp_noise_multiplier: float = 0.0,
    dp_clip_norm: float = 1.0,
    secure_aggregation: bool = False,
    secure_aggregation_bytes_factor: float = 1.0,
    seed: int = 42,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    """Run a local Flower-compatible FedAvg/FedProx simulation."""

    global_model = DelayMLP(input_dim).to(device)
    global_parameters = get_parameters(global_model)
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_parameters = [np.copy(parameter) for parameter in global_parameters]
    best_macro_f1 = -np.inf
    best_round = 0
    stale_rounds = 0

    process = psutil.Process()
    peak_memory = process.memory_info().rss
    started = time.perf_counter()

    for round_number in range(1, rounds + 1):
        round_started = time.perf_counter()
        client_results = []
        train_losses = []
        for city, client in clients.items():
            fit_config = {
                "local_epochs": local_epochs,
                "proximal_mu": proximal_mu if algorithm == "fedprox" else 0.0,
            }
            parameters, num_examples, metrics = client.fit(global_parameters, fit_config)
            if dp_noise_multiplier > 0:
                parameters = protect_client_update(
                    parameters,
                    global_parameters,
                    clip_norm=dp_clip_norm,
                    noise_multiplier=dp_noise_multiplier,
                    rng=rng,
                    num_examples=num_examples,
                )
            client_results.append((parameters, num_examples))
            train_losses.append(metrics["train_loss"])

        global_parameters = aggregate_parameters(client_results)
        set_parameters(global_model, global_parameters)
        # Evaluate the shared global model, not each stale local model.
        validation_metrics = {
            city: evaluate_model(global_model, client.client_data.x_val, client.client_data.y_val, device)
            for city, client in clients.items()
        }
        mean_macro_f1 = float(np.mean([metrics["macro_f1"] for metrics in validation_metrics.values()]))
        round_seconds = time.perf_counter() - round_started
        peak_memory = max(peak_memory, process.memory_info().rss)
        history.append(
            {
                "round": round_number,
                "algorithm": algorithm,
                "dp_noise_multiplier": float(dp_noise_multiplier),
                "secure_aggregation": bool(secure_aggregation),
                "mean_train_loss": float(np.mean(train_losses)),
                "mean_val_macro_f1": mean_macro_f1,
                "round_seconds": round_seconds,
                **{
                    f"{city}_val_macro_f1": metrics["macro_f1"]
                    for city, metrics in validation_metrics.items()
                },
            }
        )
        print(
            f"  {algorithm} round {round_number}: "
            f"val_macro_f1={mean_macro_f1:.4f}, "
            f"loss={float(np.mean(train_losses)):.4f}, "
            f"sec={round_seconds:.1f}",
            flush=True,
        )

        if mean_macro_f1 > best_macro_f1 + min_delta:
            best_macro_f1 = mean_macro_f1
            best_round = round_number
            best_parameters = [np.copy(parameter) for parameter in global_parameters]
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= patience:
                break

    set_parameters(global_model, best_parameters)
    elapsed = time.perf_counter() - started
    costs = communication_cost(global_model, clients=len(clients), rounds=len(history))
    secagg_extra_bytes = 0
    if secure_aggregation:
        secagg_extra_bytes = int(costs["model_size_bytes"] * len(clients) * len(history) * secure_aggregation_bytes_factor)
    summary = {
        "algorithm": algorithm,
        "dp_noise_multiplier": float(dp_noise_multiplier),
        "dp_clip_norm": float(dp_clip_norm),
        "secure_aggregation": bool(secure_aggregation),
        "rounds_completed": len(history),
        "best_round": best_round,
        "best_validation_macro_f1": float(best_macro_f1),
        "training_time_sec": elapsed,
        "memory_peak_mb": peak_memory / (1024 * 1024),
        **costs,
        "secure_aggregation_extra_bytes": secagg_extra_bytes,
        "total_bytes_with_overheads": costs["total_bytes"] + secagg_extra_bytes,
    }
    return global_model, history, summary


def save_torch_model(model: nn.Module, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
