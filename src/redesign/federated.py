"""FedAvg / FedProx over source cities, with a held-out city never trained on.

Internal metrics (source-city validation and internal test) and external
metrics (the held-out city, zero-shot) are computed and reported separately.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    PRIMARY_TARGET_COLUMN,
    QUANTILES,
    SECONDARY_TARGET_COLUMN,
)
from .dp import DPSettings, DPState, make_private
from .folds import FoldData
from .model import QuantileDelayMLP, multitask_loss, predict
from .reliability import pinball_loss, reliability_report, subgroup_reliability


@dataclass
class FederatedConfig:
    """One federated run configuration."""

    algorithm: str = "fedavg"
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 32)
    proximal_mu: float = 0.01
    quantiles: tuple[float, ...] = QUANTILES
    secondary_loss_weight: float = 0.5
    seed: int = 0
    target_epsilon: float = float("inf")
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    subgroup_min_rows: int = 50

    @property
    def dp_settings(self) -> DPSettings:
        return DPSettings(
            target_epsilon=self.target_epsilon,
            delta=self.delta,
            max_grad_norm=self.max_grad_norm,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "rounds": self.rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "hidden_sizes": list(self.hidden_sizes),
            "proximal_mu": self.proximal_mu if self.algorithm == "fedprox" else 0.0,
            "quantiles": list(self.quantiles),
            "secondary_loss_weight": self.secondary_loss_weight,
            "seed": self.seed,
            "target_epsilon": self.target_epsilon,
            "delta": self.delta,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass
class SplitTensors:
    """Model-ready tensors for one split."""

    frame: pd.DataFrame
    x: torch.Tensor
    y_log: torch.Tensor
    y_sla: torch.Tensor

    def __len__(self) -> int:
        return int(self.x.shape[0])


def to_tensors(frame: pd.DataFrame, fold: FoldData) -> SplitTensors:
    """Transform one split with the fold's source-train-only preprocessor."""

    features = fold.preprocessor.transform(frame)
    return SplitTensors(
        frame=frame.reset_index(drop=True),
        x=torch.from_numpy(features),
        y_log=torch.from_numpy(frame[PRIMARY_TARGET_COLUMN].to_numpy(dtype=np.float32)),
        y_sla=torch.from_numpy(frame[SECONDARY_TARGET_COLUMN].to_numpy(dtype=np.float32)),
    )


class CityClient:
    """One municipal data holder: local training only, never sees other cities."""

    def __init__(
        self,
        city: str,
        data: SplitTensors,
        config: FederatedConfig,
        input_dim: int,
        planned_steps: int,
    ) -> None:
        self.city = city
        self.data = data
        self.config = config
        self.base_model = QuantileDelayMLP(
            input_dim, config.hidden_sizes, n_quantiles=len(config.quantiles)
        )
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=config.learning_rate)
        loader = DataLoader(
            TensorDataset(data.x, data.y_log, data.y_sla),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.dp_state: DPState | None = None
        if config.dp_settings.enabled:
            self.dp_state = make_private(
                model=self.base_model,
                optimizer=optimizer,
                data_loader=loader,
                settings=config.dp_settings,
                planned_steps=planned_steps,
            )
            self.train_model: nn.Module = self.dp_state.model
            self.optimizer = self.dp_state.optimizer
            self.loader = self.dp_state.data_loader
        else:
            self.train_model = self.base_model
            self.optimizer = optimizer
            self.loader = loader

    @property
    def n_train(self) -> int:
        return len(self.data)

    def set_parameters(self, state: dict[str, torch.Tensor]) -> None:
        self.base_model.load_state_dict(copy.deepcopy(state))

    def get_parameters(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.base_model.state_dict().items()}

    def train_round(self, global_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
        """Run local epochs starting from the global parameters."""

        self.set_parameters(global_state)
        self.train_model.train()
        proximal_mu = self.config.proximal_mu if self.config.algorithm == "fedprox" else 0.0
        global_tensors = (
            {key: value.detach().clone() for key, value in global_state.items()}
            if proximal_mu > 0
            else None
        )

        losses: list[float] = []
        for _ in range(self.config.local_epochs):
            for batch_x, batch_y_log, batch_y_sla in self.loader:
                if batch_x.shape[0] == 0:
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.train_model(batch_x)
                loss = multitask_loss(
                    outputs,
                    batch_y_log,
                    batch_y_sla,
                    self.config.quantiles,
                    self.config.secondary_loss_weight,
                )
                loss.backward()
                self.optimizer.step()
                if global_tensors is not None:
                    self._apply_proximal_update(global_tensors, proximal_mu)
                losses.append(float(loss.detach().cpu()))
                if self.dp_state is not None:
                    self.dp_state.steps_taken += 1

        return self.get_parameters(), float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def _apply_proximal_update(
        self, global_tensors: dict[str, torch.Tensor], proximal_mu: float
    ) -> None:
        """FedProx proximal pull, applied as a data-independent post-step update.

        The term depends only on the global parameters, so it costs no privacy
        budget and is kept outside the clipped/noised gradient path.
        """

        state = self.base_model.state_dict()
        for key, parameter in state.items():
            if not torch.is_floating_point(parameter):
                continue
            parameter.add_(
                -self.config.learning_rate * proximal_mu * (parameter - global_tensors[key])
            )

    def dp_report(self) -> dict[str, object] | None:
        return self.dp_state.report() if self.dp_state is not None else None


def average_states(
    states: Iterable[tuple[dict[str, torch.Tensor], int]]
) -> dict[str, torch.Tensor]:
    """Sample-size weighted FedAvg aggregation."""

    states = list(states)
    total = sum(weight for _, weight in states)
    aggregated: dict[str, torch.Tensor] = {}
    for key in states[0][0]:
        stacked = torch.stack(
            [state[key].to(torch.float64) * (weight / total) for state, weight in states]
        )
        aggregated[key] = stacked.sum(dim=0).to(states[0][0][key].dtype)
    return aggregated


@dataclass
class RunResult:
    """Everything one federated run produces."""

    config: dict[str, object]
    fold: str
    held_out_city: str
    source_cities: list[str]
    history: list[dict[str, object]]
    selection: dict[str, object]
    internal: dict[str, object]
    external: dict[str, object]
    privacy: dict[str, object]
    subgroups: list[dict[str, object]] = field(default_factory=list)
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "held_out_city": self.held_out_city,
            "source_cities": self.source_cities,
            "config": self.config,
            "privacy": self.privacy,
            "selection": self.selection,
            "history": self.history,
            "metrics_internal_source_cities": self.internal,
            "metrics_external_held_out_city": self.external,
            "subgroup_reliability": self.subgroups,
            "runtime_seconds": self.runtime_seconds,
        }


def run_federated(fold: FoldData, config: FederatedConfig) -> RunResult:
    """Train FedAvg/FedProx on the source cities and evaluate both regimes."""

    started = time.perf_counter()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_tensors = {
        city: to_tensors(splits["train"], fold) for city, splits in fold.source_splits.items()
    }
    val_tensors = {
        city: to_tensors(splits["val"], fold) for city, splits in fold.source_splits.items()
    }
    test_tensors = {
        city: to_tensors(splits["internal_test"], fold)
        for city, splits in fold.source_splits.items()
    }
    external_tensors = to_tensors(fold.external, fold)

    input_dim = fold.preprocessor.output_dim
    clients: dict[str, CityClient] = {}
    for city, data in train_tensors.items():
        steps_per_epoch = max(1, int(np.ceil(len(data) / config.batch_size)))
        planned_steps = steps_per_epoch * config.local_epochs * config.rounds
        clients[city] = CityClient(city, data, config, input_dim, planned_steps)

    global_model = QuantileDelayMLP(
        input_dim, config.hidden_sizes, n_quantiles=len(config.quantiles)
    )
    global_state = {key: value.detach().clone() for key, value in global_model.state_dict().items()}

    history: list[dict[str, object]] = []
    best_score = float("inf")
    best_round = 0
    best_state = copy.deepcopy(global_state)

    for round_number in range(1, config.rounds + 1):
        round_started = time.perf_counter()
        updates: list[tuple[dict[str, torch.Tensor], int]] = []
        train_losses: dict[str, float] = {}
        for city, client in clients.items():
            parameters, loss = client.train_round(global_state)
            updates.append((parameters, client.n_train))
            train_losses[city] = loss

        global_state = average_states(updates)
        global_model.load_state_dict(global_state)

        # Model selection uses pooled SOURCE-city validation rows only.
        val_scores = {
            city: _mean_pinball(global_model, data, config) for city, data in val_tensors.items()
        }
        pooled_val = float(
            np.average(
                [val_scores[city] for city in val_scores],
                weights=[len(val_tensors[city]) for city in val_scores],
            )
        )
        history.append(
            {
                "round": round_number,
                "mean_train_loss": float(np.mean(list(train_losses.values()))),
                "train_loss_by_city": train_losses,
                "source_val_mean_pinball": pooled_val,
                "source_val_mean_pinball_by_city": val_scores,
                "seconds": time.perf_counter() - round_started,
            }
        )
        if pooled_val < best_score:
            best_score = pooled_val
            best_round = round_number
            best_state = copy.deepcopy(global_state)

    global_model.load_state_dict(best_state)

    internal = {
        "selection_rule": "lowest pooled source-city validation mean pinball loss",
        "by_city_val": {
            city: _evaluate(global_model, data, fold, config, with_subgroups=False)["report"]
            for city, data in val_tensors.items()
        },
        "by_city_internal_test": {
            city: _evaluate(global_model, data, fold, config, with_subgroups=False)["report"]
            for city, data in test_tensors.items()
        },
    }
    pooled_internal_test = _pool(test_tensors)
    internal["pooled_internal_test"] = _evaluate(
        global_model, pooled_internal_test, fold, config, with_subgroups=False
    )["report"]

    external_evaluation = _evaluate(global_model, external_tensors, fold, config, with_subgroups=True)
    external = {
        "city": fold.spec.held_out_city,
        "regime": "zero-shot transfer; this city contributed no rows to preprocessing, "
        "target policy, model selection, or training",
        "report": external_evaluation["report"],
    }

    privacy = {
        "enabled": config.dp_settings.enabled,
        "definition": (
            "Example-level (record-level) DP-SGD applied independently inside each city. "
            "Each city's reported epsilon bounds the influence of one of its own service "
            "requests on every message that city releases across all rounds."
        )
        if config.dp_settings.enabled
        else "No differential privacy (epsilon = infinity baseline).",
        "by_city": {city: client.dp_report() for city, client in clients.items()},
    }

    return RunResult(
        config=config.to_dict(),
        fold=fold.spec.name,
        held_out_city=fold.spec.held_out_city,
        source_cities=list(fold.spec.source_cities),
        history=history,
        selection={"best_round": best_round, "source_val_mean_pinball": best_score},
        internal=internal,
        external=external,
        privacy=privacy,
        subgroups=external_evaluation["subgroups"],
        runtime_seconds=time.perf_counter() - started,
    )


def _pool(tensors: dict[str, SplitTensors]) -> SplitTensors:
    frames = pd.concat([data.frame for data in tensors.values()], ignore_index=True)
    return SplitTensors(
        frame=frames,
        x=torch.cat([data.x for data in tensors.values()], dim=0),
        y_log=torch.cat([data.y_log for data in tensors.values()], dim=0),
        y_sla=torch.cat([data.y_sla for data in tensors.values()], dim=0),
    )


def _mean_pinball(model: nn.Module, data: SplitTensors, config: FederatedConfig) -> float:
    quantile_predictions, _ = predict(model, data.x, config.quantiles)
    y_true = data.y_log.numpy()
    return float(
        np.mean([pinball_loss(y_true, quantile_predictions[tau], tau) for tau in config.quantiles])
    )


def _evaluate(
    model: nn.Module,
    data: SplitTensors,
    fold: FoldData,
    config: FederatedConfig,
    *,
    with_subgroups: bool,
) -> dict[str, object]:
    quantile_predictions, probabilities = predict(model, data.x, config.quantiles)
    y_true = data.y_log.numpy()
    y_sla = data.y_sla.numpy()
    report = reliability_report(
        y_true,
        quantile_predictions,
        y_sla,
        probabilities,
        winsor_hours=fold.policy.winsor_hours,
    )
    subgroups: list[dict[str, object]] = []
    if with_subgroups:
        subgroups = subgroup_reliability(
            data.frame,
            y_true,
            quantile_predictions,
            y_sla,
            probabilities,
            min_rows=config.subgroup_min_rows,
            winsor_hours=fold.policy.winsor_hours,
        )
    return {"report": report, "subgroups": subgroups}
