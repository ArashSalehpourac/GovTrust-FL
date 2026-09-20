from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .compact import CompactRegressionDataset
from .config import T2Config
from .evaluation import macro_source_mae, regression_metrics
from .model import ResolutionMLP, regression_loss
from .preprocessing import FixedPreprocessor
from .privacy import (
    PrivacyState,
    load_plain_state_dict,
    make_training_state,
    plain_state_dict,
)


@dataclass
class ClientBundle:
    city: str
    dataset: Dataset
    privacy: PrivacyState
    steps_per_round: int
    loader_batches: int

    @property
    def n(self) -> int:
        return int(len(self.dataset))


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested != "auto":
        raise ValueError(f"unknown device mode: {requested}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_dataset(
    frame_or_dataset: pd.DataFrame | Dataset,
    pre: FixedPreprocessor,
) -> Dataset:
    if isinstance(frame_or_dataset, pd.DataFrame):
        return CompactRegressionDataset.from_frame(frame_or_dataset, pre)
    if isinstance(frame_or_dataset, Dataset):
        return frame_or_dataset
    raise TypeError("split must be a pandas DataFrame or torch Dataset")


def _predict_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    base = getattr(model, "_module", model)
    device = next(base.parameters()).device
    base.eval()
    pred_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            pred_rows.append(base(batch_x.to(device)).detach().cpu().numpy())
            target_rows.append(batch_y.detach().cpu().numpy())
    predictions = (
        np.concatenate(pred_rows) if pred_rows else np.array([], dtype=np.float32)
    )
    targets = (
        np.concatenate(target_rows) if target_rows else np.array([], dtype=np.float32)
    )
    return targets, predictions


def _evaluate_frame(
    model: torch.nn.Module,
    frame_or_dataset: pd.DataFrame | Dataset,
    pre: FixedPreprocessor,
) -> dict[str, float]:
    dataset = _as_dataset(frame_or_dataset, pre)
    y, pred = _predict_dataset(model, dataset)
    return regression_metrics(y, pred)


def _average_states(states: list[tuple[dict[str, torch.Tensor], int]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("no client states")
    total = float(sum(n for _, n in states))
    out: dict[str, torch.Tensor] = {}
    for key in states[0][0]:
        first = states[0][0][key]
        acc = torch.zeros_like(first, dtype=torch.float64)
        for state, n in states:
            acc.add_(state[key].to(torch.float64), alpha=n / total)
        out[key] = acc.to(first.dtype)
    return out


def _step_budget(loader_batches: int, config: T2Config) -> tuple[int, int]:
    if loader_batches < 1:
        raise ValueError("client loader must have at least one batch")
    if config.training_budget == "full_local_epochs":
        steps_per_round = config.local_epochs * loader_batches
    else:
        target_total_steps = max(
            1,
            math.ceil(config.total_effective_epochs * loader_batches),
        )
        steps_per_round = max(1, math.ceil(target_total_steps / config.rounds))
    return steps_per_round, steps_per_round * config.rounds


def _take_nonempty_batch(iterator, loader):
    while True:
        try:
            batch_x, batch_y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch_x, batch_y = next(iterator)
        if len(batch_x) > 0:
            return iterator, batch_x, batch_y


def train_select(
    source_splits: Mapping[str, Mapping[str, pd.DataFrame | Dataset]],
    preprocessor: FixedPreprocessor,
    config: T2Config,
) -> dict[str, object]:
    """Train/select using SOURCE cities only. No external frame is accepted."""

    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    input_dim = preprocessor.output_dim
    global_model = ResolutionMLP(input_dim, config.hidden_sizes).to(device)
    global_state = plain_state_dict(global_model)

    clients: dict[str, ClientBundle] = {}
    compact_splits: dict[str, dict[str, Dataset]] = {}
    for city, splits in source_splits.items():
        city_splits = {
            name: _as_dataset(split, preprocessor)
            for name, split in splits.items()
        }
        compact_splits[city] = city_splits
        train_dataset = city_splits["train"]
        loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        local_model = ResolutionMLP(input_dim, config.hidden_sizes).to(device)
        load_plain_state_dict(local_model, global_state)
        steps_per_round, planned_steps = _step_budget(len(loader), config)
        privacy = make_training_state(
            model=local_model,
            loader=loader,
            mode=config.mode,
            target_epsilon=config.target_epsilon,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            planned_steps=planned_steps,
        )
        clients[city] = ClientBundle(
            city,
            train_dataset,
            privacy,
            steps_per_round,
            len(loader),
        )

    best_state = copy.deepcopy(global_state)
    best_macro_val = float("inf")
    best_round = 0
    rounds_without_improvement = 0
    history: list[dict[str, object]] = []
    would_stop_round: int | None = None

    for round_number in range(1, config.rounds + 1):
        updates: list[tuple[dict[str, torch.Tensor], int]] = []
        local_losses: dict[str, float] = {}
        privacy_round: dict[str, dict[str, object]] = {}
        for city, bundle in clients.items():
            state = bundle.privacy
            load_plain_state_dict(state.model, global_state)
            state.model.train()
            losses: list[float] = []
            iterator = iter(state.loader)
            for _ in range(bundle.steps_per_round):
                iterator, batch_x, batch_y = _take_nonempty_batch(
                    iterator,
                    state.loader,
                )
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                state.optimizer.zero_grad(set_to_none=True)
                pred = state.model(batch_x)
                loss = regression_loss(pred, batch_y)
                loss.backward()
                state.optimizer.step()
                state.steps_taken += 1
                losses.append(float(loss.detach().cpu()))
            updates.append((plain_state_dict(state.model), bundle.n))
            local_losses[city] = float(np.mean(losses)) if losses else float("nan")
            privacy_round[city] = state.snapshot(round_number)

        global_state = _average_states(updates)
        load_plain_state_dict(global_model, global_state)
        val_by_city = {
            city: _evaluate_frame(global_model, compact_splits[city]["val"], preprocessor)
            for city in sorted(source_splits)
        }
        macro_val = macro_source_mae(val_by_city)
        history.append(
            {
                "round": round_number,
                "local_train_loss": local_losses,
                "source_val_by_city": val_by_city,
                "source_val_macro_mae_log1p_hours": macro_val,
                "privacy_by_city": privacy_round,
            }
        )

        if macro_val < best_macro_val:
            best_macro_val = macro_val
            best_round = round_number
            best_state = copy.deepcopy(global_state)
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= config.checkpoint_patience and would_stop_round is None:
                would_stop_round = round_number
                if config.stop_on_patience:
                    break

    load_plain_state_dict(global_model, best_state)
    internal_by_city = {
        city: _evaluate_frame(global_model, compact_splits[city]["internal_test"], preprocessor)
        for city in sorted(source_splits)
    }
    privacy_reports = {
        city: bundle.privacy.report(config.epsilon_tolerance)
        for city, bundle in clients.items()
    }
    if config.mode == "private" and not all(
        report["target_achieved_within_tolerance"] for report in privacy_reports.values()
    ):
        raise RuntimeError("realized epsilon is outside tolerance; do not label this run as target epsilon")

    return {
        "model": global_model,
        "best_state": best_state,
        "best_round": best_round,
        "device": str(device),
        "source_val_best_macro_mae_log1p_hours": best_macro_val,
        "would_stop_round": would_stop_round,
        "history": history,
        "internal_by_city": internal_by_city,
        "source_macro_mae_log1p_hours": macro_source_mae(internal_by_city),
        "privacy_by_city": privacy_reports,
        "training_budget_by_city": {
            city: {
                "loader_batches": bundle.loader_batches,
                "steps_per_round": bundle.steps_per_round,
                "planned_steps": bundle.privacy.planned_steps,
                "planned_effective_epochs": (
                    bundle.privacy.planned_steps / bundle.loader_batches
                ),
            }
            for city, bundle in clients.items()
        },
    }


def evaluate_external_after_selection(
    selected: Mapping[str, object],
    external_frame: pd.DataFrame | Dataset,
    preprocessor: FixedPreprocessor,
) -> dict[str, float]:
    """External outcomes enter only after the source-only checkpoint is frozen."""

    model = selected["model"]
    if not isinstance(model, torch.nn.Module):
        raise TypeError("selected model missing")
    return _evaluate_frame(model, external_frame, preprocessor)
