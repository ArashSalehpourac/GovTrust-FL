from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import T2Config
from .data import PRIMARY_TARGET
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
    x: torch.Tensor
    y: torch.Tensor
    privacy: PrivacyState

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


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


def _tensorize(frame: pd.DataFrame, pre: FixedPreprocessor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(pre.transform(frame))
    y = torch.from_numpy(frame[PRIMARY_TARGET].to_numpy(dtype=np.float32))
    return x, y


def _predict(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
    base = getattr(model, "_module", model)
    device = next(base.parameters()).device
    base.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size].to(device)
            rows.append(base(batch).detach().cpu().numpy())
    return np.concatenate(rows) if rows else np.array([], dtype=np.float32)


def _evaluate_frame(model: torch.nn.Module, frame: pd.DataFrame, pre: FixedPreprocessor) -> dict[str, float]:
    x, y = _tensorize(frame, pre)
    return regression_metrics(y.numpy(), _predict(model, x))


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


def train_select(
    source_splits: Mapping[str, Mapping[str, pd.DataFrame]],
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
    for city, splits in source_splits.items():
        x, y = _tensorize(splits["train"], preprocessor)
        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        local_model = ResolutionMLP(input_dim, config.hidden_sizes).to(device)
        load_plain_state_dict(local_model, global_state)
        planned_steps = config.rounds * config.local_epochs * max(1, len(loader))
        privacy = make_training_state(
            model=local_model,
            loader=loader,
            mode=config.mode,
            target_epsilon=config.target_epsilon,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            planned_steps=planned_steps,
        )
        clients[city] = ClientBundle(city, x, y, privacy)

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
            for _ in range(config.local_epochs):
                for batch_x, batch_y in state.loader:
                    if len(batch_x) == 0:
                        continue
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
            city: _evaluate_frame(global_model, source_splits[city]["val"], preprocessor)
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
        city: _evaluate_frame(global_model, source_splits[city]["internal_test"], preprocessor)
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
    }


def evaluate_external_after_selection(
    selected: Mapping[str, object],
    external_frame: pd.DataFrame,
    preprocessor: FixedPreprocessor,
) -> dict[str, float]:
    """External outcomes enter only after the source-only checkpoint is frozen."""

    model = selected["model"]
    if not isinstance(model, torch.nn.Module):
        raise TypeError("selected model missing")
    return _evaluate_frame(model, external_frame, preprocessor)
