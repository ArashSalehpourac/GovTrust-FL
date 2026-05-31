"""Differential privacy hooks for local PyTorch client training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from opacus import PrivacyEngine


@dataclass(frozen=True)
class DPTrainingConfig:
    """Opacus privacy parameters."""

    enabled: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5


def make_private_if_enabled(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    config: DPTrainingConfig,
):
    """Attach Opacus PrivacyEngine when DP is enabled."""

    if not config.enabled:
        return model, optimizer, data_loader, None

    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=config.noise_multiplier,
        max_grad_norm=config.max_grad_norm,
    )
    return model, optimizer, data_loader, privacy_engine


def epsilon_or_none(privacy_engine: PrivacyEngine | None, delta: float) -> float | None:
    """Return epsilon if an Opacus accountant is available."""

    if privacy_engine is None:
        return None
    try:
        return float(privacy_engine.get_epsilon(delta=delta))
    except Exception:
        return None
