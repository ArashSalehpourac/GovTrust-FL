"""PyTorch MLP model utilities for federated delayed-risk prediction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DelayRiskMLP(nn.Module):
    """Small binary MLP used by the Flower/FedAvg smoke pipeline."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Return model state as NumPy arrays in state-dict order."""

    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load NumPy parameters into a model."""

    state = model.state_dict()
    new_state = {
        key: torch.tensor(value, dtype=state[key].dtype)
        for key, value in zip(state.keys(), parameters, strict=True)
    }
    model.load_state_dict(new_state, strict=True)


def model_size_bytes(model: nn.Module) -> int:
    """Return trainable model size in bytes."""

    return int(sum(parameter.nelement() * parameter.element_size() for parameter in model.parameters()))
