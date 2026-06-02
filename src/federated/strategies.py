"""Federated aggregation strategy helpers."""

from __future__ import annotations

import numpy as np
import torch

from src.secure_aggregation import plain_fedavg


def fedavg_parameters(client_updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """Aggregate client parameters with standard weighted FedAvg."""

    return plain_fedavg(client_updates)


def fedprox_proximal_term(model: torch.nn.Module, global_parameters: list[torch.Tensor]) -> torch.Tensor:
    """Compute ||w_local - w_global||^2 for FedProx local objectives."""

    device = next(model.parameters()).device
    proximal = torch.zeros((), device=device)
    for parameter, global_parameter in zip(model.parameters(), global_parameters, strict=True):
        proximal = proximal + torch.sum((parameter - global_parameter.to(device)) ** 2)
    return proximal
