"""Personalized federated learning utilities."""

from __future__ import annotations

import torch

from src.fl_training import train_local
from src.dp_training import DPTrainingConfig


def fine_tune_global_model(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, float | None]:
    """Fine-tune a global model on one client's local training data."""

    global_parameters = [parameter.detach().cpu().numpy() for parameter in model.parameters()]
    return train_local(
        model,
        x_train,
        y_train,
        batch_size=batch_size,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
        proximal_mu=0.0,
        global_parameters=global_parameters,
        dp_config=DPTrainingConfig(enabled=False),
    )
