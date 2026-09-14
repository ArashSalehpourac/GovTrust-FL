"""Multi-quantile time-to-resolution model with an auxiliary SLA head."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class QuantileDelayMLP(nn.Module):
    """Predict several conditional quantiles of ``log1p(resolution_hours)``.

    The last output is an auxiliary logit for the secondary SLA-breach label.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        n_quantiles: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for index, size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous, size))
            layers.append(nn.ReLU())
            if index == 0:
                layers.append(nn.Dropout(0.1))
            previous = size
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(previous, n_quantiles + 1)
        self.n_quantiles = n_quantiles

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(features))


def split_outputs(outputs: torch.Tensor, n_quantiles: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the head into monotone quantile predictions and the SLA logit."""

    raw_quantiles = outputs[:, :n_quantiles]
    logit = outputs[:, n_quantiles]
    # Enforce non-crossing quantiles: base level plus non-negative increments.
    base = raw_quantiles[:, :1]
    increments = torch.nn.functional.softplus(raw_quantiles[:, 1:])
    quantiles = torch.cat([base, base + torch.cumsum(increments, dim=1)], dim=1)
    return quantiles, logit


def multitask_loss(
    outputs: torch.Tensor,
    y_log: torch.Tensor,
    y_sla: torch.Tensor,
    quantiles: tuple[float, ...],
    secondary_weight: float,
) -> torch.Tensor:
    """Mean pinball loss over quantile heads plus weighted BCE on the SLA head."""

    predicted, logit = split_outputs(outputs, len(quantiles))
    target = y_log.unsqueeze(1)
    residual = target - predicted
    taus = torch.tensor(quantiles, dtype=predicted.dtype, device=predicted.device).unsqueeze(0)
    pinball = torch.maximum(taus * residual, (taus - 1.0) * residual).mean()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, y_sla)
    return pinball + secondary_weight * bce


@torch.no_grad()
def predict(
    model: nn.Module,
    features: torch.Tensor,
    quantiles: tuple[float, ...],
    batch_size: int = 8192,
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """Return per-quantile predictions and SLA probabilities."""

    model.eval()
    quantile_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]
        outputs = model(batch)
        predicted, logit = split_outputs(outputs, len(quantiles))
        quantile_chunks.append(predicted.cpu().numpy())
        probability_chunks.append(torch.sigmoid(logit).cpu().numpy())
    stacked = np.vstack(quantile_chunks) if quantile_chunks else np.zeros((0, len(quantiles)))
    probabilities = (
        np.concatenate(probability_chunks) if probability_chunks else np.zeros((0,))
    )
    return {tau: stacked[:, index] for index, tau in enumerate(quantiles)}, probabilities
