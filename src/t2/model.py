from __future__ import annotations

import torch
from torch import nn


class ResolutionMLP(nn.Module):
    """Point regressor for log1p(resolution_hours)."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = (64, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(last, width), nn.ReLU()])
            last = width
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def regression_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.smooth_l1_loss(prediction, target)
