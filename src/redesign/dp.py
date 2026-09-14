"""Formal example-level DP-SGD via Opacus, with the matching accountant.

The historical pipeline added Gaussian noise to aggregated client updates with
an ad-hoc scale and no accountant. That mechanism is not used here. Each city
client runs example-level DP-SGD (Poisson sampling, per-example gradient
clipping, calibrated Gaussian noise) with an Opacus ``PrivacyEngine`` whose RDP
accountant composes across every local step of every federated round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from torch import nn
from torch.utils.data import DataLoader

ACCOUNTANT = "rdp"


@dataclass(frozen=True)
class DPSettings:
    """Target privacy for one client's local DP-SGD."""

    target_epsilon: float
    delta: float
    max_grad_norm: float
    accountant: str = ACCOUNTANT

    @property
    def enabled(self) -> bool:
        return math.isfinite(self.target_epsilon)


@dataclass
class DPState:
    """Live DP machinery plus everything needed to report the guarantee."""

    engine: PrivacyEngine
    model: nn.Module
    optimizer: torch.optim.Optimizer
    data_loader: DataLoader
    noise_multiplier: float
    sample_rate: float
    max_grad_norm: float
    delta: float
    target_epsilon: float
    planned_steps: int
    accountant: str = ACCOUNTANT
    steps_taken: int = field(default=0)

    def epsilon(self) -> float:
        return float(self.engine.get_epsilon(self.delta))

    def accounted_mechanism(self) -> list[dict[str, float]]:
        """The (noise, sample rate, steps) triples the accountant actually composed."""

        return [
            {
                "noise_multiplier": float(noise_multiplier),
                "sample_rate": float(sample_rate),
                "steps": int(steps),
            }
            for noise_multiplier, sample_rate, steps in self.engine.accountant.history
        ]

    def report(self) -> dict[str, object]:
        history = self.accounted_mechanism()
        return {
            "mechanism": "example-level DP-SGD (Opacus PrivacyEngine)",
            "accountant": self.accountant,
            "target_epsilon": self.target_epsilon,
            "epsilon_spent": self.epsilon(),
            "delta": self.delta,
            "sample_rate": history[-1]["sample_rate"] if history else self.sample_rate,
            "expected_batch_size": (history[-1]["sample_rate"] if history else self.sample_rate)
            * self.dataset_size,
            "dataset_size": self.dataset_size,
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": history[-1]["noise_multiplier"] if history else self.noise_multiplier,
            "accounted_steps": sum(entry["steps"] for entry in history),
            "accountant_history": history,
            "planned_steps": self.planned_steps,
            "steps_taken": self.steps_taken,
            "sampling": "Poisson subsampling via Opacus DPDataLoader",
            "unit_of_privacy": "one service request row within its own city dataset",
        }

    @property
    def dataset_size(self) -> int:
        return int(len(self.data_loader.dataset))


def plan_noise_multiplier(
    *,
    target_epsilon: float,
    delta: float,
    sample_rate: float,
    steps: int,
    accountant: str = ACCOUNTANT,
) -> float:
    """Calibrate the noise multiplier for the *whole* federated run."""

    if not math.isfinite(target_epsilon):
        return 0.0
    return float(
        get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant=accountant,
        )
    )


def make_private(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    settings: DPSettings,
    planned_steps: int,
) -> DPState:
    """Attach a persistent Opacus ``PrivacyEngine`` to one client."""

    dataset_size = len(data_loader.dataset)
    sample_rate = float(data_loader.batch_size) / float(dataset_size)
    noise_multiplier = plan_noise_multiplier(
        target_epsilon=settings.target_epsilon,
        delta=settings.delta,
        sample_rate=sample_rate,
        steps=planned_steps,
        accountant=settings.accountant,
    )

    engine = PrivacyEngine(accountant=settings.accountant)
    private_model, private_optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=settings.max_grad_norm,
        poisson_sampling=True,
    )
    return DPState(
        engine=engine,
        model=private_model,
        optimizer=private_optimizer,
        data_loader=private_loader,
        noise_multiplier=noise_multiplier,
        sample_rate=private_optimizer.expected_batch_size / dataset_size
        if getattr(private_optimizer, "expected_batch_size", None)
        else sample_rate,
        max_grad_norm=settings.max_grad_norm,
        delta=settings.delta,
        target_epsilon=settings.target_epsilon,
        planned_steps=planned_steps,
        accountant=settings.accountant,
    )


def recompute_epsilon(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
    accountant: str = ACCOUNTANT,
) -> float:
    """Independently recompute epsilon from the reported mechanism parameters."""

    from opacus.accountants import create_accountant

    accountant_object = create_accountant(mechanism=accountant)
    accountant_object.history = [(noise_multiplier, sample_rate, steps)]
    return float(accountant_object.get_epsilon(delta=delta))
