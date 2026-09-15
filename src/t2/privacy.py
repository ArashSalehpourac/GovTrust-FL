from __future__ import annotations

from dataclasses import dataclass, field

import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from torch import nn
from torch.utils.data import DataLoader

from .config import delta_for_client

SECURE_RNG = False


@dataclass
class PrivacyState:
    mode: str
    engine: PrivacyEngine | None
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loader: DataLoader
    dataset_size: int
    batch_size: int
    sample_rate: float
    delta: float | None
    target_epsilon: float
    noise_multiplier: float
    max_grad_norm: float
    planned_steps: int
    steps_taken: int = 0
    trace: list[dict[str, object]] = field(default_factory=list)

    def realized_epsilon(self) -> float:
        if self.mode != "private" or self.engine is None or self.delta is None:
            return float("inf")
        return float(self.engine.get_epsilon(self.delta))

    def accountant_steps(self) -> int:
        if self.engine is None:
            return 0
        return int(sum(int(steps) for _, _, steps in self.engine.accountant.history))

    def snapshot(self, round_number: int) -> dict[str, object]:
        row = {
            "round": int(round_number),
            "mode": self.mode,
            "steps_taken": int(self.steps_taken),
            "accounted_steps": self.accountant_steps(),
            "epsilon": self.realized_epsilon(),
            "delta": self.delta,
            "noise_multiplier": float(self.noise_multiplier),
            "max_grad_norm": float(self.max_grad_norm),
            "dataset_size": int(self.dataset_size),
            "batch_size_requested": int(self.batch_size),
            "sample_rate": float(self.sample_rate),
        }
        self.trace.append(row)
        return row

    def report(self, epsilon_tolerance: float) -> dict[str, object]:
        realized = self.realized_epsilon()
        achieved = (
            self.mode != "private"
            or abs(realized - self.target_epsilon) <= epsilon_tolerance
        )
        if self.mode == "private" and self.accountant_steps() != self.steps_taken:
            raise RuntimeError(
                "privacy accountant step count does not match executed optimizer steps"
            )
        return {
            "mode": self.mode,
            "unit": "one source-city training service-request row",
            "scope_note": (
                "epsilon-delta DP covers each client's training partition. "
                "Validation, internal-test, and held-out evaluation rows are not "
                "part of the protected training dataset."
            ),
            "preprocessing": (
                "data-independent fixed hashing and deterministic time encoding; "
                "no record-dependent vocabulary/category/scaler fit"
            ),
            "mechanism": (
                "Opacus example-level DP-SGD"
                if self.mode == "private"
                else (
                    "Opacus per-sample clipping with zero noise; non-private matched control"
                    if self.mode == "clipped_no_noise"
                    else "standard non-private SGD"
                )
            ),
            "accountant": "rdp" if self.mode == "private" else None,
            "target_epsilon": self.target_epsilon,
            "realized_epsilon": realized,
            "target_achieved_within_tolerance": bool(achieved),
            "epsilon_tolerance": float(epsilon_tolerance),
            "delta": self.delta,
            "noise_multiplier": float(self.noise_multiplier),
            "max_grad_norm": float(self.max_grad_norm),
            "planned_steps": int(self.planned_steps),
            "steps_taken": int(self.steps_taken),
            "accounted_steps": self.accountant_steps(),
            "dataset_size": int(self.dataset_size),
            "batch_size_requested": int(self.batch_size),
            "sample_rate": float(self.sample_rate),
            "sampling": (
                "Poisson; sample_rate exactly matches Opacus DPDataLoader "
                "construction (1 / len(original_loader))"
            ),
            "secure_rng": SECURE_RNG,
            "secure_rng_note": (
                "Opacus secure_mode is disabled for the research experiment; "
                "the accountant/mechanism is auditable, but this is not a "
                "production cryptographic deployment claim."
            ),
            "trace": self.trace,
        }


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "_module", model)


def plain_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    module = _unwrap(model)
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def load_plain_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    _unwrap(model).load_state_dict(state)


def make_training_state(
    *,
    model: nn.Module,
    loader: DataLoader,
    mode: str,
    target_epsilon: float,
    learning_rate: float,
    max_grad_norm: float,
    planned_steps: int,
) -> PrivacyState:
    n = len(loader.dataset)
    if n < 2:
        raise ValueError("client requires at least two rows")
    if len(loader) < 1:
        raise ValueError("empty loader")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    requested_batch_size = int(loader.batch_size or n)
    # Opacus DPDataLoader.from_data_loader uses sample_rate = 1 / len(loader).
    # Calibrating with any other approximation can invalidate the target-epsilon claim.
    sample_rate = 1.0 / float(len(loader))

    if mode == "nonprivate":
        return PrivacyState(
            mode,
            None,
            model,
            optimizer,
            loader,
            n,
            requested_batch_size,
            sample_rate,
            None,
            float("inf"),
            0.0,
            max_grad_norm,
            planned_steps,
        )
    if mode not in {"private", "clipped_no_noise"}:
        raise ValueError(f"unknown mode: {mode}")

    delta = delta_for_client(n) if mode == "private" else None
    noise_multiplier = 0.0
    if mode == "private":
        noise_multiplier = float(
            get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=delta,
                sample_rate=sample_rate,
                steps=planned_steps,
                accountant="rdp",
            )
        )

    engine = PrivacyEngine(accountant="rdp", secure_mode=SECURE_RNG)
    private_model, private_optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=True,
    )
    actual_rate = 1.0 / float(len(private_loader))
    if abs(actual_rate - sample_rate) > 1e-12:
        raise RuntimeError("Opacus sampling rate differs from calibrated sampling rate")
    return PrivacyState(
        mode=mode,
        engine=engine,
        model=private_model,
        optimizer=private_optimizer,
        loader=private_loader,
        dataset_size=n,
        batch_size=requested_batch_size,
        sample_rate=actual_rate,
        delta=delta,
        target_epsilon=target_epsilon if mode == "private" else float("inf"),
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        planned_steps=planned_steps,
    )
