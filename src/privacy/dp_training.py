"""Compatibility wrapper for Opacus DP training hooks."""

from src.dp_training import DPTrainingConfig, epsilon_or_none, make_private_if_enabled

__all__ = ["DPTrainingConfig", "epsilon_or_none", "make_private_if_enabled"]
