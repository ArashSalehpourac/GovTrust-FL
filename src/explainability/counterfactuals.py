"""Counterfactual explanation hooks."""

from __future__ import annotations


def dice_counterfactuals(*args, **kwargs):
    """Create DiCE counterfactuals when dice-ml is installed."""

    try:
        import dice_ml
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("Install dice-ml to generate counterfactual explanations.") from exc
    return dice_ml.Dice(*args, **kwargs)
