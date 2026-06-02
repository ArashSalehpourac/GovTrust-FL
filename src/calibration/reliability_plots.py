"""Reliability-plot helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def plot_reliability_diagram(y_true, y_score, output_path: str | Path, n_bins: int = 10) -> Path:
    """Save a binary reliability diagram and return the output path."""

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.0)
    ax.plot(prob_pred, prob_true, marker="o", color="#222222", linewidth=1.3)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output
