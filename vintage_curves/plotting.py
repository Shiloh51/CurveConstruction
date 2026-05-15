"""Matplotlib helpers — one line per vintage, optional bold aggregate overlay."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_curves(
    per_vintage: pd.DataFrame,
    title: str | None = None,
    ylabel: str | None = None,
    aggregate: pd.Series | None = None,
    aggregate_label: str = "aggregate",
):
    """Plot per-vintage curves (rows = vintages, cols = MOB) with optional aggregate.

    Returns the matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for vintage, row in per_vintage.iterrows():
        ax.plot(row.index.astype(int), row.values, label=str(vintage), alpha=0.85)
    if aggregate is not None:
        ax.plot(
            aggregate.index.astype(int),
            aggregate.values,
            color="black",
            linewidth=2.5,
            label=aggregate_label,
        )
    if title:
        ax.set_title(title)
    ax.set_xlabel("MOB (months since origination)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
