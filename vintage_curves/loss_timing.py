"""Weighted-average loss-timing curves and per-vintage CGL projection.

The function `loss_timing_projection` builds an empirical weighted-average
cumulative loss curve from the observed vintages, normalizes it into a
loss-timing curve (fraction of lifetime losses incurred by each MOB), uses
that to project each vintage's terminal cum loss from its observed cum loss
at current MOB, and back-fills the projected CGL path from current MOB out
to the terminal MOB.

Methodology
-----------
1. For each MOB t, weighted-avg monthly default rate:
       WA_monthly(t) = Σ_v∈obs(t) CO_$_v(t) / Σ_v∈obs(t) OB_v
   where obs(t) is the set of vintages with data at MOB t (optionally
   restricted to the earliest `n_vintages`).
2. Cumulate: WA_cum(t) = Σ_{s=1..t} WA_monthly(s).
3. Terminal MOB = max MOB with data in (the filtered) WA curve.
   Timing curve: timing(t) = WA_cum(t) / WA_cum(T_terminal).
4. For each vintage v with observed cum loss CGL_v(m_v) at current MOB m_v:
       projected_terminal_v = CGL_v(m_v) / timing(m_v)
       projected_CGL_v(t)   = timing(t) * projected_terminal_v   for t > m_v
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vintage_curves.metrics import VintageMetrics


@dataclass
class LossTimingResult:
    wa_monthly_loss: pd.Series
    wa_cum_loss: pd.Series
    timing_curve: pd.Series
    terminal_mob: int
    terminal_cum_loss: float
    projected_terminal_per_vintage: pd.Series
    observed_cgl: pd.DataFrame
    projected_cgl: pd.DataFrame
    vintages_used_for_wa: list = field(default_factory=list)
    png_path: Path | None = None


def loss_timing_projection(
    metrics: VintageMetrics,
    n_vintages: int | None = None,
    output_png: str | Path | None = None,
    title: str = "Loss timing and projected CGL",
) -> LossTimingResult:
    """Build a WA loss-timing curve and project per-vintage CGL to terminal MOB.

    Parameters
    ----------
    metrics
        VintageMetrics for the segment (e.g., `segment_result.metrics`).
    n_vintages
        If set, build the WA curve from only the earliest N vintages by
        origination date. Per-vintage projection still runs for every vintage.
    output_png
        If provided, write the chart to this path. Parent dirs are created.
    title
        Plot title.
    """
    monthly_loss_dollars = metrics._pivot("loss_amount")
    ob_per_vintage = metrics.vintage_original_balance
    observed_cgl = metrics.cum_gross_loss()

    all_vintages = sorted(observed_cgl.index)
    if n_vintages is not None and n_vintages < len(all_vintages):
        used = all_vintages[:n_vintages]
    else:
        used = list(all_vintages)

    monthly_for_wa = monthly_loss_dollars.loc[used]
    ob_for_wa = ob_per_vintage.loc[used]

    observed_mask = monthly_for_wa.notna()
    loss_num = monthly_for_wa.sum(axis=0, min_count=1)
    ob_broadcast = pd.DataFrame(
        np.where(observed_mask.values, ob_for_wa.to_numpy()[:, None], 0.0),
        index=observed_mask.index,
        columns=observed_mask.columns,
    )
    denom = ob_broadcast.sum(axis=0)

    wa_monthly_loss = (loss_num / denom.where(denom > 0)).sort_index().dropna()
    if wa_monthly_loss.empty:
        raise ValueError(
            "No observable (vintage, MOB) cells in the WA window — "
            "cannot build a loss-timing curve."
        )
    wa_cum_loss = wa_monthly_loss.cumsum()

    terminal_mob = int(wa_cum_loss.index.max())
    terminal_cum_loss = float(wa_cum_loss.iloc[-1])
    if terminal_cum_loss <= 0:
        raise ValueError(
            "Terminal WA cum loss is zero or negative — no losses observed "
            "in the data, so timing curve cannot be normalized."
        )

    timing_curve = wa_cum_loss / terminal_cum_loss

    projected_terminal: dict = {}
    projected_cgl = pd.DataFrame(
        index=observed_cgl.index,
        columns=sorted(timing_curve.index),
        dtype=float,
    )
    for v in observed_cgl.index:
        vintage_curve = observed_cgl.loc[v].dropna()
        if vintage_curve.empty:
            projected_terminal[v] = np.nan
            continue
        current_mob = int(vintage_curve.index.max())
        current_cgl = float(vintage_curve.iloc[-1])

        if current_mob not in timing_curve.index:
            projected_terminal[v] = np.nan
            continue
        timing_at_current = float(timing_curve.loc[current_mob])
        if timing_at_current <= 0:
            projected_terminal[v] = np.nan
            continue

        proj_terminal = current_cgl / timing_at_current
        projected_terminal[v] = proj_terminal

        future_mobs = [m for m in timing_curve.index if m > current_mob]
        for m in future_mobs:
            projected_cgl.loc[v, m] = float(timing_curve.loc[m]) * proj_terminal

    projected_terminal_per_vintage = pd.Series(
        projected_terminal, name="projected_terminal_cgl"
    ).reindex(observed_cgl.index)

    png_path: Path | None = None
    if output_png is not None:
        png_path = Path(output_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        _plot(
            observed_cgl=observed_cgl,
            projected_cgl=projected_cgl,
            wa_cum_loss=wa_cum_loss,
            title=title,
            path=png_path,
        )

    return LossTimingResult(
        wa_monthly_loss=wa_monthly_loss,
        wa_cum_loss=wa_cum_loss,
        timing_curve=timing_curve,
        terminal_mob=terminal_mob,
        terminal_cum_loss=terminal_cum_loss,
        projected_terminal_per_vintage=projected_terminal_per_vintage,
        observed_cgl=observed_cgl,
        projected_cgl=projected_cgl,
        vintages_used_for_wa=used,
        png_path=png_path,
    )


def _plot(
    observed_cgl: pd.DataFrame,
    projected_cgl: pd.DataFrame,
    wa_cum_loss: pd.Series,
    title: str,
    path: Path,
) -> None:
    vintages = list(observed_cgl.index)
    cmap = plt.get_cmap("tab10" if len(vintages) <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(len(vintages))]

    fig, ax = plt.subplots(figsize=(10, 6))

    for color, v in zip(colors, vintages):
        observed = observed_cgl.loc[v].dropna()
        if observed.empty:
            continue
        ax.plot(
            observed.index,
            observed.values * 100,
            color=color,
            linewidth=1.6,
            label=str(v),
        )
        projected = projected_cgl.loc[v].dropna()
        if not projected.empty:
            x = [int(observed.index.max())] + list(projected.index)
            y = [float(observed.iloc[-1]) * 100] + [float(p) * 100 for p in projected.values]
            ax.plot(x, y, color=color, linestyle=":", linewidth=1.6)

    ax.plot(
        wa_cum_loss.index,
        wa_cum_loss.values * 100,
        color="black",
        linewidth=2.2,
        label="WA cum loss",
    )

    ax.set_xlabel("MOB")
    ax.set_ylabel("Cumulative gross loss (%)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
