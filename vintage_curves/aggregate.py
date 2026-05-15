"""Aggregate curves across vintages.

Each call returns a DataFrame indexed by MOB with one column per requested
metric plus `n_vintages` (count of vintages observable at that MOB).

Weighting schemes (see §4 of the plan):
- "original":  weight stock metrics by OB_v; flow metrics aggregate raw $ flows.
- "beginning": weight stock metrics by BOP balance at MOB t; flow metrics
               aggregate raw $ flows (so flow metrics coincide with "original").
- "equal":     simple unweighted average across observable vintages.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd

from vintage_curves.metrics import VintageMetrics

Weighting = Literal["original", "beginning", "equal"]

STOCK_METRICS = frozenset({"cum_gross_loss", "cum_prepay", "pool_factor", "wa_coupon"})
FLOW_METRICS = frozenset({"smm", "cpr", "mdr", "cdr"})


def aggregate_curves(
    metrics: VintageMetrics,
    names: Iterable[str],
    weighting: Weighting = "original",
) -> pd.DataFrame:
    """Aggregate per-vintage curves into a single MOB-indexed series per metric."""
    names = list(names)
    if weighting not in ("original", "beginning", "equal"):
        raise ValueError(
            f"weighting must be 'original', 'beginning', or 'equal'; got {weighting!r}"
        )

    pool_factor = metrics.pool_factor()
    observable = pool_factor.notna()

    weights = _build_weights(metrics, observable, weighting)

    out: dict[str, pd.Series] = {}
    for name in names:
        if name in STOCK_METRICS:
            out[name] = _aggregate_stock(getattr(metrics, name)(), weights, weighting)
        elif name in FLOW_METRICS:
            out[name] = _aggregate_flow(metrics, name, weighting)
        else:
            raise ValueError(
                f"Unknown metric {name!r}. "
                f"Stock: {sorted(STOCK_METRICS)}; flow: {sorted(FLOW_METRICS)}."
            )

    n_vintages = observable.sum(axis=0).astype("int64")
    return pd.DataFrame(out).assign(n_vintages=n_vintages)


# --------------------------------------------------------------------- helpers


def _build_weights(
    metrics: VintageMetrics,
    observable: pd.DataFrame,
    weighting: Weighting,
) -> pd.DataFrame:
    """Wide vintage × MOB weight matrix, NaN where unobservable."""
    if weighting == "original":
        ob = metrics.vintage_original_balance.reindex(observable.index)
        w = pd.DataFrame(
            np.where(observable.values, ob.to_numpy()[:, None], np.nan),
            index=observable.index,
            columns=observable.columns,
        )
        return w
    if weighting == "beginning":
        bop = metrics._pivot("bop_bal_perf").reindex_like(observable)
        return bop.where(observable)
    # equal
    return observable.astype(float).where(observable)


def _aggregate_stock(
    per_vintage: pd.DataFrame,
    weights: pd.DataFrame,
    weighting: Weighting,
) -> pd.Series:
    """Weighted mean across vintages, skipping NaN cells."""
    if weighting == "equal":
        return per_vintage.mean(axis=0, skipna=True)
    valid = per_vintage.notna() & weights.notna()
    w_eff = weights.where(valid, 0.0)
    num = (per_vintage.where(valid, 0.0) * w_eff).sum(axis=0, min_count=1)
    denom = w_eff.sum(axis=0, min_count=1)
    return num.where(denom > 0) / denom.where(denom > 0)


def _aggregate_flow(
    metrics: VintageMetrics,
    name: str,
    weighting: Weighting,
) -> pd.Series:
    """Flow/rate metrics: $num / $denom across vintages, optionally annualized."""
    if weighting == "equal":
        per_v = getattr(metrics, name)()
        return per_v.mean(axis=0, skipna=True)

    if name in ("smm", "cpr"):
        num = metrics._pivot("unscheduled")
        denom = (
            metrics._pivot("bop_bal_perf")
            - metrics._pivot("scheduled")
            - metrics._pivot("default_bal")
        )
        annualize = name == "cpr"
    else:  # mdr / cdr
        num = metrics._pivot("default_bal")
        denom = metrics._pivot("bop_bal_perf")
        annualize = name == "cdr"

    # Mask non-positive denominators per vintage before summing
    valid = denom > 0
    num_agg = num.where(valid, 0.0).sum(axis=0, min_count=1)
    denom_agg = denom.where(valid, 0.0).sum(axis=0, min_count=1)
    rate = num_agg.where(denom_agg > 0) / denom_agg.where(denom_agg > 0)
    if annualize:
        rate = 1 - (1 - rate) ** 12
    return rate
