"""Per-vintage summary scalars: WA Coupon, latest pool factor, WAL (realized + scheduled)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vintage_curves.metrics import VintageMetrics
from vintage_curves.mob import MOB, VINTAGE
from vintage_curves.schema import CanonicalColumns


def vintage_summary(metrics: VintageMetrics) -> pd.DataFrame:
    """Return one row per vintage with summary scalars.

    Columns:
    - original_count:    number of loans in the vintage
    - original_balance:  sum of original_balance (OB_v)
    - wa_coupon:         OB-weighted average APR at origination
    - pool_factor_latest: pool factor at the last observable MOB for the vintage
    - wal_realized:      weighted average life over realized principal removals (months)
    - wal_scheduled:     weighted average life of the implied level-pay schedule (months)
    """
    loan_id = CanonicalColumns.LOAN_ID
    orig_bal = CanonicalColumns.ORIGINAL_BALANCE
    apr_col = CanonicalColumns.APR
    term = CanonicalColumns.ORIGINAL_TERM

    # Attach vintage to each loan via the first occurrence in flagged perf
    vintage_per_loan = (
        metrics.perf_flagged[[loan_id, VINTAGE]].drop_duplicates(subset=[loan_id])
    )
    loans_v = metrics.loans.merge(vintage_per_loan, on=loan_id, how="inner")

    grouped = loans_v.groupby(VINTAGE, observed=True, sort=True)
    original_count = grouped.size().rename("original_count")
    original_balance = grouped[orig_bal].sum().rename("original_balance")

    wac = grouped.apply(
        lambda g: float(np.average(g[apr_col].astype(float), weights=g[orig_bal].astype(float)))
    ).rename("wa_coupon")

    pf = metrics.pool_factor()
    # Last observable MOB per vintage (largest column index with non-NaN)
    pool_factor_latest = pf.apply(
        lambda row: row.dropna().iloc[-1] if row.notna().any() else np.nan, axis=1
    ).rename("pool_factor_latest")

    wal_real = _wal_realized(metrics).rename("wal_realized")
    wal_sched = _wal_scheduled(loans_v).rename("wal_scheduled")

    out = pd.concat(
        [original_count, original_balance, wac, pool_factor_latest, wal_real, wal_sched],
        axis=1,
    )
    return out


# ----------------------------------------------------------------- WAL helpers


def _wal_realized(metrics: VintageMetrics) -> pd.Series:
    """Σ mob · principal_removed / Σ principal_removed, per vintage.

    `principal_removed` per loan-month = max(BOP − EOP, 0), which captures
    both scheduled paydown and default write-downs and includes voluntary
    prepays naturally.
    """
    from vintage_curves.mob import BOP_BALANCE  # local to avoid import cycle clutter

    f = metrics.perf_flagged
    eop = CanonicalColumns.OUTSTANDING_BALANCE
    removed = (f[BOP_BALANCE].fillna(0.0) - f[eop].fillna(0.0)).clip(lower=0.0)
    weighted = removed * f[MOB].astype(float)
    grouped = pd.DataFrame({
        VINTAGE: f[VINTAGE],
        "removed": removed,
        "weighted": weighted,
    }).groupby(VINTAGE, observed=True, sort=True).sum()
    out = grouped["weighted"].where(grouped["removed"] > 0) / grouped["removed"].where(grouped["removed"] > 0)
    return out


def _wal_scheduled(loans_v: pd.DataFrame) -> pd.Series:
    """OB-weighted average of per-loan scheduled-WAL, per vintage.

    Per-loan scheduled WAL uses level-pay amortization with the loan's APR and
    original term: WAL = Σ t · scheduled_principal_t / original_balance.
    """
    orig_bal = CanonicalColumns.ORIGINAL_BALANCE
    apr_col = CanonicalColumns.APR
    term = CanonicalColumns.ORIGINAL_TERM

    def per_loan_wal(row: pd.Series) -> float:
        bal0 = float(row[orig_bal])
        r = float(row[apr_col]) / 12.0
        n = int(row[term])
        if n <= 0 or bal0 <= 0:
            return np.nan
        if abs(r) < 1e-12:
            # Zero-rate level pay: equal principal each period
            return (n + 1) / 2.0
        pmt = bal0 * r / (1 - (1 + r) ** -n)
        bal = bal0
        weighted_sum = 0.0
        principal_sum = 0.0
        for t in range(1, n + 1):
            interest = bal * r
            principal = pmt - interest
            if principal > bal:
                principal = bal
            weighted_sum += t * principal
            principal_sum += principal
            bal -= principal
            if bal <= 1e-9:
                break
        return weighted_sum / principal_sum if principal_sum > 0 else np.nan

    loans_v = loans_v.copy()
    loans_v["_wal"] = loans_v.apply(per_loan_wal, axis=1)

    def weighted(g: pd.DataFrame) -> float:
        w = g[orig_bal].astype(float)
        v = g["_wal"].astype(float)
        mask = v.notna() & (w > 0)
        if not mask.any():
            return np.nan
        return float(np.average(v[mask], weights=w[mask]))

    return loans_v.groupby(VINTAGE, observed=True, sort=True).apply(weighted)
