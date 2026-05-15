"""MOB, vintage, and BOP-balance derivation on a perf frame."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from vintage_curves.schema import CanonicalColumns

VintageFreq = Literal["M", "Q", "A"]

# Column names added by this module
MOB = "mob"
VINTAGE = "vintage"
BOP_BALANCE = "bop_balance"


def add_mob_and_vintage(
    perf: pd.DataFrame,
    loans: pd.DataFrame,
    vintage_freq: VintageFreq = "Q",
) -> pd.DataFrame:
    """Return a copy of `perf` with `mob`, `vintage`, and `bop_balance` added.

    Conventions
    -----------
    - MOB 1 is the first reporting period of the loan (origination month-end).
      Computed as `(as_of.to_period('M') - origination.to_period('M')).n + 1`.
    - `vintage` is `origination_date.to_period(vintage_freq)` (calendar buckets,
      not fiscal). Accepts 'M', 'Q', or 'A'.
    - `bop_balance` is the prior MOB's `outstanding_balance` within each loan_id,
      with `original_balance` substituted at MOB 1. NaN if the prior MOB is
      missing from the tape (gap).
    """
    if vintage_freq not in ("M", "Q", "A", "Y"):
        raise ValueError(f"vintage_freq must be one of 'M','Q','A'/'Y', got {vintage_freq!r}")
    pd_freq = "Y" if vintage_freq == "A" else vintage_freq

    loan_id = CanonicalColumns.LOAN_ID
    orig_dt = CanonicalColumns.ORIGINATION_DATE
    as_of = CanonicalColumns.AS_OF_DATE
    orig_bal = CanonicalColumns.ORIGINAL_BALANCE
    eop_bal = CanonicalColumns.OUTSTANDING_BALANCE

    needed = loans[[loan_id, orig_dt, orig_bal]]
    out = perf.merge(needed, on=loan_id, how="left", validate="many_to_one")

    as_of_period = out[as_of].dt.to_period("M")
    orig_period = out[orig_dt].dt.to_period("M")
    out[MOB] = (as_of_period.astype("int64") - orig_period.astype("int64")) + 1

    out[VINTAGE] = out[orig_dt].dt.to_period(pd_freq)

    out = out.sort_values([loan_id, MOB], kind="mergesort").reset_index(drop=True)

    prior_eop = out.groupby(loan_id, sort=False)[eop_bal].shift(1)
    out[BOP_BALANCE] = prior_eop.where(out[MOB] > 1, out[orig_bal])

    return out.drop(columns=[orig_dt, orig_bal])


def find_mob_gaps(perf_with_mob: pd.DataFrame) -> pd.DataFrame:
    """Return rows where a loan's MOB sequence skips a month (diagnostic only)."""
    loan_id = CanonicalColumns.LOAN_ID
    diffs = perf_with_mob.groupby(loan_id, sort=False)[MOB].diff()
    gap_mask = (diffs > 1).fillna(False)
    return perf_with_mob.loc[gap_mask, [loan_id, MOB]].assign(
        gap_size=diffs[gap_mask].astype(int)
    )
