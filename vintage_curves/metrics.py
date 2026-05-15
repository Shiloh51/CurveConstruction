"""Per-vintage performance curves.

Each public method returns a wide DataFrame indexed by `vintage`, columned by
`mob`. Cells beyond a vintage's max observable MOB are NaN.

Metric definitions follow §4 of the plan:
- Performing-at-BOP = loans that were not yet defaulted at the start of the
  period. A loan defaulting this period is performing-at-BOP but not
  strictly-performing.
- Strictly-performing = not defaulted as of EOP (excludes the default-event
  month and all later months).
- SMM denominator excludes both scheduled principal and the BOP balance of
  loans defaulting this period (PSA convention).
- Prepay numerator counts only strictly-performing loan-months so charge-off
  write-downs cannot leak into prepays.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from vintage_curves.default_policy import (
    DEFAULT_EVENT,
    DEFAULT_LOSS,
    IS_DEFAULTED,
    DefaultPolicy,
)
from vintage_curves.mob import BOP_BALANCE, MOB, VINTAGE, VintageFreq, add_mob_and_vintage
from vintage_curves.schema import CanonicalColumns
from vintage_curves.segment import Segment


@dataclass
class VintageMetrics:
    """Per-vintage curves, computed once and cached.

    Construct via `VintageMetrics.from_segment(seg, default_policy, vintage_freq=...)`.
    """

    perf_flagged: pd.DataFrame   # one row per loan-month; has mob, vintage, default flags, bop_balance
    loans: pd.DataFrame          # one row per loan_id; carries original_balance, apr, etc.

    @classmethod
    def from_segment(
        cls,
        segment: Segment,
        default_policy: DefaultPolicy,
        vintage_freq: VintageFreq = "Q",
    ) -> "VintageMetrics":
        perf = add_mob_and_vintage(segment.perf, segment.loans, vintage_freq=vintage_freq)
        flagged = default_policy.apply(perf)
        return cls(perf_flagged=flagged, loans=segment.loans)

    # ----------------------------------------------------------- per-vintage scalars

    @cached_property
    def vintage_original_balance(self) -> pd.Series:
        """OB_v: sum of original_balance per vintage (over segment loans)."""
        loan_id = CanonicalColumns.LOAN_ID
        orig_bal = CanonicalColumns.ORIGINAL_BALANCE
        # Pull vintage from the flagged perf (already computed per loan-month);
        # take the first occurrence per loan to attach vintage to each loan.
        vintage_per_loan = (
            self.perf_flagged[[loan_id, VINTAGE]].drop_duplicates(subset=[loan_id])
        )
        merged = self.loans.merge(vintage_per_loan, on=loan_id, how="inner")
        return merged.groupby(VINTAGE, observed=True)[orig_bal].sum().sort_index()

    # ------------------------------------------------------------ per-(v, mob) aggs

    @cached_property
    def aggregates(self) -> pd.DataFrame:
        """Raw per-(vintage, mob) sums used to build every curve.

        Columns:
        - bop_bal_perf:   Σ bop_balance over loans performing at BOP
        - scheduled:      Σ scheduled_principal over strictly-performing rows
        - unscheduled:    Σ max(principal_payment - scheduled_principal, 0) over strictly-performing rows
        - default_bal:    Σ bop_balance over rows where default_event = True
        - loss_amount:    Σ default_loss_amount
        - eop_bal_all:    Σ outstanding_balance over all rows (for pool factor)
        - apr_x_bop_perf: Σ apr * bop_balance over loans performing at BOP (for WA coupon)
        - n_loans:        count of loan-months (diagnostic)
        """
        loan_id = CanonicalColumns.LOAN_ID
        eop = CanonicalColumns.OUTSTANDING_BALANCE
        sched = CanonicalColumns.SCHEDULED_PRINCIPAL
        prin_pay = CanonicalColumns.PRINCIPAL_PAYMENT
        apr_col = CanonicalColumns.APR

        f = self.perf_flagged
        if apr_col not in f.columns:
            f = f.merge(self.loans[[loan_id, apr_col]], on=loan_id, how="left", validate="many_to_one")

        strictly_perf = ~f[IS_DEFAULTED]
        bop_perf = strictly_perf | f[DEFAULT_EVENT]  # performing at BOP
        de = f[DEFAULT_EVENT]

        sched_clean = f[sched].fillna(0.0)
        prin_clean = f[prin_pay].fillna(0.0)
        bop_clean = f[BOP_BALANCE].fillna(0.0)

        work = pd.DataFrame({
            VINTAGE: f[VINTAGE],
            MOB: f[MOB],
            "bop_bal_perf":   np.where(bop_perf, bop_clean, 0.0),
            "scheduled":      np.where(strictly_perf, sched_clean, 0.0),
            "unscheduled":    np.where(strictly_perf, np.clip(prin_clean - sched_clean, 0.0, None), 0.0),
            "default_bal":    np.where(de, bop_clean, 0.0),
            "loss_amount":    f[DEFAULT_LOSS].fillna(0.0).to_numpy(),
            "eop_bal_all":    f[eop].fillna(0.0).to_numpy(),
            "apr_x_bop_perf": np.where(bop_perf, f[apr_col].astype(float).fillna(0.0) * bop_clean, 0.0),
            "n_loans":        1,
        })
        agg = work.groupby([VINTAGE, MOB], observed=True, sort=True).sum()
        return agg

    # ------------------------------------------------------------------ curves

    def cum_gross_loss(self) -> pd.DataFrame:
        """Cumulative gross loss % of original balance, per vintage × MOB."""
        loss_wide = self._pivot("loss_amount")
        cum = loss_wide.cumsum(axis=1)
        return cum.div(self.vintage_original_balance, axis=0)

    def cum_prepay(self) -> pd.DataFrame:
        """Cumulative voluntary prepay % of original balance, per vintage × MOB."""
        unsched_wide = self._pivot("unscheduled")
        cum = unsched_wide.cumsum(axis=1)
        return cum.div(self.vintage_original_balance, axis=0)

    def pool_factor(self) -> pd.DataFrame:
        """EOP balance / original balance, per vintage × MOB."""
        eop_wide = self._pivot("eop_bal_all")
        return eop_wide.div(self.vintage_original_balance, axis=0)

    def smm(self) -> pd.DataFrame:
        """Single Monthly Mortality, per vintage × MOB.

        Numerator: unscheduled principal from strictly-performing loan-months.
        Denominator: BOP balance of performing loans − scheduled principal of
        strictly-performing loans − BOP balance of loans defaulting this period.
        NaN where denominator ≤ 0.
        """
        unsched = self._pivot("unscheduled")
        bop = self._pivot("bop_bal_perf")
        sched = self._pivot("scheduled")
        dbal = self._pivot("default_bal")
        denom = bop - sched - dbal
        out = unsched.where(denom > 0).div(denom.where(denom > 0))
        return out

    def cpr(self) -> pd.DataFrame:
        """Annualized CPR = 1 - (1 - SMM)^12."""
        smm = self.smm()
        return 1 - (1 - smm) ** 12

    def mdr(self) -> pd.DataFrame:
        """Monthly default rate: default_bal / BOP performing balance.

        NaN where denominator ≤ 0.
        """
        dbal = self._pivot("default_bal")
        bop = self._pivot("bop_bal_perf")
        return dbal.where(bop > 0).div(bop.where(bop > 0))

    def cdr(self) -> pd.DataFrame:
        """Annualized CDR = 1 - (1 - MDR)^12."""
        mdr = self.mdr()
        return 1 - (1 - mdr) ** 12

    def wa_coupon(self) -> pd.DataFrame:
        """Balance-weighted APR over performing-at-BOP loans, per vintage × MOB."""
        apr_bal = self._pivot("apr_x_bop_perf")
        bop = self._pivot("bop_bal_perf")
        return apr_bal.where(bop > 0).div(bop.where(bop > 0))

    # ------------------------------------------------------------------ helpers

    def _pivot(self, column: str) -> pd.DataFrame:
        """Return a wide vintage × MOB frame for an aggregate column."""
        return self.aggregates[column].unstack(MOB)

    def available_metrics(self) -> list[str]:
        return [
            "cum_gross_loss", "cum_prepay", "pool_factor",
            "smm", "cpr", "mdr", "cdr", "wa_coupon",
        ]

    def curve(self, name: str) -> pd.DataFrame:
        """Dispatch by metric name."""
        if name not in self.available_metrics():
            raise ValueError(f"Unknown metric {name!r}; available: {self.available_metrics()}")
        return getattr(self, name)()
