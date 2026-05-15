"""Default-event detection per the plan's two policies (sticky)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from vintage_curves.mob import MOB
from vintage_curves.schema import CanonicalColumns

# Columns added by `apply()`
IS_DEFAULTED = "is_defaulted"      # sticky: True from default month onward
DEFAULT_EVENT = "default_event"    # True only in the month of default
DEFAULT_LOSS = "default_loss_amount"  # populated only on the default event row

PolicyKind = Literal["charge_off_field", "dpd_threshold"]
OnConflict = Literal["raise", "charge_off_wins", "dpd_wins"]


class DefaultPolicyError(ValueError):
    """Raised when policy inputs conflict or are inapplicable."""


@dataclass(frozen=True)
class DefaultPolicy:
    kind: PolicyKind
    dpd_threshold_value: int | None = None
    on_conflict: OnConflict = "raise"

    @classmethod
    def charge_off_field(cls) -> "DefaultPolicy":
        return cls(kind="charge_off_field")

    @classmethod
    def dpd_threshold(cls, n: int, on_conflict: OnConflict = "raise") -> "DefaultPolicy":
        if n <= 0:
            raise ValueError("dpd_threshold must be positive.")
        return cls(kind="dpd_threshold", dpd_threshold_value=n, on_conflict=on_conflict)

    # ------------------------------------------------------------------ apply

    def apply(self, perf_with_mob: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `perf_with_mob` with default flags + loss amount.

        Adds columns `is_defaulted` (sticky bool), `default_event` (bool;
        True only on the first default month), and `default_loss_amount`
        (float; 0 except on the default event row).
        """
        if self.kind == "charge_off_field":
            return self._apply_charge_off(perf_with_mob)
        if self.kind == "dpd_threshold":
            return self._apply_dpd(perf_with_mob)
        raise DefaultPolicyError(f"Unknown policy kind: {self.kind!r}")

    # ----------------------------------------------------------- charge-off

    def _apply_charge_off(self, perf: pd.DataFrame) -> pd.DataFrame:
        co_date = CanonicalColumns.CHARGE_OFF_DATE
        co_amt = CanonicalColumns.CHARGE_OFF_AMOUNT
        loan_id = CanonicalColumns.LOAN_ID

        if co_date not in perf.columns:
            raise DefaultPolicyError(
                "charge_off_field policy requires `charge_off_date` in the tape."
            )

        out = perf.copy()
        out = out.sort_values([loan_id, MOB], kind="mergesort").reset_index(drop=True)

        event = out[co_date].notna()
        # Stickiness: mark only the first CO row per loan (guards against any
        # malformed tape with multiple CO rows for the same loan).
        out[DEFAULT_EVENT] = _mark_first_true_per_group(event, out[loan_id])
        out[IS_DEFAULTED] = out.groupby(loan_id, sort=False)[DEFAULT_EVENT].cummax().astype(bool)

        loss = pd.Series(0.0, index=out.index)
        if co_amt in out.columns:
            loss = out[co_amt].fillna(0.0).where(out[DEFAULT_EVENT], 0.0)
        out[DEFAULT_LOSS] = loss
        return out

    # ----------------------------------------------------------- DPD policy

    def _apply_dpd(self, perf: pd.DataFrame) -> pd.DataFrame:
        loan_id = CanonicalColumns.LOAN_ID
        dpd = CanonicalColumns.DAYS_PAST_DUE
        co_date = CanonicalColumns.CHARGE_OFF_DATE
        eop = CanonicalColumns.OUTSTANDING_BALANCE
        n = self.dpd_threshold_value
        assert n is not None

        if dpd not in perf.columns:
            raise DefaultPolicyError(
                "dpd_threshold policy requires `days_past_due` in the tape."
            )

        out = perf.copy()
        out = out.sort_values([loan_id, MOB], kind="mergesort").reset_index(drop=True)

        # Conflict detection: tape also populates charge_off_date
        if co_date in out.columns and out[co_date].notna().any():
            if self.on_conflict == "raise":
                offenders = (
                    out.loc[out[co_date].notna(), loan_id].drop_duplicates().head(5).tolist()
                )
                raise DefaultPolicyError(
                    f"dpd_threshold policy chosen but tape also populates "
                    f"charge_off_date for some loans (e.g. {offenders}). "
                    f"Pass DefaultPolicy.dpd_threshold(n, on_conflict='charge_off_wins') "
                    f"or 'dpd_wins' to resolve."
                )

        dpd_event = (out[dpd].fillna(0).astype("int64") >= n)

        if co_date in out.columns and self.on_conflict == "charge_off_wins":
            # For loans where charge_off_date is populated, use that as the event;
            # for others, use DPD threshold.
            co_loans = set(out.loc[out[co_date].notna(), loan_id].unique())
            co_event = out[co_date].notna()
            use_co = out[loan_id].isin(co_loans)
            event = np.where(use_co, co_event, dpd_event)
            out["__event_raw"] = event
            event_series = pd.Series(event, index=out.index)
        else:
            # dpd_wins (or no conflict): ignore charge_off_date, use DPD purely
            event_series = dpd_event

        out[DEFAULT_EVENT] = _mark_first_true_per_group(event_series, out[loan_id])
        out[IS_DEFAULTED] = out.groupby(loan_id, sort=False)[DEFAULT_EVENT].cummax().astype(bool)

        # Loss = EOP outstanding_balance at the default month (gross), unless
        # we're in charge_off_wins mode for a CO loan — then use charge_off_amount.
        loss = out[eop].fillna(0.0).where(out[DEFAULT_EVENT], 0.0)
        co_amt = CanonicalColumns.CHARGE_OFF_AMOUNT
        if (
            co_amt in out.columns
            and co_date in out.columns
            and self.on_conflict == "charge_off_wins"
        ):
            co_loans = set(out.loc[out[co_date].notna(), loan_id].unique())
            on_co_event = out[DEFAULT_EVENT] & out[loan_id].isin(co_loans) & out[co_date].notna()
            loss = loss.where(~on_co_event, out[co_amt].fillna(0.0))
        out[DEFAULT_LOSS] = loss

        if "__event_raw" in out.columns:
            out = out.drop(columns="__event_raw")
        return out


def _mark_first_true_per_group(flag: pd.Series, group: pd.Series) -> pd.Series:
    """For each group, return a bool series that is True only at the first
    True position within that group (and False elsewhere, including after)."""
    flag = flag.astype(bool)
    cum = flag.groupby(group, sort=False).cumsum()
    return (flag & (cum == 1))


