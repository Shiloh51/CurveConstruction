"""Tests for DefaultPolicy under both policy kinds + conflict modes."""

from __future__ import annotations

import pandas as pd
import pytest

from vintage_curves.default_policy import (
    DEFAULT_EVENT,
    DEFAULT_LOSS,
    IS_DEFAULTED,
    DefaultPolicy,
    DefaultPolicyError,
)
from vintage_curves.mob import MOB
from vintage_curves.schema import CanonicalColumns


def _perf_with_mob(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df[CanonicalColumns.AS_OF_DATE] = pd.to_datetime(df[CanonicalColumns.AS_OF_DATE])
    if CanonicalColumns.CHARGE_OFF_DATE in df.columns:
        df[CanonicalColumns.CHARGE_OFF_DATE] = pd.to_datetime(
            df[CanonicalColumns.CHARGE_OFF_DATE]
        )
    return df


def test_charge_off_field_marks_event_and_is_sticky() -> None:
    perf = _perf_with_mob([
        {"loan_id": "A", MOB: 1, "as_of_date": "2024-01-31", "outstanding_balance": 9800.0,
         "days_past_due": 0, "charge_off_date": None, "charge_off_amount": 0.0},
        {"loan_id": "A", MOB: 2, "as_of_date": "2024-02-29", "outstanding_balance": 9600.0,
         "days_past_due": 0, "charge_off_date": None, "charge_off_amount": 0.0},
        {"loan_id": "A", MOB: 3, "as_of_date": "2024-03-31", "outstanding_balance": 9600.0,
         "days_past_due": 120, "charge_off_date": "2024-03-31", "charge_off_amount": 9600.0},
        # Hypothetical post-CO observation row: should remain is_defaulted=True
        {"loan_id": "A", MOB: 4, "as_of_date": "2024-04-30", "outstanding_balance": 0.0,
         "days_past_due": 0, "charge_off_date": None, "charge_off_amount": 0.0},
    ])
    out = DefaultPolicy.charge_off_field().apply(perf)
    assert out[DEFAULT_EVENT].tolist() == [False, False, True, False]
    assert out[IS_DEFAULTED].tolist() == [False, False, True, True]
    assert out[DEFAULT_LOSS].tolist() == [0.0, 0.0, 9600.0, 0.0]


def test_dpd_threshold_marks_first_crossing_and_is_sticky_on_cure() -> None:
    # Loan crosses 120 at MOB 4, then "cures" back to current at MOB 5.
    # Stickiness: is_defaulted stays True at MOB 5.
    perf = _perf_with_mob([
        {"loan_id": "A", MOB: 1, "as_of_date": "2024-01-31", "outstanding_balance": 9800.0,
         "days_past_due": 0},
        {"loan_id": "A", MOB: 2, "as_of_date": "2024-02-29", "outstanding_balance": 9600.0,
         "days_past_due": 30},
        {"loan_id": "A", MOB: 3, "as_of_date": "2024-03-31", "outstanding_balance": 9600.0,
         "days_past_due": 90},
        {"loan_id": "A", MOB: 4, "as_of_date": "2024-04-30", "outstanding_balance": 9600.0,
         "days_past_due": 120},
        {"loan_id": "A", MOB: 5, "as_of_date": "2024-05-31", "outstanding_balance": 9400.0,
         "days_past_due": 0},  # cured
    ])
    out = DefaultPolicy.dpd_threshold(120).apply(perf)
    assert out[DEFAULT_EVENT].tolist() == [False, False, False, True, False]
    assert out[IS_DEFAULTED].tolist() == [False, False, False, True, True]
    # Loss = EOP outstanding balance at default month
    assert out[DEFAULT_LOSS].tolist() == [0.0, 0.0, 0.0, 9600.0, 0.0]


def test_dpd_threshold_conflict_raises_when_charge_off_populated() -> None:
    perf = _perf_with_mob([
        {"loan_id": "A", MOB: 1, "as_of_date": "2024-01-31", "outstanding_balance": 9800.0,
         "days_past_due": 0, "charge_off_date": None},
        {"loan_id": "A", MOB: 2, "as_of_date": "2024-02-29", "outstanding_balance": 9600.0,
         "days_past_due": 120, "charge_off_date": "2024-02-29"},
    ])
    with pytest.raises(DefaultPolicyError, match="charge_off_date"):
        DefaultPolicy.dpd_threshold(120).apply(perf)


def test_dpd_threshold_charge_off_wins_resolution() -> None:
    perf = _perf_with_mob([
        # Loan A: has charge_off_date — use it
        {"loan_id": "A", MOB: 1, "as_of_date": "2024-01-31", "outstanding_balance": 9800.0,
         "days_past_due": 0, "charge_off_date": None, "charge_off_amount": 0.0},
        {"loan_id": "A", MOB: 2, "as_of_date": "2024-02-29", "outstanding_balance": 9600.0,
         "days_past_due": 60, "charge_off_date": None, "charge_off_amount": 0.0},
        {"loan_id": "A", MOB: 3, "as_of_date": "2024-03-31", "outstanding_balance": 9600.0,
         "days_past_due": 120, "charge_off_date": "2024-03-31", "charge_off_amount": 9550.0},
        # Loan B: no charge_off_date — fall back to DPD threshold
        {"loan_id": "B", MOB: 1, "as_of_date": "2024-01-31", "outstanding_balance": 4900.0,
         "days_past_due": 120, "charge_off_date": None, "charge_off_amount": 0.0},
    ])
    out = DefaultPolicy.dpd_threshold(120, on_conflict="charge_off_wins").apply(perf)
    rows = out.set_index([CanonicalColumns.LOAN_ID, MOB])
    assert rows.loc[("A", 3), DEFAULT_EVENT]
    assert rows.loc[("A", 3), DEFAULT_LOSS] == 9550.0  # from charge_off_amount
    assert rows.loc[("B", 1), DEFAULT_EVENT]
    assert rows.loc[("B", 1), DEFAULT_LOSS] == 4900.0  # EOP balance


def test_dpd_threshold_dpd_wins_ignores_charge_off_date() -> None:
    perf = _perf_with_mob([
        {"loan_id": "A", MOB: 1, "as_of_date": "2024-01-31", "outstanding_balance": 9800.0,
         "days_past_due": 0, "charge_off_date": None, "charge_off_amount": 0.0},
        {"loan_id": "A", MOB: 2, "as_of_date": "2024-02-29", "outstanding_balance": 9600.0,
         "days_past_due": 120, "charge_off_date": None, "charge_off_amount": 0.0},
        # Spurious CO field set at MOB 3 — should be ignored under dpd_wins
        {"loan_id": "A", MOB: 3, "as_of_date": "2024-03-31", "outstanding_balance": 9600.0,
         "days_past_due": 150, "charge_off_date": "2024-03-31", "charge_off_amount": 9999.0},
    ])
    out = DefaultPolicy.dpd_threshold(120, on_conflict="dpd_wins").apply(perf)
    # Default event at MOB 2 (first DPD>=120), not MOB 3
    assert out[DEFAULT_EVENT].tolist() == [False, True, False]
    assert out.loc[out[DEFAULT_EVENT], DEFAULT_LOSS].iloc[0] == 9600.0


def test_dpd_threshold_value_must_be_positive() -> None:
    with pytest.raises(ValueError):
        DefaultPolicy.dpd_threshold(0)
