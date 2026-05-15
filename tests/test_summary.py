"""Tests for per-vintage summary scalars."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from vintage_curves import (
    DefaultPolicy,
    LoanTape,
    Segment,
    SchemaMapper,
    VintageMetrics,
    vintage_summary,
)


CANONICAL_MAPPER = SchemaMapper(mapping={c: c for c in [
    "loan_id", "origination_date", "original_balance", "apr", "original_term",
    "as_of_date", "outstanding_balance", "scheduled_principal", "principal_payment",
    "interest_payment", "days_past_due", "charge_off_date", "charge_off_amount", "fico",
]})


def _row(loan_id, as_of, eop, sched, prin_pay, dpd, co_date=None, co_amt=0.0):
    return {
        "loan_id": loan_id, "as_of_date": as_of, "outstanding_balance": eop,
        "scheduled_principal": sched, "principal_payment": prin_pay,
        "interest_payment": 0.0, "days_past_due": dpd,
        "charge_off_date": co_date, "charge_off_amount": co_amt,
    }


@pytest.fixture
def mixed_apr_metrics() -> VintageMetrics:
    # Two loans in the same vintage with different APR / OB to test WAC weighting.
    # Loan A: OB=1000, APR=0.10, observed 3 months (scheduled $100/mo, no events)
    # Loan B: OB=3000, APR=0.20, observed 3 months (scheduled $300/mo, no events)
    loans = pd.DataFrame([
        {"loan_id": "A", "origination_date": "2024-01-15",
         "original_balance": 1000.0, "apr": 0.10, "original_term": 10, "fico": 700},
        {"loan_id": "B", "origination_date": "2024-01-15",
         "original_balance": 3000.0, "apr": 0.20, "original_term": 10, "fico": 700},
    ])
    perf = pd.DataFrame([
        _row("A", "2024-01-31",  900.0, 100.0, 100.0, 0),
        _row("A", "2024-02-29",  800.0, 100.0, 100.0, 0),
        _row("A", "2024-03-31",  700.0, 100.0, 100.0, 0),
        _row("B", "2024-01-31", 2700.0, 300.0, 300.0, 0),
        _row("B", "2024-02-29", 2400.0, 300.0, 300.0, 0),
        _row("B", "2024-03-31", 2100.0, 300.0, 300.0, 0),
    ])
    raw = perf.merge(loans, on="loan_id", how="left")
    tape = LoanTape.from_dataframe(raw, CANONICAL_MAPPER)
    seg = Segment.from_tape(tape, query=None, label="all")
    return VintageMetrics.from_segment(seg, DefaultPolicy.charge_off_field(), vintage_freq="Q")


def test_summary_scalars(mixed_apr_metrics: VintageMetrics) -> None:
    s = vintage_summary(mixed_apr_metrics)
    v = pd.Period("2024Q1")
    row = s.loc[v]
    assert row["original_count"] == 2
    assert math.isclose(row["original_balance"], 4000.0, rel_tol=1e-12)
    # OB-weighted WAC: (1000*0.10 + 3000*0.20) / 4000 = 700/4000 = 0.175
    assert math.isclose(row["wa_coupon"], 0.175, rel_tol=1e-12)
    # Pool factor at MOB 3: (700 + 2100) / 4000 = 0.7
    assert math.isclose(row["pool_factor_latest"], 0.7, rel_tol=1e-12)
    # WAL realized: 100*1+100*2+100*3 = 600 for A; 300*1+300*2+300*3 = 1800 for B.
    # Combined weighted: (1*100+2*100+3*100) + (1*300+2*300+3*300) = 600 + 1800 = 2400
    # Principal removed: 300 + 900 = 1200. WAL = 2400/1200 = 2.0
    assert math.isclose(row["wal_realized"], 2.0, rel_tol=1e-12)
    # WAL scheduled exists and is positive
    assert row["wal_scheduled"] > 0
    assert row["wal_scheduled"] <= 10  # bounded by original term


def test_wal_scheduled_for_zero_rate_loan() -> None:
    # With APR=0 and term=12, level pay is equal principal each month → WAL = (n+1)/2 = 6.5
    loans = pd.DataFrame([{
        "loan_id": "Z", "origination_date": "2024-01-15",
        "original_balance": 1200.0, "apr": 0.0, "original_term": 12, "fico": 700,
    }])
    perf = pd.DataFrame([
        _row("Z", "2024-01-31", 1100.0, 100.0, 100.0, 0),
    ])
    raw = perf.merge(loans, on="loan_id", how="left")
    tape = LoanTape.from_dataframe(raw, CANONICAL_MAPPER)
    seg = Segment.from_tape(tape, query=None, label="all")
    m = VintageMetrics.from_segment(seg, DefaultPolicy.charge_off_field(), vintage_freq="Q")
    s = vintage_summary(m)
    assert math.isclose(s.iloc[0]["wal_scheduled"], 6.5, rel_tol=1e-12)
