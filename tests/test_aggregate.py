"""Tests for aggregate_curves under all three weighting schemes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vintage_curves import (
    DefaultPolicy,
    LoanTape,
    Segment,
    SchemaMapper,
    VintageMetrics,
    aggregate_curves,
)
from vintage_curves.schema import CanonicalColumns


# Reuse the 3-loan analytical fixture's idiom but with two vintages so
# aggregation across vintages is meaningful.

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
def two_vintage_metrics() -> VintageMetrics:
    # Vintage 2024Q1: 1 loan, OB=1000, pays scheduled for 3 months
    # Vintage 2024Q2: 1 loan, OB=2000, defaults at MOB 2 (CO=1900)
    loans = pd.DataFrame([
        {"loan_id": "Q1", "origination_date": "2024-01-15",
         "original_balance": 1000.0, "apr": 0.12, "original_term": 10, "fico": 700},
        {"loan_id": "Q2", "origination_date": "2024-04-15",
         "original_balance": 2000.0, "apr": 0.12, "original_term": 10, "fico": 700},
    ])
    perf = pd.DataFrame([
        _row("Q1", "2024-01-31", 900.0, 100.0, 100.0, 0),
        _row("Q1", "2024-02-29", 800.0, 100.0, 100.0, 0),
        _row("Q1", "2024-03-31", 700.0, 100.0, 100.0, 0),
        _row("Q2", "2024-04-30", 1900.0, 100.0, 100.0, 0),
        _row("Q2", "2024-05-31",    0.0, 100.0,   0.0, 120,
             co_date="2024-05-31", co_amt=1900.0),
    ])
    raw = perf.merge(loans, on="loan_id", how="left")
    tape = LoanTape.from_dataframe(raw, CANONICAL_MAPPER)
    seg = Segment.from_tape(tape, query=None, label="all")
    return VintageMetrics.from_segment(seg, DefaultPolicy.charge_off_field(), vintage_freq="Q")


def test_n_vintages_reflects_support(two_vintage_metrics: VintageMetrics) -> None:
    agg = aggregate_curves(two_vintage_metrics, ["pool_factor"], weighting="original")
    # Q1 observable for MOBs 1-3; Q2 observable for MOBs 1-2
    assert agg.loc[1, "n_vintages"] == 2
    assert agg.loc[2, "n_vintages"] == 2
    assert agg.loc[3, "n_vintages"] == 1


def test_original_weighted_pool_factor(two_vintage_metrics: VintageMetrics) -> None:
    agg = aggregate_curves(two_vintage_metrics, ["pool_factor"], weighting="original")
    # MOB 1: Q1 EOP 900 / OB 1000 = 0.9, Q2 EOP 1900 / OB 2000 = 0.95.
    # OB-weighted: (1000*0.9 + 2000*0.95) / 3000 = (900 + 1900)/3000 = 2800/3000.
    assert math.isclose(agg.loc[1, "pool_factor"], 2800 / 3000, rel_tol=1e-12)
    # MOB 2: Q1 PF=800/1000=0.8, Q2 PF=0/2000=0 (charged off).
    # OB-weighted: (1000*0.8 + 2000*0)/3000 = 800/3000.
    assert math.isclose(agg.loc[2, "pool_factor"], 800 / 3000, rel_tol=1e-12)
    # MOB 3: only Q1 observable; agg = 0.7
    assert math.isclose(agg.loc[3, "pool_factor"], 0.7, rel_tol=1e-12)


def test_equal_weighted_differs_from_original(two_vintage_metrics: VintageMetrics) -> None:
    agg_eq = aggregate_curves(two_vintage_metrics, ["pool_factor"], weighting="equal")
    agg_ob = aggregate_curves(two_vintage_metrics, ["pool_factor"], weighting="original")
    # MOB 1 equal: (0.9 + 0.95) / 2 = 0.925
    assert math.isclose(agg_eq.loc[1, "pool_factor"], 0.925, rel_tol=1e-12)
    assert agg_eq.loc[1, "pool_factor"] != agg_ob.loc[1, "pool_factor"]


def test_cum_gross_loss_aggregation(two_vintage_metrics: VintageMetrics) -> None:
    agg = aggregate_curves(two_vintage_metrics, ["cum_gross_loss"], weighting="original")
    # MOB 1: no loss anywhere → 0
    assert agg.loc[1, "cum_gross_loss"] == 0.0
    # MOB 2: Q2 loss=1900/2000=0.95, Q1 loss=0. OB-weighted = (1000*0 + 2000*0.95)/3000 = 1900/3000
    assert math.isclose(agg.loc[2, "cum_gross_loss"], 1900 / 3000, rel_tol=1e-12)
    # MOB 3: only Q1 observable, still 0 loss
    assert agg.loc[3, "cum_gross_loss"] == 0.0


def test_flow_metric_sums_dollars(two_vintage_metrics: VintageMetrics) -> None:
    # MOB 2 CDR: default_bal Q2 = 1900 (Q2 BOP at MOB 2), Q1 default_bal = 0.
    # BOP_perf Q2 = 1900, Q1 = 900. Aggregate MDR = 1900 / 2800.
    agg = aggregate_curves(two_vintage_metrics, ["mdr", "cdr"], weighting="original")
    mdr2 = 1900 / 2800
    assert math.isclose(agg.loc[2, "mdr"], mdr2, rel_tol=1e-12)
    assert math.isclose(agg.loc[2, "cdr"], 1 - (1 - mdr2) ** 12, rel_tol=1e-12)


def test_unknown_metric_and_weighting_raise(two_vintage_metrics: VintageMetrics) -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        aggregate_curves(two_vintage_metrics, ["nope"], weighting="original")
    with pytest.raises(ValueError, match="weighting"):
        aggregate_curves(two_vintage_metrics, ["pool_factor"], weighting="bogus")  # type: ignore[arg-type]


# ----------------------------------------------------------------- real-tape smoke

REPO = Path(__file__).resolve().parents[1]
TAPE = REPO / "examples" / "sample_tape.csv"
SCHEMA = REPO / "examples" / "schema.yaml"


@pytest.fixture(scope="module")
def real_metrics() -> VintageMetrics:
    tape = LoanTape.from_csv(TAPE, schema=SCHEMA)
    seg = Segment.from_tape(tape, query=None, label="all")
    return VintageMetrics.from_segment(seg, DefaultPolicy.charge_off_field(), vintage_freq="Q")


def test_real_tape_aggregate_all_weightings(real_metrics: VintageMetrics) -> None:
    metrics_to_request = ["cum_gross_loss", "cum_prepay", "pool_factor", "smm", "cpr", "mdr", "cdr"]
    for w in ("original", "beginning", "equal"):
        agg = aggregate_curves(real_metrics, metrics_to_request, weighting=w)
        assert "n_vintages" in agg.columns
        # n_vintages monotonically non-increasing as MOB grows past max observable
        nv = agg["n_vintages"].to_numpy()
        assert (np.diff(nv) <= 0).all(), f"n_vintages should not grow under {w}: {nv}"
        # CPR / CDR in [0, 1]
        for col in ("cpr", "cdr"):
            vals = agg[col].dropna()
            assert (vals >= -1e-9).all() and (vals <= 1.0 + 1e-9).all()
