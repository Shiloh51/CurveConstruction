"""Analytical-fixture tests for per-vintage metrics.

Three loans, one vintage (2024Q1), original_balance 1000 each (OB=3000),
APR 12%, 10-month term, scheduled principal $100/mo. Behavior:
- Loan A: pays scheduled for 5 months, never defaults, never prepays.
- Loan B: pays scheduled for 2 months, full voluntary prepay at MOB 3.
- Loan C: pays scheduled for 2 months, then charges off at MOB 3 (CO amount 800).

Expected metrics derived by hand; assertions check to numerical tolerance.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vintage_curves import DefaultPolicy, LoanTape, Segment, SchemaMapper, VintageMetrics
from vintage_curves.schema import CanonicalColumns


# ----------------------------------------------------------------- fixture build

CANONICAL_MAPPER = SchemaMapper(mapping={
    "loan_id": "loan_id",
    "origination_date": "origination_date",
    "original_balance": "original_balance",
    "apr": "apr",
    "original_term": "original_term",
    "as_of_date": "as_of_date",
    "outstanding_balance": "outstanding_balance",
    "scheduled_principal": "scheduled_principal",
    "principal_payment": "principal_payment",
    "interest_payment": "interest_payment",
    "days_past_due": "days_past_due",
    "charge_off_date": "charge_off_date",
    "charge_off_amount": "charge_off_amount",
    "fico": "fico",
})


def _row(loan_id, mob_label, as_of, eop, sched, prin_pay, dpd, co_date=None, co_amt=0.0):
    """Helper that emits one canonical perf-row dict."""
    return {
        "loan_id": loan_id,
        "as_of_date": as_of,
        "outstanding_balance": eop,
        "scheduled_principal": sched,
        "principal_payment": prin_pay,
        "interest_payment": 0.0,
        "days_past_due": dpd,
        "charge_off_date": co_date,
        "charge_off_amount": co_amt,
    }


@pytest.fixture
def metrics_fixture() -> VintageMetrics:
    loan_rows = [
        {"loan_id": "A", "origination_date": "2024-01-15", "original_balance": 1000.0,
         "apr": 0.12, "original_term": 10, "fico": 700},
        {"loan_id": "B", "origination_date": "2024-01-15", "original_balance": 1000.0,
         "apr": 0.12, "original_term": 10, "fico": 700},
        {"loan_id": "C", "origination_date": "2024-01-15", "original_balance": 1000.0,
         "apr": 0.12, "original_term": 10, "fico": 700},
    ]
    perf_rows = [
        # Loan A: 5 months of scheduled payments
        _row("A", 1, "2024-01-31",  900.0, 100.0, 100.0, 0),
        _row("A", 2, "2024-02-29",  800.0, 100.0, 100.0, 0),
        _row("A", 3, "2024-03-31",  700.0, 100.0, 100.0, 0),
        _row("A", 4, "2024-04-30",  600.0, 100.0, 100.0, 0),
        _row("A", 5, "2024-05-31",  500.0, 100.0, 100.0, 0),

        # Loan B: 2 scheduled then full prepay at MOB 3
        _row("B", 1, "2024-01-31",  900.0, 100.0, 100.0, 0),
        _row("B", 2, "2024-02-29",  800.0, 100.0, 100.0, 0),
        _row("B", 3, "2024-03-31",    0.0, 100.0, 800.0, 0),  # prin_pay = remaining balance

        # Loan C: 2 scheduled then charge-off at MOB 3
        _row("C", 1, "2024-01-31",  900.0, 100.0, 100.0,   0),
        _row("C", 2, "2024-02-29",  800.0, 100.0, 100.0,  30),
        _row("C", 3, "2024-03-31",    0.0, 100.0,   0.0, 120,
             co_date="2024-03-31", co_amt=800.0),
    ]
    loans = pd.DataFrame(loan_rows)
    perf = pd.DataFrame(perf_rows)
    raw = perf.merge(loans, on="loan_id", how="left")
    tape = LoanTape.from_dataframe(raw, CANONICAL_MAPPER)
    seg = Segment.from_tape(tape, query=None, label="all")
    return VintageMetrics.from_segment(seg, DefaultPolicy.charge_off_field(), vintage_freq="Q")


# ----------------------------------------------------------------- assertions

OB = 3000.0
V = pd.Period("2024Q1")


def test_pool_factor(metrics_fixture: VintageMetrics) -> None:
    pf = metrics_fixture.pool_factor().loc[V]
    expected = {
        1: 2700 / OB,        # A=900 + B=900 + C=900
        2: 2400 / OB,        # 800 + 800 + 800
        3: 700 / OB,         # 700 + 0 + 0
        4: 600 / OB,         # A only
        5: 500 / OB,
    }
    for mob, val in expected.items():
        assert math.isclose(pf[mob], val, rel_tol=1e-12), (mob, pf[mob], val)


def test_cum_gross_loss(metrics_fixture: VintageMetrics) -> None:
    cgl = metrics_fixture.cum_gross_loss().loc[V]
    assert cgl[1] == 0.0
    assert cgl[2] == 0.0
    assert math.isclose(cgl[3], 800 / OB, rel_tol=1e-12)
    assert math.isclose(cgl[4], 800 / OB, rel_tol=1e-12)  # sticky
    assert math.isclose(cgl[5], 800 / OB, rel_tol=1e-12)


def test_cum_prepay(metrics_fixture: VintageMetrics) -> None:
    cpp = metrics_fixture.cum_prepay().loc[V]
    assert cpp[1] == 0.0
    assert cpp[2] == 0.0
    # B prepays 700 of unscheduled (800 paid - 100 scheduled) at MOB 3
    assert math.isclose(cpp[3], 700 / OB, rel_tol=1e-12)
    assert math.isclose(cpp[4], 700 / OB, rel_tol=1e-12)
    assert math.isclose(cpp[5], 700 / OB, rel_tol=1e-12)


def test_smm_excludes_defaults_from_denominator(metrics_fixture: VintageMetrics) -> None:
    smm = metrics_fixture.smm().loc[V]
    # MOB 3: numerator = 700 (B's prepay), denom = BOP_perf 2400 - sched_perf 200 - default_bal 800 = 1400
    assert math.isclose(smm[3], 700 / 1400, rel_tol=1e-12)
    # Other months: no prepay → 0
    assert smm[1] == 0.0
    assert smm[2] == 0.0
    assert smm[4] == 0.0
    assert smm[5] == 0.0


def test_cpr_annualization(metrics_fixture: VintageMetrics) -> None:
    cpr = metrics_fixture.cpr().loc[V]
    smm3 = 700 / 1400
    assert math.isclose(cpr[3], 1 - (1 - smm3) ** 12, rel_tol=1e-12)


def test_mdr_uses_performing_bop_denominator(metrics_fixture: VintageMetrics) -> None:
    mdr = metrics_fixture.mdr().loc[V]
    # MOB 3: default_bal 800 / BOP_perf 2400 = 1/3
    assert math.isclose(mdr[3], 800 / 2400, rel_tol=1e-12)
    assert mdr[1] == 0.0
    assert mdr[2] == 0.0


def test_cdr_annualization(metrics_fixture: VintageMetrics) -> None:
    cdr = metrics_fixture.cdr().loc[V]
    mdr3 = 800 / 2400
    assert math.isclose(cdr[3], 1 - (1 - mdr3) ** 12, rel_tol=1e-12)


def test_wa_coupon_constant_when_all_loans_share_apr(metrics_fixture: VintageMetrics) -> None:
    wac = metrics_fixture.wa_coupon().loc[V]
    # All performing-at-BOP loans share apr=0.12, so WAC is exactly 0.12 wherever defined
    for mob in [1, 2, 3, 4, 5]:
        assert math.isclose(wac[mob], 0.12, rel_tol=1e-12)


def test_curve_dispatch_and_listing(metrics_fixture: VintageMetrics) -> None:
    assert "cum_gross_loss" in metrics_fixture.available_metrics()
    direct = metrics_fixture.pool_factor()
    via = metrics_fixture.curve("pool_factor")
    pd.testing.assert_frame_equal(direct, via)
    with pytest.raises(ValueError, match="Unknown metric"):
        metrics_fixture.curve("not_a_metric")


def test_truncation_nans_beyond_max_observable_mob(metrics_fixture: VintageMetrics) -> None:
    pf = metrics_fixture.pool_factor()
    # Vintage's max observable MOB in this fixture is 5; cells beyond should be absent.
    assert pf.columns.max() == 5
    # Within columns, all entries should be non-NaN (vintage is observable for all 5 MOBs)
    assert pf.loc[V].notna().all()


# ----------------------------------------------------------------- sanity on real tape

REPO = Path(__file__).resolve().parents[1]
TAPE = REPO / "examples" / "sample_tape.csv"
SCHEMA = REPO / "examples" / "schema.yaml"


@pytest.fixture(scope="module")
def real_tape() -> LoanTape:
    return LoanTape.from_csv(TAPE, schema=SCHEMA)


def test_real_tape_metrics_smoke(real_tape: LoanTape) -> None:
    seg = Segment.from_tape(real_tape, query=None, label="all")
    m = VintageMetrics.from_segment(seg, DefaultPolicy.charge_off_field(), vintage_freq="Q")
    pf = m.pool_factor()
    cgl = m.cum_gross_loss()
    cpr = m.cpr()
    # Pool factor starts near 1 at MOB 1 and only declines monotonically per vintage
    for v in pf.index:
        row = pf.loc[v].dropna()
        assert row.iloc[0] <= 1.0 and row.iloc[0] > 0.9, (v, row.iloc[0])
        # Allow tiny float wiggle: assert non-increasing within 1e-9
        diffs = row.diff().dropna()
        assert (diffs <= 1e-9).all(), (v, diffs[diffs > 1e-9])
    # Cum loss is non-decreasing per vintage
    for v in cgl.index:
        row = cgl.loc[v].dropna()
        diffs = row.diff().dropna()
        assert (diffs >= -1e-12).all()
    # CPR is finite and in [0, 1] where defined
    finite = cpr.stack().dropna()
    assert (finite >= -1e-9).all() and (finite <= 1.0 + 1e-9).all()
