"""Tests for delinquency bucket assignment and roll-rate transition matrices."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vintage_curves import DefaultPolicy, LoanTape, Segment, SchemaMapper, VintageMetrics
from vintage_curves.default_policy import IS_DEFAULTED
from vintage_curves.mob import add_mob_and_vintage
from vintage_curves.rolls import (
    BUCKET,
    CO_BUCKET,
    DEFAULT_BOUNDARIES,
    assign_buckets,
    transition_matrix,
)
from vintage_curves.schema import CanonicalColumns


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
def flagged_perf() -> pd.DataFrame:
    """Three loans with different roll behaviors."""
    loans = pd.DataFrame([
        {"loan_id": "A", "origination_date": "2024-01-15",
         "original_balance": 1000.0, "apr": 0.12, "original_term": 10, "fico": 700},
        {"loan_id": "B", "origination_date": "2024-01-15",
         "original_balance": 1000.0, "apr": 0.12, "original_term": 10, "fico": 700},
        {"loan_id": "C", "origination_date": "2024-01-15",
         "original_balance": 1000.0, "apr": 0.12, "original_term": 10, "fico": 700},
    ])
    perf = pd.DataFrame([
        # A: stays current
        _row("A", "2024-01-31", 900.0, 100.0, 100.0,   0),
        _row("A", "2024-02-29", 800.0, 100.0, 100.0,   0),
        _row("A", "2024-03-31", 700.0, 100.0, 100.0,   0),
        # B: Current → 30-59 → 60-89 → cures back to Current
        _row("B", "2024-01-31", 900.0, 100.0, 100.0,   0),
        _row("B", "2024-02-29", 900.0, 100.0,   0.0,  30),
        _row("B", "2024-03-31", 900.0, 100.0,   0.0,  60),
        _row("B", "2024-04-30", 800.0, 100.0, 200.0,   0),
        # C: 30-59 → CO via DPD on month 3 (with charge_off_date populated)
        _row("C", "2024-01-31", 900.0, 100.0, 100.0,  30),
        _row("C", "2024-02-29", 900.0, 100.0,   0.0,  90),
        _row("C", "2024-03-31",   0.0, 100.0,   0.0, 120,
             co_date="2024-03-31", co_amt=900.0),
    ])
    raw = perf.merge(loans, on="loan_id", how="left")
    tape = LoanTape.from_dataframe(raw, CANONICAL_MAPPER)
    seg = Segment.from_tape(tape, query=None, label="all")
    perf_v = add_mob_and_vintage(seg.perf, seg.loans, vintage_freq="Q")
    return DefaultPolicy.charge_off_field().apply(perf_v)


def test_bucket_assignment_basic(flagged_perf: pd.DataFrame) -> None:
    buckets = assign_buckets(flagged_perf)
    flagged_perf = flagged_perf.assign(b=buckets.values)
    # Loan A always Current
    assert (flagged_perf[flagged_perf["loan_id"] == "A"]["b"] == "Current").all()
    # Loan B sequence
    seq_b = flagged_perf[flagged_perf["loan_id"] == "B"].sort_values("mob")["b"].tolist()
    assert seq_b == ["Current", "30-59", "60-89", "Current"]
    # Loan C ends in CO (sticky from MOB 3 onward — only MOB 3 row here)
    seq_c = flagged_perf[flagged_perf["loan_id"] == "C"].sort_values("mob")["b"].tolist()
    assert seq_c == ["30-59", "90+", "CO"]


def test_co_is_sticky_overrides_dpd() -> None:
    # Hand-build a row where DPD is 0 but is_defaulted is True (post-CO survivor row)
    df = pd.DataFrame({
        "loan_id": ["X", "X"],
        "mob": [1, 2],
        CanonicalColumns.DAYS_PAST_DUE: [0, 0],
        IS_DEFAULTED: [False, True],
    })
    buckets = assign_buckets(df)
    assert buckets.tolist() == ["Current", "CO"]


def test_transition_matrix_counts(flagged_perf: pd.DataFrame) -> None:
    tm = transition_matrix(flagged_perf, weight="count")
    # Rows are from-bucket, columns are to-bucket. Rows sum to 1.
    row_sums = tm.sum(axis=1)
    for label, s in row_sums.items():
        # Buckets with no observed transitions sum to 0; the rest sum to 1.
        assert s in (0.0, pytest.approx(1.0, abs=1e-12)), (label, s)

    # Specific observed transitions:
    # Current→Current: A has 2 such (mob 1→2, 2→3), B has 1 (mob 4 is last so no next), so 2 + ...
    # Adjacency requires next_mob - mob == 1. Last row of each loan is dropped.
    # A: (Current→Current) x2 ; B: (Current→30-59), (30-59→60-89), (60-89→Current); C: (30-59→90+), (90+→CO)
    # So Current→Current = 2, Current→30-59 = 1, 30-59→60-89 = 1, 30-59→90+ = 1,
    # 60-89→Current = 1, 90+→CO = 1.
    # Row totals: Current=3, 30-59=2, 60-89=1, 90+=1
    assert math.isclose(tm.loc["Current", "Current"], 2 / 3, rel_tol=1e-12)
    assert math.isclose(tm.loc["Current", "30-59"], 1 / 3, rel_tol=1e-12)
    assert math.isclose(tm.loc["30-59", "60-89"], 0.5, rel_tol=1e-12)
    assert math.isclose(tm.loc["30-59", "90+"], 0.5, rel_tol=1e-12)
    assert math.isclose(tm.loc["60-89", "Current"], 1.0, rel_tol=1e-12)
    assert math.isclose(tm.loc["90+", "CO"], 1.0, rel_tol=1e-12)


def test_transition_matrix_balance_weight(flagged_perf: pd.DataFrame) -> None:
    tm = transition_matrix(flagged_perf, weight="balance")
    # Each non-zero row still sums to 1
    row_sums = tm.sum(axis=1)
    for label, s in row_sums.items():
        assert s in (0.0, pytest.approx(1.0, abs=1e-12)), (label, s)


def test_per_mob_transition_matrix(flagged_perf: pd.DataFrame) -> None:
    tm = transition_matrix(flagged_perf, per_mob=True)
    assert tm.index.names == ["mob", BUCKET]
    # Each (mob, from_bucket) row that has any flow normalizes to 1
    for idx, row in tm.iterrows():
        total = row.sum()
        assert total in (0.0, pytest.approx(1.0, abs=1e-12)), (idx, total)


def test_invalid_weight_raises(flagged_perf: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="weight"):
        transition_matrix(flagged_perf, weight="weird")  # type: ignore[arg-type]


# ----------------------------------------------------------------- real-tape smoke

REPO = Path(__file__).resolve().parents[1]
TAPE = REPO / "examples" / "sample_tape.csv"
SCHEMA = REPO / "examples" / "schema.yaml"


def test_rolls_on_real_tape() -> None:
    tape = LoanTape.from_csv(TAPE, schema=SCHEMA)
    seg = Segment.from_tape(tape, query=None, label="all")
    perf_v = add_mob_and_vintage(seg.perf, seg.loans, vintage_freq="Q")
    flagged = DefaultPolicy.charge_off_field().apply(perf_v)
    tm = transition_matrix(flagged, weight="count")
    # Most loans stay Current — diagonal Current→Current is large
    assert tm.loc["Current", "Current"] > 0.9
    # Once defaulted, sticky → CO→CO ≈ 1
    if tm.loc[CO_BUCKET].sum() > 0:
        assert math.isclose(tm.loc[CO_BUCKET, CO_BUCKET], 1.0, rel_tol=1e-9)
