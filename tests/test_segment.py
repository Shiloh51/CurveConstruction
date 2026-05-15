"""Tests for Segment / SegmentSet filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from vintage_curves import LoanTape, Segment, SegmentSet
from vintage_curves.schema import CanonicalColumns
from vintage_curves.segment import SegmentQueryError

REPO = Path(__file__).resolve().parents[1]
TAPE = REPO / "examples" / "sample_tape.csv"
SCHEMA = REPO / "examples" / "schema.yaml"


@pytest.fixture(scope="module")
def tape() -> LoanTape:
    return LoanTape.from_csv(TAPE, schema=SCHEMA)


def test_segment_filters_loans_and_perf(tape: LoanTape) -> None:
    seg = Segment.from_tape(tape, "fico >= 720 and fico <= 759", label="prime")
    assert seg.label == "prime"
    assert seg.n_loans > 0
    assert seg.n_loans < tape.n_loans
    # All matched loans satisfy the predicate
    assert seg.loans[CanonicalColumns.FICO].between(720, 759).all()
    # Perf rows belong only to matched loans
    assert seg.perf[CanonicalColumns.LOAN_ID].isin(
        seg.loans[CanonicalColumns.LOAN_ID]
    ).all()


def test_empty_query_returns_full_tape(tape: LoanTape) -> None:
    seg = Segment.from_tape(tape, None, label="all")
    assert seg.n_loans == tape.n_loans
    assert seg.n_loan_months == tape.n_loan_months


def test_bad_query_raises(tape: LoanTape) -> None:
    with pytest.raises(SegmentQueryError):
        Segment.from_tape(tape, "no_such_column == 1", label="bad")


def test_segment_set_iteration(tape: LoanTape) -> None:
    sset = SegmentSet.from_tape(tape, [
        ("super_prime", "fico >= 760"),
        ("prime", "fico >= 700 and fico < 760"),
        ("subprime", "fico < 700"),
    ])
    assert len(sset) == 3
    labels = [s.label for s in sset]
    assert labels == ["super_prime", "prime", "subprime"]
    # Partition (disjoint + covers all FICO-bearing loans)
    total = sum(s.n_loans for s in sset)
    assert total == tape.n_loans
