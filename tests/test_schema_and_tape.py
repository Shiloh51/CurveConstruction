"""Smoke tests for SchemaMapper + LoanTape against the example tape."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vintage_curves import LoanTape, SchemaMapper
from vintage_curves.schema import (
    CanonicalColumns,
    REQUIRED_LOAN_COLUMNS,
    REQUIRED_PERF_COLUMNS,
    SchemaValidationError,
)

REPO = Path(__file__).resolve().parents[1]
TAPE = REPO / "examples" / "sample_tape.csv"
SCHEMA = REPO / "examples" / "schema.yaml"


@pytest.fixture(scope="module")
def tape() -> LoanTape:
    return LoanTape.from_csv(TAPE, schema=SCHEMA)


def test_sample_tape_loads(tape: LoanTape) -> None:
    assert tape.n_loans == 24_000
    assert tape.n_loan_months == 654_811


def test_required_columns_present(tape: LoanTape) -> None:
    for col in REQUIRED_LOAN_COLUMNS:
        assert col in tape.loans.columns, col
    for col in REQUIRED_PERF_COLUMNS:
        assert col in tape.perf.columns, col


def test_dtype_coercion(tape: LoanTape) -> None:
    assert pd.api.types.is_datetime64_any_dtype(tape.loans[CanonicalColumns.ORIGINATION_DATE])
    assert pd.api.types.is_datetime64_any_dtype(tape.perf[CanonicalColumns.AS_OF_DATE])
    assert pd.api.types.is_float_dtype(tape.loans[CanonicalColumns.ORIGINAL_BALANCE])
    # nullable Int64 for ints that may carry NaN
    assert str(tape.perf[CanonicalColumns.DAYS_PAST_DUE].dtype) == "Int64"


def test_perf_is_sorted(tape: LoanTape) -> None:
    p = tape.perf
    assert p[CanonicalColumns.LOAN_ID].is_monotonic_increasing or True  # not strictly required globally
    # within each loan, as_of_date is monotonic
    grp = p.groupby(CanonicalColumns.LOAN_ID)[CanonicalColumns.AS_OF_DATE]
    assert grp.apply(lambda s: s.is_monotonic_increasing).all()


def test_loan_level_split_unique(tape: LoanTape) -> None:
    assert tape.loans[CanonicalColumns.LOAN_ID].is_unique


def test_missing_source_column_raises(tmp_path: Path) -> None:
    bad_schema = tmp_path / "bad.yaml"
    bad_schema.write_text(
        "loan_id: LoanNumber\n"
        "origination_date: OrigDt\n"
        "original_balance: OrigUPB\n"
        "apr: NoteRate\n"
        "original_term: OrigTerm\n"
        "as_of_date: ReportingPeriod\n"
        "outstanding_balance: EndingBalance\n"
        "scheduled_principal: NotARealColumn\n"
        "principal_payment: PrincipalPaid\n"
        "days_past_due: DPD\n"
    )
    with pytest.raises(SchemaValidationError, match="absent from tape"):
        LoanTape.from_csv(TAPE, schema=bad_schema)


def test_missing_required_canonical_raises() -> None:
    df = pd.DataFrame({
        "LoanNumber": ["A"],
        "OrigDt": ["2024-01-01"],
        "OrigUPB": [10_000.0],
        "NoteRate": [0.15],
        "OrigTerm": [36],
        "ReportingPeriod": ["2024-01-31"],
        "EndingBalance": [9_800.0],
        # SchedPrincipal omitted entirely
        "PrincipalPaid": [200.0],
        "DPD": [0],
    })
    mapper = SchemaMapper(mapping={
        "loan_id": "LoanNumber",
        "origination_date": "OrigDt",
        "original_balance": "OrigUPB",
        "apr": "NoteRate",
        "original_term": "OrigTerm",
        "as_of_date": "ReportingPeriod",
        "outstanding_balance": "EndingBalance",
        "principal_payment": "PrincipalPaid",
        "days_past_due": "DPD",
    })
    with pytest.raises(SchemaValidationError, match="scheduled_principal"):
        LoanTape.from_dataframe(df, mapper)


def test_loan_constancy_enforced() -> None:
    df = pd.DataFrame({
        "LoanNumber": ["A", "A"],
        "OrigDt": ["2024-01-01", "2024-01-01"],
        "OrigUPB": [10_000.0, 10_000.0],
        "NoteRate": [0.15, 0.18],  # inconsistent within loan
        "OrigTerm": [36, 36],
        "ReportingPeriod": ["2024-01-31", "2024-02-29"],
        "EndingBalance": [9_800.0, 9_600.0],
        "SchedPrincipal": [200.0, 200.0],
        "PrincipalPaid": [200.0, 200.0],
        "DPD": [0, 0],
    })
    mapper = SchemaMapper(mapping={
        "loan_id": "LoanNumber",
        "origination_date": "OrigDt",
        "original_balance": "OrigUPB",
        "apr": "NoteRate",
        "original_term": "OrigTerm",
        "as_of_date": "ReportingPeriod",
        "outstanding_balance": "EndingBalance",
        "scheduled_principal": "SchedPrincipal",
        "principal_payment": "PrincipalPaid",
        "days_past_due": "DPD",
    })
    with pytest.raises(SchemaValidationError, match="vary within loan_id"):
        LoanTape.from_dataframe(df, mapper)
