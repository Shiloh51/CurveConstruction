"""LoanTape: load a CSV tape, apply schema, split into loan-level + perf frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vintage_curves.schema import (
    LOAN_LEVEL_COLUMNS,
    PERF_LEVEL_COLUMNS,
    CanonicalColumns,
    SchemaMapper,
    SchemaValidationError,
)


@dataclass
class LoanTape:
    """Canonicalized tape split into loan-level and perf-level frames.

    `loans`: one row per `loan_id` with loan + borrower characteristics.
    `perf`:  one row per (loan_id, as_of_date), sorted.
    """

    loans: pd.DataFrame
    perf: pd.DataFrame

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        schema: str | Path | SchemaMapper,
        **read_csv_kwargs,
    ) -> "LoanTape":
        mapper = schema if isinstance(schema, SchemaMapper) else SchemaMapper.from_yaml(schema)
        raw = pd.read_csv(path, **read_csv_kwargs)
        return cls.from_dataframe(raw, mapper)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, mapper: SchemaMapper) -> "LoanTape":
        renamed = mapper.rename(df)
        canonical = mapper.validate_and_coerce(renamed)
        loans, perf = _split(canonical)
        _validate_loan_constancy(canonical, loans)
        perf = perf.sort_values(
            [CanonicalColumns.LOAN_ID, CanonicalColumns.AS_OF_DATE],
            kind="mergesort",
        ).reset_index(drop=True)
        return cls(loans=loans, perf=perf)

    @property
    def n_loans(self) -> int:
        return len(self.loans)

    @property
    def n_loan_months(self) -> int:
        return len(self.perf)


def _split(canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    loan_cols = [c for c in canonical.columns if c in LOAN_LEVEL_COLUMNS]
    perf_cols = [c for c in canonical.columns if c in PERF_LEVEL_COLUMNS]
    if CanonicalColumns.LOAN_ID not in loan_cols:
        raise SchemaValidationError("loan_id column missing after canonicalization.")

    loans = (
        canonical[loan_cols]
        .drop_duplicates(subset=[CanonicalColumns.LOAN_ID])
        .reset_index(drop=True)
    )
    perf = canonical[[CanonicalColumns.LOAN_ID, *perf_cols]].copy()
    return loans, perf


def _validate_loan_constancy(canonical: pd.DataFrame, loans: pd.DataFrame) -> None:
    """Confirm loan-level columns don't vary within a loan_id."""
    loan_id = CanonicalColumns.LOAN_ID
    cols_to_check = [c for c in loans.columns if c != loan_id]
    if not cols_to_check:
        return
    # nunique across rows of the same loan_id; >1 means inconsistent
    nun = canonical.groupby(loan_id)[cols_to_check].nunique(dropna=False)
    bad = nun[(nun > 1).any(axis=1)]
    if not bad.empty:
        offenders = bad.index.tolist()[:5]
        raise SchemaValidationError(
            f"Loan-level columns vary within loan_id for {len(bad)} loan(s); "
            f"first offenders: {offenders}"
        )
