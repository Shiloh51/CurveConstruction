"""Segment: a labeled filter over a LoanTape using a pandas query string."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from vintage_curves.schema import CanonicalColumns
from vintage_curves.tape import LoanTape


@dataclass
class Segment:
    """A subset of a LoanTape selected by a pandas-`query()` string on loan-level attrs.

    The filter operates on `tape.loans` (one row per loan_id); the matching
    perf rows are derived by `loan_id` membership. Time-varying attributes
    (e.g. days_past_due) are deliberately not queryable here — segmentation
    is a loan-cohort concept.
    """

    label: str
    loans: pd.DataFrame
    perf: pd.DataFrame
    query: str | None = None

    @classmethod
    def from_tape(
        cls,
        tape: LoanTape,
        query: str | None,
        label: str,
    ) -> "Segment":
        if query is None or query.strip() == "":
            loans = tape.loans
        else:
            try:
                loans = tape.loans.query(query)
            except Exception as e:  # noqa: BLE001 — re-raise with context
                raise SegmentQueryError(
                    f"Failed to evaluate segment query {query!r}: {e}"
                ) from e
        loan_ids = loans[CanonicalColumns.LOAN_ID]
        perf = tape.perf[tape.perf[CanonicalColumns.LOAN_ID].isin(loan_ids)].copy()
        return cls(label=label, loans=loans.copy(), perf=perf, query=query)

    @property
    def n_loans(self) -> int:
        return len(self.loans)

    @property
    def n_loan_months(self) -> int:
        return len(self.perf)


class SegmentQueryError(ValueError):
    """Raised when a segment query string cannot be evaluated against loan-level attrs."""


@dataclass
class SegmentSet:
    """A collection of named segments for batch metric production."""

    segments: list[Segment]

    @classmethod
    def from_tape(
        cls,
        tape: LoanTape,
        specs: Iterable[tuple[str, str]],
    ) -> "SegmentSet":
        """`specs` is an iterable of (label, query) pairs."""
        return cls(segments=[Segment.from_tape(tape, q, label=lbl) for lbl, q in specs])

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)
