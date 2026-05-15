"""Canonical column names, schema mapping, and tape validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml


class CanonicalColumns:
    # Loan-level (constant per loan_id)
    LOAN_ID = "loan_id"
    ORIGINATION_DATE = "origination_date"
    ORIGINAL_BALANCE = "original_balance"
    ORIGINAL_TERM = "original_term"
    APR = "apr"
    PRODUCT_TYPE = "product_type"
    DEFERRAL_PERIOD = "deferral_period"

    # Borrower-level (constant per loan_id)
    FICO = "fico"
    DTI = "dti"
    INCOME = "income"
    STATE = "state"

    # Performance (one row per loan-month)
    AS_OF_DATE = "as_of_date"
    OUTSTANDING_BALANCE = "outstanding_balance"
    SCHEDULED_PRINCIPAL = "scheduled_principal"
    PRINCIPAL_PAYMENT = "principal_payment"
    INTEREST_PAYMENT = "interest_payment"
    FEES_PAID = "fees_paid"
    DAYS_PAST_DUE = "days_past_due"
    CHARGE_OFF_DATE = "charge_off_date"
    CHARGE_OFF_AMOUNT = "charge_off_amount"


LOAN_LEVEL_COLUMNS = frozenset({
    CanonicalColumns.LOAN_ID,
    CanonicalColumns.ORIGINATION_DATE,
    CanonicalColumns.ORIGINAL_BALANCE,
    CanonicalColumns.ORIGINAL_TERM,
    CanonicalColumns.APR,
    CanonicalColumns.PRODUCT_TYPE,
    CanonicalColumns.DEFERRAL_PERIOD,
    CanonicalColumns.FICO,
    CanonicalColumns.DTI,
    CanonicalColumns.INCOME,
    CanonicalColumns.STATE,
})

PERF_LEVEL_COLUMNS = frozenset({
    CanonicalColumns.AS_OF_DATE,
    CanonicalColumns.OUTSTANDING_BALANCE,
    CanonicalColumns.SCHEDULED_PRINCIPAL,
    CanonicalColumns.PRINCIPAL_PAYMENT,
    CanonicalColumns.INTEREST_PAYMENT,
    CanonicalColumns.FEES_PAID,
    CanonicalColumns.DAYS_PAST_DUE,
    CanonicalColumns.CHARGE_OFF_DATE,
    CanonicalColumns.CHARGE_OFF_AMOUNT,
})

REQUIRED_LOAN_COLUMNS = frozenset({
    CanonicalColumns.LOAN_ID,
    CanonicalColumns.ORIGINATION_DATE,
    CanonicalColumns.ORIGINAL_BALANCE,
    CanonicalColumns.ORIGINAL_TERM,
    CanonicalColumns.APR,
})

REQUIRED_PERF_COLUMNS = frozenset({
    CanonicalColumns.AS_OF_DATE,
    CanonicalColumns.OUTSTANDING_BALANCE,
    CanonicalColumns.SCHEDULED_PRINCIPAL,
    CanonicalColumns.PRINCIPAL_PAYMENT,
    CanonicalColumns.DAYS_PAST_DUE,
})

DATE_COLUMNS = frozenset({
    CanonicalColumns.ORIGINATION_DATE,
    CanonicalColumns.AS_OF_DATE,
    CanonicalColumns.CHARGE_OFF_DATE,
})

FLOAT_COLUMNS = frozenset({
    CanonicalColumns.ORIGINAL_BALANCE,
    CanonicalColumns.APR,
    CanonicalColumns.DTI,
    CanonicalColumns.INCOME,
    CanonicalColumns.OUTSTANDING_BALANCE,
    CanonicalColumns.SCHEDULED_PRINCIPAL,
    CanonicalColumns.PRINCIPAL_PAYMENT,
    CanonicalColumns.INTEREST_PAYMENT,
    CanonicalColumns.FEES_PAID,
    CanonicalColumns.CHARGE_OFF_AMOUNT,
})

INT_COLUMNS = frozenset({
    CanonicalColumns.ORIGINAL_TERM,
    CanonicalColumns.DEFERRAL_PERIOD,
    CanonicalColumns.FICO,
    CanonicalColumns.DAYS_PAST_DUE,
})

STR_COLUMNS = frozenset({
    CanonicalColumns.LOAN_ID,
    CanonicalColumns.PRODUCT_TYPE,
    CanonicalColumns.STATE,
})


class SchemaValidationError(ValueError):
    """Raised when a tape fails canonical-schema validation."""


@dataclass
class SchemaMapper:
    """Renames vendor columns to canonical names and coerces dtypes.

    The mapping dict is keyed by canonical name and valued by the source
    column name in the user's tape.
    """

    mapping: Mapping[str, str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SchemaMapper":
        with open(path) as f:
            mapping = yaml.safe_load(f) or {}
        if not isinstance(mapping, dict):
            raise SchemaValidationError(
                f"Schema YAML at {path} must be a mapping of canonical->source names."
            )
        unknown = set(mapping) - LOAN_LEVEL_COLUMNS - PERF_LEVEL_COLUMNS
        if unknown:
            raise SchemaValidationError(
                f"Unknown canonical columns in schema: {sorted(unknown)}"
            )
        return cls(mapping=dict(mapping))

    def rename(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with source columns renamed to canonical names.

        Source columns not referenced in the mapping are dropped — the analyzer
        only operates on the canonical schema.
        """
        # canonical -> source, invert to source -> canonical, keeping only sources present
        present = {src: canon for canon, src in self.mapping.items() if src in df.columns}
        missing_sources = [src for src in self.mapping.values() if src not in df.columns]
        if missing_sources:
            raise SchemaValidationError(
                f"Source columns referenced in schema but absent from tape: {missing_sources}"
            )
        keep = list(present.keys())
        return df[keep].rename(columns=present)

    def validate_and_coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns are present and non-empty; coerce dtypes."""
        missing_required = (REQUIRED_LOAN_COLUMNS | REQUIRED_PERF_COLUMNS) - set(df.columns)
        if missing_required:
            raise SchemaValidationError(
                f"Required canonical columns missing from tape after mapping: "
                f"{sorted(missing_required)}"
            )

        out = df.copy()

        for col in DATE_COLUMNS & set(out.columns):
            out[col] = pd.to_datetime(out[col], errors="coerce")

        for col in FLOAT_COLUMNS & set(out.columns):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

        for col in INT_COLUMNS & set(out.columns):
            # Nullable Int64 so optional ints (e.g. deferral_period) can carry NaN
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

        for col in STR_COLUMNS & set(out.columns):
            out[col] = out[col].astype("string")

        # All-NaN check on hard-required columns
        for col in REQUIRED_LOAN_COLUMNS | REQUIRED_PERF_COLUMNS:
            if col in out.columns and out[col].isna().all():
                raise SchemaValidationError(
                    f"Required column '{col}' is present but entirely NaN/empty."
                )

        return out
