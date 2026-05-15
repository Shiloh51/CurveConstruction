"""Tests for MOB / vintage / BOP-balance derivation."""

from __future__ import annotations

import pandas as pd
import pytest

from vintage_curves.mob import BOP_BALANCE, MOB, VINTAGE, add_mob_and_vintage, find_mob_gaps
from vintage_curves.schema import CanonicalColumns


def _tiny_tape() -> tuple[pd.DataFrame, pd.DataFrame]:
    loans = pd.DataFrame({
        CanonicalColumns.LOAN_ID: ["L1", "L2"],
        CanonicalColumns.ORIGINATION_DATE: pd.to_datetime(["2024-01-15", "2024-02-20"]),
        CanonicalColumns.ORIGINAL_BALANCE: [10_000.0, 5_000.0],
    })
    perf = pd.DataFrame({
        CanonicalColumns.LOAN_ID: ["L1", "L1", "L1", "L2", "L2"],
        CanonicalColumns.AS_OF_DATE: pd.to_datetime([
            "2024-01-31", "2024-02-29", "2024-03-31",
            "2024-02-29", "2024-03-31",
        ]),
        CanonicalColumns.OUTSTANDING_BALANCE: [9_700.0, 9_400.0, 9_100.0, 4_800.0, 4_600.0],
    })
    return perf, loans


def test_mob_starts_at_one() -> None:
    perf, loans = _tiny_tape()
    out = add_mob_and_vintage(perf, loans)
    assert out.loc[out[CanonicalColumns.LOAN_ID] == "L1", MOB].tolist() == [1, 2, 3]
    assert out.loc[out[CanonicalColumns.LOAN_ID] == "L2", MOB].tolist() == [1, 2]


def test_bop_balance_uses_original_at_mob_1_then_prior_eop() -> None:
    perf, loans = _tiny_tape()
    out = add_mob_and_vintage(perf, loans)
    l1 = out[out[CanonicalColumns.LOAN_ID] == "L1"].sort_values(MOB)
    assert l1[BOP_BALANCE].tolist() == [10_000.0, 9_700.0, 9_400.0]


def test_vintage_period_buckets() -> None:
    perf, loans = _tiny_tape()
    out_q = add_mob_and_vintage(perf, loans, vintage_freq="Q")
    assert (out_q[out_q[CanonicalColumns.LOAN_ID] == "L1"][VINTAGE].iloc[0] == pd.Period("2024Q1"))
    out_m = add_mob_and_vintage(perf, loans, vintage_freq="M")
    assert (out_m[out_m[CanonicalColumns.LOAN_ID] == "L1"][VINTAGE].iloc[0] == pd.Period("2024-01"))
    out_a = add_mob_and_vintage(perf, loans, vintage_freq="A")
    assert (out_a[out_a[CanonicalColumns.LOAN_ID] == "L1"][VINTAGE].iloc[0] == pd.Period("2024"))


def test_invalid_vintage_freq() -> None:
    perf, loans = _tiny_tape()
    with pytest.raises(ValueError, match="vintage_freq"):
        add_mob_and_vintage(perf, loans, vintage_freq="W")  # type: ignore[arg-type]


def test_gap_detection() -> None:
    loans = pd.DataFrame({
        CanonicalColumns.LOAN_ID: ["L1"],
        CanonicalColumns.ORIGINATION_DATE: pd.to_datetime(["2024-01-15"]),
        CanonicalColumns.ORIGINAL_BALANCE: [10_000.0],
    })
    perf = pd.DataFrame({
        CanonicalColumns.LOAN_ID: ["L1", "L1"],
        CanonicalColumns.AS_OF_DATE: pd.to_datetime(["2024-01-31", "2024-03-31"]),
        CanonicalColumns.OUTSTANDING_BALANCE: [9_700.0, 9_100.0],
    })
    out = add_mob_and_vintage(perf, loans)
    gaps = find_mob_gaps(out)
    assert len(gaps) == 1
    assert int(gaps["gap_size"].iloc[0]) == 2
