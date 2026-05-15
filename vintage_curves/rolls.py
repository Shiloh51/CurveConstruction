"""Delinquency bucket assignment and roll-rate transition matrices."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd

from vintage_curves.default_policy import IS_DEFAULTED
from vintage_curves.mob import MOB
from vintage_curves.schema import CanonicalColumns

BUCKET = "delinquency_bucket"
CO_BUCKET = "CO"

# Default bucket boundaries (per the refined plan): Current = 0–29 DPD.
DEFAULT_BOUNDARIES: tuple[tuple[str, int, int | None], ...] = (
    ("Current",  0,   30),
    ("30-59",    30,  60),
    ("60-89",    60,  90),
    ("90+",      90,  None),
)


WeightMode = Literal["count", "balance"]


def assign_buckets(
    perf_flagged: pd.DataFrame,
    boundaries: Sequence[tuple[str, int, int | None]] = DEFAULT_BOUNDARIES,
) -> pd.Series:
    """Return a categorical Series of bucket labels per row.

    Defaulted rows (sticky) are labeled 'CO' regardless of DPD. Non-defaulted
    rows are bucketed by DPD using half-open intervals [lo, hi).
    """
    dpd = CanonicalColumns.DAYS_PAST_DUE
    labels = [b[0] for b in boundaries] + [CO_BUCKET]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Bucket labels must be unique; got {labels}")

    dpd_vals = perf_flagged[dpd].fillna(0).astype("int64").to_numpy()
    result = np.full(len(perf_flagged), "", dtype=object)
    for label, lo, hi in boundaries:
        if hi is None:
            mask = dpd_vals >= lo
        else:
            mask = (dpd_vals >= lo) & (dpd_vals < hi)
        result[mask] = label

    # Override with CO for defaulted rows (sticky)
    if IS_DEFAULTED in perf_flagged.columns:
        is_co = perf_flagged[IS_DEFAULTED].to_numpy()
        result[is_co] = CO_BUCKET

    return pd.Series(
        pd.Categorical(result, categories=labels, ordered=True),
        index=perf_flagged.index,
        name=BUCKET,
    )


def transition_matrix(
    perf_flagged: pd.DataFrame,
    weight: WeightMode = "count",
    per_mob: bool = False,
    boundaries: Sequence[tuple[str, int, int | None]] = DEFAULT_BOUNDARIES,
) -> pd.DataFrame:
    """Roll-rate matrix from bucket(t) → bucket(t+1) within each loan_id.

    - weight="count":   each transition contributes 1.
    - weight="balance": each transition contributes the EOP balance at MOB t.
    - per_mob=False:    one pooled matrix indexed by from_bucket, columned by to_bucket.
    - per_mob=True:     MultiIndex (mob, from_bucket); rows normalize to 1 within (mob, from_bucket).
    """
    loan_id = CanonicalColumns.LOAN_ID
    eop = CanonicalColumns.OUTSTANDING_BALANCE
    labels = [b[0] for b in boundaries] + [CO_BUCKET]

    f = perf_flagged.sort_values([loan_id, MOB], kind="mergesort").reset_index(drop=True)
    f = f.assign(**{BUCKET: assign_buckets(f, boundaries)})
    f["__next_bucket"] = f.groupby(loan_id, sort=False)[BUCKET].shift(-1)
    f["__next_mob"] = f.groupby(loan_id, sort=False)[MOB].shift(-1)

    # Keep only adjacent transitions (next observation is mob+1 in the tape)
    transitions = f[(f["__next_mob"] - f[MOB]) == 1].copy()
    if transitions.empty:
        idx = pd.CategoricalIndex(labels, categories=labels, ordered=True, name=BUCKET)
        return pd.DataFrame(0.0, index=idx, columns=idx)

    if weight == "count":
        transitions["__w"] = 1.0
    elif weight == "balance":
        transitions["__w"] = transitions[eop].fillna(0.0).astype(float)
    else:
        raise ValueError(f"weight must be 'count' or 'balance', got {weight!r}")

    # Force both bucket columns to the same ordered Categorical so crosstab
    # emits a full label × label matrix (zeros where no flow occurred).
    cat = pd.CategoricalDtype(categories=labels, ordered=True)
    transitions[BUCKET] = transitions[BUCKET].astype(cat)
    transitions["__next_bucket"] = transitions["__next_bucket"].astype(cat)

    if per_mob:
        wide = pd.crosstab(
            index=[transitions[MOB], transitions[BUCKET]],
            columns=transitions["__next_bucket"],
            values=transitions["__w"],
            aggfunc="sum",
            dropna=False,
        ).fillna(0.0)
        # Ensure every (observed mob, bucket) combination is present
        mobs = sorted(transitions[MOB].unique())
        full_index = pd.MultiIndex.from_product([mobs, labels], names=[MOB, BUCKET])
        wide = wide.reindex(index=full_index, columns=labels, fill_value=0.0)
    else:
        wide = pd.crosstab(
            index=transitions[BUCKET],
            columns=transitions["__next_bucket"],
            values=transitions["__w"],
            aggfunc="sum",
            dropna=False,
        ).fillna(0.0)
        wide = wide.reindex(index=labels, columns=labels, fill_value=0.0)

    wide.columns.name = BUCKET + "_next"

    # Normalize rows: each from-bucket sums to 1 (or 0 if no flows).
    # Force plain (non-categorical) index to avoid categorical-alignment quirks.
    wide.index = wide.index if per_mob else pd.Index(list(wide.index), name=wide.index.name)
    wide.columns = pd.Index(list(wide.columns), name=wide.columns.name)
    row_sums = wide.sum(axis=1)
    denom = row_sums.where(row_sums > 0)
    normalized = wide.divide(denom, axis="index").fillna(0.0)
    return normalized
