"""Output helpers: collect curves into wide frames and export to CSV/Excel."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

# Excel sheet-name max length
_SHEET_NAME_MAX = 31


def to_excel(
    path: str | Path,
    curves: Mapping[str, pd.DataFrame],
    summary: pd.DataFrame | None = None,
    rolls: pd.DataFrame | None = None,
    aggregates: Mapping[str, pd.DataFrame] | None = None,
) -> Path:
    """Write a multi-sheet workbook.

    - `curves`: {metric_name: wide vintage × MOB frame} — one sheet per metric.
    - `summary`: optional per-vintage scalars frame → sheet "summary".
    - `rolls`: optional transition matrix → sheet "roll_rates".
    - `aggregates`: optional {weighting: aggregate frame} → sheets
      "agg_<weighting>".
    """
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in curves.items():
            df.to_excel(xw, sheet_name=_safe_sheet(name))
        if summary is not None:
            summary.to_excel(xw, sheet_name="summary")
        if rolls is not None:
            rolls.to_excel(xw, sheet_name="roll_rates")
        if aggregates is not None:
            for weighting, df in aggregates.items():
                df.to_excel(xw, sheet_name=_safe_sheet(f"agg_{weighting}"))
    return path


def to_csv_dir(
    out_dir: str | Path,
    curves: Mapping[str, pd.DataFrame],
    summary: pd.DataFrame | None = None,
    rolls: pd.DataFrame | None = None,
    aggregates: Mapping[str, pd.DataFrame] | None = None,
) -> Path:
    """Write one CSV per curve under `out_dir`, plus optional summary/rolls."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in curves.items():
        df.to_csv(out_dir / f"{name}.csv")
    if summary is not None:
        summary.to_csv(out_dir / "summary.csv")
    if rolls is not None:
        rolls.to_csv(out_dir / "roll_rates.csv")
    if aggregates is not None:
        for weighting, df in aggregates.items():
            df.to_csv(out_dir / f"agg_{weighting}.csv")
    return out_dir


def _safe_sheet(name: str) -> str:
    return name[:_SHEET_NAME_MAX]
