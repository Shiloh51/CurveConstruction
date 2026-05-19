"""End-to-end test against the example sample tape via VintageAnalyzer."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import pandas as pd
import pytest

from vintage_curves import DefaultPolicy, VintageAnalyzer

REPO = Path(__file__).resolve().parents[1]
TAPE = REPO / "examples" / "sample_tape.csv"
SCHEMA = REPO / "examples" / "schema.yaml"


@pytest.fixture(scope="module")
def analyzer() -> VintageAnalyzer:
    return VintageAnalyzer.from_csv(
        TAPE, schema=SCHEMA, vintage_freq="Q",
        default_policy=DefaultPolicy.charge_off_field(),
    )


def test_segment_curve_and_aggregate(analyzer: VintageAnalyzer) -> None:
    seg = analyzer.segment("fico >= 700", label="prime+")
    cgl = seg.curve("cum_gross_loss")
    assert isinstance(cgl, pd.DataFrame)
    assert cgl.shape[0] == 6  # six vintages

    agg = seg.aggregate(["cum_gross_loss", "cpr", "cdr"], weighting="original")
    assert {"cum_gross_loss", "cpr", "cdr", "n_vintages"}.issubset(agg.columns)
    assert (agg["n_vintages"] >= 1).all()


def test_summary_and_rolls(analyzer: VintageAnalyzer) -> None:
    seg = analyzer.all()
    summ = seg.summary()
    assert summ.shape == (6, 6)
    assert (summ["original_count"] == 4_000).all()

    rolls = seg.roll_rates(by_mob=False, weight="count")
    # Each row sums to 0 or 1
    row_sums = rolls.sum(axis=1)
    assert ((row_sums == 0) | (row_sums.round(9) == 1)).all()


def test_plot_returns_figure(analyzer: VintageAnalyzer) -> None:
    seg = analyzer.all()
    fig = seg.plot("cum_gross_loss", include_agg=True, weighting="original")
    assert fig is not None
    assert len(fig.axes) == 1


def test_excel_export_roundtrip(tmp_path: Path, analyzer: VintageAnalyzer) -> None:
    seg = analyzer.segment("original_term == 36", label="36mo")
    out = seg.export_excel(
        tmp_path / "out.xlsx",
        metrics=["cum_gross_loss", "pool_factor", "cpr"],
    )
    assert out.exists()
    book = pd.ExcelFile(out)
    expected = {
        "cum_gross_loss", "pool_factor", "cpr",
        "summary", "roll_rates",
        "agg_original", "agg_beginning", "agg_equal",
    }
    assert expected.issubset(set(book.sheet_names))

    # Spot-check: agg_original sheet contains an n_vintages column
    agg_sheet = pd.read_excel(out, sheet_name="agg_original", index_col=0)
    assert "n_vintages" in agg_sheet.columns


def test_csv_dir_export(tmp_path: Path, analyzer: VintageAnalyzer) -> None:
    seg = analyzer.all()
    out_dir = seg.export_csv_dir(
        tmp_path / "out", metrics=["cum_gross_loss", "cpr"]
    )
    files = {p.name for p in out_dir.iterdir()}
    assert {"cum_gross_loss.csv", "cpr.csv", "summary.csv", "roll_rates.csv",
            "agg_original.csv", "agg_beginning.csv", "agg_equal.csv"}.issubset(files)
