"""Worked example: full vintage-curve analysis of `sample_tape.csv`.

Usage:
    python3 examples/run_analysis.py [--outdir examples/output]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from vintage_curves import DefaultPolicy, VintageAnalyzer

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", default=HERE / "sample_tape.csv")
    ap.add_argument("--schema", default=HERE / "schema.yaml")
    ap.add_argument("--outdir", default=DEFAULT_OUT, type=Path)
    args = ap.parse_args()

    t0 = time.time()
    va = VintageAnalyzer.from_csv(
        args.tape,
        schema=args.schema,
        vintage_freq="Q",
        default_policy=DefaultPolicy.charge_off_field(),
    )
    print(f"loaded {va.tape.n_loans:,} loans, "
          f"{va.tape.n_loan_months:,} loan-months in {time.time()-t0:.2f}s")

    seg = va.all()

    print("\nper-vintage cum gross loss (%):")
    print((seg.curve("cum_gross_loss") * 100).round(2).iloc[:, [5, 11, 17, 23]])

    print("\nper-vintage CPR (%) at MOB 6/12/18:")
    cpr = seg.curve("cpr") * 100
    print(cpr[[c for c in cpr.columns if c in (6, 12, 18)]].round(2))

    print("\naggregate (OB-weighted) at MOB 6/12/18/24:")
    agg = seg.aggregate(
        ["cum_gross_loss", "cum_prepay", "cpr", "cdr", "pool_factor"],
        weighting="original",
    )
    print(agg.loc[[m for m in (6, 12, 18, 24) if m in agg.index]].round(4))

    print("\nvintage summary:")
    print(seg.summary().round(3))

    print("\ntransition matrix (count-weighted, pooled):")
    print(seg.roll_rates().round(4))

    args.outdir.mkdir(parents=True, exist_ok=True)
    xlsx = seg.export_excel(args.outdir / "analysis.xlsx")
    print(f"\nwrote {xlsx}")
    fig = seg.plot("cum_gross_loss", include_agg=True, weighting="original")
    fig.savefig(args.outdir / "cum_gross_loss.png", dpi=120)
    print(f"wrote {args.outdir / 'cum_gross_loss.png'}")


if __name__ == "__main__":
    main()
