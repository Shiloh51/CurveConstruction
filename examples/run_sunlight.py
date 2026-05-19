"""Sunlight tape: cum_gross_loss / pool_factor / cpr across 12 cohorts.

Cohorts: Product == "Installment", FICO buckets {680-719, 720-759, 760+},
Terms {60, 120, 144, 180}. All quarterly vintages.

Usage (from the repo root on your Windows machine):
    python examples/run_sunlight.py
    python examples/run_sunlight.py --tape <path> --schema <path> --outdir <path>

Defaults assume the layout you described:
    C:\\Users\\aesses\\CurveConstruction\\sunlight_test\\
        <your tape file>            <-- set TAPE_FILENAME below
        sunlight_schema.yaml
        output/                     <-- created on first run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from vintage_curves import DefaultPolicy, VintageAnalyzer

# ----------------------------------------------------------------- config
# Update TAPE_FILENAME to match your actual file name in sunlight_test/.
SUNLIGHT_DIR = Path(r"C:\Users\aesses\CurveConstruction\sunlight_test")
TAPE_FILENAME = "SLF_Full.csv"        # <-- change if your file is named differently
SCHEMA_FILENAME = "sunlight_schema.yaml"

# Canonical column names after schema mapping. If your schema maps the
# product column to a different canonical name, adjust here.
PRODUCT_COL = "product_type"
PRODUCT_VALUE = "Installment"

FICO_BUCKETS = [
    ("fico_680_719",  "680 <= fico <= 719"),
    ("fico_720_759",  "720 <= fico <= 759"),
    ("fico_760_plus", "fico >= 760"),
]
TERMS = [36, 60, 120, 144, 180]
METRICS = ["cum_gross_loss", "pool_factor", "cpr"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape",   default=SUNLIGHT_DIR / TAPE_FILENAME, type=Path)
    ap.add_argument("--schema", default=SUNLIGHT_DIR / SCHEMA_FILENAME, type=Path)
    ap.add_argument("--outdir", default=SUNLIGHT_DIR / "output", type=Path)
    args = ap.parse_args()

    print(f"Loading tape: {args.tape}")
    va = VintageAnalyzer.from_csv(
        args.tape,
        schema=args.schema,
        vintage_freq="Q",
        default_policy=DefaultPolicy.charge_off_field(),
    )
    print(f"  {va.tape.n_loans:,} loans, {va.tape.n_loan_months:,} loan-months")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for fico_tag, fico_query in FICO_BUCKETS:
        for term in TERMS:
            label = f"installment_{fico_tag}_term_{term}"
            query = (
                f'{PRODUCT_COL} == "{PRODUCT_VALUE}" '
                f'and original_term == {term} '
                f'and {fico_query}'
            )
            print(f"\n[{label}]  query: {query}")

            try:
                r = va.segment(query=query, label=label)
            except Exception as e:
                print(f"  SKIPPED — segment build failed: {e}")
                continue

            n_loans = r.metrics.loans.shape[0]
            n_vintages = r.metrics.cum_gross_loss().shape[0]
            print(f"  {n_loans:,} loans across {n_vintages} quarterly vintages")
            summary_rows.append({
                "segment":    label,
                "fico":       fico_tag,
                "term":       term,
                "n_loans":    n_loans,
                "n_vintages": n_vintages,
            })

            if n_loans == 0:
                print("  EMPTY segment — no rows written")
                continue

            seg_dir = args.outdir / label
            seg_dir.mkdir(parents=True, exist_ok=True)

            for metric in METRICS:
                curve = r.curve(metric)
                curve.to_csv(seg_dir / f"{metric}.csv")
                try:
                    fig = r.plot(metric, include_agg=True, weighting="original")
                    fig.savefig(seg_dir / f"{metric}.png", dpi=120, bbox_inches="tight")
                except Exception as e:
                    print(f"  plot {metric} failed: {e}")

            r.summary().to_csv(seg_dir / "vintage_summary.csv")
            print(f"  wrote -> {seg_dir}")

    if summary_rows:
        import pandas as pd
        pd.DataFrame(summary_rows).to_csv(args.outdir / "_cohorts_summary.csv", index=False)
        print(f"\nWrote cohort summary: {args.outdir / '_cohorts_summary.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
