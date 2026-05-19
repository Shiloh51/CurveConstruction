# CurveConstruction

`vintage_curves` is a Python package that turns a consumer-loan performance
tape (one row per loan-month) into vintage-level performance curves —
cumulative loss, prepay, SMM/CPR, MDR/CDR, pool factor, WAC — plus aggregate
curves across vintages (three weighting schemes), per-vintage summary
scalars, and roll-rate transition matrices.

The design is documented in [`VINTAGE_CURVES_PLAN.md`](VINTAGE_CURVES_PLAN.md).

## Quick start

```python
from vintage_curves import VintageAnalyzer, DefaultPolicy

va = VintageAnalyzer.from_csv(
    "examples/sample_tape.csv",
    schema="examples/schema.yaml",
    vintage_freq="Q",
    default_policy=DefaultPolicy.charge_off_field(),
)

seg = va.segment("fico >= 720", label="prime+")

seg.curve("cum_gross_loss")                          # wide vintage x MOB
seg.aggregate(["cum_gross_loss", "cpr", "cdr"],      # MOB x metric (+ n_vintages)
              weighting="original")
seg.summary()                                        # per-vintage scalars
seg.roll_rates(by_mob=False)                         # transition matrix
seg.plot("cum_gross_loss", include_agg=True).savefig("loss.png")
seg.export_excel("analysis.xlsx")
```

Or run the worked example end-to-end:

```
pip install -e .
python3 examples/run_analysis.py
```

## Install

```
pip install -e .
```

Python 3.11+. Depends on pandas≥2, numpy, pyyaml, matplotlib, openpyxl.

## Tests

```
pip install -e ".[test]"
pytest
```

The full suite includes a hand-derived analytical fixture for every metric
plus end-to-end coverage on the bundled ~655k-row synthetic tape.
