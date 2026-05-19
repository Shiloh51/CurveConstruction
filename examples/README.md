# Sample tape

A small deterministic synthetic tape for exercising the vintage-curves plan.

## Files
- `generate_sample_tape.py` — deterministic generator (seed `20260515`).
- `sample_tape.csv` — 24,000 loans across 4 cohorts × 6 quarterly vintages (2023Q1 – 2024Q2), monthly observations through 2026-04-30. ~655k loan-month rows. Vendor-style (non-canonical) column names so the schema mapper has real work to do.
- `schema.yaml` — mapping from the tape's columns to the canonical schema.

## What's in it
Four cohorts, 1,000 loans per cohort per vintage. Lifetime default rate (DR) is the probability a loan charges off; defaults follow a 30→60→90→120 DPD ramp with charge-off in the 120-DPD month. Voluntary prepays are a geometric draw from monthly SMM:

| Product    | FICO    | Lifetime DR | Annual CPR | APR     |
|------------|---------|-------------|------------|---------|
| Standard36 | 680–719 | 13%         | 8%         | 15.99%  |
| Standard36 | 720–759 | 6%          | 10%        | 11.99%  |
| Standard60 | 680–719 | 22%         | 5%         | 17.99%  |
| Standard60 | 720–759 | 11%         | 7%         | 13.99%  |

Defaults concentrate in MOB 6–18 (36-mo) or 8–30 (60-mo). DTI 0.10–0.45, eight states, level-pay amortization.

## Regenerate
```
python3 examples/generate_sample_tape.py
```

The script is deterministic — re-running produces the identical CSV.
