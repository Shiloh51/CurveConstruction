# Sample tape

A small deterministic synthetic tape for exercising the vintage-curves plan.

## Files
- `generate_sample_tape.py` — deterministic generator (seed `20260515`).
- `sample_tape.csv` — 45 loans × 3 quarterly vintages (2024Q1 / Q2 / Q3), 36-month term, monthly observations through 2026-04-30. Vendor-style (non-canonical) column names so the schema mapper has real work to do.
- `schema.yaml` — mapping from the tape's columns to the canonical schema.

## What's in it
- ~10% / 15% / 5% default rates across the three vintages (defaults concentrated in MOB 6–18, with a 30→60→90→120 DPD ramp and a charge-off in the 120 DPD month).
- ~6% / 4% / 8% annualized voluntary-prepay rates (full prepay, geometric draw from SMM).
- Realistic level-pay amortization at 14.99% APR.
- FICO 620–800, DTI 0.10–0.45, eight states, single `PersonalLoan` product.

## Regenerate
```
python3 examples/generate_sample_tape.py
```

The script is deterministic — re-running produces the identical CSV.
