# Vintage Performance Curves — Plan of Action

A Python library for taking a consumer-loan performance tape (loan + borrower
characteristics + monthly performance data) and producing vintage-level
performance curves, with arbitrary segmentation and standard ABS metrics.

---

## 1. Scope & Goals

**In scope (v1):**
- Single flat CSV input (one row per loan-month).
- User-supplied column-name mapping (YAML or dict) → canonical schema.
- Arbitrary pandas-`query()` style segmentation strings.
- Configurable default definition: `Charge-Off Date` field **OR** a DPD threshold (sticky — once defaulted, stays defaulted even on cure).
- Monthly, quarterly, **or** annual vintage grouping, anchored on origination date.
- MOB = months since origination.
- Per-vintage curves **and** aggregate curves across vintages, with three weighting schemes (original balance / beginning balance at MOB / equal). Each aggregate is published alongside an `n_vintages` series so consumers can see where support collapses.
- Truncate each vintage at its maximum observable MOB.
- Wide matrix output (vintage × MOB) per metric, plus matplotlib plotting and CSV/Excel export.
- Portfolio summary stats per vintage (WA Coupon, Pool Factor, WAL).
- Delinquency transition / roll-rate matrices.
- Unit tests against a small synthetic tape.

**Out of scope (v1):**
- Multi-file inputs, Parquet/Excel readers (can add later via a thin loader layer).
- CLI / dashboard.
- Net loss curves (gross only for v1; explicitly deferred).
- Polars / DuckDB backends. Code will be structured so the engine could be swapped, but v1 ships pandas only.

---

## 2. Canonical Schema

All internal logic operates on canonical column names. Users provide a mapping
from their tape's columns to these names.

### Loan characteristics (one value per Loan_ID, constant over time)
| Canonical | Type | Notes |
|---|---|---|
| `loan_id` | str | Primary key |
| `origination_date` | date | Used to derive vintage |
| `original_balance` | float | Denominator for cum loss/prepay |
| `original_term` | int | Months |
| `apr` | float | Decimal (e.g., 0.1499) |
| `product_type` | str | Optional |
| `deferral_period` | int | Optional, months |

### Borrower characteristics (one value per Loan_ID)
| Canonical | Type |
|---|---|
| `fico` | int |
| `dti` | float |
| `income` | float |
| `state` | str |

### Performance (one row per loan-month)
| Canonical | Type | Notes |
|---|---|---|
| `as_of_date` | date | Month-end of the observation |
| `outstanding_balance` | float | Period-end |
| `scheduled_principal` | float | Scheduled principal for the period |
| `principal_payment` | float | Actual principal paid (total) |
| `interest_payment` | float | Actual interest paid |
| `fees_paid` | float | Optional |
| `days_past_due` | int | DPD at period end |
| `charge_off_date` | date or NaT | Filled when CO occurs |
| `charge_off_amount` | float | Loss amount at CO (gross) |


(Recoveries / `recovery_amount` deliberately omitted from v1 — net loss is out of scope.)

Mapping config example (`schema.yaml`):
```yaml
loan_id: LoanNumber
origination_date: OrigDt
original_balance: OrigUPB
apr: NoteRate
fico: BorrowerFICO
as_of_date: ReportingPeriod
outstanding_balance: EndingBalance
charge_off_date: ChargeOffDt
charge_off_amount: ChargeOffAmt
days_past_due: DPD
principal_payment: PrincipalPaid
interest_payment: InterestPaid
```

A `SchemaMapper` class loads the YAML, renames columns, validates required
fields are present, and coerces dtypes.

---

## 3. Architecture

```
vintage_curves/
├── __init__.py
├── schema.py          # SchemaMapper, canonical column constants, required-field validation
├── tape.py            # LoanTape: load CSV, apply schema, split into loan-level vs perf-level frames
├── default_policy.py  # DefaultPolicy: charge_off_field | dpd_threshold(n)
├── segment.py         # Segment: query string + label; SegmentSet for batches
├── mob.py             # add_mob(perf, loans) -> adds 'mob' and 'vintage' columns
├── metrics.py         # Per-vintage metric calcs (cum gross loss, cum prepay, SMM/CPR, CDR, etc.)
├── aggregate.py       # Balance-weighted aggregation across vintages
├── summary.py         # WA Coupon, Pool Factor, WAL per vintage
├── rolls.py           # Delinquency bucket assignment + transition matrices
├── output.py          # Wide-matrix builder, CSV/Excel export
├── plotting.py        # plot_curves(matrix, title=...) matplotlib helpers
└── analyzer.py        # VintageAnalyzer — the public façade
tests/
├── synthetic.py       # Generates a small deterministic tape
├── test_metrics.py
├── test_segment.py
├── test_rolls.py
└── test_end_to_end.py
```

### Public API (sketch)

```python
from vintage_curves import VintageAnalyzer, DefaultPolicy

va = VintageAnalyzer.from_csv(
    "tape.csv",
    schema="schema.yaml",
    vintage_freq="Q",                       # "M", "Q", or "A"
    default_policy=DefaultPolicy.charge_off_field(),  # or .dpd_threshold(120)
)

# Segment
seg = va.segment("fico >= 720 and fico <= 759 and original_term == 120",
                 label="720-759 / 120mo")

# Per-vintage curves: rows = vintages, cols = MOB
cum_gross_loss = seg.curve("cum_gross_loss")    # DataFrame (wide)
cpr            = seg.curve("cpr")
cdr            = seg.curve("cdr")
pool_factor    = seg.curve("pool_factor")

# Aggregate across vintages — one row per MOB
# weighting: "original" (default) weights each vintage by its original balance;
#            "beginning" uses begin-of-period balance at each MOB;
#            "equal" simple-averages across vintages (sanity check).
# Every aggregate frame includes an `n_vintages` column showing support at each MOB.
agg_orig = seg.aggregate(metrics=["cum_gross_loss", "smm", "cdr"], weighting="original")
agg_bop  = seg.aggregate(metrics=["cum_gross_loss", "smm", "cdr"], weighting="beginning")
agg_eq   = seg.aggregate(metrics=["cum_gross_loss", "smm", "cdr"], weighting="equal")

# Per-vintage summary stats
summary = seg.summary()      # wa_coupon, pool_factor_latest, wal, original_count, original_bal

# Roll-rate transition matrix (current→30→60→90→CO)
rolls = seg.roll_rates(by_mob=True)

# Plot & export
seg.plot("cum_gross_loss").savefig("loss.png")
seg.export_excel("output.xlsx")    # one sheet per metric + summary + rolls
```

---

## 4. Metric Definitions

Let original balance for vintage v be `OB_v = Σ original_balance` for loans in v.
Let `bal_{l,t}` be outstanding (end-of-period) balance for loan l at MOB t.

**Beginning-of-period (BOP) balance** is defined as the prior period's EOP balance,
with `original_balance` substituted at MOB 1 (and MOB 0 treated as the origination
snapshot). If the tape has a gap for a loan-month, BOP for the next observed month
falls back to the most recent prior EOP; gaps are flagged in a load-time diagnostic.

**Performing-loan set** at MOB t for vintage v: all loans in v that have not
defaulted (per the active `DefaultPolicy`) and have not fully prepaid as of the
start of period t. Defaults are **sticky**: once a loan defaults, it is removed
from the performing set for all subsequent MOBs regardless of any later cure.

### Per-vintage curves
- **Cumulative gross loss %** `cum_gross_loss_{v,t} = (Σ_{l∈v, mob≤t} charge_off_amount_l) / OB_v`
- **Cumulative prepayments %** `cum_prepay_{v,t} = (Σ unscheduled principal) / OB_v` where unscheduled = `principal_payment − scheduled_principal` (floored at 0), summed **only over performing loan-months** — the default month and all later months for a defaulted loan are excluded from the prepay numerator so charge-off write-downs are never counted as prepays.
- **SMM (Single Monthly Mortality)** at MOB t for vintage v, restricted to performing loans:
  `SMM_{v,t} = unscheduled_principal_{v,t} / (BOP_bal_{v,t} − scheduled_principal_{v,t} − default_bal_{v,t})`
  where `default_bal_{v,t}` is the BOP balance of loans that default during period t (PSA convention — keeps prepay rate uncontaminated by defaults).
- **CPR (annualized)** `CPR_{v,t} = 1 − (1 − SMM_{v,t})^12`
- **MDR / CDR (annualized)** `MDR_{v,t} = default_bal_{v,t} / BOP_bal_performing_{v,t}`, then `CDR_{v,t} = 1 − (1 − MDR_{v,t})^12`. Denominator is the BOP **actual** balance of performing loans for vintage v (same set used for SMM, before the scheduled-principal carve-out).
- **Pool Factor** `pool_factor_{v,t} = (Σ bal_{l,t}) / OB_v`
- **WA Coupon** balance-weighted `apr` at MOB t (matrix or single per-vintage value)
- **WAL** per vintage, in months. Two flavors are emitted side-by-side because realized-only WAL is biased downward by defaults/prepays:
  - `wal_realized`: `Σ t · principal_received_t / Σ principal_received_t` over observed history.
  - `wal_scheduled`: same formula but using the **scheduled** principal stream (level-pay implied by `original_balance`, `apr`, `original_term`), ignoring defaults and prepays.

### Default policy
- `DefaultPolicy.charge_off_field()`: a loan defaults in the month its `charge_off_date` falls; loss = `charge_off_amount`.
- `DefaultPolicy.dpd_threshold(n)`: a loan defaults the first MOB where `days_past_due ≥ n`; loss = **EOP `outstanding_balance` of the default month** (gross).
- **Stickiness**: under either policy, once a loan is flagged defaulted it stays defaulted for all subsequent MOBs even if it later cures. The loan is removed from performing denominators (SMM, CDR, pool factor performing-share) from the default month forward.
- **Coexistence of fields**: if the user selects `dpd_threshold(n)` but the tape **also** populates `charge_off_date` on some loans, schema validation raises by default. The user can pass `on_conflict="charge_off_wins"` or `"dpd_wins"` to resolve explicitly; the resolution choice is logged.

### Aggregate across vintages (three weighting schemes)

For metric M at MOB t, restricted to vintages where MOB t is observable:

- **`weighting="original"` (default)** — weight each vintage by its original balance `OB_v`.
  - Stock metrics (cum loss %, cum prepay %, pool factor): `M_agg_t = Σ_v OB_v · M_{v,t} / Σ_v OB_v`.
  - Flow / rate metrics (SMM, CPR, MDR, CDR): aggregate raw $ numerator and $ denominator separately, then form the ratio — equivalent to dollar-weighting by original balance contributions.
  - Interpretation: "the average originated dollar's lifetime experience" — keeps each vintage's footprint proportional to how big it was at origination.

- **`weighting="beginning"`** — weight each vintage at MOB t by its **beginning-of-period balance** at that MOB.
  - Stock metrics: `M_agg_t = Σ_v BOP_{v,t} · M_{v,t} / Σ_v BOP_{v,t}`.
  - Flow / rate metrics: numerator = Σ_v flow$_{v,t}; denominator = Σ_v BOP_{v,t} (or `BOP − scheduled_principal` for SMM, consistent with the per-vintage definition).
  - Interpretation: "the average outstanding dollar's instantaneous behavior" — reweights toward vintages that still have balance at MOB t. Useful for current-portfolio behavior; CPR/CDR comparisons can differ noticeably from the original-weighted view late in a curve.

- **`weighting="equal"`** — simple unweighted average of `M_{v,t}` across vintages observable at MOB t. Useful as a sanity check against original-weighted aggregates that one large vintage can dominate.

All three views are always available; Excel export emits each as a separate sheet. Every aggregate frame carries an `n_vintages` column (count of vintages contributing at each MOB) so users can see exactly where support thins out.

### Truncation
For each vintage v with last `as_of_date = T_v`, the maximum observable MOB is
`(T_v − origination_month_v)` in months. Cells beyond `mob_max_v` are NaN
(and dropped from aggregate denominators for that MOB).

### Delinquency buckets & roll rates
Buckets: `Current` (DPD=0–29), `30-59`, `60-89`, `90+`, `CO`. (Standard consumer-credit convention folds 1–29 DPD into Current; the bucket boundaries are configurable on `roll_rates(...)` for users who want finer-grained early-stage tracking.)
Transition matrix: count or balance share moving from bucket_i (MOB t) →
bucket_j (MOB t+1), normalized within row. Optionally per-MOB or pooled.

---

## 5. Implementation Notes

- **Engine**: pandas. Group operations are vectorized; one merge of loan-level
  attrs onto perf at load time, then group by `vintage` / `mob`.
- **MOB**: `mob = (as_of_date.year - orig.year) * 12 + (as_of_date.month - orig.month)`.
- **Vintage**: `vintage = orig.to_period("M", "Q", or "A")` (calendar quarters/years, not fiscal).
- **BOP balance**: derived once at load time as the prior period's EOP `outstanding_balance` per loan, with `original_balance` used at MOB 1. Stored as a column `bop_balance` on the perf frame. Gaps in the per-loan monthly sequence are flagged in a load diagnostic.
- **Memory hook**: load function accepts `usecols`, `dtype`, and `chunksize`
  parameters as future swap-in points (documented, not exercised in v1).
- **Required columns** (hard-fail in schema validation if missing or all-NaN):
  loan-level — `loan_id`, `origination_date`, `original_balance`, `original_term`, `apr`;
  perf-level — `as_of_date`, `outstanding_balance`, `scheduled_principal`, `principal_payment`, `days_past_due`.
  No silent fallback to a computed level-pay schedule for `scheduled_principal`; users needing that must precompute the column.
- **Sign conventions**: payments positive, balances non-negative.
- **Plotting**: each `plot_curves` returns a `matplotlib.figure.Figure`. One
  line per vintage; aggregate overlaid in bold black when `include_agg=True`.

---

## 6. Synthetic Test Tape

`tests/synthetic.py` generates a deterministic tape:
- 3 vintages × 50 loans each, 60-month term, 12% APR.
- Configurable default rate per vintage to produce known cum-loss curves.
- A few prepays per vintage to produce a known CPR.
- Seeded RNG for full determinism.
- Unit tests assert metric values with relative tolerance `1e-9` for stock metrics (cum loss, pool factor) and `1e-6` for annualized rate metrics (CPR/CDR) to absorb the `1−(1−x)^12` float drift.
- Explicit fixtures for: (a) a loan that cures after crossing the DPD threshold (verifies stickiness), (b) a default-month with nonzero `principal_payment` (verifies prepay numerator exclusion), (c) a vintage that doesn't reach the max MOB (verifies `n_vintages` reporting).

---

## 7. Deliverables & Sequence

1. `vintage_curves/` package skeleton + `schema.py`, `tape.py`.
2. `mob.py`, `default_policy.py`, `segment.py`.
3. `metrics.py` (per-vintage curves) + tests.
4. `aggregate.py` (balance-weighted) + tests.
5. `summary.py`, `rolls.py` + tests.
6. `output.py`, `plotting.py`.
7. `analyzer.py` façade + end-to-end test.
8. `examples/` — a notebook-style script using the synthetic tape.
9. Top-level `README` section for the package.

---

## 8. Open Items / Future Work

- Net-loss curves with recovery lag modeling.
- Multi-file & Parquet loaders.
- Polars backend.
- Streamlit dashboard.
- Stratification reports (segment × segment grids).
- Statistical curve fitting (logistic / Weibull) for forecasting.
