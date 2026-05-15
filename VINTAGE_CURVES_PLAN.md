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
- Configurable default definition: `Charge-Off Date` field **OR** a DPD threshold.
- Monthly **or** quarterly vintage grouping, anchored on origination date.
- MOB = months since origination.
- Per-vintage curves **and** aggregate curves across vintages, with two weighting schemes (original balance / beginning balance at MOB).
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
    vintage_freq="Q",                       # "M" or "Q"
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
#            "beginning" uses begin-of-period balance at each MOB.
agg_orig = seg.aggregate(metrics=["cum_gross_loss", "smm", "cdr"], weighting="original")
agg_bop  = seg.aggregate(metrics=["cum_gross_loss", "smm", "cdr"], weighting="beginning")

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
Let `bal_{l,t}` be outstanding balance for loan l at MOB t.

### Per-vintage curves
- **Cumulative gross loss %** `cum_gross_loss_{v,t} = (Σ_{l∈v, mob≤t} charge_off_amount_l) / OB_v`
- **Cumulative prepayments %** `cum_prepay_{v,t} = (Σ unscheduled principal) / OB_v` where unscheduled = `principal_payment − scheduled_principal` (floored at 0).
- **SMM (Single Monthly Mortality)** at MOB t for vintage v:
  `SMM_{v,t} = unscheduled_principal_{v,t} / (begin_bal_{v,t} − scheduled_principal_{v,t})`
- **CPR (annualized)** `CPR_{v,t} = 1 − (1 − SMM_{v,t})^12`
- **CDR (annualized)** Default $ in MOB t divided by begin-of-period scheduled balance, then `1 − (1 − MDR)^12`.
- **Pool Factor** `pool_factor_{v,t} = (Σ bal_{l,t}) / OB_v`
- **WA Coupon** balance-weighted `apr` at MOB t (matrix or single per-vintage value)
- **WAL** (weighted average life, per vintage, over realized cashflows): `Σ t · principal_received_t / Σ principal_received_t`, in months.

### Default policy
- `DefaultPolicy.charge_off_field()`: a loan defaults in the month its `charge_off_date` falls; loss = `charge_off_amount`.
- `DefaultPolicy.dpd_threshold(n)`: a loan defaults the first MOB where `days_past_due ≥ n`; loss = outstanding_balance at that month (gross). Subsequent months for that loan are excluded from active denominators.

### Aggregate across vintages (two weighting schemes)

For metric M at MOB t, restricted to vintages where MOB t is observable:

- **`weighting="original"` (default)** — weight each vintage by its original balance `OB_v`.
  - Stock metrics (cum loss %, cum prepay %, pool factor): `M_agg_t = Σ_v OB_v · M_{v,t} / Σ_v OB_v`.
  - Flow / rate metrics (SMM, CPR, MDR, CDR): aggregate raw $ numerator and $ denominator separately, then form the ratio — equivalent to dollar-weighting by original balance contributions.
  - Interpretation: "the average originated dollar's lifetime experience" — keeps each vintage's footprint proportional to how big it was at origination.

- **`weighting="beginning"`** — weight each vintage at MOB t by its **beginning-of-period balance** at that MOB.
  - Stock metrics: `M_agg_t = Σ_v BOP_{v,t} · M_{v,t} / Σ_v BOP_{v,t}`.
  - Flow / rate metrics: numerator = Σ_v flow$_{v,t}; denominator = Σ_v BOP_{v,t} (or `BOP − scheduled_principal` for SMM, consistent with the per-vintage definition).
  - Interpretation: "the average outstanding dollar's instantaneous behavior" — reweights toward vintages that still have balance at MOB t. Useful for current-portfolio behavior; CPR/CDR comparisons can differ noticeably from the original-weighted view late in a curve.

Both views are always available; Excel export can emit each as a separate sheet.

### Truncation
For each vintage v with last `as_of_date = T_v`, the maximum observable MOB is
`(T_v − origination_month_v)` in months. Cells beyond `mob_max_v` are NaN
(and dropped from aggregate denominators for that MOB).

### Delinquency buckets & roll rates
Buckets: `Current` (DPD=0), `1-29`, `30-59`, `60-89`, `90+`, `CO`.
Transition matrix: count or balance share moving from bucket_i (MOB t) →
bucket_j (MOB t+1), normalized within row. Optionally per-MOB or pooled.

---

## 5. Implementation Notes

- **Engine**: pandas. Group operations are vectorized; one merge of loan-level
  attrs onto perf at load time, then group by `vintage` / `mob`.
- **MOB**: `mob = (as_of_date.year - orig.year) * 12 + (as_of_date.month - orig.month)`.
- **Vintage**: `vintage = orig.to_period("M" or "Q")`.
- **Memory hook**: load function accepts `usecols`, `dtype`, and `chunksize`
  parameters as future swap-in points (documented, not exercised in v1).
- **Required performance columns**: `scheduled_principal` is required — if the
  column is missing or all-NaN, schema validation raises a hard error rather
  than silently falling back to a computed level-pay schedule. (Users who
  need a fallback can precompute the column before handing the tape to the
  analyzer.)
- **Sign conventions**: payments positive, balances non-negative.
- **Plotting**: each `plot_curves` returns a `matplotlib.figure.Figure`. One
  line per vintage; aggregate overlaid in bold black when `include_agg=True`.

---

## 6. Synthetic Test Tape

`tests/synthetic.py` generates a deterministic tape:
- 3 vintages × 50 loans each, 60-month term, 12% APR.
- Configurable default rate per vintage to produce known cum-loss curves.
- A few prepays per vintage to produce a known CPR.
- Unit tests assert metric values to within 1e-6 of analytical expectations.

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
