"""Generate a synthetic loan performance tape with 4 cohorts x 6 vintages.

Cohorts (segmentation dimensions):
  - Term: 36mo, 60mo
  - FICO: 680-719, 720-759

Vintages: 2024Q1 .. 2025Q2 (six quarterly origination dates).
750 loans per cohort-vintage cell -> 24 cells -> 18,000 loans total.

Default behavior interacts with both FICO and term:
  - Lower FICO has ~2x the lifetime default rate of higher FICO.
  - 60mo term has higher lifetime default than 36mo (more time to default).
  - Prepay is roughly FICO/term-neutral with mild differences.

Output: examples/cohort_tape.csv (vendor-style column names; mapped via
examples/schema.yaml).
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

SEED = 20260519
OUT = Path(__file__).parent / "cohort_tape.csv"

VINTAGES = [
    date(2024, 1, 1),
    date(2024, 4, 1),
    date(2024, 7, 1),
    date(2024, 10, 1),
    date(2025, 1, 1),
    date(2025, 4, 1),
]
LOANS_PER_CELL = 750
AS_OF_END = date(2026, 4, 30)
STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA"]

# Cohort definitions: (term_months, fico_low, fico_high_inclusive)
COHORTS = [
    (36, 680, 719),
    (36, 720, 759),
    (60, 680, 719),
    (60, 720, 759),
]

# APR depends on credit/term (lower FICO + longer term => higher rate).
APR_BY_COHORT = {
    (36, 680, 719): 0.1799,
    (36, 720, 759): 0.1399,
    (60, 680, 719): 0.1999,
    (60, 720, 759): 0.1599,
}

# Lifetime default probability per cohort. FICO + term both matter.
LIFETIME_DEFAULT_BY_COHORT = {
    (36, 680, 719): 0.12,
    (36, 720, 759): 0.06,
    (60, 680, 719): 0.18,
    (60, 720, 759): 0.09,
}

# Annualized voluntary prepay rate per cohort (mild FICO/term effect).
PREPAY_ANNUAL_BY_COHORT = {
    (36, 680, 719): 0.05,
    (36, 720, 759): 0.07,
    (60, 680, 719): 0.06,
    (60, 720, 759): 0.08,
}

# Vintage multipliers — let later vintages show modestly worse default
# experience so segmentation has something interesting to show.
VINTAGE_DEFAULT_MULT = {
    date(2024, 1, 1): 0.90,
    date(2024, 4, 1): 1.00,
    date(2024, 7, 1): 1.05,
    date(2024, 10, 1): 1.10,
    date(2025, 1, 1): 1.15,
    date(2025, 4, 1): 1.20,
}


def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    return date(y, m % 12 + 1, 1)


def month_end(d: date) -> date:
    nxt = add_months(date(d.year, d.month, 1), 1)
    return nxt - timedelta(days=1)


def level_payment(bal: float, apr: float, n: int) -> float:
    r = apr / 12
    return bal * r / (1 - (1 + r) ** -n)


@dataclass
class Loan:
    loan_id: str
    orig_dt: date
    orig_bal: float
    apr: float
    term: int
    fico: int
    dti: float
    income: float
    state: str
    product: str
    default_mob: int | None
    prepay_mob: int | None


def build_loans(rng: random.Random) -> list[Loan]:
    loans: list[Loan] = []
    for cohort in COHORTS:
        term, fico_lo, fico_hi = cohort
        apr = APR_BY_COHORT[cohort]
        prepay_annual = PREPAY_ANNUAL_BY_COHORT[cohort]
        smm = 1 - (1 - prepay_annual) ** (1 / 12)
        base_default = LIFETIME_DEFAULT_BY_COHORT[cohort]

        for v in VINTAGES:
            dr = min(0.40, base_default * VINTAGE_DEFAULT_MULT[v])
            # Defaults seasoning window scales with term.
            default_window = (6, min(term - 3, 18 if term == 36 else 30))

            for i in range(LOANS_PER_CELL):
                loan_id = f"L{v.year}{v.month:02d}-T{term}-F{fico_lo}-{i+1:04d}"
                # Balance scales modestly with term.
                if term == 36:
                    bal = round(rng.uniform(8_000, 25_000), 2)
                else:
                    bal = round(rng.uniform(15_000, 40_000), 2)
                fico = rng.randint(fico_lo, fico_hi)
                dti = round(rng.uniform(0.10, 0.45), 3)
                income = round(rng.uniform(40_000, 150_000), 0)
                st = rng.choice(STATES)
                product = "PersonalLoan"

                default_mob: int | None = None
                prepay_mob: int | None = None

                if rng.random() < dr:
                    default_mob = rng.randint(default_window[0], default_window[1])
                else:
                    for m in range(1, term):
                        if rng.random() < smm:
                            prepay_mob = m
                            break

                loans.append(
                    Loan(
                        loan_id=loan_id,
                        orig_dt=v,
                        orig_bal=bal,
                        apr=apr,
                        term=term,
                        fico=fico,
                        dti=dti,
                        income=income,
                        state=st,
                        product=product,
                        default_mob=default_mob,
                        prepay_mob=prepay_mob,
                    )
                )
    return loans


def perf_rows(loan: Loan) -> list[dict]:
    rows: list[dict] = []
    pmt = level_payment(loan.orig_bal, loan.apr, loan.term)
    r = loan.apr / 12
    bal = loan.orig_bal
    dpd = 0

    for mob in range(1, loan.term + 1):
        period_end = month_end(add_months(loan.orig_dt, mob - 1))
        if period_end > AS_OF_END:
            break

        interest = round(bal * r, 2)
        sched_prin = round(pmt - interest, 2)
        if sched_prin > bal:
            sched_prin = round(bal, 2)

        principal_paid = sched_prin
        interest_paid = interest
        fees = 0.0
        co_date = ""
        co_amt = 0.0

        if (
            loan.default_mob is not None
            and mob >= loan.default_mob - 3
            and mob <= loan.default_mob
        ):
            ramp = mob - (loan.default_mob - 3)
            dpd = min(120, 30 * (ramp + 1))
            principal_paid = 0.0
            interest_paid = 0.0
            sched_prin = round(pmt - round(bal * r, 2), 2)
            if mob == loan.default_mob:
                co_date = period_end.isoformat()
                co_amt = round(bal, 2)
                end_bal = 0.0
                rows.append(
                    _row(loan, period_end, end_bal, sched_prin, principal_paid,
                         interest_paid, fees, dpd, co_date, co_amt)
                )
                break
            end_bal = round(bal, 2)
        elif loan.prepay_mob is not None and mob == loan.prepay_mob:
            principal_paid = round(bal, 2)
            interest_paid = interest
            end_bal = 0.0
            dpd = 0
            rows.append(
                _row(loan, period_end, end_bal, sched_prin, principal_paid,
                     interest_paid, fees, dpd, "", 0.0)
            )
            break
        else:
            end_bal = round(bal - sched_prin, 2)
            dpd = 0

        rows.append(
            _row(loan, period_end, end_bal, sched_prin, principal_paid,
                 interest_paid, fees, dpd, "", 0.0)
        )

        bal = end_bal
        if bal <= 0.005:
            break

    return rows


def _row(loan: Loan, period_end: date, end_bal: float, sched_prin: float,
         principal_paid: float, interest_paid: float, fees: float, dpd: int,
         co_date: str, co_amt: float) -> dict:
    return dict(
        LoanNumber=loan.loan_id,
        OrigDt=loan.orig_dt.isoformat(),
        OrigUPB=loan.orig_bal,
        NoteRate=loan.apr,
        OrigTerm=loan.term,
        Product=loan.product,
        BorrowerFICO=loan.fico,
        BorrowerDTI=loan.dti,
        BorrowerIncome=loan.income,
        BorrowerState=loan.state,
        ReportingPeriod=period_end.isoformat(),
        EndingBalance=end_bal,
        SchedPrincipal=sched_prin,
        PrincipalPaid=principal_paid,
        InterestPaid=interest_paid,
        FeesPaid=fees,
        DPD=dpd,
        ChargeOffDt=co_date,
        ChargeOffAmt=co_amt,
    )


def main() -> None:
    rng = random.Random(SEED)
    loans = build_loans(rng)

    fieldnames = [
        "LoanNumber", "OrigDt", "OrigUPB", "NoteRate", "OrigTerm", "Product",
        "BorrowerFICO", "BorrowerDTI", "BorrowerIncome", "BorrowerState",
        "ReportingPeriod", "EndingBalance", "SchedPrincipal",
        "PrincipalPaid", "InterestPaid", "FeesPaid",
        "DPD", "ChargeOffDt", "ChargeOffAmt",
    ]

    n_rows = 0
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for loan in loans:
            for row in perf_rows(loan):
                w.writerow(row)
                n_rows += 1

    n_default = sum(1 for l in loans if l.default_mob is not None)
    n_prepay = sum(1 for l in loans if l.prepay_mob is not None)
    print(f"Wrote {OUT} — {len(loans)} loans, {n_rows} loan-months")
    print(f"  defaults: {n_default}, voluntary prepays: {n_prepay}")
    print(f"  cohorts: {len(COHORTS)}, vintages: {len(VINTAGES)}, "
          f"loans per cell: {LOANS_PER_CELL}")


if __name__ == "__main__":
    main()
