"""Generate a deterministic synthetic loan performance tape.

Produces `sample_tape.csv` — one row per loan-month — using vendor-style
(non-canonical) column names so the SchemaMapper has real work to do.

The tape spans four cohorts (2 products x 2 FICO buckets) and six
quarterly vintages, with cohort-specific default rate, prepay rate, and
APR so each (Product, FICO bucket) slice has a distinct, hand-checkable
performance profile.

Cohorts (DR is lifetime cumulative default probability):
    Standard36 / 680-719   DR=13%  CPR_annual=8%   APR=15.99%
    Standard36 / 720-759   DR= 6%  CPR_annual=10%  APR=11.99%
    Standard60 / 680-719   DR=22%  CPR_annual=5%   APR=17.99%
    Standard60 / 720-759   DR=11%  CPR_annual=7%   APR=13.99%
"""

from __future__ import annotations

import csv
import datetime as dt
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SEED = 20260515
OUT = Path(__file__).parent / "sample_tape.csv"

VINTAGES = [
    date(2023, 1, 1),
    date(2023, 4, 1),
    date(2023, 7, 1),
    date(2023, 10, 1),
    date(2024, 1, 1),
    date(2024, 4, 1),
]
LOANS_PER_COHORT_PER_VINTAGE = 1_000
AS_OF_END = date(2026, 4, 30)  # observation horizon
STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA"]

# (product_name, term_months, fico_low, fico_high_inclusive,
#  lifetime_default_rate, annual_prepay_rate, apr)
COHORTS = [
    ("Standard36", 36, 680, 719, 0.13, 0.08, 0.1599),
    ("Standard36", 36, 720, 759, 0.06, 0.10, 0.1199),
    ("Standard60", 60, 680, 719, 0.22, 0.05, 0.1799),
    ("Standard60", 60, 720, 759, 0.11, 0.07, 0.1399),
]

# Original-balance ranges by product (longer term -> larger loans on average)
ORIG_BAL_BY_PRODUCT = {
    "Standard36": (5_000, 25_000),
    "Standard60": (10_000, 40_000),
}


def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    return date(y, m % 12 + 1, 1)


def month_end(d: date) -> date:
    nxt = add_months(date(d.year, d.month, 1), 1)
    return nxt - dt.timedelta(days=1)


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
    default_mob: int | None  # MOB at which DPD first crosses 120 (and CO recognized)
    prepay_mob: int | None   # MOB at which loan voluntarily pays off in full


def level_payment(bal: float, apr: float, n: int) -> float:
    r = apr / 12
    return bal * r / (1 - (1 + r) ** -n)


def build_loans(rng: random.Random) -> list[Loan]:
    loans: list[Loan] = []
    for v in VINTAGES:
        for (product, term, fico_lo, fico_hi,
             lifetime_dr, annual_prepay, apr) in COHORTS:
            smm = 1 - (1 - annual_prepay) ** (1 / 12)
            bal_lo, bal_hi = ORIG_BAL_BY_PRODUCT[product]
            # Default ramp window: front-loaded but wider for longer terms
            def_lo, def_hi = (6, 18) if term == 36 else (8, 30)
            for i in range(LOANS_PER_COHORT_PER_VINTAGE):
                loan_id = (
                    f"L{v.year}{v.month:02d}-{product[-2:]}"
                    f"-{fico_lo}-{i+1:04d}"
                )
                bal = round(rng.uniform(bal_lo, bal_hi), 2)
                fico = rng.randint(fico_lo, fico_hi)
                dti = round(rng.uniform(0.10, 0.45), 3)
                income = round(rng.uniform(40_000, 150_000), 0)
                st = rng.choice(STATES)

                default_mob: int | None = None
                prepay_mob: int | None = None

                if rng.random() < lifetime_dr:
                    default_mob = rng.randint(def_lo, min(def_hi, term - 1))
                else:
                    # Independent prepay risk via geometric draw from SMM.
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
    """Walk monthly cashflows for one loan up to AS_OF_END."""
    rows: list[dict] = []
    pmt = level_payment(loan.orig_bal, loan.apr, loan.term)
    r = loan.apr / 12
    bal = loan.orig_bal
    dpd = 0

    for mob in range(1, loan.term + 1):
        period_end = add_months(loan.orig_dt, mob) - dt.timedelta(days=1)
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

        # Default handling: DPD ramps to 120 by default_mob, then charge-off
        if (loan.default_mob is not None
                and mob >= loan.default_mob - 3
                and mob <= loan.default_mob):
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
                    dict(
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
                )
                break
            end_bal = round(bal, 2)  # no paydown while delinquent
        elif loan.prepay_mob is not None and mob == loan.prepay_mob:
            principal_paid = round(bal, 2)
            interest_paid = interest
            end_bal = 0.0
            dpd = 0
            rows.append(
                dict(
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
                    ChargeOffDt="",
                    ChargeOffAmt=0.0,
                )
            )
            break
        else:
            end_bal = round(bal - sched_prin, 2)
            dpd = 0

        rows.append(
            dict(
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
                ChargeOffDt="",
                ChargeOffAmt=0.0,
            )
        )

        bal = end_bal
        if bal <= 0.005:
            break

    return rows


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
    by_cohort: dict[tuple[str, int], int] = {}
    for l in loans:
        bucket = (l.product, 680 if l.fico < 720 else 720)
        by_cohort[bucket] = by_cohort.get(bucket, 0) + 1
    for k, v in sorted(by_cohort.items()):
        print(f"  cohort {k[0]} / FICO {k[1]}-{k[1]+39}: {v} loans")


if __name__ == "__main__":
    main()
