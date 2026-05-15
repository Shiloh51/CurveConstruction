"""Generate a deterministic synthetic loan performance tape.

Produces `sample_tape.csv` — one row per loan-month — using vendor-style
(non-canonical) column names so the SchemaMapper has real work to do when the
plan is implemented. Three quarterly vintages with known default + prepay
behavior so cum-loss / CPR / CDR can be sanity-checked by hand.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SEED = 20260515
OUT = Path(__file__).parent / "sample_tape.csv"

VINTAGES = [date(2024, 1, 1), date(2024, 4, 1), date(2024, 7, 1)]
LOANS_PER_VINTAGE = 11_200
TERM_MONTHS = 36
APR = 0.1499
AS_OF_END = date(2026, 4, 30)  # observation horizon
STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA"]

# Per-vintage scenarios: (default_rate, annualized_prepay_rate)
SCENARIOS = {
    date(2024, 1, 1): (0.10, 0.06),
    date(2024, 4, 1): (0.15, 0.04),
    date(2024, 7, 1): (0.05, 0.08),
}


def month_end(y: int, m: int) -> date:
    if m == 12:
        return date(y, 12, 31)
    nxt = date(y, m + 1, 1)
    return date(nxt.year, nxt.month, 1).replace(day=1).fromordinal(nxt.toordinal() - 1)


def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    return date(y, m % 12 + 1, 1)


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
    # behavior
    default_mob: int | None  # MOB at which DPD first crosses 120 (and CO recognized)
    prepay_mob: int | None   # MOB at which loan voluntarily pays off in full


def level_payment(bal: float, apr: float, n: int) -> float:
    r = apr / 12
    return bal * r / (1 - (1 + r) ** -n)


def build_loans(rng: random.Random) -> list[Loan]:
    loans: list[Loan] = []
    for v in VINTAGES:
        dr, prepay_annual = SCENARIOS[v]
        smm = 1 - (1 - prepay_annual) ** (1 / 12)
        for i in range(LOANS_PER_VINTAGE):
            loan_id = f"L{v.year}{v.month:02d}-{i+1:03d}"
            bal = round(rng.uniform(8_000, 25_000), 2)
            fico = rng.randint(620, 800)
            dti = round(rng.uniform(0.10, 0.45), 3)
            income = round(rng.uniform(40_000, 150_000), 0)
            st = rng.choice(STATES)
            product = "PersonalLoan"

            default_mob: int | None = None
            prepay_mob: int | None = None

            if rng.random() < dr:
                # Defaults concentrated in months 6-18 (typical front-loaded curve)
                default_mob = rng.randint(6, 18)
            else:
                # Independent prepay risk via geometric draw from SMM
                # First "success" month index, capped at TERM-1
                for m in range(1, TERM_MONTHS):
                    if rng.random() < smm:
                        prepay_mob = m
                        break

            loans.append(
                Loan(
                    loan_id=loan_id,
                    orig_dt=v,
                    orig_bal=bal,
                    apr=APR,
                    term=TERM_MONTHS,
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
        period_end = add_months(loan.orig_dt, mob) - __import__("datetime").timedelta(days=1)
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
        if loan.default_mob is not None and mob >= loan.default_mob - 3 and mob <= loan.default_mob:
            # ramp DPD: 30, 60, 90, 120
            ramp = mob - (loan.default_mob - 3)
            dpd = min(120, 30 * (ramp + 1))
            principal_paid = 0.0
            interest_paid = 0.0
            sched_prin = round(pmt - round(bal * r, 2), 2)  # what was scheduled
            if mob == loan.default_mob:
                # Charge off: write down remaining balance
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
            # Voluntary full prepay
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


if __name__ == "__main__":
    main()
