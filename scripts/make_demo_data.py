"""Generate a self-contained synthetic demo pack in `data/`.

The generator creates realistic panel dynamics — delinquency cascades, defaults,
prepayments and amortisation — so the pipeline's models have genuine signal to learn.
It then injects the exact kinds of messiness the challenge asks for: negative balances,
broken dates, missing documents, duplicate rows and conflicting servicer updates.
Targets are computed from the *true* realised states, so features can be messy while
labels stay honest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = {
    "credit_score_band": [(0, 620, "<620"), (620, 680, "620-679"), (680, 740, "680-739"), (740, 900, "740+")],
    "ltv_band": [(0, 60, "<=60"), (60, 80, "61-80"), (80, 100, "81-100"), (100, 999, "100+")],
    "dti_band": [(0, 20, "<=20"), (20, 35, "21-35"), (35, 43, "36-43"), (43, 200, "44-50")],
}


def band_of(value: float, spec) -> str:
    for lo, hi, label in spec:
        if lo <= value < hi:
            return label
    return spec[-1][2]


def hazard_to_30dpd(credit: float, ltv: float, rate: float, frailty: float = 0.0) -> float:
    """Monthly entry-into-delinquency hazard. Coefficients are calibrated to realistic
    magnitudes: base rate ~1%/month, roughly a 2-4x spread across the observed credit range.
    Observable features dominate; the unobserved frailty adds realistic residual noise."""
    z = -4.5 - 0.005 * (credit - 700) + 0.010 * (ltv - 70) + 0.08 * (rate - 6.0) + 0.35 * frailty
    return float(np.clip(1 / (1 + np.exp(-z)), 0.0005, 0.04))


def prepay_hazard(credit: float, rate: float, frailty: float = 0.0) -> float:
    z = -4.4 + 0.0035 * (credit - 700) - 0.15 * (rate - 6.0) - 0.3 * frailty
    return float(np.clip(1 / (1 + np.exp(-z)), 0.002, 0.06))


def main(rows: int):
    rng = np.random.default_rng(2026)
    root = Path("data")
    root.mkdir(exist_ok=True)
    n_loans = max(250, rows // 14)
    months = pd.date_range("2023-01-01", periods=30, freq="MS")
    states = ["CA", "TX", "FL", "NY", "IL", "GA", "WA"]
    purposes = ["purchase", "refinance", "cash_out"]
    occupancies = ["owner", "investor", "second_home"]
    properties = ["sfr", "condo", "townhouse"]
    servicers = ["Atlas", "Beacon", "Cedar"]

    # ------------------------------------------------------- static attributes
    credit = rng.integers(580, 820, n_loans)
    ltv = np.clip(rng.normal(72, 15, n_loans), 30, 115)
    dti = np.clip(rng.normal(33, 9, n_loans), 8, 55)
    origination = rng.choice(months[:12], n_loans)
    static = pd.DataFrame({
        "loan_id": [f"L{i:06}" for i in range(n_loans)],
        "credit_score": credit,
        "credit_score_band": [band_of(c, BANDS["credit_score_band"]) for c in credit],
        "ltv": ltv.round(1),
        "ltv_band": [band_of(v, BANDS["ltv_band"]) for v in ltv],
        "dti": dti.round(1),
        "dti_band": [band_of(v, BANDS["dti_band"]) for v in dti],
        "state": rng.choice(states, n_loans),
        "loan_purpose": rng.choice(purposes, n_loans, p=[0.5, 0.35, 0.15]),
        "occupancy_type": rng.choice(occupancies, n_loans, p=[0.85, 0.1, 0.05]),
        "property_type": rng.choice(properties, n_loans, p=[0.7, 0.2, 0.1]),
        "original_balance": rng.integers(80_000, 650_000, n_loans),
        "interest_rate": np.clip(np.round(np.clip(rng.normal(6.0, 1.2, n_loans) - 0.008 * (credit - 700), 2.5, 10) * 4) / 4, 2.5, 10),
        "term_months": rng.choice([180, 240, 360], n_loans, p=[0.15, 0.25, 0.6]),
        "origination_month": origination,
    })
    static["vintage"] = pd.to_datetime(static.origination_month).dt.year

    # ------------------------------------------------------- monthly dynamics
    panel_rows = []
    loan_ids = static.loan_id.tolist()
    credits = dict(zip(static.loan_id, static.credit_score))
    ltvs = dict(zip(static.loan_id, static.ltv))
    rates = dict(zip(static.loan_id, static.interest_rate))
    terms = dict(zip(static.loan_id, static.term_months))
    originals = dict(zip(static.loan_id, static.original_balance))
    originations = dict(zip(static.loan_id, static.origination_month))
    servicer_map = dict(zip(static.loan_id, rng.choice(servicers, n_loans)))
    # Unobserved per-loan frailty: the realistic source of irreducible prediction error.
    frailties = dict(zip(static.loan_id, rng.normal(0, 1.1, n_loans)))

    for loan_id in loan_ids:
        orig = pd.Timestamp(originations[loan_id])
        eligible = [m for m in months if m >= orig]
        if len(eligible) < 4:
            continue
        # Randomly censor the loan's observation window to create censored panels.
        stop = len(eligible) if rng.random() > 0.12 else rng.integers(4, len(eligible) + 1)
        state, dpd = "Current", 0
        balance = float(originals[loan_id])
        prev_balance = balance
        for i, month in enumerate(eligible[:stop]):
            age = i
            # Unobservable per-loan-month shocks make outcomes genuinely stochastic:
            # no feature set can pin the label down deterministically.
            shock = float(rng.uniform(0.3, 2.5))
            if state == "Current":
                if rng.random() < 0.0008 * shock:  # rare unobservable income shock
                    state, dpd = "Default", 120
                elif rng.random() < hazard_to_30dpd(credits[loan_id], ltvs[loan_id], rates[loan_id], frailties[loan_id]) * shock:
                    state, dpd = "30 DPD", 30
                elif rng.random() < prepay_hazard(credits[loan_id], rates[loan_id], frailties[loan_id]) * shock:
                    state, dpd = "Prepaid", 0
            elif state in ("30 DPD", "60 DPD", "90 DPD"):
                # Direct default hazard from any delinquent stage. Observable risk factors
                # (credit, LTV, rate) dominate the unobserved frailty, so a model trained on
                # features alone has genuine cross-loan signal — the loan-disjoint ablation
                # must land clearly above the 0.5 coin-toss line.
                p_default = float(np.clip(0.06 * np.exp(
                    0.45 * frailties[loan_id] + 0.008 * (720 - credits[loan_id])
                    + 0.012 * (ltvs[loan_id] - 70) + 0.10 * (rates[loan_id] - 6.0)), 0.008, 0.30)) * shock
                r = rng.random()
                if r < p_default:
                    state, dpd = "Default", 120
                elif r < p_default + 0.5:
                    state, dpd = "Current", 0
                else:
                    dpd = min(dpd + 30, 90)
                    state = f"{dpd} DPD"
            elif state in ("Default", "Prepaid"):
                pass
            # Amortisation
            if state not in ("Default", "Prepaid"):
                payment = float(originals[loan_id]) / terms[loan_id]
                balance = max(balance - payment * (1 + rng.normal(0, 0.15)), 0)
                if balance <= 1:
                    state, dpd = "Prepaid", 0
            panel_rows.append({
                "loan_id": loan_id,
                "reporting_month": month,
                "origination_month": orig,
                "month_index": age,
                "loan_age_months": age,
                "remaining_term_months": max(terms[loan_id] - age, 0),
                "original_balance": originals[loan_id],
                "current_balance": round(balance, 2),
                "interest_rate": rates[loan_id],
                # Credit scores refresh over time in real data: the observed score is the
                # true score plus monthly refresh noise, so it is NOT a constant loan key.
                "credit_score": int(np.clip(credits[loan_id] + rng.normal(0, 12), 560, 840)),
                # Current LTV moves with the amortised balance (property value held fixed).
                "ltv": round(ltvs[loan_id] * (balance / originals[loan_id]), 1),
                "servicer_name": servicer_map[loan_id],
                "current_status": state,
                "days_past_due": dpd,
                "document_status": rng.choice(["complete", "missing", "pending"], p=[0.90, 0.03, 0.07]),
                "last_updated_at": month + pd.Timedelta(days=int(rng.integers(1, 20))),
                "source_system": "core",
            })
            prev_balance = balance
    panel = pd.DataFrame(panel_rows)
    panel = panel[panel.origination_month.notna()]
    if len(panel) > rows:
        panel = panel.sample(rows, random_state=2026).sort_values(["loan_id", "reporting_month"])

    # ------------------------------------------------------- targets from truth
    def lookahead_flag(loan_series: pd.Series, window: int, condition) -> list:
        out = []
        values = loan_series.tolist()
        for i in range(len(values)):
            look = values[i + 1: i + 1 + window]
            out.append(1 if any(condition(v) for v in look) else 0)
        return out

    g = panel.sort_values(["loan_id", "reporting_month"]).groupby("loan_id", sort=False)
    panel["next_3m_delinquency_flag"] = g["days_past_due"].transform(lambda s: lookahead_flag(s, 3, lambda v: v > 0))
    panel["next_6m_delinquency_flag"] = g["days_past_due"].transform(lambda s: lookahead_flag(s, 6, lambda v: v > 0))
    panel["next_12m_default_flag"] = g["current_status"].transform(lambda s: lookahead_flag(s, 12, lambda v: v == "Default"))
    panel["next_12m_prepayment_flag"] = g["current_status"].transform(lambda s: lookahead_flag(s, 12, lambda v: v == "Prepaid"))
    next_state = g["current_status"].shift(-1)
    panel["next_state"] = next_state.astype(str).str.lower().str.replace(" ", "_")
    panel.loc[panel.next_state.isna(), "next_state"] = np.nan

    # Targets are only defined for loans still at risk: a loan already in Default has
    # trivially defaulted, and one already Prepaid has trivially prepaid. Setting the
    # corresponding target to NaN (and documenting it) mirrors how these flags are used
    # in production and stops absorbed rows from inflating model metrics.
    panel.loc[panel.current_status == "Default",
              ["next_3m_delinquency_flag", "next_6m_delinquency_flag", "next_12m_default_flag"]] = np.nan
    panel.loc[panel.current_status == "Prepaid", "next_12m_prepayment_flag"] = np.nan

    # ------------------------------------------------------- inject messiness
    n = len(panel)
    neg_idx = rng.choice(panel.index, max(5, n // 300), replace=False)
    panel.loc[neg_idx, "current_balance"] = -25
    fut_idx = rng.choice(panel.index, max(5, n // 400), replace=False)
    panel.loc[fut_idx, "last_updated_at"] = panel.loc[fut_idx, "reporting_month"] + pd.Timedelta(days=120)
    drop_idx = rng.choice(panel.index, max(5, n // 250), replace=False)
    panel.loc[drop_idx, "current_balance"] = np.nan
    panel.loc[drop_idx[: max(1, len(drop_idx) // 2)], "current_status"] = np.nan
    panel.loc[rng.choice(panel.index, max(3, n // 600), replace=False), "days_past_due"] = -10

    # ------------------------------------------------------- exceptions (targets)
    panel["exception_type"] = "none"
    panel.loc[neg_idx, "exception_type"] = "data_break"
    panel.loc[panel.document_status == "missing", "exception_type"] = "missing_docs"
    panel.loc[panel.current_status.isna(), "exception_type"] = "data_break"

    # ------------------------------------------------------- observation noise
    # Reporting errors on *features only*: targets above were computed from the true
    # states, so models have realistic, imperfect signal to learn from. Absorbing states
    # (Default / Prepaid) are never flipped by status or DPD noise: a defaulted or paid-off
    # loan cannot silently "cure" in a servicer file, and flipping them would pollute the
    # transition matrix with impossible default -> current transitions.
    active_mask = ~panel.current_status.isin(["Default", "Prepaid"])
    status_noise = rng.choice(panel.index[active_mask], max(60, int(active_mask.sum()) // 12), replace=False)
    panel.loc[status_noise, "current_status"] = np.where(
        panel.loc[status_noise, "current_status"] == "Current", "Delinquent", "Current")
    dpd_noise = rng.choice(panel.index[active_mask], max(60, int(active_mask.sum()) // 20), replace=False)
    panel.loc[dpd_noise, "days_past_due"] = (
        panel.loc[dpd_noise, "days_past_due"] + rng.choice([-45, -30, -15, 15, 30, 45], len(dpd_noise))).clip(lower=0)
    balance_noise = rng.choice(panel.index, max(40, n // 30), replace=False)
    panel.loc[balance_noise, "current_balance"] = (
        panel.loc[balance_noise, "current_balance"] * rng.uniform(0.85, 1.15, len(balance_noise))).round(2)

    # ------------------------------------------------------- servicer conflicts
    conflict_loans = rng.choice(panel.loan_id.unique(), max(20, int(panel.loan_id.nunique() * 0.12)), replace=False)
    updates_rows = []
    for loan_id in conflict_loans:
        loan_rows = panel[panel.loan_id == loan_id]
        latest = loan_rows.sort_values("reporting_month").iloc[-1]
        balance_shift = rng.normal(0, 2500)
        status_shift = rng.choice([None, "Current", "Delinquent"])
        updates_rows.append({
            "loan_id": loan_id,
            "current_balance": round(latest.current_balance + balance_shift, 2),
            "current_status": status_shift or latest.current_status,
            "last_updated_at": latest.reporting_month + pd.Timedelta(days=int(rng.integers(1, 40))),
            "source_system": "servicer",
        })
    servicer_updates = pd.DataFrame(updates_rows)
    panel.loc[panel.loan_id.isin(servicer_updates.loan_id), "exception_type"] = \
        panel.loc[panel.loan_id.isin(servicer_updates.loan_id), "exception_type"].replace("none", "source_conflict")
    # Real exception workflows are noisy: not every genuine issue is flagged, and some
    # flags are raised on clean records. Computed after all issue sources are known.
    true_exception = (panel.exception_type != "none")
    panel["exception_required"] = np.where(
        rng.random(len(panel)) < 0.02, 1,
        np.where(true_exception & (rng.random(len(panel)) < 0.85), 1, 0)).astype(int)

    # Duplicate loan-month rows injected last so earlier positional indices stay valid.
    # Duplicates are injected into the TRAIN partition only: the profiling engine must flag
    # them, while the submission file should never carry duplicated scoring rows.
    cutoff_for_dups = pd.Timestamp("2025-01-01")
    train_idx = panel.index[panel.reporting_month < cutoff_for_dups]
    dup_idx = rng.choice(train_idx, max(4, len(train_idx) // 500), replace=False)
    panel = pd.concat([panel, panel.loc[dup_idx]], ignore_index=True)

    # ------------------------------------------------------- split and write
    cutoff = pd.Timestamp("2025-01-01")
    train = panel[panel.reporting_month < cutoff]
    test = panel[panel.reporting_month >= cutoff]
    drop_targets = [c for c in panel.columns if c.startswith("next_") or c.startswith("exception_")]
    train.to_csv(root / "loan_monthly_performance_train.csv", index=False)
    test.drop(columns=drop_targets).to_csv(root / "loan_monthly_performance_test.csv", index=False)
    static.to_csv(root / "loan_static_attributes.csv", index=False)
    servicer_updates.to_csv(root / "servicer_updates.csv", index=False)

    pd.DataFrame([
        {"scenario": "Base", "credit_score_shock": 0.0, "prepayment_uplift": 0.0},
        {"scenario": "Adverse Credit", "credit_score_shock": -35.0, "prepayment_uplift": -0.01},
        {"scenario": "High Prepayment", "credit_score_shock": 0.0, "prepayment_uplift": 0.06},
    ]).to_csv(root / "macro_scenarios.csv", index=False)

    (root / "data_dictionary.md").write_text(
        "# Data dictionary\n"
        "- `loan_id`: unique loan identifier.\n"
        "- `reporting_month`: calendar month the row describes.\n"
        "- `origination_month`: month the loan was originated.\n"
        "- `loan_age_months`: months since origination.\n"
        "- `remaining_term_months`: contractual months remaining.\n"
        "- `original_balance`: principal at origination.\n"
        "- `current_balance`: unpaid principal at the reporting month.\n"
        "- `interest_rate`: note rate in percent.\n"
        "- `credit_score`: borrower credit score at observation.\n"
        "- `credit_score_band`: discretised credit score.\n"
        "- `ltv_band`: loan-to-value band at origination.\n"
        "- `dti_band`: debt-to-income band at origination.\n"
        "- `state`: property state.\n"
        "- `loan_purpose`: purchase / refinance / cash-out.\n"
        "- `occupancy_type`: owner / investor / second home.\n"
        "- `property_type`: single-family / condo / townhouse.\n"
        "- `servicer_name`: servicing entity.\n"
        "- `current_status`: performing status or delinquency stage.\n"
        "- `days_past_due`: days past due at the reporting month.\n"
        "- `document_status`: completeness of loan documents.\n"
        "- `last_updated_at`: timestamp of the last system update.\n"
        "- `source_system`: originating system of the row.\n"
    )
    (root / "validation_rules.json").write_text(json.dumps({
        "rules": [
            "current_balance >= 0",
            "origination_month <= reporting_month",
            "prepaid status requires near-zero balance",
            "days_past_due in [0, 720]",
            "remaining_term_months in [0, 600]",
            "interest_rate in [0, 30]",
            "no duplicate loan_id + reporting_month rows",
            "current_balance <= 105% of original_balance",
        ]
    }, indent=2))

    template_cols = [
        "loan_id", "next_3m_delinquency_prob", "next_6m_delinquency_prob",
        "next_12m_default_prob", "next_12m_prepayment_prob", "next_state",
        "next_state_confidence", "exception_type", "exception_type_confidence",
        "anomaly_score", "anomaly_reason", "top_drivers", "recommended_action", "confidence",
    ]
    pd.DataFrame(columns=template_cols).to_csv(root / "submission_template.csv", index=False)

    print(f"Wrote synthetic pack to {root}/")
    print(f"  train={len(train):,} rows ({train.loan_id.nunique():,} loans) | test={len(test):,} rows "
          f"| servicer updates={len(servicer_updates):,} | static={len(static):,}")
    print(f"  event rates (train): default {train.next_12m_default_flag.mean():.3f} | "
          f"delinquency {train.next_3m_delinquency_flag.mean():.3f} | prepay {train.next_12m_prepayment_flag.mean():.3f} "
          f"| exceptions {train.exception_required.mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=12000)
    main(parser.parse_args().rows)
