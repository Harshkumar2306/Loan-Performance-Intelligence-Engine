"""Data ingestion, schema gates, time-aware splitting and deterministic feature engineering.

Design rules
------------
* Targets and post-event flags are categorically excluded from model features.
  `assert_no_leakage` is a hard gate: if a banned column survives into the
  feature set, the pipeline fails instead of silently training on future data.
* Engineered features are contemporaneously observable only. They use the
  current row and strictly historical rows of the same loan. Nothing uses the
  row's own target or any future monthly observation.
* Band fields (credit / LTV / DTI) are parsed into documented numeric midpoints
  so models can exploit ordinal structure without trusting exact values.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
    "next_state",
    "exception_required",
    "exception_type",
]
# Outcome columns that are realised after the observation date and therefore leak.
POST_EVENT_COLUMNS = {"default_flag", "prepayment_flag", "loss_severity_band", "modification_flag"}
IDENTIFIERS = {"loan_id", "reporting_month", "month_index", "last_updated_at", "origination_month"}

CORE_FEATURES = [
    "loan_age_months",
    "remaining_term_months",
    "original_balance",
    "current_balance",
    "interest_rate",
    "credit_score",
    "days_past_due",
    "credit_score_band",
    "ltv_band",
    "dti_band",
    "state",
    "loan_purpose",
    "occupancy_type",
    "property_type",
    "servicer_name",
    "current_status",
    "document_status",
    "source_system",
]

# Engineered, contemporaneously observable features.
ENGINEERED_FEATURES = [
    "credit_score_numeric",
    "ltv_numeric",
    "dti_numeric",
    "balance_ratio",
    "term_ratio",
    "dpd_velocity",
    "ever_dpd",
    "paydown_rate",
    "vintage_year",
]

FEATURE_DOC = {
    "credit_score_numeric": "Credit-score band parsed to its documented midpoint (ordinal fallback when the raw score is absent).",
    "ltv_numeric": "Loan-to-value band parsed to its documented midpoint.",
    "dti_numeric": "Debt-to-income band parsed to its documented midpoint.",
    "balance_ratio": "current_balance / original_balance, clipped to [0, 2]. How much principal remains.",
    "term_ratio": "remaining_term_months / (loan_age_months + remaining_term_months). Share of contractual life remaining.",
    "dpd_velocity": "days_past_due minus the previous month's days_past_due for the same loan (0 for the first observation).",
    "ever_dpd": "Whether the loan has ever been past due up to and including the current row (strictly historical).",
    "paydown_rate": "(previous balance - current balance) / previous balance, 0 for the first observation.",
    "vintage_year": "Calendar year of origination, as a categorical cohort label.",
}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def read_pack(paths: dict) -> dict:
    """Load the organiser data pack from a directory. `train` and `test` are required."""
    root = Path(paths["data_dir"])
    pack: dict = {}
    for key in ("train", "test", "static", "updates", "scenarios", "template"):
        path = root / paths[key]
        if path.exists():
            pack[key] = pd.read_csv(path)
    for key in ("dictionary", "rules"):
        path = root / paths[key]
        if path.exists():
            pack[key] = json.loads(path.read_text()) if path.suffix == ".json" else path.read_text()
    missing = {"train", "test"} - set(pack)
    if missing:
        raise FileNotFoundError(f"Missing required data files: {sorted(missing)} in {root.resolve()}")
    return pack


REQUIRED_PANEL_COLUMNS = ("loan_id", "reporting_month")


def validate_schema(frame: pd.DataFrame, name: str, required: tuple = REQUIRED_PANEL_COLUMNS) -> None:
    """Fail fast with a clear message when a required panel column is missing or fully empty."""
    if frame is None or len(frame) == 0:
        raise ValueError(f"Schema gate: '{name}' is empty (0 rows). Refusing to run on an empty dataset.")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Schema gate: '{name}' is missing required column(s) {missing}. "
                         f"Columns present: {list(frame.columns)[:12]}")
    for c in required:
        if frame[c].isna().all():
            raise ValueError(f"Schema gate: '{name}' column '{c}' is entirely null.")


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names, timestamps and string columns."""
    df = frame.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    for col in ("reporting_month", "origination_month", "last_updated_at"):
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in (
        "loan_id", "state", "servicer_name", "current_status", "document_status",
        "credit_score_band", "ltv_band", "dti_band", "loan_purpose", "occupancy_type",
        "property_type", "source_system", "vintage_year",
    ):
        if col in df:
            df[col] = df[col].astype("string")
    return df


def merge_attributes(monthly: pd.DataFrame, static: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join one origination record per loan; the monthly file wins on overlapping fields."""
    if static is None or "loan_id" not in static:
        return monthly
    static = normalise(static).drop_duplicates("loan_id")
    overlap = [c for c in static.columns if c != "loan_id" and c in monthly.columns]
    return monthly.merge(static.drop(columns=overlap), on="loan_id", how="left", validate="m:1")


# ---------------------------------------------------------------------------
# Leakage gate and time-aware split
# ---------------------------------------------------------------------------

def assert_no_leakage(features: list[str], available_columns: list[str]) -> list[str]:
    """Hard gate: no target, post-event or identifier column may be used as a feature.

    Returns the columns that were *excluded* for the audit trail. Raises if an
    engineered feature name collides with a banned name.
    """
    banned = set(TARGETS) | POST_EVENT_COLUMNS | IDENTIFIERS
    excluded = [c for c in available_columns if c in banned or c.startswith("next_") or c.startswith("exception_")]
    for feature in features:
        if feature in banned or feature.startswith("next_") or feature.startswith("exception_"):
            raise ValueError(f"Leakage gate failure: banned column '{feature}' reached the feature set.")
    return sorted(set(excluded))


def time_split(train: pd.DataFrame, months: int, time_col: str = "reporting_month") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Strict contiguous out-of-time split: the last `months` distinct reporting months are validation.

    Panel loans legitimately span the boundary (the same loan keeps reporting), so loan-level
    overlap is expected and is reported as evidence rather than treated as leakage. The critical
    invariant is that no *future* monthly row ever enters training.
    """
    df = train.dropna(subset=[time_col]).sort_values(time_col).copy()
    distinct = df[time_col].drop_duplicates().sort_values()
    if months >= len(distinct):
        raise ValueError(
            f"time_split: validation_months={months} must be smaller than the number of distinct "
            f"{time_col} values ({len(distinct)}).")
    cutoff = distinct.iloc[-months]
    fit, valid = df[df[time_col] < cutoff].copy(), df[df[time_col] >= cutoff].copy()
    overlap = set(fit["loan_id"]).intersection(valid["loan_id"]) if "loan_id" in df else set()
    assert fit[time_col].max() < valid[time_col].min(), "Time split invariant broken."
    return fit, valid, {
        "cutoff": str(cutoff.date()),
        "fit_rows": len(fit),
        "validation_rows": len(valid),
        "validation_months": int(valid[time_col].nunique()),
        "loans_spanning_boundary": len(overlap),
        "method": "strict reporting-month out-of-time split; no future rows in training",
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_BAND_MIDPOINT_MAPS = {
    "credit_score_band": {"<620": 590.0, "620-679": 650.0, "680-739": 710.0, "740+": 790.0, ">=740": 790.0, "740-850": 790.0},
    "ltv_band": {"<=60": 50.0, "61-80": 70.5, "81-100": 90.5, "100+": 110.0, ">100": 110.0, "101-120": 110.0},
    "dti_band": {"<=20": 15.0, "21-35": 28.0, "36-43": 39.5, "44-50": 47.0, ">50": 55.0, "51-60": 55.0},
}


def band_midpoint(value, band_name: str) -> float:
    """Parse a band label into a documented midpoint, falling back to a generic parser."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if band_name in _BAND_MIDPOINT_MAPS and text in _BAND_MIDPOINT_MAPS[band_name]:
        return float(_BAND_MIDPOINT_MAPS[band_name][text])
    numbers = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    if len(numbers) == 2:
        return (float(numbers[0]) + float(numbers[1])) / 2.0
    if len(numbers) == 1:
        n = float(numbers[0])
        if text.strip().startswith("<") or text.strip().startswith("<="):
            return max(n - 25.0, 0.0)
        if ">" in text or text.strip().endswith("+"):
            return n + 25.0
        return n
    return np.nan


def _numeric_or_band(df: pd.DataFrame, numeric_col: str, band_col: str, band_name: str) -> pd.Series:
    """Prefer the raw numeric column; otherwise parse the band."""
    if numeric_col in df.columns:
        return pd.to_numeric(df[numeric_col], errors="coerce")
    if band_col in df.columns:
        return df[band_col].apply(lambda v: band_midpoint(v, band_name)).astype(float)
    return pd.Series(np.nan, index=df.index)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add deterministic, contemporaneously observable features. Returns (frame, new column names)."""
    out = df.copy()
    out = out.sort_values(["loan_id", "reporting_month"]).reset_index(drop=True)

    out["credit_score_numeric"] = _numeric_or_band(out, "credit_score", "credit_score_band", "credit_score_band")
    out["ltv_numeric"] = _numeric_or_band(out, "ltv", "ltv_band", "ltv_band")
    out["dti_numeric"] = _numeric_or_band(out, "dti", "dti_band", "dti_band")

    original = pd.to_numeric(out.get("original_balance", pd.Series(np.nan, index=out.index)), errors="coerce")
    current = pd.to_numeric(out.get("current_balance", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["balance_ratio"] = (current / original.replace(0, np.nan)).clip(0, 2)

    age = pd.to_numeric(out.get("loan_age_months", pd.Series(np.nan, index=out.index)), errors="coerce")
    term = pd.to_numeric(out.get("remaining_term_months", pd.Series(np.nan, index=out.index)), errors="coerce")
    denom = (age + term).replace(0, np.nan)
    out["term_ratio"] = (term / denom).clip(0, 1)

    dpd = pd.to_numeric(out.get("days_past_due", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
    group = out.groupby("loan_id", sort=False) if "loan_id" in out else None
    if group is not None:
        prev_dpd = pd.to_numeric(group["days_past_due"].shift(1), errors="coerce")
        prev_bal = pd.to_numeric(group["current_balance"].shift(1), errors="coerce")
        # First observation of a loan has no history: velocity is 0, not dpd - 0.
        velocity = (dpd - prev_dpd).clip(-500, 500)
        out["dpd_velocity"] = velocity.mask(prev_dpd.isna(), 0.0).fillna(0)
        out["ever_dpd"] = group["days_past_due"].transform(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).gt(0).cummax()).astype(int)
        out["paydown_rate"] = ((prev_bal - current) / prev_bal.replace(0, np.nan)).clip(-1, 1).fillna(0)
    else:
        out["dpd_velocity"] = 0.0
        out["ever_dpd"] = (dpd > 0).astype(int)
        out["paydown_rate"] = 0.0

    if "origination_month" in out.columns:
        out["vintage_year"] = pd.to_datetime(out["origination_month"], errors="coerce").dt.year.astype("string")
    return out, ENGINEERED_FEATURES


def engineer_panel(train: pd.DataFrame, test: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Engineer features on train+test together, then split back.

    History features (`dpd_velocity`, `ever_dpd`, `paydown_rate`) must see the loan's full
    prior history, including the rows that land in the test file — otherwise the first test
    row of every loan is scored with a false "no history" signal (train/serve skew).
    Engineering is strictly historical, so combining frames cannot leak future data.
    """
    if test is None or "loan_id" not in train or "loan_id" not in test:
        return engineer_features(train)[0], (engineer_features(test)[0] if test is not None else None)
    marker = "_lp_side"
    combined = pd.concat(
        [train.assign(**{marker: "train"}), test.assign(**{marker: "test"})], ignore_index=True)
    combined, _ = engineer_features(combined)
    train_out = combined[combined[marker] == "train"].drop(columns=[marker]).reset_index(drop=True)
    test_out = combined[combined[marker] == "test"].drop(columns=[marker]).reset_index(drop=True)
    return train_out, test_out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Approved feature list: curated contemporaneous columns plus engineered features.

    Raw columns are taken from an explicit allow-list (`CORE_FEATURES`) so that an
    unexpected target-shaped column can never enter the model through enumeration.
    Expects an already-engineered frame (`engineer_features` applied); engineering is
    deliberately NOT repeated here so 1M-row panels stay cheap.
    """
    present = [c for c in CORE_FEATURES if c in df.columns]
    present += [c for c in ENGINEERED_FEATURES if c in df.columns]
    # `credit_score_numeric` subsumes the raw column (numeric preferred, band-midpoint fallback),
    # so never feed both to the model.
    if "credit_score" in present and "credit_score_numeric" in present:
        present.remove("credit_score")
    banned = set(TARGETS) | POST_EVENT_COLUMNS | IDENTIFIERS
    features = [c for c in present if c not in banned and not c.startswith("next_") and not c.startswith("exception_")]
    return features
