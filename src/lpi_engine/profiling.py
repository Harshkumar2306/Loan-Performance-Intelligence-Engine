"""Data intelligence: profiling, outlier detection, association/drift checks and quality scoring.

The quality model is deliberately transparent: every row-level penalty is the sum of
named, deterministic checks with weights documented in the configuration. There is no
secret score — a reviewer can always reconstruct why a record scored badly.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from lpi_engine.data import IDENTIFIERS

# ---------------------------------------------------------------------------
# Deterministic rule engine
# ---------------------------------------------------------------------------

def _column(d: pd.DataFrame, name: str, default):
    return d[name] if name in d else pd.Series(default, index=d.index)


def _num(d: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    return pd.to_numeric(_column(d, name, default), errors="coerce")


def _months_between(a: pd.Series, b: pd.Series) -> pd.Series:
    """Whole-month distance between two timestamp series (pandas 3 no longer supports 'M' timedeltas)."""
    a, b = pd.to_datetime(a, errors="coerce"), pd.to_datetime(b, errors="coerce")
    return (b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)


RULES: dict[str, tuple] = {
    "negative_balance": (lambda d: _num(d, "current_balance") < 0, "current_balance is negative"),
    "balance_exceeds_original": (lambda d: _num(d, "current_balance") > _num(d, "original_balance", np.inf) * 1.05,
                                 "current_balance exceeds 105% of original_balance"),
    "dpd_status_break": (lambda d: (_num(d, "days_past_due") > 0)
                         & _column(d, "current_status", "").astype(str).str.contains("current", case=False, na=False),
                         "days past due > 0 while status says current"),
    "dpd_impossible": (lambda d: (_num(d, "days_past_due") < 0) | (_num(d, "days_past_due") > 720),
                       "days_past_due outside the plausible range [0, 720]"),
    "invalid_term": (lambda d: _num(d, "remaining_term_months", 1) < 0, "remaining_term_months is negative"),
    "term_implausible": (lambda d: _num(d, "remaining_term_months", 1) > 600, "remaining_term_months exceeds 600"),
    "invalid_date_order": (lambda d: pd.to_datetime(_column(d, "origination_month", pd.NaT), errors="coerce")
                           > pd.to_datetime(_column(d, "reporting_month", pd.NaT), errors="coerce"),
                           "origination_month is after reporting_month"),
    "age_date_break": (lambda d: (_months_between(pd.to_datetime(_column(d, "origination_month", pd.NaT), errors="coerce"),
                                                  pd.to_datetime(_column(d, "reporting_month", pd.NaT), errors="coerce"))
                                  - _num(d, "loan_age_months").round()).abs() > 3,
                       "loan_age_months disagrees with origination->reporting distance by >3 months"),
    "future_update": (lambda d: pd.to_datetime(_column(d, "last_updated_at", pd.NaT), errors="coerce")
                      > pd.to_datetime(_column(d, "reporting_month", pd.NaT), errors="coerce") + pd.Timedelta(days=45),
                      "last_updated_at is more than 45 days after reporting_month"),
    "prepaid_positive_balance": (lambda d: _column(d, "current_status", "").astype(str).str.contains("prepaid", case=False, na=False)
                                 & (_num(d, "current_balance") > 1),
                                 "prepaid/paid-off status but current_balance > 1"),
    "default_no_dpd": (lambda d: _column(d, "current_status", "").astype(str).str.contains("default|foreclosure", case=False, na=False)
                       & (_num(d, "days_past_due") < 60),
                       "default status but days_past_due < 60"),
    "negative_rate": (lambda d: (_num(d, "interest_rate") < 0) | (_num(d, "interest_rate") > 30),
                      "interest_rate outside the plausible range [0, 30]"),
    "missing_core": (lambda d: _num(d, "current_balance", np.nan).isna() | _column(d, "current_status", None).isna(),
                     "missing current_balance or current_status"),
    "duplicate_row": (lambda d: d.duplicated(subset=[c for c in ("loan_id", "reporting_month") if c in d.columns], keep=False)
                      if "loan_id" in d else pd.Series(False, index=d.index),
                      "duplicate loan_id + reporting_month row"),
}

RULE_WEIGHTS = {
    "negative_balance": 15, "balance_exceeds_original": 8, "dpd_status_break": 12, "dpd_impossible": 15,
    "invalid_term": 12, "term_implausible": 5, "invalid_date_order": 15, "age_date_break": 10,
    "future_update": 8, "prepaid_positive_balance": 12, "default_no_dpd": 10, "negative_rate": 8,
    "missing_core": 12, "duplicate_row": 15,
}

# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------

def psi(expected: pd.Series, actual: pd.Series, bins: int = 20) -> float:
    """Population stability index between train and test for numeric or categorical data."""
    a = pd.to_numeric(expected, errors="coerce").dropna() if pd.api.types.is_numeric_dtype(expected) else expected.dropna().astype(str)
    b = pd.to_numeric(actual, errors="coerce").dropna() if pd.api.types.is_numeric_dtype(actual) else actual.dropna().astype(str)
    if len(a) < 30 or len(b) < 30:
        return 0.0
    if pd.api.types.is_numeric_dtype(a):
        edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        ea = np.histogram(a, edges)[0] / len(a)
        eb = np.histogram(b, edges)[0] / len(b)
    else:
        cats = sorted(set(a) | set(b))
        if len(cats) > 50:
            return 0.0
        ea = np.array([(a == c).mean() for c in cats])
        eb = np.array([(b == c).mean() for c in cats])
    return float(np.sum((eb - ea) * np.log((eb + 1e-6) / (ea + 1e-6))))


def robust_outliers(df: pd.DataFrame, columns: list[str], threshold: float = 5.0) -> tuple[pd.DataFrame, dict]:
    """Flag robust-z (median/MAD) outliers per numeric column. Returns (flags frame, per-column counts)."""
    flags = pd.DataFrame(index=df.index)
    counts = {}
    for col in columns:
        x = pd.to_numeric(df[col], errors="coerce")
        valid = x.dropna()
        if len(valid) == 0:
            continue
        med = valid.median()
        diff = (valid - med).abs()
        if len(diff) == 0:
            continue
        mad = diff.median()
        if pd.isna(mad) or mad == 0:
            continue
        z = (x - med) / (1.4826 * mad)
        flags[f"outlier_{col}"] = z.abs() > threshold
        counts[col] = int(flags[f"outlier_{col}"].sum())
    return flags, counts


def cramers_v(frame: pd.DataFrame, cat_columns: list[str], sample: int = 20000, max_pairs: int = 12) -> list[dict]:
    """Cramér's V for the strongest categorical pairs (chi-square based)."""
    if len(cat_columns) < 2:
        return []
    sub = frame[cat_columns].dropna()
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=2026)
    rows = []
    for a, b in combinations(cat_columns, 2):
        table = pd.crosstab(sub[a], sub[b])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(table.values)
        n = table.values.sum()
        v = float(np.sqrt(chi2 / (n * min(table.shape[0], table.shape[1]) - 1)))
        rows.append({"pair": f"{a}__{b}", "cramers_v": round(v, 4), "chi2_p": round(float(p), 6)})
    return sorted(rows, key=lambda r: -r["cramers_v"])[:max_pairs]


def association_rules(frame: pd.DataFrame, cat_columns: list[str], sample: int = 20000,
                      max_pairs: int = 8, top_rules: int = 12) -> list[dict]:
    """Literal association rules (brief Section 3: "association rules"): top lift co-occurrences
    between categorical value pairs, e.g. `document_status=missing -> exception_required`.

    Lift = P(A & B) / (P(A) * P(B)); a lift far above 1 means the two values co-occur more
    often than independence predicts. Computed with pandas only — no extra dependency.
    """
    if len(cat_columns) < 2:
        return []
    sub = frame[cat_columns].dropna()
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=2026)
    n = len(sub)
    pairs = list(combinations(cat_columns, 2))[:max_pairs]
    rules = []
    for a, b in pairs:
        if sub[a].nunique() > 25 or sub[b].nunique() > 25:
            continue
        table = pd.crosstab(sub[a], sub[b])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        pa = sub[a].value_counts(normalize=True)
        pb = sub[b].value_counts(normalize=True)
        for va in table.index:
            for vb in table.columns:
                support = float(table.loc[va, vb] / n)
                confidence = float(table.loc[va, vb] / (sub[a] == va).sum())
                lift = confidence / float(pb[vb]) if pb[vb] > 0 else 0.0
                rules.append({"rule": f"{a}={va} -> {b}={vb}", "support": round(support, 4),
                              "confidence": round(confidence, 4), "lift": round(lift, 3)})
    return sorted(rules, key=lambda r: -r["lift"])[:top_rules]


# ---------------------------------------------------------------------------
# Source reconciliation
# ---------------------------------------------------------------------------

def source_conflicts(monthly: pd.DataFrame, updates: pd.DataFrame | None) -> pd.DataFrame:
    """Flag disagreements with the latest servicer update, without overwriting source data."""
    flags = pd.DataFrame(index=monthly.index)
    for col in ("balance_source_conflict", "status_source_conflict", "stale_record"):
        flags[col] = False
    if updates is None or updates.empty or "loan_id" not in updates or "loan_id" not in monthly:
        return flags
    latest = updates.copy()
    if "last_updated_at" in latest:
        latest["last_updated_at"] = pd.to_datetime(latest["last_updated_at"], errors="coerce")
        latest = latest.sort_values("last_updated_at").drop_duplicates("loan_id", keep="last")
    else:
        latest = latest.drop_duplicates("loan_id", keep="last")
    base_columns = [c for c in ["loan_id", "current_balance", "current_status"] if c in monthly]
    joined = monthly[base_columns].merge(latest, on="loan_id", how="left", suffixes=("", "_servicer"))
    if {"current_balance", "current_balance_servicer"}.issubset(joined):
        left = pd.to_numeric(joined.current_balance, errors="coerce")
        right = pd.to_numeric(joined.current_balance_servicer, errors="coerce")
        flags["balance_source_conflict"] = (left.sub(right).abs() > 1).fillna(False).to_numpy()
    if {"current_status", "current_status_servicer"}.issubset(joined):
        flags["status_source_conflict"] = (joined.current_status.astype(str).str.lower()
                                           != joined.current_status_servicer.astype(str).str.lower()).fillna(False).to_numpy()
    if {"last_updated_at", "reporting_month"}.issubset(monthly):
        age = pd.to_datetime(monthly.reporting_month, errors="coerce") - pd.to_datetime(monthly.last_updated_at, errors="coerce")
        flags["stale_record"] = (age.dt.days > 90).fillna(False)
    return flags


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def quality_scores(train: pd.DataFrame, outlier_flags: pd.DataFrame, conflict_flags: pd.DataFrame) -> pd.DataFrame:
    """Row-level data-quality score with fully documented weights.

    score = 100 - Σ(rule weights for breaches) - 8*min(n_outlier_flags, 3) - 10*min(n_conflicts, 2) - 20*missing_rate
    """
    quality = pd.DataFrame(index=train.index)
    for name, (check, _) in RULES.items():
        quality[name] = check(train).fillna(False).astype(bool)
    for col in outlier_flags.columns:
        quality[col] = outlier_flags[col].reindex(train.index).fillna(False).astype(bool)
    for col in conflict_flags.columns:
        quality[col] = conflict_flags[col].reindex(train.index).fillna(False).astype(bool)

    rule_penalty = sum(quality[name].astype(int) * RULE_WEIGHTS[name] for name in RULES)
    outlier_penalty = 8 * quality[[c for c in outlier_flags.columns]].sum(axis=1).clip(upper=3)
    conflict_penalty = 10 * quality[[c for c in conflict_flags.columns]].sum(axis=1).clip(upper=2)
    missing_penalty = 20 * train.isna().mean(axis=1)

    score = (100 - rule_penalty - outlier_penalty - conflict_penalty - missing_penalty).clip(0, 100)
    quality["data_quality_score"] = score.round(1)
    quality["n_breaches"] = quality[[c for c in quality.columns if c != "data_quality_score"]].sum(axis=1)
    return quality


def score_test_frame(test: pd.DataFrame, updates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Quality scores for an unlabelled scoring frame (test) using the same documented model."""
    numeric_cols = [c for c in test.select_dtypes(include="number").columns if c not in IDENTIFIERS]
    outlier_flags, _ = robust_outliers(test, numeric_cols)
    conflict_flags = source_conflicts(test, updates)
    return quality_scores(test, outlier_flags, conflict_flags)


def batch_kpi(train: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Batch-level data-health KPI per reporting month."""
    if "reporting_month" not in train:
        return pd.DataFrame()
    tmp = train[["reporting_month"]].copy()
    tmp["data_quality_score"] = quality["data_quality_score"].reindex(train.index).values
    tmp["n_breaches"] = quality["n_breaches"].reindex(train.index).values
    g = tmp.groupby("reporting_month")
    kpi = pd.DataFrame({
        "reporting_month": g["reporting_month"].first(),
        "rows": g.size(),
        "mean_quality_score": g["data_quality_score"].mean().round(2),
        "median_quality_score": g["data_quality_score"].median().round(2),
        "total_breaches": g["n_breaches"].sum(),
        "breaches_per_1k_rows": (g["n_breaches"].sum() / g.size() * 1000).round(1),
    }).reset_index(drop=True)
    return kpi.sort_values("reporting_month")


# ---------------------------------------------------------------------------
# Master profile
# ---------------------------------------------------------------------------

def profile(train: pd.DataFrame, test: pd.DataFrame, updates: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Full data-intelligence profile: rules, outliers, associations, drift, missingness and quality."""
    numeric = train.select_dtypes(include="number")
    numeric_cols = [c for c in numeric.columns if c not in IDENTIFIERS]
    categorical_cols = [c for c in train.columns
                        if c not in numeric_cols and c not in IDENTIFIERS and train[c].nunique() <= 30 and train[c].nunique() > 1]

    outlier_flags, outlier_counts = robust_outliers(train, numeric_cols)
    conflict_flags = source_conflicts(train, updates)
    quality = quality_scores(train, outlier_flags, conflict_flags)

    missing = train.isna().mean().sort_values(ascending=False)

    corr = numeric.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().sort_values(ascending=False)
    top_correlations = {f"{a}__{b}": round(float(v), 3) for (a, b), v in upper.head(15).items()}
    highly_dependent = [k for k, v in top_correlations.items() if float(v) > 0.95]

    drift_numeric = {c: round(psi(train[c], test[c]), 4) for c in numeric_cols if c in test.columns}
    drift_categorical = {c: round(psi(train[c], test[c]), 4) for c in categorical_cols if c in test.columns}
    associations = cramers_v(train, categorical_cols)
    highly_dependent += [a["pair"] for a in associations if a["cramers_v"] > 0.9]
    rules = association_rules(train, categorical_cols)

    # Missingness patterns: most common missing-column combinations.
    pattern_counts = train.isna().astype(int).groupby(list(train.isna().columns)).size() if len(train) <= 200_000 else None
    missing_patterns = {}
    if pattern_counts is not None:
        top_patterns = pattern_counts.sort_values(ascending=False).head(8)
        for cols_tuple, count in top_patterns.items():
            names = tuple(train.columns[i] for i, v in enumerate(cols_tuple) if v == 1) if isinstance(cols_tuple, tuple) else ()
            missing_patterns["+".join(names) if names else "complete_row"] = int(count)

    report = {
        "rows": int(len(train)),
        "columns": int(train.shape[1]),
        "train_months": int(train["reporting_month"].nunique()) if "reporting_month" in train else None,
        "missingness_top_15": missing.head(15).round(4).to_dict(),
        "missing_patterns_top_8": missing_patterns,
        "rule_breaches": {name: int(quality[name].sum()) for name in RULES},
        "outlier_counts": outlier_counts,
        "highest_numeric_correlations": top_correlations,
        "strongest_categorical_associations": associations,
        "top_association_rules_by_lift": rules,
        "highly_dependent_fields": sorted(set(highly_dependent)),
        "train_test_numeric_psi": drift_numeric,
        "train_test_categorical_psi": drift_categorical,
        "source_reconciliation": {c: int(conflict_flags[c].sum()) for c in conflict_flags},
        "quality_score_distribution": quality["data_quality_score"].describe().round(2).to_dict(),
        "share_rows_below_60_quality": round(float((quality["data_quality_score"] < 60).mean()), 4),
    }
    return report, quality, batch_kpi(train, quality)
