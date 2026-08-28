"""Supervised modelling: time-aware evaluation, calibration, imbalance handling, multi-outcome prediction.

All models are non-LLM classical ML. Metrics are computed only on the contiguous
out-of-time validation period, never on training data.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

try:  # Optional candidate family; the pipeline never depends on it.
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover
    lgb = None
    HAS_LIGHTGBM = False

BINARY_TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
    "exception_required",
]


def _cat_to_object(X):
    """Convert nullable string columns to plain object dtype with np.nan, which sklearn's imputer can handle."""
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == "string" or X[c].dtype == object:
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([
            ("to_object", FunctionTransformer(_cat_to_object, validate=False, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
        ]), cat),
    ], remainder="drop")


def binary_metrics(y: pd.Series, p: np.ndarray, threshold: float = 0.5) -> dict:
    if y.nunique() < 2:
        return {"note": "single class in validation"}
    pred = p >= threshold
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "event_rate": round(float(y.mean()), 4),
    }


def recall_at_fixed_precision(y: pd.Series, p: np.ndarray, target_precision: float = 0.6) -> dict:
    """Scan thresholds on validation and report the best recall achievable at >= target precision.

    Precision is not monotone in threshold, so ALL candidate thresholds are scanned and the
    lowest threshold meeting the target (hence the highest recall) is kept — the previous
    implementation broke out of the scan at the first precision dip and could falsely report
    the target as unreachable.
    """
    if y.nunique() < 2 or len(y) < 50:
        return {"note": "insufficient validation data for threshold scan"}
    thresholds = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 60)))
    best = None
    for t in sorted(thresholds):  # ascending: keep the lowest threshold that still meets precision
        pred = p >= t
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        if prec >= target_precision:
            best = {"threshold": round(float(t), 4), "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4), "f1": round(float(f1_score(y, pred, zero_division=0)), 4)}
    if best is None:
        best = {"note": f"precision target {target_precision} unreachable; max precision "
                        f"{round(float(precision_score(y, p >= 0.5, zero_division=0)), 4)} at 0.5"}
    best["target_precision"] = target_precision
    return best


def best_f1_threshold(y: pd.Series, p: np.ndarray) -> dict:
    """Threshold maximising validation F1 (fallback operating point when the precision target
    cannot be met, e.g. a heavily compressed low-event-rate probability distribution)."""
    thresholds = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 80)))
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        pred = p >= t
        f1 = float(f1_score(y, pred, zero_division=0))
        if f1 > best["f1"]:
            best = {"threshold": round(float(t), 4), "f1": round(f1, 4),
                    "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
                    "recall": round(float(recall_score(y, pred, zero_division=0)), 4)}
    return best


def threshold_metrics(y: pd.Series, p: np.ndarray, threshold: float) -> dict:
    """F1 / precision / recall at a chosen operating threshold (both models, same threshold)."""
    pred = p >= threshold
    return {
        "threshold": round(float(threshold), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "n_flagged": int(pred.sum()),
    }


def calibration_diagnostics(y: pd.Series, p: np.ndarray, bins: int = 10) -> tuple[dict, pd.DataFrame]:
    """Reliability diagram and expected calibration error (ECE) on validation."""
    if y.nunique() < 2:
        return {"note": "single class"}, pd.DataFrame()
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if mask.sum() == 0:
            continue
        rows.append({"bin": i, "lower": round(float(edges[i]), 2), "upper": round(float(edges[i + 1]), 2),
                     "count": int(mask.sum()), "mean_predicted": round(float(p[mask].mean()), 4),
                     "observed_rate": round(float(y[mask].mean()), 4)})
    table = pd.DataFrame(rows)
    ece = float((np.abs(table["mean_predicted"] - table["observed_rate"]) * table["count"]).sum() / table["count"].sum())
    return {"ece": round(ece, 4), "bins": int(len(table))}, table


def calibration_by_segment(valid: pd.DataFrame, predictions: pd.DataFrame, target: str,
                           segment_columns: tuple, bins: int = 10, min_rows: int = 60) -> pd.DataFrame:
    """ECE broken down by segment (credit band / vintage / state), on validation only.

    Implements the brief's advanced feature "Model calibration by vintage or credit band":
    a globally well-calibrated model can be miscalibrated inside specific segments, and
    this table is what surfaces it.
    """
    prob_col = target.replace("_flag", "_prob")
    if target not in valid or prob_col not in predictions:
        return pd.DataFrame()
    clean = valid.dropna(subset=[target])
    rows = []
    for seg in segment_columns:
        if seg not in clean:
            continue
        for value, group in clean.groupby(seg):
            if len(group) < min_rows:
                continue
            y = group[target].astype(int)
            p = predictions.loc[group.index, prob_col].to_numpy()
            if y.nunique() < 2:
                continue
            diag, _ = calibration_diagnostics(y, p, bins)
            rows.append({"target": target, "segment": seg, "segment_value": str(value),
                         "n": int(len(group)), "event_rate": round(float(y.mean()), 4),
                         "ece": diag["ece"], "bins": diag["bins"]})
    return pd.DataFrame(rows)


def metric_bootstrap_ci(y: pd.Series, p: np.ndarray, metric=roc_auc_score,
                        n_boot: int = 500, seed: int = 2026) -> dict:
    """Bootstrap confidence interval for a validation metric (samples rows with replacement).

    Implements the brief's advanced feature "Model confidence intervals": the point estimate
    in metrics.json is accompanied by a percentile CI, so reviewers see the estimate's spread.
    """
    if y.nunique() < 2:
        return {"note": "single class"}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    y_arr = y.to_numpy()
    boots = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(y), replace=True)
        if np.unique(y_arr[b]).size < 2:
            continue
        boots.append(float(metric(y_arr[b], p[b])))
    if not boots:
        return {"note": "bootstrap degenerate"}
    return {"point": round(float(metric(y_arr, p)), 4),
            "ci_low": round(float(np.percentile(boots, 2.5)), 4),
            "ci_high": round(float(np.percentile(boots, 97.5)), 4),
            "n_boot": int(len(boots))}


def fit_binary(Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series, seed: int, cfg: dict):
    """Baseline random forest vs calibrated gradient boosting, with full metric set."""
    prep = preprocessor(Xtr)
    base = Pipeline([("prep", prep), ("model", RandomForestClassifier(
        n_estimators=120, max_depth=6, class_weight="balanced_subsample", n_jobs=-1, random_state=seed))])
    base.fit(Xtr, ytr)
    pb = base.predict_proba(Xva)[:, 1]

    improved = Pipeline([("prep", preprocessor(Xtr)), ("model", CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=cfg["max_iter"], max_leaf_nodes=cfg["max_leaf_nodes"],
                                       min_samples_leaf=cfg["min_samples_leaf"], l2_regularization=1.0,
                                       class_weight="balanced", random_state=seed),
        method="sigmoid", cv=cfg["calibration_cv"]))])
    improved.fit(Xtr, ytr)
    pi = improved.predict_proba(Xva)[:, 1]

    metrics = {
        "baseline_rf": binary_metrics(yva, pb),
        "calibrated_hgb": binary_metrics(yva, pi),
        "recall_at_fixed_precision": recall_at_fixed_precision(yva, pi, cfg.get("operating_precision", 0.6)),
        "calibration": calibration_diagnostics(yva, pi, cfg.get("calibration_bins", 10))[0],
        "fit_event_rate": round(float(ytr.mean()), 4),
        "valid_event_rate": round(float(yva.mean()), 4),
        "roc_auc_ci": metric_bootstrap_ci(yva, pi, roc_auc_score, cfg.get("bootstrap_samples", 400), seed),
        "pr_auc_ci": metric_bootstrap_ci(yva, pi, average_precision_score, cfg.get("bootstrap_samples", 400), seed),
    }
    # Operating-point metrics at one documented threshold so a calibrated (compressed)
    # probability range is not misread as a broken model. Threshold selection uses
    # validation labels only: the recall@precision scan, falling back to best-F1.
    raf = metrics["recall_at_fixed_precision"]
    if "threshold" in raf:
        op_threshold = raf["threshold"]
        op_method = f"recall_at_precision_{cfg.get('operating_precision', 0.6)}"
    else:
        bf1 = best_f1_threshold(yva, pi)
        op_threshold = bf1["threshold"]
        op_method = "best_validation_f1_fallback"
    metrics["operating_point"] = {"threshold": op_threshold, "method": op_method}
    metrics["threshold_metrics"] = {
        "baseline_rf_at_threshold": threshold_metrics(yva, pb, op_threshold),
        "calibrated_hgb_at_threshold": threshold_metrics(yva, pi, op_threshold),
    }
    if HAS_LIGHTGBM:
        try:
            lgbm = Pipeline([("prep", preprocessor(Xtr)), ("model", CalibratedClassifierCV(
                lgb.LGBMClassifier(objective="binary", n_estimators=cfg.get("lgbm_estimators", 200),
                                   learning_rate=cfg.get("lgbm_learning_rate", 0.05),
                                   num_leaves=cfg.get("lgbm_num_leaves", 31),
                                   min_child_samples=cfg.get("min_samples_leaf", 40),
                                   class_weight="balanced", random_state=seed, verbosity=-1),
                method="sigmoid", cv=cfg["calibration_cv"]))])
            lgbm.fit(Xtr, ytr)
            pl = lgbm.predict_proba(Xva)[:, 1]
            metrics["lgbm_candidate"] = binary_metrics(yva, pl)
            metrics["lgbm_candidate"]["calibration"] = \
                calibration_diagnostics(yva, pl, cfg.get("calibration_bins", 10))[0]
        except Exception as exc:  # never let an optional candidate break the run
            metrics["lgbm_candidate"] = {"note": f"LightGBM candidate skipped: {type(exc).__name__}"}
    else:
        metrics["lgbm_candidate"] = {"note": "lightgbm not installed (optional dependency)"}
    return improved, metrics, pi


def fit_next_state(Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series, seed: int):
    """Multiclass next-state model with a majority-class baseline for honest uplift comparison."""
    model = Pipeline([("prep", preprocessor(Xtr)), ("model", RandomForestClassifier(
        n_estimators=220, min_samples_leaf=8, class_weight="balanced_subsample", n_jobs=-1, random_state=seed))])
    model.fit(Xtr, ytr)
    predicted = model.predict(Xva)
    majority = yva.mode().iloc[0]
    baseline_pred = pd.Series(majority, index=yva.index)
    metrics = {
        "macro_f1": round(float(f1_score(yva, predicted, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(yva, predicted, average="weighted", zero_division=0)), 4),
        "majority_baseline_macro_f1": round(float(f1_score(yva, baseline_pred, average="macro", zero_division=0)), 4),
        "accuracy": round(float((predicted == yva).mean()), 4),
        "classes": sorted(model.classes_.tolist()),
    }
    probabilities = model.predict_proba(Xva)
    return model, metrics, predicted, probabilities.max(axis=1)


def fit_exception_type(Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series, seed: int):
    """Multiclass exception-type model (only rows flagged as exceptions, plus a 'none' class when present)."""
    model = Pipeline([("prep", preprocessor(Xtr)), ("model", RandomForestClassifier(
        n_estimators=180, min_samples_leaf=10, class_weight="balanced_subsample", n_jobs=-1, random_state=seed))])
    model.fit(Xtr, ytr)
    predicted = model.predict(Xva)
    per_class = {}
    for cls in sorted(set(yva)):
        mask = yva == cls
        per_class[str(cls)] = {
            "support": int(mask.sum()),
            "precision": round(float(precision_score(yva == cls, predicted == cls, zero_division=0)), 4),
            "recall": round(float(recall_score(yva == cls, predicted == cls, zero_division=0)), 4),
        }
    metrics = {
        "macro_f1": round(float(f1_score(yva, predicted, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(yva, predicted, average="weighted", zero_division=0)), 4),
        "per_class": per_class,
        "classes": sorted(model.classes_.tolist()),
    }
    probabilities = model.predict_proba(Xva)
    return model, metrics, predicted, probabilities.max(axis=1)


def monthly_validation_metrics(valid: pd.DataFrame, predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    """AUC/Brier per validation reporting month — temporal stability evidence.

    `target` is the LABEL column name; the probability column is derived as
    `target.replace('_flag', '_prob')` and read from `predictions`.
    """
    prob_col = target.replace("_flag", "_prob")
    if target not in valid or prob_col not in predictions:
        return pd.DataFrame()
    rows = []
    for month, group in valid.groupby("reporting_month"):
        group = group.dropna(subset=[target])
        if group.empty:
            continue
        p = predictions.loc[group.index, prob_col]
        y = group[target].astype(int)
        entry = {"reporting_month": month, "rows": int(len(group)), "event_rate": round(float(y.mean()), 4)}
        if y.nunique() >= 2:
            entry["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
            entry["brier"] = round(float(brier_score_loss(y, p)), 4)
        rows.append(entry)
    return pd.DataFrame(rows).sort_values("reporting_month")


def loan_disjoint_ablation(train: pd.DataFrame, features: list[str], target: str, seed: int,
                           n_splits: int = 5) -> dict:
    """Evaluation audit: how much of the time-split performance comes from panel overlap?

    Trains a quick model on a deterministic 70% of loans and scores the remaining 30% of
    loans — which share no loan with training. Repeated over `n_splits` deterministic
    partition variants, reporting the mean and min-max spread, because a single split can
    wobble around the coin-toss line when events are sparse. Reported, never hidden.
    """
    if target not in train or "loan_id" not in train:
        return {"note": "target or loan_id unavailable"}
    df = train.dropna(subset=[target])
    if len(df) < 200:
        return {"note": "insufficient rows for ablation"}
    base_hash = df.loan_id.astype(str).apply(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16))
    aucs = []
    for k in range(n_splits):
        bucket = (base_hash + k * 7919) % 100
        tr = df[bucket < 70]
        va = df[bucket >= 70]
        if len(tr) < 100 or len(va) < 50 or va[target].nunique() < 2:
            continue
        model = Pipeline([("prep", preprocessor(tr[features])), ("model", HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=20, min_samples_leaf=30, class_weight="balanced", random_state=seed))])
        model.fit(tr[features], tr[target].astype(int))
        p = model.predict_proba(va[features])[:, 1]
        aucs.append(float(roc_auc_score(va[target].astype(int), p)))
    if not aucs:
        return {"note": "ablation split degenerate"}
    mean_auc = float(np.mean(aucs))
    return {
        "target": target,
        "auc_loan_disjoint_mean": round(mean_auc, 4),
        "auc_loan_disjoint_min": round(min(aucs), 4),
        "auc_loan_disjoint_max": round(max(aucs), 4),
        "auc_loan_disjoint_splits": [round(a, 4) for a in aucs],
        "n_splits": len(aucs),
        "train_loans": int(df.loan_id.nunique()),
        "interpretation": "The time-split AUC includes legitimate panel overlap; this loan-disjoint "
                          "estimate removes it and quantifies genuine cross-loan signal. Values near 0.5 "
                          "mean the signal is carried by the loan's own history; values well above 0.5 "
                          "mean features generalise across loans. Both numbers are shipped on purpose.",
    }


def train_all(fit: pd.DataFrame, valid: pd.DataFrame, features: list[str], seed: int, cfg: dict):
    """Train every outcome model on the out-of-time fit period and score the validation period."""
    models: dict = {}
    metrics: dict = {}
    predictions = pd.DataFrame(index=valid.index)

    for target in BINARY_TARGETS:
        if target not in fit or fit[target].dropna().nunique() < 2:
            continue
        tr = fit.dropna(subset=[target])
        va = valid.dropna(subset=[target])
        if len(va) < 30:
            continue
        model, score, p = fit_binary(tr[features], tr[target].astype(int), va[features], va[target].astype(int), seed, cfg)
        models[target] = model
        metrics[target] = score
        predictions.loc[va.index, target.replace("_flag", "_prob")] = p

    if "next_state" in fit and fit["next_state"].dropna().nunique() > 1:
        tr, va = fit.dropna(subset=["next_state"]), valid.dropna(subset=["next_state"])
        if len(va) >= 30:
            model, score, pred, confidence = fit_next_state(
                tr[features], tr.next_state.astype(str), va[features], va.next_state.astype(str), seed)
            models["next_state"] = model
            metrics["next_state"] = score
            predictions.loc[va.index, "next_state"] = pred
            predictions.loc[va.index, "next_state_confidence"] = confidence

    if "exception_type" in fit and fit["exception_type"].dropna().nunique() > 1:
        tr, va = fit.dropna(subset=["exception_type"]), valid.dropna(subset=["exception_type"])
        if len(va) >= 30:
            model, score, pred, confidence = fit_exception_type(
                tr[features], tr.exception_type.astype(str), va[features], va.exception_type.astype(str), seed)
            models["exception_type"] = model
            metrics["exception_type"] = score
            predictions.loc[va.index, "exception_type"] = pred
            predictions.loc[va.index, "exception_type_confidence"] = confidence

    # Temporal stability evidence for the two headline outcomes.
    for target in ("next_3m_delinquency_flag", "next_12m_default_flag"):
        if target in valid and target.replace("_flag", "_prob") in predictions:
            metrics[f"monthly_{target}"] = monthly_validation_metrics(
                valid, predictions, target).to_dict("records")

    return models, metrics, predictions
