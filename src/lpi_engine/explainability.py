"""Explainability and error analysis: global importance, local driver attribution, FP/FN analysis, uncertainty.

Local explanations prefer TreeSHAP when the `shap` package is installed; otherwise they
fall back to an honest deviation-based attribution (robust z-score x global importance
rank), clearly labelled so reviewers never mistake it for model-derived SHAP values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:  # Optional dependency — never required for the core pipeline.
    import shap
    HAS_SHAP = True
except Exception:  # pragma: no cover
    shap = None
    HAS_SHAP = False


def global_importance(models: dict, valid: pd.DataFrame, features: list[str], limit: int = 4000) -> pd.DataFrame:
    """Validation-only permutation importances (average-precision scoring), sampled for large panels."""
    sample = valid.sample(min(len(valid), limit), random_state=2026)
    rows = []
    for target, model in models.items():
        if target not in sample or target in ("next_state", "exception_type"):
            continue
        subset = sample.dropna(subset=[target])
        if subset[target].nunique() < 2:
            continue
        try:
            result = permutation_importance(model, subset[features], subset[target], n_repeats=3,
                                            scoring="average_precision", random_state=2026, n_jobs=-1)
            rows.extend({"target": target, "feature": feature,
                         "permutation_importance": round(float(score), 6),
                         "importance_std": round(float(std), 6)}
                        for feature, score, std in zip(features, result.importances_mean, result.importances_std))
        except ValueError:
            continue
    return pd.DataFrame(rows).sort_values(["target", "permutation_importance"], ascending=[True, False]) if rows \
        else pd.DataFrame(columns=["target", "feature", "permutation_importance", "importance_std"])


def _extract_base_estimator(model) -> tuple[object, str]:
    """Unwrap Pipeline -> CalibratedClassifierCV -> first fold's base estimator for TreeSHAP."""
    est = model
    if hasattr(est, "named_steps") and "model" in est.named_steps:
        est = est.named_steps["model"]
    if hasattr(est, "calibrated_classifiers_") and len(est.calibrated_classifiers_) > 0:
        est = est.calibrated_classifiers_[0].estimator
    if hasattr(est, "named_steps"):
        est = est.named_steps.get("model", est)
    return est, type(est).__name__


def _transform_and_map(model, X: pd.DataFrame):
    """Transform raw features through the fitted preprocessor and map output columns to raw feature names."""
    prep = model.named_steps.get("prep") if hasattr(model, "named_steps") else None
    if prep is None:
        return X, list(X.columns), {c: c for c in X.columns}
    X_trans = prep.transform(X)
    names = list(prep.get_feature_names_out())
    mapping = {}
    for name in names:
        if name.startswith("num__"):
            raw = name[5:]
        elif name.startswith("cat__"):
            raw = name[5:].rsplit("_", 1)[0]
        else:
            raw = name
        mapping[name] = raw
    return pd.DataFrame(X_trans, columns=names), names, mapping


def _aggregate_shap(shap_row, names: list[str], mapping: dict) -> dict[str, float]:
    """Sum one-hot SHAP values back to raw-feature contributions."""
    agg: dict[str, float] = {}
    for name, value in zip(names, shap_row):
        raw = mapping.get(name, name)
        agg[raw] = agg.get(raw, 0.0) + float(value)
    return agg


def local_explanations(models: dict, sample: pd.DataFrame, features: list[str],
                       n_rows: int = 25, background: int = 150) -> pd.DataFrame:
    """Per-record, per-target feature attributions for a reviewer sample.

    Method column is explicit: `shap_tree` (model-derived TreeSHAP on the calibrated base
    estimator, aggregated back to raw features) or `deviation_rank` (robust z-score x global
    importance — an honest heuristic fallback).
    """
    rows = []
    subset = sample.head(n_rows).reset_index(drop=True)
    for target, model in models.items():
        if target in ("next_state", "exception_type"):
            continue
        X = subset[features]
        method = "deviation_rank"
        shap_values = None
        names, mapping = [], {}
        if HAS_SHAP:
            try:
                est, _ = _extract_base_estimator(model)
                X_trans, names, mapping = _transform_and_map(model, X)
                explainer = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
                sample_X = X_trans.iloc[:min(background, len(X_trans))]
                shap_values = explainer.shap_values(sample_X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                method = "shap_tree"
            except Exception:
                shap_values = None
        if shap_values is not None:
            for j in range(min(len(shap_values), len(subset))):
                agg = _aggregate_shap(shap_values[j], names, mapping)
                order = sorted(agg, key=lambda f: -abs(agg[f]))[:3]
                for k in order:
                    rows.append({
                        "target": target, "row": j, "loan_id": subset.iloc[j].get("loan_id"),
                        "feature": k,
                        "value": round(agg[k], 5),
                        "top_3": ", ".join(order),
                        "method": method,
                        "direction": "increases_risk" if agg[k] > 0 else "decreases_risk",
                    })
            continue
        # Deviation-based fallback: |robust z| of the record scaled by the feature's global
        # permutation importance rank. Direction is unavailable; labelled accordingly.
        imp = global_importance({target: model}, subset, features, limit=min(len(subset), 500))
        weight = {}
        ranked = imp.sort_values("permutation_importance", ascending=False)["feature"].tolist()
        for i, f in enumerate(ranked):
            weight[f] = max(1.0 - i / len(ranked), 0.05)
        for j in range(len(subset)):
            row = subset.iloc[j]
            zs = {}
            for f in features:
                if pd.api.types.is_numeric_dtype(X[f]):
                    med = X[f].median()
                    mad = (X[f] - med).abs().median()
                    if mad and mad > 0:
                        zs[f] = abs((float(row[f]) - med) / (1.4826 * mad))
            top = sorted(zs, key=lambda f: -zs[f] * weight.get(f, 0.05))[:3]
            rows.append({"target": target, "row": j, "loan_id": row.get("loan_id"),
                         "feature": top[0] if top else None, "value": round(zs.get(top[0], 0), 3) if top else None,
                         "top_3": ", ".join(top), "method": method, "direction": "unknown"})
    return pd.DataFrame(rows)


def error_analysis(valid: pd.DataFrame, predictions: pd.DataFrame, target: str,
                   threshold: float = 0.5) -> dict:
    """False-positive / false-negative analysis for one binary target on validation."""
    prob_col = target.replace("_flag", "_prob")
    if target not in valid or prob_col not in predictions:
        return {}
    clean = valid.dropna(subset=[target])
    y = clean[target].astype(int)
    p = predictions.loc[clean.index, prob_col]
    pred = (p >= threshold).astype(int)
    fp_mask = (pred == 1) & (y == 0)
    fn_mask = (pred == 0) & (y == 1)
    segments = [c for c in ("credit_score_band", "state", "servicer_name", "ltv_band") if c in valid]
    summary = {
        "target": target,
        "threshold": threshold,
        "n_false_positives": int(fp_mask.sum()),
        "n_false_negatives": int(fn_mask.sum()),
        "fp_rate_at_threshold": round(float(fp_mask.mean()), 4),
        "fn_rate_at_threshold": round(float(fn_mask.mean()), 4),
    }
    for label, mask in (("false_positives", fp_mask), ("false_negatives", fn_mask)):
        if mask.sum() == 0:
            continue
        sub = clean.loc[mask]
        concentration = {}
        for seg in segments:
            if seg in sub and seg in clean:
                table = (sub[seg].astype(str).value_counts() / clean[seg].astype(str).value_counts()).replace(np.inf, 0).fillna(0)
                top = table.sort_values(ascending=False).head(3)
                concentration[seg] = {str(k): round(float(v), 4) for k, v in top.items()}
        summary[f"{label}_segment_concentration"] = concentration
        summary[f"{label}_mean_quality"] = round(float(clean.loc[mask, "data_quality_score"].mean()), 2) \
            if "data_quality_score" in clean else None
    return summary


def error_examples(valid: pd.DataFrame, predictions: pd.DataFrame, target: str,
                   threshold: float = 0.5, limit: int = 200) -> pd.DataFrame:
    """Row-level FP/FN examples with evidence columns for the reviewer."""
    prob_col = target.replace("_flag", "_prob")
    if target not in valid or prob_col not in predictions:
        return pd.DataFrame()
    clean = valid.dropna(subset=[target])
    y = clean[target].astype(int)
    p = predictions.loc[clean.index, prob_col]
    pred = (p >= threshold).astype(int)
    mask = (pred != y)
    cols = [c for c in ("loan_id", "reporting_month", "days_past_due", "current_balance", "credit_score_band",
                        "state", "servicer_name", "data_quality_score", "anomaly_score") if c in clean.columns]
    out = clean.loc[mask, cols].copy()
    out[target] = y.loc[mask]
    out[prob_col] = p.loc[mask].round(4)
    out["error_type"] = np.where(pred.loc[mask] == 1, "false_positive", "false_negative")
    out = out.rename(columns={prob_col: "predicted_probability"})
    return out.head(limit)


def fairness_report(valid: pd.DataFrame, predictions: pd.DataFrame, target: str,
                    segments: tuple = ("credit_score_band", "state"),
                    threshold: float = 0.5) -> pd.DataFrame:
    """Basic segment parity analysis on validation: event rate, predicted positive rate, FPR and FNR.

    This is a descriptive monitoring table, not a legal fairness assessment — the model card
    states that full bias/fairness review is a pre-production gate.
    """
    prob_col = target.replace("_flag", "_prob")
    if target not in valid or prob_col not in predictions:
        return pd.DataFrame()
    clean = valid.dropna(subset=[target])
    p = predictions.loc[clean.index, prob_col]
    y = clean[target].astype(int)
    pred = (p >= threshold).astype(int)
    rows = []
    for seg in segments:
        if seg not in clean:
            continue
        for value, mask in clean.groupby(seg):
            m = mask.index
            rows.append({
                "target": target,
                "segment": seg,
                "segment_value": str(value),
                "n": int(len(m)),
                "event_rate": round(float(y.loc[m].mean()), 4),
                "predicted_positive_rate": round(float(pred.loc[m].mean()), 4),
                "fpr": round(float(((pred == 1) & (y == 0)).loc[m].mean()), 4),
                "fnr": round(float(((pred == 0) & (y == 1)).loc[m].mean()), 4),
                "mean_predicted_probability": round(float(p.loc[m].mean()), 4),
            })
    return pd.DataFrame(rows)


DEFAULT_COUNTERFACTUALS = [
    ("days_past_due", 0.0, "cure delinquency (DPD → 0)"),
    ("credit_score_numeric", None, "credit score +40 points"),
    ("ltv_numeric", None, "LTV −10 points"),
    ("interest_rate", None, "rate −1 pp"),
    ("balance_ratio", 0.8, "20% principal paydown"),
]


def counterfactual_explanations(records: pd.DataFrame, model, features: list[str],
                                target: str, n: int = 20) -> pd.DataFrame:
    """What-if analysis for reviewer-prioritised records (brief: "Counterfactual explanations").

    For each intervention (documented in DEFAULT_COUNTERFACTUALS) the feature is moved to a
    safer value, the model is re-scored, and the probability delta is reported. Relative
    interventions (credit +40, LTV −10, rate −1) are anchored to the record's own value.
    Also reports the combined "all interventions" cure.
    """
    prob_col = target.replace("_flag", "_prob")
    subset = records.head(n).copy()
    if subset.empty:
        return pd.DataFrame()
    rows = []
    for _, record in subset.iterrows():
        x0 = pd.DataFrame([record[features].to_dict()])
        p0 = float(model.predict_proba(x0)[0, 1])
        entry = {"target": target, "loan_id": record.get("loan_id"),
                 "baseline_probability": round(p0, 4), "best_counterfactual": "",
                 "best_delta": 0.0, "all_cured_probability": None}
        cured = x0.copy()
        best_delta = -np.inf
        for col, value, label in DEFAULT_COUNTERFACTUALS:
            if col not in x0.columns:
                continue
            base_val = x0[col].iloc[0]
            if pd.isna(base_val):
                continue
            if value is None:
                delta_amount = {"credit_score_numeric": 40.0, "ltv_numeric": -10.0,
                                 "interest_rate": -1.0}.get(col, 0.0)
                new_val = base_val + delta_amount
            else:
                new_val = value
            x1 = x0.copy()
            x1[col] = new_val
            p1 = float(model.predict_proba(x1)[0, 1])
            entry[f"{col}_prob"] = round(p1, 4)
            entry[f"{col}_delta"] = round(p1 - p0, 4)
            if p1 - p0 > best_delta:
                best_delta = p1 - p0
                entry["best_counterfactual"] = label
                entry["best_delta"] = round(p1 - p0, 4)
            cured[col] = new_val
        p_cured = float(model.predict_proba(cured)[0, 1])
        entry["all_cured_probability"] = round(p_cured, 4)
        entry["all_cured_delta"] = round(p_cured - p0, 4)
        rows.append(entry)
    return pd.DataFrame(rows)


def model_confidence(probabilities: dict, quality_scores: pd.Series) -> pd.DataFrame:
    """Documented confidence: 70% distance from the coin-flip + 30% data quality.

    `model_confidence = 2*|p-0.5|` captures how far the model is from indecision;
    `data_confidence = quality/100` captures how much the record can be trusted.
    The blend is explicit and auditable, not a black-box score.
    """
    conf = pd.DataFrame(index=quality_scores.index)
    model_terms = [pd.Series(np.abs(2 * (prob - 0.5)), index=prob.index) for prob in probabilities.values() if len(prob) == len(conf)]
    if model_terms:
        conf["model_confidence"] = pd.concat(model_terms, axis=1).mean(axis=1).round(3)
    else:
        conf["model_confidence"] = 0.0
    conf["data_confidence"] = (quality_scores / 100).clip(0, 1).round(3)
    conf["confidence"] = (0.7 * conf["model_confidence"] + 0.3 * conf["data_confidence"]).round(3)
    return conf
