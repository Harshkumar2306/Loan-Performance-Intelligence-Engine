"""End-to-end reproducible pipeline: ingest -> profile -> split -> train -> explain -> simulate -> submit.

Run:  python scripts/run_pipeline.py --config config/default.yaml
Every artefact is written to `outputs/`. Metrics are only ever computed on the
contiguous out-of-time validation period, never on training data.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpi_engine.data import (
    assert_no_leakage,
    engineer_panel,
    feature_columns,
    merge_attributes,
    normalise,
    read_pack,
    time_split,
    validate_schema,
)
from lpi_engine.profiling import profile, score_test_frame
from lpi_engine.modeling import (
    BINARY_TARGETS,
    calibration_by_segment,
    calibration_diagnostics,
    loan_disjoint_ablation,
    train_all,
)
from lpi_engine.intelligence import (
    anomaly_scores,
    competing_risk_curves,
    scenario_cache,
    scenario_drivers,
    scenario_monte_carlo,
    scenario_table,
    stress_sensitivity_by_cluster,
    transition_curves,
    transition_model,
)
from lpi_engine.copilot import demonstrate_governance, log_audit, parse_rules
from lpi_engine.dashboard import render as render_dashboard
from lpi_engine.explainability import (
    counterfactual_explanations,
    error_analysis,
    error_examples,
    fairness_report,
    global_importance,
    local_explanations,
    model_confidence,
)
from lpi_engine.tracking import log_run

STANDARD_SUBMISSION_COLUMNS = [
    "loan_id", "next_3m_delinquency_prob", "next_6m_delinquency_prob",
    "next_12m_default_prob", "next_12m_prepayment_prob", "next_state",
    "next_state_confidence", "exception_type", "exception_type_confidence",
    "anomaly_score", "anomaly_reason", "top_drivers", "recommended_action", "confidence",
]


def _fmt_cell(value):
    if isinstance(value, float):
        return round(value, 4)
    if pd.isna(value):
        return ""
    return value


def _json_dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=str))


def _markdown_data_intelligence(report: dict) -> str:
    lines = [
        "# Data Intelligence Report",
        "",
        f"*Generated {datetime.now(timezone.utc).date()} by `run_pipeline.py`.*",
        "",
        f"- **Rows profiled:** {report['rows']:,}  |  **Columns:** {report['columns']}  |  **Reporting months:** {report['train_months']}",
        f"- **Rows below 60/100 data quality:** {report['share_rows_below_60_quality']:.1%}",
        "",
        "## Deterministic rule breaches",
        "",
        "| Rule | Breaches |",
        "|---|---|",
    ]
    lines += [f"| {name} | {count:,} |" for name, count in sorted(report["rule_breaches"].items(), key=lambda kv: -kv[1])]
    lines += [
        "",
        "## Outlier flags (robust z > 5)",
        "",
        "| Column | Flags |",
        "|---|---|",
    ]
    lines += [f"| {col} | {count:,} |" for col, count in sorted(report["outlier_counts"].items(), key=lambda kv: -kv[1])]
    lines += [
        "",
        "## Source reconciliation (servicer updates)",
        "",
        "| Conflict | Rows |",
        "|---|---|",
    ]
    lines += [f"| {col} | {count:,} |" for col, count in report["source_reconciliation"].items()]
    lines += [
        "",
        "## Highest numeric correlations",
        "",
        "| Pair | |corr| |",
        "|---|---|",
    ]
    lines += [f"| {pair} | {value} |" for pair, value in list(report["highest_numeric_correlations"].items())[:10]]
    lines += [
        "",
        "## Strongest categorical associations (Cramér's V)",
        "",
        "| Pair | V | p |",
        "|---|---|---|",
    ]
    lines += [f"| {a['pair']} | {a['cramers_v']} | {a['chi2_p']} |" for a in report["strongest_categorical_associations"]]
    lines += [
        "",
        "## Top association rules (by lift)",
        "",
        "| Rule | Support | Confidence | Lift |",
        "|---|---|---|---|",
    ]
    lines += [f"| `{r['rule']}` | {r['support']} | {r['confidence']} | {r['lift']} |"
              for r in report.get("top_association_rules_by_lift", [])]
    lines += [
        "",
        "## Highly dependent fields (|corr| or V > 0.9)",
        "",
        ", ".join(report["highly_dependent_fields"]) or "None detected.",
        "",
        "## Train vs test drift (PSI)",
        "",
        "| Column | PSI |",
        "|---|---|",
    ]
    lines += [f"| {c} | {v} |" for c, v in sorted(report["train_test_numeric_psi"].items(), key=lambda kv: -kv[1]) if v > 0.01]
    lines += [f"| {c} | {v} |" for c, v in sorted(report["train_test_categorical_psi"].items(), key=lambda kv: -kv[1]) if v > 0.01]
    lines += [
        "",
        "## Missingness patterns (top 8)",
        "",
        "| Pattern | Rows |",
        "|---|---|",
    ]
    lines += [f"| `{pattern}` | {count:,} |" for pattern, count in report["missing_patterns_top_8"].items()]
    lines += [
        "",
        "## Data-quality score distribution",
        "",
        "```",
        json.dumps(report["quality_score_distribution"], indent=2),
        "```",
    ]
    return "\n".join(lines)


def _markdown_explainability(global_df: pd.DataFrame, error_summaries: dict, targets: list[str]) -> str:
    lines = [
        "# Explainability Report",
        "",
        f"*Generated {datetime.now(timezone.utc).date()} by `run_pipeline.py`.*",
        "",
        "## Global permutation importance (validation, average-precision scoring)",
        "",
        "| Target | Feature | Importance ± std |",
        "|---|---|---|",
    ]
    for _, row in global_df.head(30).iterrows():
        lines.append(f"| {row['target']} | {row['feature']} | {row['permutation_importance']} ± {row['importance_std']} |")
    lines += ["", "## Error analysis (false positives / false negatives, validation period)", ""]
    for target in targets:
        summary = error_summaries.get(target, {})
        if not summary:
            continue
        lines += [f"### {target}", "",
                  f"- False positives: {summary.get('n_false_positives')} "
                  f"(rate {summary.get('fp_rate_at_threshold')})",
                  f"- False negatives: {summary.get('n_false_negatives')} "
                  f"(rate {summary.get('fn_rate_at_threshold')})",
                  f"- Threshold: {summary.get('threshold')}"]
        for label in ("false_positives_segment_concentration", "false_negatives_segment_concentration"):
            concentration = summary.get(label)
            if concentration:
                lines.append(f"- {label.replace('_', ' ').title()}: {json.dumps(concentration)}")
        lines.append("")
    lines += [
        "## Local explanations",
        "",
        "See `local_explanations.csv` — per-record, per-target attributions with an explicit",
        "`method` column (`shap_tree` when TreeSHAP is available, otherwise an honestly labelled",
        "`deviation_rank` heuristic). See `validation_predictions.csv` and `error_examples.csv`",
        "for the underlying records.",
        "",
        "## Uncertainty",
        "",
        "`confidence = 0.7 * model_confidence + 0.3 * data_confidence`, where",
        "`model_confidence = 2*|p-0.5|` and `data_confidence = data_quality_score/100`.",
        "The formula is auditable in `explainability.model_confidence`.",
    ]
    return "\n".join(lines)


def _markdown_scenario(scenario_df: pd.DataFrame, drivers_df: pd.DataFrame, mc_df: pd.DataFrame) -> str:
    lines = [
        "# Scenario & Stress Report",
        "",
        f"*Generated {datetime.now(timezone.utc).date()} by `run_pipeline.py`.*",
        "",
        "Scenarios are transparent feature shocks followed by the same fitted models, so every",
        "number is auditable. Segment rows live in `scenario_report.csv`.",
        "",
        "## Scenario-level projected default and prepayment rates (bootstrap 5–95 bounds)",
        "",
        "| Scenario | Mean default | Default 5th | Default 95th | Mean prepay | Prepay 5th | Prepay 95th | Sims |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in mc_df.iterrows():
        lines.append(f"| {row['scenario']} | {row['mean_default_rate']} | {row['p05_default_rate']} | "
                     f"{row['p95_default_rate']} | {row['mean_prepayment_rate']} | "
                     f"{row['p05_prepayment_rate']} | {row['p95_prepayment_rate']} | {row['simulations']} |")
    lines += [
        "",
        "## Driver attribution (one-feature-at-a-time counterfactuals)",
        "",
        "| Scenario | Δ default (pp) | Credit shock (pp) | Prepay-uplift effect on default (pp) | Δ prepayment (pp) | Uplift applied (pp) |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in drivers_df.iterrows():
        lines.append(f"| {row['scenario']} | {row['delta_default_pp']} | {row['credit_shock_contribution_pp']} | "
                     f"{row['prepayment_uplift_contribution_pp']} | {row['delta_prepayment_pp']} | "
                     f"{row['prepayment_uplift_applied_pp']} |")
    lines += [
        "",
        "## Segment-level default-rate impacts (top 12 largest moves vs Base)",
        "",
        "| Scenario | Segment | Value | Default rate | Delinquency | Prepayment |",
        "|---|---|---|---|---|---|",
    ]
    if not scenario_df.empty:
        base = scenario_df[scenario_df.scenario == "Base"]
        adverse = scenario_df[scenario_df.scenario != "Base"]
        merged = adverse.merge(base, on=["segment", "segment_value"], suffixes=("", "_base"))
        merged["delta"] = merged["next_12m_default_prob"] - merged["next_12m_default_prob_base"]
        for _, row in merged.nlargest(12, "delta").iterrows():
            lines.append(f"| {row['scenario']} | {row['segment']} | {row['segment_value']} | "
                         f"{row['next_12m_default_prob']:.4f} | {row['next_3m_delinquency_prob']:.4f} | "
                         f"{row['next_12m_prepayment_prob']:.4f} |")
    lines += [
        "",
        "## Segment-level prepayment-rate impacts (top 12 largest moves vs Base)",
        "",
        "| Scenario | Segment | Value | Base prepay | Scenario prepay | Δ (pp) |",
        "|---|---|---|---|---|---|",
    ]
    if not scenario_df.empty:
        base = scenario_df[scenario_df.scenario == "Base"]
        adverse = scenario_df[scenario_df.scenario != "Base"]
        merged = adverse.merge(base, on=["segment", "segment_value"], suffixes=("", "_base"))
        merged["delta_pre"] = merged["next_12m_prepayment_prob"] - merged["next_12m_prepayment_prob_base"]
        for _, row in merged.nlargest(12, "delta_pre").iterrows():
            lines.append(f"| {row['scenario']} | {row['segment']} | {row['segment_value']} | "
                         f"{row['next_12m_prepayment_prob_base']:.4f} | {row['next_12m_prepayment_prob']:.4f} | "
                         f"{row['delta_pre'] * 100:.2f} |")
    lines += ["", "See `scenario_drivers.csv` for the full attribution and `scenario_mc.csv` for bootstrap bounds."]
    return "\n".join(lines)


def _markdown_model_card(metrics: dict, features: list[str], split: dict, report: dict,
                         segment_cal_df: pd.DataFrame | None = None) -> str:
    lines = [
        "# Model Card — Loan Performance Intelligence Engine (generated)",
        "",
        f"*Generated {datetime.now(timezone.utc).date()} from the actual pipeline run. Values are never pre-filled.*",
        "",
        "## Intended use",
        "Rank monthly loan observations for a human reviewer. Not for autonomous servicing, credit",
        "approval, or any consumer-impacting decision.",
        "",
        "## Data & validation",
        f"- Time split: {split['method']} (cutoff {split['cutoff']}).",
        f"- Fit rows: {split['fit_rows']:,}  |  Validation rows: {split['validation_rows']:,} across {split['validation_months']} months.",
        f"- Loans spanning the boundary (expected in panel data): {split['loans_spanning_boundary']:,}.",
        f"- Feature count: {len(features)} (see `features_used.json`).",
        "",
        "## Metrics (out-of-time validation, threshold-free)",
        "",
        "| Target | Model | ROC-AUC | PR-AUC | Brier |",
        "|---|---|---|---|---|",
    ]
    for target, entry in metrics.items():
        if not isinstance(entry, dict) or "baseline_rf" not in entry:
            continue
        base, cal = entry["baseline_rf"], entry["calibrated_hgb"]
        lines.append(f"| {target} | baseline RF | {base.get('roc_auc')} | {base.get('pr_auc')} | {base.get('brier')} |")
        lines.append(f"| {target} | calibrated HGB | {cal.get('roc_auc')} | {cal.get('pr_auc')} | {cal.get('brier')} |")
        lgbm_entry = entry.get("lgbm_candidate", {})
        if isinstance(lgbm_entry, dict) and "roc_auc" in lgbm_entry:
            lines.append(f"| {target} | LightGBM (optional) | {lgbm_entry.get('roc_auc')} | {lgbm_entry.get('pr_auc')} | "
                         f"{lgbm_entry.get('brier')} |")
    for target in ("next_state", "exception_type"):
        if target in metrics:
            m = metrics[target]
            lines.append(f"| {target} | random forest (multiclass) | macro-F1 {m.get('macro_f1')} | "
                         f"weighted-F1 {m.get('weighted_f1')} | classes {m.get('classes')} |")
    lines += [
        "",
        "## Operating point (threshold chosen on validation only)",
        "",
        "Per-target thresholds come from the recall-at-fixed-precision scan, falling back to the",
        "validation best-F1 threshold when the precision target is unreachable (e.g. a strongly",
        "compressed probability range on a low-event-rate target). Both models are evaluated at",
        "the SAME threshold.",
        "",
        "| Target | Threshold | Method | Model | F1 | Precision | Recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for target, entry in metrics.items():
        if not isinstance(entry, dict) or "threshold_metrics" not in entry:
            continue
        op = entry.get("operating_point", {})
        for model_key, label in (("baseline_rf_at_threshold", "baseline RF"),
                                 ("calibrated_hgb_at_threshold", "calibrated HGB")):
            tm = entry["threshold_metrics"].get(model_key, {})
            if tm:
                lines.append(f"| {target} | {op.get('threshold')} | {op.get('method')} | {label} | "
                             f"{tm.get('f1')} | {tm.get('precision')} | {tm.get('recall')} |")
    lines += [
        "",
        "## Evaluation audit (panel-overlap memorisation)",
        "",
        "| Target | Time-split ROC-AUC | Loan-disjoint ROC-AUC | Interpretation |",
        "|---|---|---|---|",
    ]
    time_auc = {t: metrics[t]["calibrated_hgb"].get("roc_auc") for t in metrics
                if isinstance(metrics.get(t), dict) and "calibrated_hgb" in metrics[t]}
    for audit in metrics.get("evaluation_audit", []):
        if "auc_loan_disjoint_mean" not in audit:
            continue
        t = audit["target"]
        lines.append(f"| {t} | {time_auc.get(t)} | {audit['auc_loan_disjoint_mean']} "
                     f"(range {audit['auc_loan_disjoint_min']}–{audit['auc_loan_disjoint_max']}) "
                     f"| {audit['interpretation']} |")
    lines += [
        "",
        "## Bootstrap confidence intervals (validation ROC-AUC, 2.5–97.5 percentile)",
        "",
        "| Target | Point | CI low | CI high |",
        "|---|---|---|---|",
    ]
    for target, entry in metrics.items():
        if not isinstance(entry, dict) or "roc_auc_ci" not in entry:
            continue
        ci = entry["roc_auc_ci"]
        if "point" in ci:
            lines.append(f"| {target} | {ci['point']} | {ci['ci_low']} | {ci['ci_high']} |")
    lines += [
        "",
        "## Calibration by segment (ECE per credit band / vintage / state)",
        "",
        "Worst-calibrated segments first; the full table is in `calibration_by_segment.csv`.",
        "",
        "| Target | Segment | Value | n | ECE |",
        "|---|---|---|---|---|",
    ]
    if segment_cal_df is not None and not segment_cal_df.empty:
        for _, row in segment_cal_df.sort_values("ece", ascending=False).head(12).iterrows():
            lines.append(f"| {row['target']} | {row['segment']} | {row['segment_value']} | "
                         f"{row['n']} | {row['ece']} |")
    else:
        lines.append("| — | no segment with enough rows for a reliable ECE | | | |")
    lines += [
        "",
        "## Calibration & recall-at-fixed-precision",
        "",
        "| Target | ECE | Recall@precision≥0.6 | Threshold |",
        "|---|---|---|---|",
    ]
    for target, entry in metrics.items():
        if not isinstance(entry, dict) or "calibration" not in entry:
            continue
        cal = entry["calibration"]
        raf = entry.get("recall_at_fixed_precision", {})
        if "note" in raf:
            lines.append(f"| {target} | {cal.get('ece')} | *{raf['note']}* | — |")
        else:
            lines.append(f"| {target} | {cal.get('ece')} | {raf.get('recall')} | {raf.get('threshold')} |")
    lines += [
        "",
        "## Limitations",
        "- Probabilities are statistical estimates; calibration degrades under regime change.",
        "- Sparse default tails make extreme-decile metrics unstable.",
        "- The synthetic demo pack is a functional test only, not evidence of real-world performance.",
        "",
        "## Controls",
        "- Targets and post-event flags are excluded by a hard schema gate (`assert_no_leakage`).",
        "- No LLM participates in prediction; the copilot is read-only and advisory.",
        "- Fairness/bias testing, legal review and approval thresholds are required before any production use.",
    ]
    return "\n".join(lines)


def main(config_path: str, data_dir: str | None = None, output_dir: str | None = None, open_browser: bool = True):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if data_dir:
        cfg["paths"]["data_dir"] = data_dir
    if output_dir:
        cfg["paths"]["output_dir"] = output_dir
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(exist_ok=True)
    seed = cfg["seed"]

    # ---------------------------------------------------------------- ingest
    pack = read_pack(cfg["paths"])
    updates = normalise(pack["updates"]) if "updates" in pack else None
    train = normalise(pack["train"])
    test = normalise(pack["test"])
    validate_schema(train, "loan_monthly_performance_train.csv")
    validate_schema(test, "loan_monthly_performance_test.csv")
    if "static" in pack:
        static = normalise(pack["static"])
        train = merge_attributes(train, static)
        test = merge_attributes(test, static)
    train, test = engineer_panel(train, test)

    # ------------------------------------------------------------- profiling
    report, quality, batch = profile(train, test, updates)
    excluded = assert_no_leakage(feature_columns(train), list(train.columns))
    report["excluded_columns_gate"] = excluded
    _json_dump(report, out / "data_intelligence_report.json")
    (out / "data_intelligence_report.md").write_text(_markdown_data_intelligence(report))
    batch.to_csv(out / "batch_quality_kpi.csv", index=False)
    quality.to_csv(out / "train_quality_scores.csv", index=False)
    test_quality_frame = score_test_frame(test, updates)
    test_quality_frame.to_csv(out / "test_quality_scores.csv", index=False)

    # ------------------------------------------------------------- time split
    fit, valid, split = time_split(train, cfg["split"]["validation_months"])
    features = feature_columns(train)
    features = [f for f in features if f in fit.columns]

    # -------------------------------------------------------------- modelling
    models, metrics, valid_pred = train_all(fit, valid, features, seed, cfg["models"])
    skipped_binary = [t for t in BINARY_TARGETS if t not in metrics]
    if skipped_binary:
        print(f"  WARNING: skipped binary target(s) with degenerate labels in the data: {skipped_binary}")
    metrics["time_split"] = split
    metrics["excluded_columns_gate"] = excluded
    # Honest evaluation audit: quantify how much time-split performance is panel-overlap
    # memorisation by re-evaluating on loans disjoint from training.
    metrics["evaluation_audit"] = [
        loan_disjoint_ablation(fit, features, t, seed, n_splits=cfg["models"].get("audit_splits", 5))
        for t in ("next_12m_default_flag", "next_3m_delinquency_flag")
    ]
    _json_dump(metrics, out / "metrics.json")
    valid_pred.to_csv(out / "validation_predictions.csv", index=False)
    (out / "features_used.json").write_text(json.dumps({"features": features}, indent=2))

    # calibration tables per target
    cal_rows = []
    for target, entry in metrics.items():
        if isinstance(entry, dict) and "calibration" in entry:
            yva = valid.dropna(subset=[target])[target].astype(int)
            p = valid_pred.loc[yva.index, target.replace("_flag", "_prob")]
            _, table = calibration_diagnostics(yva, p.values, cfg["models"].get("calibration_bins", 10))
            table.insert(0, "target", target)
            cal_rows.append(table)
    if cal_rows:
        pd.concat(cal_rows, ignore_index=True).to_csv(out / "calibration.csv", index=False)

    # ECE broken down by segment (brief: "Model calibration by vintage or credit band").
    segment_cal_frames = []
    segment_cal_df = None
    for target in BINARY_TARGETS:
        if target in metrics and isinstance(metrics[target], dict) and "calibrated_hgb" in metrics[target]:
            segment_cal_frames.append(calibration_by_segment(
                valid, valid_pred, target, tuple(cfg["models"].get("segment_calibration", [])),
                bins=cfg["models"].get("calibration_bins", 10)))
    if segment_cal_frames:
        segment_cal_df = pd.concat(segment_cal_frames, ignore_index=True)
        segment_cal_df.to_csv(out / "calibration_by_segment.csv", index=False)

    # ---------------------------------------------------- time-to-event stack
    matrix = transition_model(train)
    matrix.to_csv(out / "transition_matrix.csv", index=False)
    markov = transition_curves(matrix, cfg["survival"]["horizon"])
    markov.to_csv(out / "transition_curves.csv", index=False)
    km = competing_risk_curves(train, cfg["survival"]["horizon"])
    if not km.empty and not markov.empty:
        km = km.merge(markov, on="month", how="left")
    km.to_csv(out / "km_curves.csv", index=False)

    # -------------------------------------------------------- explainability
    global_df = global_importance(models, valid, features)
    global_df.to_csv(out / "explainability.csv", index=False)
    queue_preview = None  # filled below
    local_df = local_explanations(models, valid.sample(min(len(valid), 200), random_state=seed),
                                  features, n_rows=cfg["explainability"]["local_rows"],
                                  background=cfg["explainability"]["shap_background"])
    local_df.to_csv(out / "local_explanations.csv", index=False)

    valid = valid.join(quality[["data_quality_score"]])
    error_summaries = {}
    error_frames = []
    for target in BINARY_TARGETS:
        if target not in metrics or not isinstance(metrics[target], dict) or "baseline_rf" not in metrics[target]:
            continue
        threshold = metrics[target].get("recall_at_fixed_precision", {}).get("threshold", 0.5)
        summary = error_analysis(valid, valid_pred, target, threshold)
        if summary:
            error_summaries[target] = summary
        error_frames.append(error_examples(valid, valid_pred, target, threshold))
    if error_frames:
        pd.concat(error_frames, ignore_index=True).to_csv(out / "error_examples.csv", index=False)
    _json_dump(error_summaries, out / "error_analysis.json")

    # Basic segment parity tables for the two headline outcomes (descriptive monitoring,
    # not a legal fairness assessment — see the model card).
    fairness_frames = []
    for target in ("next_12m_default_flag", "next_3m_delinquency_flag"):
        if target in metrics and isinstance(metrics[target], dict) and "baseline_rf" in metrics[target]:
            threshold = metrics[target].get("recall_at_fixed_precision", {}).get("threshold", 0.5)
            fairness_frames.append(fairness_report(valid, valid_pred, target, threshold=threshold))
    fairness_df = pd.concat(fairness_frames, ignore_index=True) if fairness_frames else pd.DataFrame()
    if not fairness_df.empty:
        fairness_df.to_csv(out / "fairness_report.csv", index=False)

    (out / "explainability_report.md").write_text(
        _markdown_explainability(global_df, error_summaries, list(error_summaries)))
    with (out / "explainability_report.md").open("a") as f:
        f.write("\n## Counterfactual explanations\n\n"
                "`counterfactuals.csv` holds what-if re-scores for the top-priority records: "
                "each documented intervention (cure DPD, credit +40, LTV −10, rate −1, 20% paydown) "
                "is applied and the probability delta reported, plus the combined cure. "
                "This answers \"what would change the model's mind\" for a reviewer.\n")
    if not fairness_df.empty:
        with (out / "explainability_report.md").open("a") as f:
            f.write("\n## Segment parity (descriptive monitoring)\n\n"
                    "| Target | Segment | Value | n | Event rate | Predicted rate | FPR | FNR |\n"
                    "|---|---|---|---|---|---|---|---|\n")
            for _, row in fairness_df.iterrows():
                f.write(f"| {row['target']} | {row['segment']} | "
                        f"{row['segment_value']} | {row['n']} | {row['event_rate']} | "
                        f"{row['predicted_positive_rate']} | {row['fpr']} | {row['fnr']} |\n")
            f.write("\n*Descriptive parity monitoring only; full bias/fairness review is a documented "
                    "pre-production gate (see the model card).*\n")

    # ---------------------------------------------------------------- scoring
    scored = test.copy()
    for target, model in models.items():
        if target in ("next_state", "exception_type"):
            label, conf = model.predict(test[features]), model.predict_proba(test[features]).max(axis=1)
            scored[target] = label
            scored[f"{target}_confidence"] = conf.round(4)
        else:
            scored[target.replace("_flag", "_prob")] = model.predict_proba(test[features])[:, 1]

    anomalies = anomaly_scores(test, test_quality_frame, seed)
    scored = scored.join(anomalies)
    scored = scored.join(test_quality_frame[["data_quality_score"]])
    probs_for_conf = {t: scored[t] for t in (
        "next_3m_delinquency_prob", "next_6m_delinquency_prob", "next_12m_default_prob",
        "next_12m_prepayment_prob", "exception_required_prob") if t in scored}
    confidence = model_confidence(probs_for_conf, test_quality_frame["data_quality_score"])
    scored = scored.join(confidence)

    scored["n_breaches"] = test_quality_frame["n_breaches"].values
    # Vectorised action assignment (row-wise apply is too slow at organiser scale).
    anomaly_values = scored["anomaly_score"].to_numpy()
    scored["recommended_action"] = np.select(
        [anomaly_values >= cfg["review"]["escalate_threshold"],
         (anomaly_values >= cfg["review"]["reconcile_threshold"]) | (scored["n_breaches"].to_numpy() > 0)],
        ["escalate_human_review", "reconcile_record"],
        default="monitor")
    scored["top_drivers"] = scored["anomaly_reason"]

    queue = scored.sort_values("anomaly_score", ascending=False).head(cfg["review"]["anomaly_queue_size"])
    queue.to_csv(out / "review_queue.csv", index=False)

    # Counterfactual explanations for the highest-risk records (brief: "Counterfactual explanations").
    if "next_12m_default_flag" in models:
        cf_input = scored.sort_values("next_12m_default_prob", ascending=False).head(
            cfg["explainability"].get("counterfactual_rows", 20) * 3)
        cf_rows = counterfactual_explanations(
            cf_input, models["next_12m_default_flag"], features, "next_12m_default_flag",
            n=cfg["explainability"].get("counterfactual_rows", 20))
        cf_rows.to_csv(out / "counterfactuals.csv", index=False)

    # Anomaly examples with true labels on the validation side (reviewer-ready evidence).
    valid_anom = anomaly_scores(valid.drop(columns=["data_quality_score"], errors="ignore"), quality.reindex(valid.index), seed)
    valid_examples = valid.join(valid_anom).sort_values("anomaly_score", ascending=False).head(cfg["review"]["anomaly_queue_size"])
    keep = [c for c in ("loan_id", "reporting_month", "days_past_due", "current_balance", "current_status",
                        "data_quality_score", "anomaly_score", "anomaly_reason", "next_12m_default_flag",
                        "next_3m_delinquency_flag", "exception_required") if c in valid_examples.columns]
    valid_examples[keep].to_csv(out / "validation_anomaly_examples.csv", index=False)
    # Clean up the temporarily joined columns so the frame stays single-source.
    valid = valid.drop(columns=["data_quality_score"], errors="ignore")

    # ---------------------------------------------------------------- scenarios
    scenario_cache_map = scenario_cache(test, models, features, pack.get("scenarios"))
    scenario_df = scenario_table(test, models, features, pack.get("scenarios"),
                                 cfg["scenarios"]["segment_columns"], cache=scenario_cache_map)
    scenario_df.to_csv(out / "scenario_report.csv", index=False)
    drivers_df = scenario_drivers(test, models, features, pack.get("scenarios"), cache=scenario_cache_map)
    drivers_df.to_csv(out / "scenario_drivers.csv", index=False)
    mc_df = scenario_monte_carlo(test, models, features, pack.get("scenarios"),
                                 cfg["scenarios"]["mc_sims"], cfg["scenarios"]["mc_sample"], seed,
                                 cache=scenario_cache_map)
    mc_df.to_csv(out / "scenario_mc.csv", index=False)
    cluster_df = stress_sensitivity_by_cluster(scenario_cache_map, test, pack.get("scenarios"))
    if not cluster_df.empty:
        cluster_df.to_csv(out / "stress_sensitivity_clusters.csv", index=False)
    (out / "scenario_report.md").write_text(_markdown_scenario(scenario_df, drivers_df, mc_df))
    if not cluster_df.empty:
        with (out / "scenario_report.md").open("a") as f:
            f.write("\n## Stress sensitivity by feature cluster\n\n"
                    "`stress_sensitivity_clusters.csv` splits the stress impact by credit band × LTV band, "
                    "showing which segments absorb the scenario instead of only the portfolio average.\n")

    # ---------------------------------------------------------------- submission
    template_cols = None
    if "template" in pack and pack["template"] is not None:
        template_cols = [str(c).strip().lower().replace(" ", "_") for c in pack["template"].columns]
    submission_cols = template_cols or STANDARD_SUBMISSION_COLUMNS
    missing_sub_cols = [c for c in submission_cols if c not in scored.columns]
    if missing_sub_cols:
        print(f"  WARNING: submission column(s) unavailable (degenerate targets skipped): {missing_sub_cols}")
    submission_cols = [c for c in submission_cols if c in scored.columns]
    scored[submission_cols].to_csv(out / "submission.csv", index=False)

    # ------------------------------------------------------------------ copilot
    rules = parse_rules(pack.get("rules")) if "rules" in pack else []
    dictionary = pack.get("dictionary", "")
    audit_path = out / "llm_audit_log.jsonl"
    audit_path.write_text("")  # fresh audit trail per run
    note = ""  # stays empty when no review queue exists, so the dashboard never references an undefined name
    if len(queue):
        record = queue.iloc[0]
        note, audit_entries = demonstrate_governance(record, dictionary, rules, cfg.get("llm"))
        (out / "reviewer_note.md").write_text(note + "\n")
        for entry in audit_entries:
            log_audit(entry, audit_path)
        queue_preview = record[["loan_id", "anomaly_score", "anomaly_reason"]].to_dict()

    (out / "model_card.md").write_text(_markdown_model_card(metrics, features, split, report, segment_cal_df))

    # ------------------------------------------------------------ dashboard
    batch_payload = []
    for _, row in batch.iterrows():
        batch_payload.append({
            "month_ordinal": int(pd.Timestamp(row["reporting_month"]).to_period("M").ordinal),
            "month": str(row["reporting_month"])[:7],
            "mean_quality_score": row["mean_quality_score"],
            "breaches_per_1k_rows": row["breaches_per_1k_rows"],
        })
    metric_rows = [["Target", "Model", "ROC-AUC", "PR-AUC", "Brier", "ECE"]]
    for target, entry in metrics.items():
        if not isinstance(entry, dict):
            continue
        if "calibrated_hgb" in entry:
            m = entry["calibrated_hgb"]
            metric_rows.append([target, "calibrated HGB", m.get("roc_auc"), m.get("pr_auc"),
                                m.get("brier"), entry.get("calibration", {}).get("ece")])
            b = entry["baseline_rf"]
            metric_rows.append([target, "baseline RF", b.get("roc_auc"), b.get("pr_auc"),
                                b.get("brier"), "—"])
            lgbm_entry = entry.get("lgbm_candidate", {})
            if isinstance(lgbm_entry, dict) and "roc_auc" in lgbm_entry:
                metric_rows.append([target, "LightGBM", lgbm_entry.get("roc_auc"), lgbm_entry.get("pr_auc"),
                                    lgbm_entry.get("brier"), lgbm_entry.get("calibration", {}).get("ece")])
    if "next_state" in metrics:
        metric_rows.append(["next_state", "RF multiclass", f"macro-F1 {metrics['next_state']['macro_f1']}",
                            f"weighted-F1 {metrics['next_state']['weighted_f1']}", "—", "—"])
    drift_rows = [["Column", "PSI"]]
    drift_rows += [[c, v] for c, v in sorted(report["train_test_numeric_psi"].items(), key=lambda kv: -kv[1])
                   if v > 0.01][:10]
    drift_rows += [[c, v] for c, v in sorted(report["train_test_categorical_psi"].items(), key=lambda kv: -kv[1])
                   if v > 0.01][:5]
    queue_cols = [c for c in ("loan_id", "anomaly_score", "anomaly_reason", "next_12m_default_prob",
                              "data_quality_score", "recommended_action") if c in queue.columns]
    queue_rows = [queue_cols] + [[_fmt_cell(queue.iloc[i][c]) for c in queue_cols]
                                 for i in range(min(15, len(queue)))]
    audit_rows = []
    for line in audit_path.read_text().splitlines():
        if line.strip():
            audit_rows.append(json.loads(line))
    top_imp = []
    if "global_df" in locals() and not global_df.empty:
        for t in ("next_12m_default_flag", "exception_required"):
            sub = global_df[global_df["target"] == t].sort_values("permutation_importance", ascending=False).head(4)
            for _, r in sub.iterrows():
                top_imp.append({"target": t.replace("_flag", ""), "feature": str(r["feature"]), "importance": round(float(r["permutation_importance"]), 4)})

    top_cf = []
    if "cf_rows" in locals() and not cf_rows.empty:
        for _, r in cf_rows.head(5).iterrows():
            top_cf.append({
                "loan_id": str(r["loan_id"]),
                "baseline_prob": round(float(r["baseline_probability"]), 4),
                "best_counterfactual": str(r["best_counterfactual"]),
                "all_cured_prob": round(float(r["all_cured_probability"]), 4),
                "all_cured_delta": round(float(r["all_cured_delta"]), 4),
            })

    payload = {
        "meta": {"cutoff": split["cutoff"], "n_features": len(features), "seed": seed,
                 "generated_utc": datetime.now(timezone.utc).isoformat()},
        "kpis": {
            "rows": {"label": "Train rows", "value": f"{report['rows']:,}"},
            "loans": {"label": "Train loans", "value": f"{train.loan_id.nunique():,}" if "loan_id" in train else "n/a"},
            "features": {"label": "Model features", "value": str(len(features))},
            "quality_low": {"label": "Rows < 60 quality", "value": f"{report['share_rows_below_60_quality']:.1%}"},
            "queue": {"label": "Review queue", "value": str(len(queue))},
        },
        "metrics": metric_rows,
        "km_curves": km.head(cfg["survival"]["horizon"]).to_dict("records"),
        "batch_quality": batch_payload,
        "scenarios_mc": mc_df.to_dict("records"),
        "drift": drift_rows,
        "review_queue": queue_rows,
        "audit": audit_rows,
        "reviewer_note": note,
        "explainability": top_imp,
        "counterfactuals": top_cf,
    }
    (out / "dashboard.html").write_text(render_dashboard(payload))

    # ------------------------------------------------------------- runs log
    run_entry = {
        "kind": "pipeline_run",
        "config": str(Path(config_path).name),
        "seed": seed,
        "train_rows": report["rows"],
        "test_rows": len(test),
        "n_features": len(features),
        "split_cutoff": split["cutoff"],
        "metrics": {t: entry for t, entry in metrics.items() if t not in (
            "time_split", "evaluation_audit", "excluded_columns_gate") and not str(t).startswith("monthly_")},
        "evaluation_audit": metrics.get("evaluation_audit"),
    }
    tracking_result = log_run(run_entry, out / "runs_log.jsonl", cfg.get("tracking"))
    if tracking_result.get("wandb") and tracking_result["wandb"] != "not_configured":
        print(f"  Tracking: {tracking_result['wandb']}")

    print("=" * 78)
    print(f"Loan Performance Intelligence Engine — run complete at {datetime.now(timezone.utc).isoformat()}")
    print(f"Outputs: {out.resolve()}")
    print(f"Data: {report['rows']:,} train rows, {len(test):,} test rows | features: {len(features)} | "
          f"split cutoff: {split['cutoff']}")
    for target, entry in metrics.items():
        if isinstance(entry, dict) and "calibrated_hgb" in entry:
            m = entry["calibrated_hgb"]
            if "roc_auc" in m:
                print(f"  {target:<28} ROC-AUC {m['roc_auc']:<7} PR-AUC {m['pr_auc']:<7} Brier {m['brier']:<7} "
                      f"ECE {entry['calibration'].get('ece', 'n/a')}")
    if "next_state" in metrics:
        print(f"  {'next_state':<28} macro-F1 {metrics['next_state']['macro_f1']} "
              f"(majority baseline {metrics['next_state']['majority_baseline_macro_f1']})")
    if queue_preview:
        print(f"  Review queue head: {queue_preview}")
    print(f"  Anomaly queue rows: {len(queue)} | submission rows: {len(scored)} | "
          f"scenarios: {scenario_df['scenario'].nunique() if not scenario_df.empty else 0}")
    print("=" * 78)

    dashboard_file = (out / "dashboard.html").resolve()
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(dashboard_file.as_uri())
            print(f"🚀 Automatically opened dashboard in your browser: {dashboard_file.as_uri()}")
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Performance Intelligence Engine Pipeline")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML configuration file")
    parser.add_argument("--data-dir", default=None, help="Optional override for input data directory")
    parser.add_argument("--output-dir", default=None, help="Optional override for output directory")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open dashboard in browser")
    args = parser.parse_args()
    main(args.config, data_dir=args.data_dir, output_dir=args.output_dir, open_browser=not args.no_browser)
