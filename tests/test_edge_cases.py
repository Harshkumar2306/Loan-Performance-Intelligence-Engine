"""Edge-case and adversarial tests: state mapping, absorbing transitions, monthly metrics,
operating points, schema gates, LLM fallback honesty, scenario prepayment impact, security
hygiene, and a full end-to-end pipeline run on a tiny synthetic pack."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lpi_engine.copilot import grounded_note
from lpi_engine.data import engineer_features, validate_schema
from lpi_engine.intelligence import (
    scenario_drivers,
    scenario_monte_carlo,
    state_of,
    transition_model,
)
from lpi_engine.modeling import (
    best_f1_threshold,
    monthly_validation_metrics,
    recall_at_fixed_precision,
    threshold_metrics,
)
from lpi_engine.tracking import log_run

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- state mapping

def test_state_buckets_match_status_vocabulary():
    """A '30 DPD' status must map to state 30_dpd (previously it mapped to 60_dpd)."""
    assert state_of("30 DPD", 30) == "30_dpd"
    assert state_of("60 DPD", 60) == "60_dpd"
    assert state_of("90 DPD", 90) == "90_dpd"
    assert state_of("Current", 0) == "current"
    assert state_of("Current", 15) == "30_dpd"
    # Only an explicit default status marks the absorbing default state.
    assert state_of("Default", 120) == "default"
    assert state_of("Current", 120) == "90_dpd"
    assert state_of("Prepaid", 0) == "prepaid"
    assert state_of("Paid Off", 0) == "prepaid"


def test_transition_matrix_absorbing_rows_are_pinned_after_renormalisation():
    d = pd.DataFrame({
        "loan_id": ["A", "A", "B", "B", "C", "C"],
        "reporting_month": pd.to_datetime(["2024-01-01", "2024-02-01"] * 3),
        "current_status": ["Current", "Default", "Current", "Prepaid", "Default", "Default"],
        "days_past_due": [0, 120, 0, 0, 120, 120],
    })
    mat = transition_model(d).set_index("from_state")
    assert mat.loc["default", "default"] == 1.0
    assert mat.loc["prepaid", "prepaid"] == 1.0
    assert np.allclose(mat.sum(axis=1), 1.0)


# ---------------------------------------------------------------- monthly metrics bug

def test_monthly_validation_metrics_produce_rows_for_label_target():
    rng = np.random.default_rng(2)
    valid = pd.DataFrame({
        "reporting_month": pd.to_datetime(["2024-07-01"] * 30 + ["2024-08-01"] * 30),
        "next_3m_delinquency_flag": np.r_[rng.integers(0, 2, 30), rng.integers(0, 2, 30)],
    })
    predictions = pd.DataFrame(
        {"next_3m_delinquency_prob": rng.random(60)}, index=valid.index)
    out = monthly_validation_metrics(valid, predictions, "next_3m_delinquency_flag")
    assert len(out) == 2
    assert {"reporting_month", "rows", "roc_auc", "brier"}.issubset(out.columns)
    assert out.roc_auc.between(0, 1).all() and out.brier.between(0, 1).all()
    # The probability column must never be read from the label frame.
    assert monthly_validation_metrics(valid, predictions, "next_3m_delinquency_prob").empty


# ---------------------------------------------------------------- operating points

def test_recall_at_fixed_precision_scans_without_premature_break():
    # Precision dips mid-scan; the target remains reachable at a lower threshold.
    y = pd.Series([1] * 10 + [0] * 90)
    p = np.r_[np.linspace(0.95, 0.85, 10), np.linspace(0.0, 0.5, 90)]
    out = recall_at_fixed_precision(y, p, 0.9)
    assert "threshold" in out and out["precision"] >= 0.9 and out["recall"] > 0.0


def test_recall_at_fixed_precision_reports_unreachable_honestly():
    y = pd.Series([1] * 5 + [0] * 95)
    # All positives sit at the very bottom of the score distribution, so no threshold
    # can reach the precision target — the scan must say so instead of breaking early.
    p = np.r_[np.full(5, 0.01), np.full(95, 0.30)]
    out = recall_at_fixed_precision(y, p, 0.6)
    assert "note" in out and "unreachable" in out["note"]


def test_best_f1_threshold_and_threshold_metrics_bounded():
    rng = np.random.default_rng(4)
    y = pd.Series(rng.integers(0, 2, 300))
    p = np.clip(y * 0.7 + rng.normal(0.2, 0.25, 300), 0, 1)
    bf1 = best_f1_threshold(y, p)
    assert bf1["f1"] >= 0 and 0 <= bf1["threshold"] <= 1
    tm = threshold_metrics(y, p, bf1["threshold"])
    assert 0 <= tm["precision"] <= 1 and 0 <= tm["recall"] <= 1
    assert tm["n_flagged"] >= 0


# ---------------------------------------------------------------- feature engineering

def test_dpd_velocity_is_zero_for_first_observation_even_when_dpd_is_positive():
    df = pd.DataFrame({
        "loan_id": ["L1", "L1", "L2"],
        "reporting_month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]),
        "origination_month": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-06-01"]),
        "loan_age_months": [12, 13, 7], "remaining_term_months": [348, 347, 353],
        "original_balance": [200_000, 200_000, 150_000], "current_balance": [190_000, 188_000, 150_000],
        "days_past_due": [30, 60, 0], "credit_score_band": ["680-739", "680-739", "<620"],
        "ltv_band": ["61-80", "61-80", "81-100"], "dti_band": ["36-43", "36-43", "<=20"],
    })
    out, _ = engineer_features(df)
    assert out.loc[0, "dpd_velocity"] == 0.0  # first observation: no history -> velocity 0
    assert out.loc[1, "dpd_velocity"] == 30.0
    assert out.loc[2, "dpd_velocity"] == 0.0


# ---------------------------------------------------------------- schema gates

def test_validate_schema_rejects_empty_and_missing_columns():
    with pytest.raises(ValueError, match="empty"):
        validate_schema(pd.DataFrame(), "train.csv")
    with pytest.raises(ValueError, match="loan_id"):
        validate_schema(pd.DataFrame({"reporting_month": [1]}), "train.csv")
    with pytest.raises(ValueError, match="entirely null"):
        validate_schema(pd.DataFrame({"loan_id": [None, None],
                                      "reporting_month": ["2024-01-01", "2024-02-01"]}), "train.csv")


# ---------------------------------------------------------------- tracking

def test_log_run_writes_real_timestamp(tmp_path):
    result = log_run({"kind": "pipeline_run", "seed": 1}, tmp_path / "runs.jsonl")
    entry = json.loads((tmp_path / "runs.jsonl").read_text())
    assert entry["timestamp_utc"] and entry["timestamp_utc"] != "null"
    assert result["jsonl"] is True


# ---------------------------------------------------------------- copilot honesty

def test_offline_audit_never_claims_a_configured_llm_model():
    record = pd.Series({"loan_id": "L1", "days_past_due": 30, "current_balance": 100,
                        "anomaly_score": 0.8, "anomaly_reason": "dpd_status_break",
                        "next_12m_default_prob": 0.5, "data_quality_score": 60.0,
                        "top_drivers": "dpd_status_break", "confidence": 0.6})
    note, audit = grounded_note(record, "", [], cfg={"provider": "auto", "base_url": "",
                                                     "model": "qwen/qwen3.8-27b"})
    assert audit["used_llm"] is False
    assert audit["model"] == "grounded-template-v1"  # not the configured LLM name
    assert note.startswith("RECOMMENDATION ONLY")


def test_copilot_never_logs_api_key(monkeypatch, tmp_path):
    from lpi_engine.copilot import log_audit
    record = pd.Series({"loan_id": "L1", "days_past_due": 30, "current_balance": 100,
                        "anomaly_score": 0.8, "anomaly_reason": "dpd_status_break",
                        "next_12m_default_prob": 0.5, "data_quality_score": 60.0,
                        "top_drivers": "dpd_status_break", "confidence": 0.6})
    monkeypatch.setenv("OPENAI_API_KEY", "gsk_SECRET_PLACEHOLDER_12345")
    note, audit = grounded_note(record, "", [], cfg={"provider": "auto",
                                                     "base_url": "https://api.invalid.test",
                                                     "model": "qwen/qwen3.8-27b"})
    blob = json.dumps(audit)
    assert "gsk_SECRET_PLACEHOLDER_12345" not in blob
    audit_path = tmp_path / "llm_audit_log.jsonl"
    log_audit(audit, audit_path)
    assert "gsk_SECRET_PLACEHOLDER_12345" not in audit_path.read_text()


# ---------------------------------------------------------------- scenario impact

def _dummy_scenario_setup():
    train = pd.DataFrame({
        "loan_id": ["A", "B"], "reporting_month": [pd.Timestamp("2024-01-01")] * 2,
        "days_past_due": [0, 0], "current_balance": [100.0, 90.0], "current_status": ["Current"] * 2,
        "loan_age_months": [10, 10], "remaining_term_months": [350, 350],
        "original_balance": [100_000, 100_000], "interest_rate": [6.0, 5.5],
        "credit_score_numeric": [700.0, 650.0], "credit_score_band": ["680-739", "620-679"]})
    features = [c for c in train.columns if c != "loan_id" and not c.startswith("next_")]

    class DummyModel:
        def predict_proba(self, X):
            risk = 0.4 + np.clip((700 - pd.to_numeric(X["credit_score_numeric"])) * 0.001, -0.2, 0.2)
            return np.column_stack([1 - risk, risk])

    return train, features, {"next_12m_default_flag": DummyModel()}


def test_scenario_drivers_report_prepayment_impact():
    train, features, models = _dummy_scenario_setup()
    scenarios = pd.DataFrame([
        {"scenario": "Base", "credit_score_shock": 0.0, "prepayment_uplift": 0.0},
        {"scenario": "High Prepayment", "credit_score_shock": 0.0, "prepayment_uplift": 0.06},
    ])
    drivers = scenario_drivers(train, models, features, scenarios)
    hp = drivers[drivers.scenario == "High Prepayment"].iloc[0]
    assert "delta_prepayment_pp" in drivers.columns
    assert hp["delta_prepayment_pp"] > 0  # the uplift must move the prepayment projection
    assert hp["delta_default_pp"] == 0.0


def test_scenario_monte_carlo_reports_prepayment_bounds():
    train, features, models = _dummy_scenario_setup()
    mc = scenario_monte_carlo(train, models, features, None, n_sims=10, sample=2, seed=1)
    assert "mean_prepayment_rate" in mc.columns
    assert mc.mean_prepayment_rate.between(0, 1).all()
    assert (mc.p05_prepayment_rate <= mc.mean_prepayment_rate).all()
    assert (mc.mean_prepayment_rate <= mc.p95_prepayment_rate).all()


# ---------------------------------------------------------------- security hygiene

def test_no_hardcoded_secrets_in_repository():
    suspicious = []
    patterns = ["BEGIN PRIVATE KEY", "AKIA[0-9A-Z]{16}", "sk-live-", "-----BEGIN"]
    roots = [REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "docs", REPO_ROOT / "config"]
    files = [REPO_ROOT / "README.md", REPO_ROOT / "pyproject.toml"]
    for root in roots:
        files += [p for p in root.rglob("*") if p.is_file()]
    for path in files:
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for pat in patterns:
            if pat in text:
                suspicious.append((str(path), pat))
    assert not suspicious, f"possible secrets: {suspicious}"


# ---------------------------------------------------------------- end-to-end pipeline

def _write_tiny_pack(tmp_path):
    """Minimal organiser-style pack: 16 loans x 14 months train + 2 months test."""
    rng = np.random.default_rng(2026)
    loans = [f"T{i:03}" for i in range(16)]
    rows = []
    for li, loan in enumerate(loans):
        for m in range(14):
            dpd = 0 if rng.random() > 0.1 else int(rng.choice([30, 60, 90, 120]))
            status = "Current" if dpd == 0 else f"{dpd} DPD"
            rows.append({
                "loan_id": loan, "reporting_month": pd.Timestamp("2024-01-01") + pd.DateOffset(months=m),
                "origination_month": pd.Timestamp("2023-06-01"), "month_index": m,
                "loan_age_months": m, "remaining_term_months": 360 - m,
                "original_balance": 200_000.0, "current_balance": 200_000 - m * 500 - (25 if rng.random() < 0.05 else 0),
                "interest_rate": 6.0, "credit_score": int(rng.integers(600, 780)), "ltv": 70.0,
                "servicer_name": "Atlas", "current_status": status, "days_past_due": dpd,
                "document_status": rng.choice(["complete", "missing", "pending"], p=[0.9, 0.05, 0.05]),
                "last_updated_at": pd.Timestamp("2024-01-01") + pd.DateOffset(months=m, days=5),
                "source_system": "core",
            })
    train = pd.DataFrame(rows)
    train["next_3m_delinquency_flag"] = rng.integers(0, 2, len(train))
    train["next_6m_delinquency_flag"] = rng.integers(0, 2, len(train))
    train["next_12m_default_flag"] = rng.integers(0, 2, len(train))
    train["next_12m_prepayment_flag"] = rng.integers(0, 2, len(train))
    train["next_state"] = rng.choice(["current", "30_dpd", "default", "prepaid"], len(train))
    train["exception_type"] = rng.choice(["none", "missing_docs", "data_break"], len(train), p=[0.8, 0.1, 0.1])
    train["exception_required"] = (train.exception_type != "none").astype(int)

    test = pd.DataFrame(rows).head(0)
    test_rows = []
    for li, loan in enumerate(loans):
        for m in range(14, 16):
            test_rows.append({
                "loan_id": loan, "reporting_month": pd.Timestamp("2024-01-01") + pd.DateOffset(months=m),
                "origination_month": pd.Timestamp("2023-06-01"), "month_index": m,
                "loan_age_months": m, "remaining_term_months": 360 - m,
                "original_balance": 200_000.0, "current_balance": 200_000 - m * 500,
                "interest_rate": 6.0, "credit_score": int(rng.integers(600, 780)), "ltv": 70.0,
                "servicer_name": "Atlas", "current_status": "Current", "days_past_due": 0,
                "document_status": "complete",
                "last_updated_at": pd.Timestamp("2024-01-01") + pd.DateOffset(months=m, days=5),
                "source_system": "core",
            })
    # One deliberately broken row so the review queue / copilot demo is guaranteed non-empty.
    test_rows.append({
        "loan_id": "T999", "reporting_month": pd.Timestamp("2025-03-01"),
        "origination_month": pd.Timestamp("2023-06-01"), "month_index": 14,
        "loan_age_months": 14, "remaining_term_months": -5,
        "original_balance": 200_000.0, "current_balance": -25.0,
        "interest_rate": 99.0, "credit_score": 300, "ltv": 250.0,
        "servicer_name": "Atlas", "current_status": "Current", "days_past_due": 800,
        "document_status": "missing",
        "last_updated_at": pd.Timestamp("2025-06-01"),
        "source_system": "core",
    })
    test = pd.DataFrame(test_rows)
    static = pd.DataFrame({
        "loan_id": loans, "credit_score": [700] * 16, "credit_score_band": ["680-739"] * 16,
        "ltv": [70.0] * 16, "ltv_band": ["61-80"] * 16, "dti": [33.0] * 16, "dti_band": ["21-35"] * 16,
        "state": ["CA"] * 16, "loan_purpose": ["purchase"] * 16, "occupancy_type": ["owner"] * 16,
        "property_type": ["sfr"] * 16, "original_balance": [200_000.0] * 16, "interest_rate": [6.0] * 16,
        "term_months": [360] * 16, "origination_month": [pd.Timestamp("2023-06-01")] * 16,
        "vintage": [2023] * 16})
    updates = pd.DataFrame({
        "loan_id": loans[:3], "current_balance": [200_100.0, 195_000.0, 210_000.0],
        "current_status": ["Current"] * 3,
        "last_updated_at": [pd.Timestamp("2024-06-10")] * 3, "source_system": ["servicer"] * 3})
    template = pd.DataFrame(columns=[
        "loan_id", "next_3m_delinquency_prob", "next_6m_delinquency_prob", "next_12m_default_prob",
        "next_12m_prepayment_prob", "next_state", "next_state_confidence", "exception_type",
        "exception_type_confidence", "anomaly_score", "anomaly_reason", "top_drivers",
        "recommended_action", "confidence"])

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train.to_csv(data_dir / "loan_monthly_performance_train.csv", index=False)
    test.to_csv(data_dir / "loan_monthly_performance_test.csv", index=False)
    static.to_csv(data_dir / "loan_static_attributes.csv", index=False)
    updates.to_csv(data_dir / "servicer_updates.csv", index=False)
    template.to_csv(data_dir / "submission_template.csv", index=False)
    (data_dir / "data_dictionary.md").write_text(
        "- `loan_id`: unique loan identifier.\n- `days_past_due`: days past due.\n")
    (data_dir / "validation_rules.json").write_text(
        json.dumps({"rules": ["current_balance >= 0", "days_past_due in [0, 720]"]}))
    return data_dir


def _tiny_config(tmp_path):
    cfg = yaml_safe_load(REPO_ROOT / "config" / "default.yaml")
    cfg["split"]["validation_months"] = 6
    cfg["models"]["bootstrap_samples"] = 100
    cfg["models"]["audit_splits"] = 3
    cfg["scenarios"]["mc_sims"] = 5
    cfg["scenarios"]["mc_sample"] = 20
    cfg["explainability"]["local_rows"] = 5
    cfg["explainability"]["counterfactual_rows"] = 3
    cfg["review"]["anomaly_queue_size"] = 20
    cfg_path = tmp_path / "config_tiny.yaml"
    cfg_path.write_text(yaml_safe_dump(cfg))
    return str(cfg_path)


def yaml_safe_load(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def yaml_safe_dump(obj):
    import yaml
    return yaml.safe_dump(obj)


def test_end_to_end_pipeline_produces_valid_submission(tmp_path):
    from scripts.run_pipeline import main as pipeline_main
    data_dir = _write_tiny_pack(tmp_path)
    out_dir = tmp_path / "out"
    config_path = _tiny_config(tmp_path)
    pipeline_main(config_path, data_dir=str(data_dir), output_dir=str(out_dir), open_browser=False)

    sub = pd.read_csv(out_dir / "submission.csv")
    template = pd.read_csv(data_dir / "submission_template.csv")
    test = pd.read_csv(data_dir / "loan_monthly_performance_test.csv")
    assert list(sub.columns) == list(template.columns)
    assert len(sub) == len(test)
    assert sub.duplicated().sum() == 0
    for c in ("next_3m_delinquency_prob", "next_6m_delinquency_prob", "next_12m_default_prob",
              "next_12m_prepayment_prob", "anomaly_score", "confidence"):
        assert sub[c].between(0, 1).all() and sub[c].notna().all()
    assert sub.loan_id.notna().all() and sub.anomaly_reason.notna().all()
    assert set(sub.recommended_action.unique()) <= {"monitor", "reconcile_record", "escalate_human_review"}

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "time_split" in metrics and metrics["time_split"]["fit_rows"] > 0
    # Monthly temporal-stability metrics must be present and non-empty for the headline targets.
    assert metrics["monthly_next_3m_delinquency_flag"], "monthly metrics were silently empty"
    assert metrics["monthly_next_12m_default_flag"], "monthly metrics were silently empty"

    audit = [json.loads(line) for line in (out_dir / "llm_audit_log.jsonl").read_text().splitlines() if line.strip()]
    assert audit and all("decision_status" in a and "used_llm" in a for a in audit)
    assert any(a["decision_status"] == "rejected_by_human" for a in audit)
    assert all(a["used_llm"] is False for a in audit)  # offline run must not claim a live call

    runs = [json.loads(line) for line in (out_dir / "runs_log.jsonl").read_text().splitlines() if line.strip()]
    assert runs and all(r.get("timestamp_utc") for r in runs)

    for artifact in ("transition_matrix.csv", "transition_curves.csv", "km_curves.csv", "review_queue.csv",
                     "scenario_report.csv", "scenario_drivers.csv", "scenario_mc.csv", "model_card.md",
                     "data_intelligence_report.md", "explainability_report.md", "scenario_report.md"):
        assert (out_dir / artifact).exists(), f"missing {artifact}"

    mc = pd.read_csv(out_dir / "scenario_mc.csv")
    assert "mean_prepayment_rate" in mc.columns
    high_prepay = mc[mc.scenario == "High Prepayment"].iloc[0]
    base = mc[mc.scenario == "Base"].iloc[0]
    assert high_prepay["mean_prepayment_rate"] > base["mean_prepayment_rate"] + 0.01
    drivers = pd.read_csv(out_dir / "scenario_drivers.csv")
    assert "delta_prepayment_pp" in drivers.columns
    tm = pd.read_csv(out_dir / "transition_matrix.csv")
    tm = tm.set_index("from_state")
    for absorbing in ("default", "prepaid", "paid_off"):
        assert tm.loc[absorbing, absorbing] == 1.0
    assert np.allclose(tm.sum(axis=1), 1.0)


def test_pipeline_skips_target_with_no_positive_examples(tmp_path):
    """A binary target with a single class must be skipped, not crash or fabricate metrics."""
    from scripts.run_pipeline import main as pipeline_main
    data_dir = _write_tiny_pack(tmp_path)
    train = pd.read_csv(data_dir / "loan_monthly_performance_train.csv")
    train["next_12m_default_flag"] = 0  # no positives anywhere
    train.to_csv(data_dir / "loan_monthly_performance_train.csv", index=False)
    out_dir = tmp_path / "out2"
    pipeline_main(_tiny_config(tmp_path), data_dir=str(data_dir), output_dir=str(out_dir), open_browser=False)
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "next_12m_default_flag" not in metrics
    sub = pd.read_csv(out_dir / "submission.csv")
    # No fabricated probability column for the skipped target.
    assert "next_12m_default_prob" not in sub.columns


def test_pipeline_handles_one_row_test(tmp_path):
    from scripts.run_pipeline import main as pipeline_main
    data_dir = _write_tiny_pack(tmp_path)
    test = pd.read_csv(data_dir / "loan_monthly_performance_test.csv").head(1)
    test.to_csv(data_dir / "loan_monthly_performance_test.csv", index=False)
    out_dir = tmp_path / "out3"
    pipeline_main(_tiny_config(tmp_path), data_dir=str(data_dir), output_dir=str(out_dir), open_browser=False)
    sub = pd.read_csv(out_dir / "submission.csv")
    assert len(sub) == 1
    assert sub[["next_3m_delinquency_prob", "anomaly_score", "confidence"]].notna().all().all()


def test_unseen_categorical_values_do_not_break_prediction():
    from lpi_engine.modeling import preprocessor
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr = pd.DataFrame({"state": ["CA", "TX", "NY", "CA"], "x": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 0, 1])
    model = Pipeline([("prep", preprocessor(Xtr)), ("model", HistGradientBoostingClassifier(max_iter=30))])
    model.fit(Xtr, y)
    Xte = pd.DataFrame({"state": ["ZZ"], "x": [2.5]})  # unseen category
    p = model.predict_proba(Xte)
    assert p.shape == (1, 2) and 0 <= p[0, 1] <= 1


def test_time_split_on_single_month_raises_clear_error():
    from lpi_engine.data import time_split
    d = pd.DataFrame({"loan_id": [1, 2], "reporting_month": pd.to_datetime(["2024-01-01", "2024-01-01"])})
    with pytest.raises(ValueError, match="distinct"):
        time_split(d, 1)
