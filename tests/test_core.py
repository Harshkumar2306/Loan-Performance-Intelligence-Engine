"""Core invariant tests: leakage gates, time splits, censoring, competing risks, quality scoring, governance."""
import json

import numpy as np
import pandas as pd
import pytest

from lpi_engine.copilot import grounded_note, parse_dictionary, parse_rules, rejected_example_audit
from lpi_engine.dashboard import render as render_dashboard
from lpi_engine.data import (
    TARGETS,
    assert_no_leakage,
    band_midpoint,
    engineer_features,
    engineer_panel,
    feature_columns,
    time_split,
)
from lpi_engine.explainability import fairness_report, model_confidence
from lpi_engine.intelligence import (
    anomaly_scores,
    competing_risk_curves,
    scenario_cache,
    scenario_drivers,
    scenario_table,
    stress_sensitivity_by_cluster,
    transition_model,
)
from lpi_engine.modeling import calibration_diagnostics, recall_at_fixed_precision
from lpi_engine.profiling import profile, psi, quality_scores, score_test_frame


def _panel() -> pd.DataFrame:
    """Small synthetic panel with one censored loan, one default and one prepayment."""
    rows = []
    # Loan A: current for 3 months (censored at end)
    for i in range(3):
        rows.append({"loan_id": "A", "reporting_month": pd.Timestamp(f"2024-0{i+1}-01"),
                     "month_index": i, "days_past_due": 0, "current_status": "Current",
                     "current_balance": 100.0, "loan_age_months": i, "remaining_term_months": 30 - i})
    # Loan B: current -> default at month 2
    rows.append({"loan_id": "B", "reporting_month": pd.Timestamp("2024-01-01"), "month_index": 0,
                 "days_past_due": 0, "current_status": "Current", "current_balance": 100.0,
                 "loan_age_months": 0, "remaining_term_months": 30})
    rows.append({"loan_id": "B", "reporting_month": pd.Timestamp("2024-02-01"), "month_index": 1,
                 "days_past_due": 120, "current_status": "Default", "current_balance": 100.0,
                 "loan_age_months": 1, "remaining_term_months": 29})
    rows.append({"loan_id": "B", "reporting_month": pd.Timestamp("2024-03-01"), "month_index": 2,
                 "days_past_due": 120, "current_status": "Default", "current_balance": 100.0,
                 "loan_age_months": 2, "remaining_term_months": 28})
    # Loan C: current -> prepaid at month 2
    rows.append({"loan_id": "C", "reporting_month": pd.Timestamp("2024-01-01"), "month_index": 0,
                 "days_past_due": 0, "current_status": "Current", "current_balance": 100.0,
                 "loan_age_months": 0, "remaining_term_months": 30})
    rows.append({"loan_id": "C", "reporting_month": pd.Timestamp("2024-02-01"), "month_index": 1,
                 "days_past_due": 0, "current_status": "Prepaid", "current_balance": 0.0,
                 "loan_age_months": 1, "remaining_term_months": 29})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- leakage gates

def test_targets_are_not_features():
    cols = feature_columns(pd.DataFrame({"loan_id": [1], "credit_score": [700],
                                         "next_12m_default_flag": [1], "default_flag": [0]}))
    assert "next_12m_default_flag" not in cols and "default_flag" not in cols
    assert cols == ["credit_score"]


def test_leakage_gate_rejects_banned_feature():
    with pytest.raises(ValueError):
        assert_no_leakage(["next_12m_default_flag"], ["loan_id", "credit_score"])


def test_leakage_gate_reports_excluded_columns():
    excluded = assert_no_leakage(["credit_score"], ["loan_id", "credit_score", "next_state", "loss_severity_band"])
    assert "next_state" in excluded and "loss_severity_band" in excluded


def test_time_split_is_chronological_and_has_no_future_rows_in_fit():
    d = pd.DataFrame({"loan_id": [1, 1, 2, 2],
                      "reporting_month": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-02-01", "2024-04-01"])})
    a, b, evidence = time_split(d, 2)
    assert a.reporting_month.max() < b.reporting_month.min()
    assert evidence["method"].startswith("strict")


# ---------------------------------------------------------------- feature engineering

def test_band_midpoint_parsing():
    assert band_midpoint("<620", "credit_score_band") == 590.0
    assert band_midpoint("680-739", "credit_score_band") == 710.0
    assert band_midpoint("81-100", "ltv_band") == 90.5
    assert band_midpoint("unknown-55", "ltv_band") == 55.0  # generic single-number fallback


def test_engineered_features_are_bounded_and_historical():
    df = pd.DataFrame({
        "loan_id": ["L1", "L1", "L2"], "reporting_month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]),
        "origination_month": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-06-01"]),
        "loan_age_months": [12, 13, 7], "remaining_term_months": [348, 347, 353],
        "original_balance": [200_000, 200_000, 150_000], "current_balance": [190_000, 188_000, 150_000],
        "days_past_due": [0, 30, 0], "credit_score_band": ["680-739", "680-739", "<620"],
        "ltv_band": ["61-80", "61-80", "81-100"], "dti_band": ["36-43", "36-43", "<=20"],
    })
    out, new_cols = engineer_features(df)
    assert out.balance_ratio.between(0, 2).all()
    assert out.term_ratio.between(0, 1).all()
    assert out.loc[0, "dpd_velocity"] == 0  # first observation has no history
    assert out.loc[1, "dpd_velocity"] == 30
    assert out.loc[1, "ever_dpd"] == 1 and out.loc[0, "ever_dpd"] == 0
    assert out.loc[0, "vintage_year"] == "2023"
    assert set(new_cols) == {"credit_score_numeric", "ltv_numeric", "dti_numeric", "balance_ratio",
                             "term_ratio", "dpd_velocity", "ever_dpd", "paydown_rate", "vintage_year"}


# ---------------------------------------------------------------- transitions & survival

def test_transition_excludes_censored_final_observation():
    d = pd.DataFrame({"loan_id": [1, 1], "reporting_month": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                      "current_status": ["Current", "30 DPD"], "days_past_due": [0, 15]})
    matrix = transition_model(d)
    assert float(matrix.loc[matrix.from_state == "current", "30_dpd"].iloc[0]) == 1.0


def test_transition_matrix_rows_sum_to_one_and_absorbing_persist():
    d = _panel()
    matrix = transition_model(d).set_index("from_state")
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert matrix.loc["default", "default"] == 1.0
    assert matrix.loc["prepaid", "prepaid"] == 1.0


def test_competing_risk_curves_exact_on_small_panel():
    curves = competing_risk_curves(_panel(), horizon=3)
    assert curves["km_survival"].is_monotonic_decreasing
    assert (curves["cif_default"].between(0, 1).all()) and (curves["cif_prepay"].between(0, 1).all())
    # Three loans at risk at month 1: B defaults, C prepays -> CIF 1/3 each; S drops to 1/3.
    assert curves.loc[curves.month == 1, "cif_default"].iloc[0] == pytest.approx(1 / 3, abs=1e-4)
    assert curves.loc[curves.month == 1, "cif_prepay"].iloc[0] == pytest.approx(1 / 3, abs=1e-4)
    assert curves.loc[curves.month == 1, "km_survival"].iloc[0] == pytest.approx(1 / 3, abs=1e-4)
    # Censoring-blind naive rate (2 default rows / 8) must differ from the censoring-aware CIF.
    assert not np.isclose(curves["naive_default_rate"].iloc[0], curves["cif_default"].iloc[-1])


# ---------------------------------------------------------------- quality & profiling

def test_quality_scores_bounded_and_penalise_duplicates():
    clean = pd.DataFrame({"loan_id": ["A", "B"], "reporting_month": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                          "current_balance": [100, 200], "current_status": ["Current", "Current"],
                          "days_past_due": [0, 0], "loan_age_months": [1, 2],
                          "remaining_term_months": [29, 28], "origination_month": pd.to_datetime(["2023-12-01", "2023-12-01"]),
                          "last_updated_at": pd.to_datetime(["2024-01-10", "2024-02-10"])})
    dup = pd.concat([clean, clean.iloc[[0]]], ignore_index=True)
    quality = score_test_frame(dup)
    assert quality.data_quality_score.between(0, 100).all()
    # Both copies of the duplicated row are flagged and scored below a clean row.
    assert quality.loc[0, "duplicate_row"] and quality.loc[2, "duplicate_row"]
    assert quality.loc[0, "data_quality_score"] < quality.loc[1, "data_quality_score"]
    assert quality.loc[2, "data_quality_score"] < quality.loc[1, "data_quality_score"]


def test_profile_reports_quality_and_drift():
    train = pd.DataFrame({"loan_id": ["A"] * 10, "reporting_month": pd.date_range("2024-01-01", periods=10, freq="MS"),
                          "current_balance": np.linspace(100, 200, 10), "current_status": ["Current"] * 10,
                          "days_past_due": [0] * 10, "loan_age_months": range(10), "remaining_term_months": range(30, 20, -1),
                          "origination_month": pd.date_range("2023-01-01", periods=10, freq="MS"),
                          "last_updated_at": pd.date_range("2024-01-10", periods=10, freq="MS")})
    test = train.copy()
    test["current_balance"] = test["current_balance"] * 2  # strong drift
    report, quality, batch = profile(train, test)
    assert report["quality_score_distribution"]["max"] <= 100
    assert "train_test_numeric_psi" in report
    assert quality.data_quality_score.between(0, 100).all()
    assert len(batch) == 10


def test_psi_is_zero_for_identical_and_positive_for_shifted():
    a = pd.Series(np.random.default_rng(0).normal(0, 1, 1000))
    assert psi(a, a) == pytest.approx(0.0, abs=1e-6)
    assert psi(a, a + 3) > 0.5


def test_association_rules_emit_lift():
    from lpi_engine.profiling import association_rules
    rng = np.random.default_rng(3)
    n = 2000
    # exception_required is strongly associated with document_status=missing
    frame = pd.DataFrame({"document_status": rng.choice(["complete", "missing"], n, p=[0.9, 0.1])})
    frame["exception_required"] = np.where(
        (frame.document_status == "missing") | (rng.random(n) < 0.05), 1, 0)
    rules = association_rules(frame, ["document_status", "exception_required"])
    assert rules, "rules should be produced"
    assert all("lift" in r and "rule" in r and "support" in r for r in rules)
    best = rules[0]
    assert "document_status=missing" in best["rule"] and best["lift"] > 1.0


def test_copilot_auto_provider_falls_back_without_key(monkeypatch):
    record = pd.Series({"loan_id": "L1", "days_past_due": 30, "current_balance": 100,
                        "anomaly_score": 0.8, "anomaly_reason": "dpd_status_break",
                        "next_12m_default_prob": 0.5, "data_quality_score": 60.0,
                        "top_drivers": "dpd_status_break", "confidence": 0.6})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    note, audit = grounded_note(record, "", [], cfg={"provider": "auto", "base_url": "", "model": "gpt-4o-mini"})
    assert note.startswith("RECOMMENDATION ONLY")
    assert audit["used_llm"] is False  # no key -> honest fallback to the grounded template


# ---------------------------------------------------------------- modelling metrics

def test_recall_at_fixed_precision_degrades_with_higher_target():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, 800))
    p = np.clip(y.astype(float) * 0.7 + rng.normal(0.2, 0.3, 800), 0, 1)
    loose = recall_at_fixed_precision(y, p, 0.3)
    tight = recall_at_fixed_precision(y, p, 0.8)
    assert loose["recall"] >= tight["recall"] - 1e-6


def test_calibration_ece_bounded_and_zero_for_perfect():
    y = pd.Series([0] * 50 + [1] * 50)
    p = np.array([0.1] * 50 + [0.9] * 50)
    diag, table = calibration_diagnostics(y, p, bins=10)
    assert 0.0 <= diag["ece"] <= 1.0
    assert len(table) >= 2


# ---------------------------------------------------------------- anomaly & scenario

def test_anomaly_scores_bounded_and_rule_breaches_rank_higher():
    df = _panel().copy()
    df["current_balance"] = df["current_balance"].astype(float)
    quality = score_test_frame(df)
    scores = anomaly_scores(df, quality, seed=42)
    assert scores.anomaly_score.between(0, 1).all()
    assert scores.anomaly_reason.notna().all()
    breached = quality[quality.negative_balance | quality.dpd_impossible].index
    clean = quality[~(quality.negative_balance | quality.dpd_impossible)].index
    if len(breached) and len(clean):
        assert scores.loc[breached, "anomaly_score"].mean() >= scores.loc[clean, "anomaly_score"].mean()


def test_scenario_adverse_credit_never_lowers_default_risk():
    train = pd.DataFrame({"loan_id": ["A"], "reporting_month": [pd.Timestamp("2024-01-01")],
                          "days_past_due": [0], "current_balance": [100.0], "current_status": ["Current"],
                          "loan_age_months": [10], "remaining_term_months": [350], "original_balance": [100_000],
                          "interest_rate": [6.0], "credit_score": [700], "credit_score_numeric": [700.0],
                          "credit_score_band": ["680-739"], "origination_month": [pd.Timestamp("2023-03-01")]})
    features = [c for c in train.columns if c not in TARGETS and c != "loan_id"]
    class DummyModel:
        def predict_proba(self, X):
            risk = 0.4 + np.clip((700 - pd.to_numeric(X["credit_score_numeric"])) * 0.001, -0.2, 0.2)
            return np.column_stack([1 - risk, risk])
    models = {"next_12m_default_flag": DummyModel()}
    table = scenario_table(train, models, features, None, ["credit_score_band"])
    base = table[(table.scenario == "Base") & (table.segment == "credit_score_band")].next_12m_default_prob.mean()
    adverse = table[(table.scenario == "Adverse Credit") & (table.segment == "credit_score_band")].next_12m_default_prob.mean()
    assert adverse >= base - 1e-9
    drivers = scenario_drivers(train, models, features, None)
    adverse_drivers = drivers[drivers.scenario == "Adverse Credit"].iloc[0]
    assert adverse_drivers.credit_shock_contribution_pp > 0


def test_model_confidence_blend_bounded():
    quality = pd.Series([100.0, 60.0, 30.0], index=[0, 1, 2])
    probs = {f"t{i}": pd.Series([0.5, 0.9, 0.1], index=[0, 1, 2]) for i in range(2)}
    conf = model_confidence(probs, quality)
    assert conf.confidence.between(0, 1).all()
    assert conf.loc[1, "model_confidence"] > conf.loc[0, "model_confidence"]


# ---------------------------------------------------------------- copilot governance

def test_copilot_note_is_advisory_and_grounded():
    record = pd.Series({"loan_id": "L9", "days_past_due": 45, "current_balance": 123_000,
                        "anomaly_score": 0.81, "anomaly_reason": "dpd_status_break",
                        "next_12m_default_prob": 0.62, "data_quality_score": 55.0,
                        "top_drivers": "45 days past due", "confidence": 0.6})
    dictionary = "# Data dictionary\n- `days_past_due`: days since due.\n- `current_balance`: unpaid principal.\n"
    note, audit = grounded_note(record, dictionary, ["days_past_due in [0, 720]"], cfg={"provider": "template"})
    assert note.startswith("RECOMMENDATION ONLY")
    assert audit["decision_status"] == "recommendation_pending_human_review"
    assert audit["used_llm"] is False


def test_copilot_retrieval_and_rejected_example():
    assert parse_dictionary("- `loan_id`: unique identifier.") == {"loan_id": "unique identifier."}
    assert len(parse_rules({"rules": ["a >= 0", "b <= 1"]})) == 2
    rejected = rejected_example_audit()
    assert rejected["decision_status"] == "rejected_by_human"
    assert "rejection_reason" in rejected
    # Honesty guard: the offline simulation must never claim a model was called.
    assert rejected["used_llm"] is False
    assert rejected["scenario"] == "offline_governance_simulation"


def test_demonstrate_governance_offline_labels_honestly(monkeypatch):
    from lpi_engine.copilot import demonstrate_governance
    record = pd.Series({"loan_id": "L9", "days_past_due": 45, "current_balance": 123_000,
                        "anomaly_score": 0.81, "anomaly_reason": "dpd_status_break",
                        "next_12m_default_prob": 0.62, "data_quality_score": 55.0,
                        "top_drivers": "45 days past due", "confidence": 0.6})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    note, entries = demonstrate_governance(record, "", [], cfg={"provider": "auto", "base_url": ""})
    assert note.startswith("RECOMMENDATION ONLY")
    assert len(entries) == 2
    grounded, rejected = entries
    assert grounded["used_llm"] is False and grounded["decision_status"] == "recommendation_pending_human_review"
    assert rejected["used_llm"] is False and rejected["decision_status"] == "rejected_by_human"
    assert rejected["scenario"] == "offline_governance_simulation"


def test_copilot_maps_rule_names_to_validation_rules():
    record = pd.Series({"loan_id": "L1", "days_past_due": 0, "current_balance": -5,
                        "anomaly_score": 0.9, "anomaly_reason": "negative_balance",
                        "next_12m_default_prob": 0.5, "data_quality_score": 40.0,
                        "top_drivers": "negative_balance", "confidence": 0.7})
    _, audit = grounded_note(record, "", [], cfg={"provider": "template"})
    assert any("current_balance >= 0" in r for r in audit["grounding"]["matched_rules"])
    assert audit["grounding"]["evidence"]["next_12m_default_prob"] == 0.5


def test_km_curves_have_finite_confidence_bands():
    curves = competing_risk_curves(_panel(), horizon=4)
    for col in ("km_survival_ci_low", "km_survival_ci_high", "cif_default_ci_low", "cif_default_ci_high",
                "cif_prepay_ci_low", "cif_prepay_ci_high"):
        assert curves[col].between(0, 1).all() and curves[col].notna().all()


def test_loan_disjoint_ablation_reports_lower_bound():
    from lpi_engine.modeling import loan_disjoint_ablation
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({
        "loan_id": np.repeat([f"L{i}" for i in range(30)], 10),
        "credit_score": rng.normal(700, 60, n),
        "interest_rate": rng.normal(6, 1, n),
        "days_past_due": rng.integers(0, 60, n),
    })
    df["next_12m_default_flag"] = rng.binomial(1, np.clip((df.credit_score < 650).astype(float) * 0.6 + 0.1, 0, 1))
    result = loan_disjoint_ablation(df, ["credit_score", "interest_rate", "days_past_due"], "next_12m_default_flag", seed=1)
    assert 0.0 <= result["auc_loan_disjoint_mean"] <= 1.0
    assert result["auc_loan_disjoint_min"] <= result["auc_loan_disjoint_mean"] <= result["auc_loan_disjoint_max"]
    assert len(result["auc_loan_disjoint_splits"]) >= 3  # repeated splits for stability


def test_scenario_monte_carlo_produces_ordered_bounds():
    from lpi_engine.intelligence import scenario_monte_carlo
    train = pd.DataFrame({"loan_id": ["A"], "reporting_month": [pd.Timestamp("2024-01-01")],
                          "days_past_due": [0], "current_balance": [100.0], "current_status": ["Current"],
                          "loan_age_months": [10], "remaining_term_months": [350], "original_balance": [100_000],
                          "interest_rate": [6.0], "credit_score": [700], "credit_score_numeric": [700.0],
                          "credit_score_band": ["680-739"]})
    features = [c for c in train.columns if c != "loan_id" and not c.startswith("next_")]
    class DummyModel:
        def predict_proba(self, X):
            risk = 0.5 + np.clip((700 - pd.to_numeric(X["credit_score_numeric"])) * 0.001, -0.2, 0.2)
            return np.column_stack([1 - risk, risk])
    models = {"next_12m_default_flag": DummyModel()}
    mc = scenario_monte_carlo(train, models, features, None, n_sims=20, sample=2, seed=3)
    assert mc.p05_default_rate.min() >= 0 and mc.p95_default_rate.max() <= 1
    assert (mc.p05_default_rate <= mc.mean_default_rate).all()
    assert (mc.mean_default_rate <= mc.p95_default_rate).all()


# ---------------------------------------------------------------- train/serve consistency

def test_engineer_panel_preserves_loan_history_across_split():
    train = pd.DataFrame({
        "loan_id": ["L1", "L1"], "reporting_month": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "origination_month": pd.to_datetime(["2023-01-01", "2023-01-01"]),
        "loan_age_months": [12, 13], "remaining_term_months": [348, 347],
        "original_balance": [200_000, 200_000], "current_balance": [195_000, 190_000],
        "days_past_due": [0, 30], "credit_score_band": ["680-739", "680-739"],
        "ltv_band": ["61-80", "61-80"], "dti_band": ["36-43", "36-43"],
    })
    test = pd.DataFrame({
        "loan_id": ["L1"], "reporting_month": pd.to_datetime(["2024-03-01"]),
        "origination_month": pd.to_datetime(["2023-01-01"]),
        "loan_age_months": [14], "remaining_term_months": [346],
        "original_balance": [200_000], "current_balance": [185_000],
        "days_past_due": [0], "credit_score_band": ["680-739"],
        "ltv_band": ["61-80"], "dti_band": ["36-43"],
    })
    train_out, test_out = engineer_panel(train, test)
    # The loan's February row had 30 DPD, so the test row's velocity must see that history (-30),
    # and paydown must use the February balance — not reset to zero.
    assert test_out.loc[0, "dpd_velocity"] == -30
    assert test_out.loc[0, "ever_dpd"] == 1
    assert abs(test_out.loc[0, "paydown_rate"] - (190_000 - 185_000) / 190_000) < 1e-9
    assert len(train_out) == 2 and len(test_out) == 1


def test_feature_columns_deduplicates_credit_score():
    df, _ = engineer_features(pd.DataFrame({
        "loan_id": ["A"], "reporting_month": [pd.Timestamp("2024-01-01")], "credit_score": [700],
        "credit_score_band": ["680-739"], "days_past_due": [0], "current_balance": [100.0],
        "loan_age_months": [1], "remaining_term_months": [29], "origination_month": [pd.Timestamp("2023-12-01")],
        "last_updated_at": [pd.Timestamp("2024-01-10")], "current_status": ["Current"]}))
    cols = feature_columns(df)
    assert "credit_score_numeric" in cols
    assert "credit_score" not in cols  # subsumed by the engineered parse


def test_time_split_rejects_too_many_validation_months():
    d = pd.DataFrame({"loan_id": [1, 2], "reporting_month": pd.to_datetime(["2024-01-01", "2024-02-01"])})
    with pytest.raises(ValueError):
        time_split(d, 3)


# ---------------------------------------------------------------- competing-risk identity

def test_cif_identity_holds_with_censoring():
    curves = competing_risk_curves(_panel(), horizon=4)
    for _, row in curves.iterrows():
        # Aalen-Johansen identity: CIF_default + CIF_prepay == 1 - KM survival (any event).
        assert row["cif_default"] + row["cif_prepay"] == pytest.approx(1 - row["km_survival"], abs=1e-4)


def test_fairness_report_bounded_and_labelled():
    valid = pd.DataFrame({"loan_id": ["A", "B", "C", "D"],
                          "credit_score_band": ["<620", "<620", "740+", "740+"],
                          "state": ["CA", "CA", "TX", "TX"],
                          "next_12m_default_flag": [1, 0, 1, 0]})
    predictions = pd.DataFrame({"next_12m_default_prob": [0.8, 0.2, 0.7, 0.3]}, index=valid.index)
    report = fairness_report(valid, predictions, "next_12m_default_flag", threshold=0.5)
    assert set(report.columns) >= {"target", "segment", "segment_value", "fpr", "fnr"}
    assert report.fpr.between(0, 1).all() and report.fnr.between(0, 1).all()
    assert report.predicted_positive_rate.between(0, 1).all()


def test_scenario_cache_matches_uncached():
    train = pd.DataFrame({"loan_id": ["A", "B"], "reporting_month": [pd.Timestamp("2024-01-01")] * 2,
                          "days_past_due": [0, 0], "current_balance": [100.0, 90.0], "current_status": ["Current"] * 2,
                          "loan_age_months": [10, 10], "remaining_term_months": [350, 350],
                          "original_balance": [100_000, 100_000], "interest_rate": [6.0, 5.5],
                          "credit_score_numeric": [700.0, 650.0], "credit_score_band": ["680-739", "620-679"]})
    features = [c for c in train.columns if c != "loan_id" and not c.startswith("next_")]
    class DummyModel:
        def __init__(self):
            self.calls = 0
        def predict_proba(self, X):
            self.calls += 1
            risk = 0.4 + np.clip((700 - pd.to_numeric(X["credit_score_numeric"])) * 0.001, -0.2, 0.2)
            return np.column_stack([1 - risk, risk])
    model = DummyModel()
    models = {"next_12m_default_flag": model}
    cache = scenario_cache(train, models, features, None)  # one predict per scenario combo
    assert model.calls == 3
    cached = scenario_table(train, models, features, None, ["credit_score_band"], cache=cache)
    assert model.calls == 3  # cache path runs no new model calls
    direct = scenario_table(train, models, features, None, ["credit_score_band"])
    assert model.calls == 6
    pd.testing.assert_frame_equal(cached.reset_index(drop=True), direct.reset_index(drop=True))


# ---------------------------------------------------------------- advanced features

def test_calibration_by_segment_bounded():
    from lpi_engine.modeling import calibration_by_segment
    rng = np.random.default_rng(5)
    n = 600
    valid = pd.DataFrame({
        "credit_score_band": np.repeat(["<620", "620-679", "740+"], 200),
        "vintage_year": np.repeat(["2023", "2024", "2023", "2024", "2023", "2024"], 100),
        "next_12m_default_flag": rng.integers(0, 2, n),
    })
    predictions = pd.DataFrame({"next_12m_default_prob": np.clip(valid.next_12m_default_flag * 0.7 + rng.normal(0.2, 0.25, n), 0, 1)},
                               index=valid.index)
    report = calibration_by_segment(valid, predictions, "next_12m_default_flag",
                                    ("credit_score_band", "vintage_year"))
    assert set(report.columns) >= {"target", "segment", "segment_value", "n", "ece"}
    assert report.ece.between(0, 1).all()
    assert report.segment.nunique() == 2  # both segment dimensions represented


def test_metric_bootstrap_ci_ordered():
    from lpi_engine.modeling import metric_bootstrap_ci
    rng = np.random.default_rng(9)
    y = pd.Series(rng.integers(0, 2, 400))
    p = np.clip(y.astype(float) * 0.65 + rng.normal(0.2, 0.3, 400), 0, 1)
    ci = metric_bootstrap_ci(y, p, n_boot=100, seed=1)
    assert 0.0 <= ci["ci_low"] <= ci["point"] <= ci["ci_high"] <= 1.0


def test_counterfactual_explanations_what_if_deltas():
    from lpi_engine.explainability import counterfactual_explanations
    from lpi_engine.modeling import preprocessor
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(11)
    n = 500
    df = pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(n)],
        "days_past_due": rng.integers(0, 90, n),
        "credit_score_numeric": rng.integers(560, 800, n).astype(float),
        "ltv_numeric": rng.normal(70, 12, n),
        "interest_rate": rng.normal(6, 1, n),
        "balance_ratio": rng.normal(0.9, 0.08, n),
        "current_status": rng.choice(["Current", "30 DPD", "60 DPD"], n),
    })
    risk = np.clip(0.3 + df.days_past_due / 300 + (700 - df.credit_score_numeric) / 1000, 0, 1)
    y = pd.Series(rng.binomial(1, risk))
    features = list(df.columns)
    model = Pipeline([("prep", preprocessor(df[features])), ("model", HistGradientBoostingClassifier(
        max_iter=80, min_samples_leaf=20, random_state=0))])
    model.fit(df[features], y)
    top = df.assign(next_12m_default_prob=model.predict_proba(df[features])[:, 1]).sort_values(
        "next_12m_default_prob", ascending=False)
    cf = counterfactual_explanations(top, model, features, "next_12m_default_flag", n=10)
    assert len(cf) == 10
    assert "days_past_due_delta" in cf.columns and "all_cured_delta" in cf.columns
    assert cf.baseline_probability.between(0, 1).all()
    assert cf.best_counterfactual.notna().all()
    # Curing a delinquent record must not *increase* the default probability on average.
    assert cf.all_cured_probability.mean() <= cf.baseline_probability.mean() + 1e-6


def test_stress_sensitivity_by_cluster_rows():
    train = pd.DataFrame({"loan_id": ["A", "B", "C", "D"],
                          "reporting_month": [pd.Timestamp("2024-01-01")] * 4,
                          "days_past_due": [0] * 4, "current_balance": [100.0] * 4,
                          "current_status": ["Current"] * 4, "loan_age_months": [10] * 4,
                          "remaining_term_months": [350] * 4, "original_balance": [100_000] * 4,
                          "interest_rate": [6.0] * 4, "credit_score_numeric": [700.0, 650.0, 700.0, 650.0],
                          "credit_score_band": ["680-739", "620-679", "680-739", "620-679"],
                          "ltv_band": ["<=60", "<=60", "81-100", "81-100"]})
    features = [c for c in train.columns if c != "loan_id" and not c.startswith("next_")]
    class DummyModel:
        def predict_proba(self, X):
            risk = 0.4 + np.clip((700 - pd.to_numeric(X["credit_score_numeric"])) * 0.001, -0.2, 0.2)
            return np.column_stack([1 - risk, risk])
    models = {"next_12m_default_flag": DummyModel()}
    cache = scenario_cache(train, models, features, None)
    clusters = stress_sensitivity_by_cluster(cache, train, None)
    assert not clusters.empty
    assert "delta_pp" in clusters.columns and "credit_score_band" in clusters.columns
    # Adverse Credit raises default rate for the low-credit cluster.
    adverse = clusters[(clusters.scenario == "Adverse Credit") & (clusters.credit_score_band == "620-679")]
    assert adverse.delta_pp.iloc[0] >= 0


def test_tracking_log_run_jsonl_and_graceful_wandb(tmp_path):
    from lpi_engine.tracking import log_run
    runs_path = tmp_path / "runs_log.jsonl"
    result = log_run({"kind": "pipeline_run", "seed": 1, "metrics": {"a/b": {"x": 0.5}}},
                     runs_path, {"enabled": False})
    assert result == {"jsonl": True, "wandb": "not_configured"}
    assert runs_path.exists() and runs_path.read_text().strip()
    entry = json.loads(runs_path.read_text())
    assert entry["tracking"]["jsonl"] == "logged"
    # Enabled without wandb installed must still not raise and must record the skip reason.
    result2 = log_run({"kind": "pipeline_run", "seed": 2}, runs_path, {"enabled": True, "project": "x"})
    assert result2["jsonl"] is True and str(result2["wandb"]).startswith("skipped")
    assert runs_path.read_text().count("\n") == 2


def test_copilot_cites_dictionary_tokens_from_drivers():
    record = pd.Series({"loan_id": "L1", "days_past_due": 0, "current_balance": 100,
                        "anomaly_score": 0.9, "anomaly_reason": "outlier_remaining_term_months",
                        "top_drivers": "outlier_remaining_term_months; negative_balance",
                        "next_12m_default_prob": 0.5, "data_quality_score": 40.0, "confidence": 0.7})
    dictionary = "# dict\n- `remaining_term_months`: months left on the contract.\n"
    _, audit = grounded_note(record, dictionary, [], cfg={"provider": "template"})
    cited = audit["grounding"]["dictionary_citations"]
    assert "remaining_term_months" in cited
    assert cited["remaining_term_months"] == "months left on the contract."


def test_dashboard_render_contains_sections():
    html = render_dashboard({
        "meta": {"cutoff": "2024-07-01", "n_features": 27, "seed": 2026},
        "kpis": {"rows": {"label": "Train rows", "value": "12,000"}},
        "metrics": [["Target", "Model", "ROC-AUC"], ["t", "HGB", 0.9]],
        "km_curves": [{"month": 1, "km_survival": 0.98, "cif_default": 0.01, "cif_prepay": 0.01}],
        "batch_quality": [{"month_ordinal": 1, "mean_quality_score": 90.0, "breaches_per_1k_rows": 5.0}],
        "scenarios_mc": [],
        "drift": [["Column", "PSI"], ["x", 0.2]],
        "review_queue": [["loan_id", "anomaly_score"], ["L1", 0.9]],
        "audit": [{"timestamp_utc": "t", "model": "template", "decision_status": "recommendation_pending_human_review"}],
    })
    assert "<title>Loan Performance Intelligence" in html
    assert "Competing-risk survival curves" in html
    assert "Reviewer queue" in html
    assert "Copilot audit trail" in html
    assert "L1" in html
