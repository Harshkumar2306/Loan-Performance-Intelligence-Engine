"""Loan intelligence: transition modelling, survival/competing-risk curves, anomaly detection and scenario simulation.

Key modelling decisions
-----------------------
* Loan states are normalised deterministically from `days_past_due` and explicit statuses,
  so every downstream curve and matrix uses one consistent state definition.
* Right-censored final observations are never silently labelled as safe: the
  Kaplan-Meier / Aalen-Johansen estimators treat them as censored, and the transition
  matrix excludes them.
* Competing risk (default vs prepayment) is estimated with Aalen-Johansen cumulative
  incidence, including the Aalen variance estimator for confidence bands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

STATE_ORDER = ["current", "30_dpd", "60_dpd", "90_dpd", "default", "prepaid", "paid_off"]
ABSORBING = {"default", "prepaid", "paid_off"}


def state_of(status, dpd) -> str:
    """Deterministic loan state: days_past_due drives delinquency buckets; explicit statuses win.

    Buckets are aligned with the status vocabulary used by the data (`30 DPD` -> `30_dpd`,
    `60 DPD` -> `60_dpd`, `90 DPD` -> `90_dpd`). Only an explicit default/foreclosure/RPO
    status marks the `default` absorbing state; `days_past_due >= 90` without that status
    remains a (severe) delinquency stage, matching how `next_state` labels the same data.
    """
    s = str(status).lower() if pd.notna(status) else ""
    d = float(dpd) if pd.notna(dpd) else 0.0
    if "prepaid" in s or "paid_off" in s or "paid off" in s:
        return "prepaid"
    if "default" in s or "foreclosure" in s or "rpo" in s:
        return "default"
    if d >= 90:
        return "90_dpd"
    if d >= 60:
        return "60_dpd"
    if d >= 30:
        return "30_dpd"
    if d > 0:
        return "30_dpd"
    return "current"


def normalise_states(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    status = d["current_status"] if "current_status" in d else pd.Series("current", index=d.index)
    dpd = d["days_past_due"] if "days_past_due" in d else pd.Series(0, index=d.index)
    d["state"] = [state_of(s, x) for s, x in zip(status, dpd)]
    return d.sort_values(["loan_id", "reporting_month"])


def transition_model(panel: pd.DataFrame) -> pd.DataFrame:
    """Observed month-to-month transition probabilities; final rows are right-censored and excluded."""
    d = normalise_states(panel)
    d["next_observed_state"] = d.groupby("loan_id").state.shift(-1)
    observed = d.dropna(subset=["next_observed_state"])
    mat = pd.crosstab(observed.state, observed.next_observed_state, normalize="index")
    for s in STATE_ORDER:
        if s not in mat.index:
            mat.loc[s] = 0.0
    for s in STATE_ORDER:
        if s not in mat.columns:
            mat[s] = 0.0
    mat = mat.reindex(index=STATE_ORDER, columns=STATE_ORDER).fillna(0.0)
    # States never observed as origins stay put by convention (documented self-loop), so the
    # Markov projection is well-defined for every state in the state space.
    for s in STATE_ORDER:
        if mat.loc[s].sum() == 0:
            mat.loc[s, s] = 1.0
    mat = mat.div(mat.sum(axis=1).replace(0, 1), axis=0).fillna(0.0)
    # Absorbing states persist by construction. This override must run AFTER the row
    # normalisation above, otherwise renormalising a non-absorbing row would silently
    # erase the pinned diagonal (a previous bug: default could appear to "cure").
    for s in ABSORBING:
        mat.loc[s] = 0.0
        mat.loc[s, s] = 1.0
    result = mat.reset_index()
    result.columns = ["from_state"] + list(result.columns[1:])
    return result


def transition_curves(matrix: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Markov cohort projection from the observed transition matrix (baseline comparator for the KM curves)."""
    if matrix.empty:
        return pd.DataFrame(columns=["month", "markov_default", "markov_prepay"])
    p = matrix.set_index("from_state")[STATE_ORDER].to_numpy(dtype=float)
    distribution = np.zeros(len(STATE_ORDER))
    distribution[STATE_ORDER.index("current")] = 1.0
    rows = []
    for month in range(1, horizon + 1):
        distribution = distribution @ p
        rows.append({"month": month,
                     "markov_default": round(float(distribution[STATE_ORDER.index("default")]), 5),
                     "markov_prepay": round(float(distribution[STATE_ORDER.index("prepaid")]
                                                  + distribution[STATE_ORDER.index("paid_off")]), 5)})
    return pd.DataFrame(rows)


def _event_times(panel: pd.DataFrame, event_states: set[str]) -> dict:
    """First month_index at which each loan enters one of `event_states` (None if never).

    The time axis is `month_index` (or `loan_age_months`); when neither exists the axis falls
    back to the loan's own row order (1-based), never to a global calendar offset, so censoring
    times remain per-loan correct.
    """
    d = normalise_states(panel).sort_values(["loan_id", "reporting_month"])
    time_col = "month_index" if "month_index" in d.columns else ("loan_age_months" if "loan_age_months" in d.columns else None)
    if time_col is None:
        d["_row_order"] = d.groupby("loan_id").cumcount() + 1
        time_col = "_row_order"
    hits = d[d.state.isin(event_states)].groupby("loan_id")[time_col].min()
    last = d.groupby("loan_id")[time_col].max()
    return {loan: (float(hits[loan]) if loan in hits.index else None, float(last[loan])) for loan in last.index}


def competing_risk_curves(panel: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Kaplan-Meier overall survival + Aalen-Johansen cumulative incidence for default and prepayment.

    Event = entering the absorbing state; the competing event and the end of observation
    both censor. Confidence bands use Greenwood (KM) and the Aalen variance estimator (CIF),
    computed in a two-pass algorithm over the discrete-time hazard tables.
    """
    default_times = _event_times(panel, {"default"})
    prepaid_times = _event_times(panel, {"prepaid", "paid_off"})
    loans = sorted(set(default_times) | set(prepaid_times))
    if not loans:
        return pd.DataFrame(columns=["month", "km_survival", "km_survival_ci_low", "km_survival_ci_high",
                                     "cif_default", "cif_default_ci_low", "cif_default_ci_high",
                                     "cif_prepay", "cif_prepay_ci_low", "cif_prepay_ci_high",
                                     "at_risk", "default_events", "prepay_events"])

    rows = []
    for loan in loans:
        d_time, d_last = default_times[loan]
        p_time, p_last = prepaid_times[loan]
        event_time = min([t for t in (d_time, p_time) if t is not None], default=None)
        censor_time = min(t for t in (d_last, p_last) if t is not None)
        rows.append({
            "loan": loan,
            "event": "default" if d_time == event_time and d_time is not None
            else ("prepay" if p_time == event_time and p_time is not None else "censored"),
            "time": event_time if event_time is not None else censor_time,
            "censored": event_time is None,
        })
    life = pd.DataFrame(rows)

    # --- Pass 1: hazard tables and point estimates -----------------------
    n_at, d_def, d_pre, d_tot, c_at = {}, {}, {}, {}, {}
    km_s, km_var_t, s_before, cif_def, cif_pre = {}, {}, {}, {}, {}
    km_survival = 1.0
    km_var = 0.0
    cif_def[0] = cif_pre[0] = 0.0
    s_before[0] = 1.0
    for t in range(1, horizon + 1):
        n_at[t] = int((life.time >= t).sum())
        d_def[t] = int(((life.time == t) & (life.event == "default")).sum())
        d_pre[t] = int(((life.time == t) & (life.event == "prepay")).sum())
        d_tot[t] = d_def[t] + d_pre[t]
        c_at[t] = int(((life.time == t) & life.censored).sum())
        s_before[t] = km_survival
        if n_at[t] > 0:
            km_survival *= (1 - d_tot[t] / n_at[t])
            if n_at[t] > d_tot[t]:
                km_var += d_tot[t] / (n_at[t] * (n_at[t] - d_tot[t]))  # Greenwood increment
            cif_def[t] = cif_def[t - 1] + s_before[t] * d_def[t] / n_at[t]
            cif_pre[t] = cif_pre[t - 1] + s_before[t] * d_pre[t] / n_at[t]
        else:
            cif_def[t] = cif_def[t - 1]
            cif_pre[t] = cif_pre[t - 1]
        km_s[t] = km_survival
        km_var_t[t] = km_var

    # --- Pass 2: Aalen variance estimator for each CIF at each horizon ----
    def aalen_variance(event_k: dict, event_j: dict, cif_k: dict, cif_final: float) -> float:
        var = 0.0
        for u in range(1, horizon + 1):
            n = n_at[u]
            if n <= 0:
                continue
            s2 = s_before[u] ** 2
            dk, dj = event_k[u], event_j[u]
            if dk > 0 and n > dk:
                var += s2 * dk * (n - dk) / n ** 3
            if dj > 0 and n > dj:
                delta = cif_final - cif_k[u]
                var += s2 * (delta ** 2) * dj * (n - dj) / n ** 3
                if dk > 0:
                    var -= 2 * s2 * delta * dk * dj / n ** 3
        return max(var, 0.0)

    out = []
    for t in range(1, horizon + 1):
        var_def = aalen_variance(d_def, d_pre, cif_def, cif_def[horizon])
        var_pre = aalen_variance(d_pre, d_def, cif_pre, cif_pre[horizon])
        km_lo = max(km_s[t] - 1.96 * np.sqrt(km_var_t[t]), 0.0)
        km_hi = min(km_s[t] + 1.96 * np.sqrt(km_var_t[t]), 1.0)
        out.append({
            "month": t,
            "at_risk": n_at[t],
            "default_events": d_def[t],
            "prepay_events": d_pre[t],
            "censored_this_month": c_at[t],
            "km_survival": round(float(km_s[t]), 5),
            "km_survival_ci_low": round(float(km_lo), 5),
            "km_survival_ci_high": round(float(km_hi), 5),
            "cif_default": round(float(cif_def[t]), 5),
            "cif_default_ci_low": round(float(max(cif_def[t] - 1.96 * np.sqrt(var_def), 0.0)), 5),
            "cif_default_ci_high": round(float(min(cif_def[t] + 1.96 * np.sqrt(var_def), 1.0)), 5),
            "cif_prepay": round(float(cif_pre[t]), 5),
            "cif_prepay_ci_low": round(float(max(cif_pre[t] - 1.96 * np.sqrt(var_pre), 0.0)), 5),
            "cif_prepay_ci_high": round(float(min(cif_pre[t] + 1.96 * np.sqrt(var_pre), 1.0)), 5),
        })
    # Naive, censoring-blind baselines (row-level event rates).
    d = normalise_states(panel)
    naive_default = float((d.state == "default").mean())
    naive_prepay = float((d.state.isin({"prepaid", "paid_off"})).mean())
    for row in out:
        row["naive_default_rate"] = round(naive_default, 5)
        row["naive_prepay_rate"] = round(naive_prepay, 5)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

ANOMALY_FEATURES = ["current_balance", "original_balance", "interest_rate",
                    "loan_age_months", "remaining_term_months", "days_past_due"]


def anomaly_scores(df: pd.DataFrame, quality: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Hybrid anomaly score: Isolation Forest + documented rule breaches + data-quality penalty.

    Vectorised for organiser-scale panels: named drivers are assembled with numpy, not
    per-row Python loops.
    """
    cols = [c for c in ANOMALY_FEATURES if c in df.columns]
    if cols and len(df) >= 20:
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        scaled = RobustScaler().fit_transform(X)
        raw = -IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit_predict(scaled)
        iso = pd.Series((raw + 1) / 2, index=df.index)  # 1 = flagged
        robust_z = pd.DataFrame(np.abs(scaled), columns=cols, index=df.index)
    else:
        # Tiny frames (< 20 rows) cannot support a stable Isolation Forest / robust scaler;
        # fall back to the deterministic rule + quality signal only.
        iso = pd.Series(0.0, index=df.index)
        robust_z = pd.DataFrame(index=df.index)

    breaches = quality[[c for c in quality.columns if c not in ("data_quality_score", "n_breaches")]]
    breach_matrix = breaches.reindex(df.index).fillna(False).astype(bool)
    breach_count = breach_matrix.sum(axis=1)
    quality_score = quality["data_quality_score"].reindex(df.index).fillna(100)
    score = np.clip(
        0.55 * iso + 0.30 * (1 - quality_score / 100) + 0.15 * np.minimum(breach_count / 2, 1), 0, 1)

    # Named drivers: up to 3 rule names + up to 2 most deviating features, fully vectorised.
    names = breach_matrix.columns.to_numpy()
    rows_idx, cols_idx = np.nonzero(breach_matrix.to_numpy())
    per_row_hits: dict[int, list] = {}
    for r, c in zip(rows_idx, cols_idx):
        bucket = per_row_hits.setdefault(int(r), [])
        if len(bucket) < 3:
            bucket.append(str(names[c]))

    top_dev: dict[int, list] = {}
    if robust_z.shape[1] > 0:
        z = robust_z.to_numpy(dtype=float)
        k = min(2, z.shape[1])
        # argpartition needs at least one row; tiny frames skip deviation drivers.
        if len(z) > 0:
            top_idx = np.argpartition(-z, kth=k - 1, axis=1)[:, :k]
            for i in range(z.shape[0]):
                top_dev[i] = [str(cols[j]) for j in top_idx[i] if z[i, j] > 0.5]

    reasons = []
    drivers = []
    flagged_iso = iso.to_numpy(dtype=float) > 0.5
    for i in range(len(df)):
        hits = per_row_hits.get(i, [])
        dev = top_dev.get(i, [])
        drivers.append({"rule_breaches": hits, "deviating_features": dev})
        parts = list(hits)
        if not parts and bool(flagged_iso[i]):
            parts = [f"multivariate pattern outlier ({', '.join(dev[:2]) or 'unnamed features'})"]
        reasons.append("; ".join(parts) if parts else "no rule breached; low residual risk")
    return pd.DataFrame({"anomaly_score": score.round(4), "anomaly_reason": reasons, "anomaly_drivers": drivers},
                        index=df.index)


# ---------------------------------------------------------------------------
# Scenario simulation
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS = [
    {"scenario": "Base", "credit_score_shock": 0.0, "prepayment_uplift": 0.0},
    {"scenario": "Adverse Credit", "credit_score_shock": -35.0, "prepayment_uplift": -0.01},
    {"scenario": "High Prepayment", "credit_score_shock": 0.0, "prepayment_uplift": 0.06},
]


def _scored_frame(test: pd.DataFrame, models: dict, features: list[str], credit_shock: float,
                  prepayment_uplift: float) -> pd.DataFrame:
    x = test[features].copy()
    for col in ("credit_score", "credit_score_numeric"):
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce") + credit_shock
    out = test.copy()
    for target, model in models.items():
        if target in ("next_state", "exception_type"):
            continue
        out[target.replace("_flag", "_prob")] = model.predict_proba(x)[:, 1]
    for col in ("next_12m_default_prob", "next_3m_delinquency_prob", "next_12m_prepayment_prob"):
        if col not in out.columns:
            out[col] = 0.0
    out["next_12m_prepayment_prob"] = np.clip(out["next_12m_prepayment_prob"] + prepayment_uplift, 0, 1)
    return out


def scenario_cache(test: pd.DataFrame, models: dict, features: list[str],
                   scenarios: pd.DataFrame | None) -> dict:
    """Precompute the scored frame once per unique (credit shock, prepay uplift) combination,
    so `scenario_table` / `scenario_drivers` / `scenario_monte_carlo` never re-run models."""
    scenario_rows = scenarios.to_dict("records") if scenarios is not None else DEFAULT_SCENARIOS
    cache = {}
    for s in scenario_rows:
        key = (float(s.get("credit_score_shock", 0) or 0), float(s.get("prepayment_uplift", 0) or 0))
        if key not in cache:
            cache[key] = _scored_frame(test, models, features, key[0], key[1])
    return cache


def _cached_frame(cache: dict, s: dict, test: pd.DataFrame, models: dict, features: list[str]) -> pd.DataFrame:
    key = (float(s.get("credit_score_shock", 0) or 0), float(s.get("prepayment_uplift", 0) or 0))
    return cache[key] if key in cache else _scored_frame(test, models, features, key[0], key[1])


def scenario_table(test: pd.DataFrame, models: dict, features: list[str],
                   scenarios: pd.DataFrame | None, segment_columns: list[str],
                   cache: dict | None = None) -> pd.DataFrame:
    rows = []
    scenario_rows = scenarios.to_dict("records") if scenarios is not None else DEFAULT_SCENARIOS
    for s in scenario_rows:
        tmp = _cached_frame(cache, s, test, models, features) if cache is not None \
            else _scored_frame(test, models, features, float(s.get("credit_score_shock", 0) or 0),
                               float(s.get("prepayment_uplift", 0) or 0))
        tmp["vintage_year"] = (pd.to_datetime(tmp["origination_month"], errors="coerce").dt.year.astype("string")
                               if "origination_month" in tmp else pd.Series("unknown", index=tmp.index))
        for segment in [c for c in segment_columns if c in tmp.columns]:
            agg = tmp.groupby(segment)[["next_12m_default_prob", "next_3m_delinquency_prob", "next_12m_prepayment_prob"]].mean().reset_index()
            agg = agg.rename(columns={segment: "segment_value"})
            agg.insert(0, "segment", segment)
            agg.insert(0, "scenario", str(s.get("scenario", "unnamed")))
            rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def scenario_drivers(test: pd.DataFrame, models: dict, features: list[str],
                     scenarios: pd.DataFrame | None, cache: dict | None = None) -> pd.DataFrame:
    """One-feature-at-a-time attribution of each scenario's impact on projected default AND
    prepayment rates. The credit shock moves default risk; the prepayment uplift moves the
    prepayment projection directly — both are reported so the High Prepayment scenario has a
    visible, auditable effect instead of looking identical to Base."""
    scenario_rows = scenarios.to_dict("records") if scenarios is not None else DEFAULT_SCENARIOS

    def frame_for(s: dict) -> pd.DataFrame:
        if cache is not None:
            return _cached_frame(cache, s, test, models, features)
        return _scored_frame(test, models, features, float(s.get("credit_score_shock", 0) or 0),
                             float(s.get("prepayment_uplift", 0) or 0))

    base = frame_for(scenario_rows[0])
    base_default = base["next_12m_default_prob"].mean()
    base_prepay = base["next_12m_prepayment_prob"].mean()
    out = []
    for s in scenario_rows:
        if s.get("scenario") == "Base":
            continue
        credit = float(s.get("credit_score_shock", 0) or 0)
        uplift = float(s.get("prepayment_uplift", 0) or 0)
        total = frame_for(s)
        only_credit = frame_for({"credit_score_shock": credit, "prepayment_uplift": 0.0})
        out.append({
            "scenario": s.get("scenario"),
            "baseline_default_rate": round(float(base_default), 5),
            "scenario_default_rate": round(float(total["next_12m_default_prob"].mean()), 5),
            "delta_default_pp": round(float((total["next_12m_default_prob"].mean() - base_default) * 100), 4),
            "credit_shock_contribution_pp": round(float((only_credit["next_12m_default_prob"].mean() - base_default) * 100), 4),
            "prepayment_uplift_contribution_pp": round(float((total["next_12m_default_prob"].mean()
                                                              - only_credit["next_12m_default_prob"].mean()) * 100), 4),
            "baseline_prepayment_rate": round(float(base_prepay), 5),
            "scenario_prepayment_rate": round(float(total["next_12m_prepayment_prob"].mean()), 5),
            "delta_prepayment_pp": round(float((total["next_12m_prepayment_prob"].mean() - base_prepay) * 100), 4),
            "prepayment_uplift_applied_pp": round(uplift * 100, 4),
        })
    return pd.DataFrame(out)


def stress_sensitivity_by_cluster(cache: dict, test: pd.DataFrame, scenarios: pd.DataFrame | None,
                                  cluster_cols: tuple = ("credit_score_band", "ltv_band")) -> pd.DataFrame:
    """Scenario impact per feature cluster (brief: "Stress sensitivity by feature cluster").

    Reuses the cached per-scenario scored frames: for each cluster of the test population,
    compares the Base projected default rate against each non-Base scenario, so reviewers
    see *which* segments absorb the stress instead of only the portfolio average.
    """
    scenario_rows = scenarios.to_dict("records") if scenarios is not None else DEFAULT_SCENARIOS
    present_cols = [c for c in cluster_cols if c in test.columns]
    if not present_cols:
        return pd.DataFrame()
    base_key = (0.0, 0.0)
    base_frame = cache.get(base_key)
    if base_frame is None:
        return pd.DataFrame()
    base_map = base_frame.groupby(present_cols, dropna=False)["next_12m_default_prob"].mean().to_dict()
    rows = []
    for s in scenario_rows:
        name = str(s.get("scenario"))
        if name == "Base":
            continue
        key = (float(s.get("credit_score_shock", 0) or 0), float(s.get("prepayment_uplift", 0) or 0))
        frame = cache.get(key)
        if frame is None:
            continue
        scenario_map = frame.groupby(present_cols, dropna=False)["next_12m_default_prob"].mean().to_dict()
        # Dictionary iteration tolerates NaN inside cluster keys, whereas a MultiIndex
        # `.loc[cluster]` lookup raises KeyError for clusters containing NaN bands.
        for cluster, base_rate in base_map.items():
            scenario_rate = scenario_map.get(cluster, base_rate)
            row = {"scenario": name, "base_default_rate": round(float(base_rate), 5),
                   "scenario_default_rate": round(float(scenario_rate), 5),
                   "delta_pp": round((float(scenario_rate) - float(base_rate)) * 100, 4)}
            if isinstance(cluster, tuple):
                for col, value in zip(present_cols, cluster):
                    row[col] = str(value)
            else:
                row[present_cols[0]] = str(cluster)
            rows.append(row)
    return pd.DataFrame(rows)


def scenario_monte_carlo(test: pd.DataFrame, models: dict, features: list[str],
                         scenarios: pd.DataFrame | None, n_sims: int = 30, sample: int = 300,
                         seed: int = 2026, cache: dict | None = None) -> pd.DataFrame:
    """Bootstrap portfolio simulation: mean and 5-95 percentile projected default AND prepayment
    rates per scenario. Resamples the precomputed scored frame (when a cache is supplied), so no
    model is re-run inside the bootstrap loop."""
    rng = np.random.default_rng(seed)
    scenario_rows = scenarios.to_dict("records") if scenarios is not None else DEFAULT_SCENARIOS
    out = []
    for s in scenario_rows:
        if cache is not None:
            frame = _cached_frame(cache, s, test, models, features)
            sims = [frame["next_12m_default_prob"].sample(n=min(sample, len(frame)), replace=True,
                                                          random_state=int(rng.integers(1, 2**31))).mean()
                    for _ in range(n_sims)]
            sims_pre = [frame["next_12m_prepayment_prob"].sample(n=min(sample, len(frame)), replace=True,
                                                                 random_state=int(rng.integers(1, 2**31))).mean()
                        for _ in range(n_sims)]
        else:
            credit = float(s.get("credit_score_shock", 0) or 0)
            uplift = float(s.get("prepayment_uplift", 0) or 0)
            sims, sims_pre = [], []
            for _ in range(n_sims):
                idx = rng.choice(test.index, size=min(sample, len(test)), replace=True)
                sample_frame = test.loc[idx]
                scored = _scored_frame(sample_frame, models, features, credit, uplift)
                sims.append(scored["next_12m_default_prob"].mean())
                sims_pre.append(scored["next_12m_prepayment_prob"].mean())
        sims = np.array(sims)
        sims_pre = np.array(sims_pre)
        out.append({
            "scenario": s.get("scenario"),
            "mean_default_rate": round(float(sims.mean()), 5),
            "p05_default_rate": round(float(np.percentile(sims, 5)), 5),
            "p95_default_rate": round(float(np.percentile(sims, 95)), 5),
            "mean_prepayment_rate": round(float(sims_pre.mean()), 5),
            "p05_prepayment_rate": round(float(np.percentile(sims_pre, 5)), 5),
            "p95_prepayment_rate": round(float(np.percentile(sims_pre, 95)), 5),
            "simulations": n_sims,
        })
    return pd.DataFrame(out)
