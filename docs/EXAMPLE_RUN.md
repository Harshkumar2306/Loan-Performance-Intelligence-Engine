# Example run — synthetic demo pack (2026-08-31)

> **Not a performance claim.** These numbers come from the built-in synthetic generator
> (`make_demo_data.py --rows 16000`), which exists to exercise the pipeline end-to-end.
> The organiser's real data pack will produce the submission metrics. Regenerate any time:
> `python scripts/make_demo_data.py --rows 16000 && python scripts/run_pipeline.py`.

## Setup
- Data: 12,393 train rows / 3,631 test rows, 1,142 loans, 24 reporting months.
- Split: strict reporting-month holdout, **cutoff 2024-01-01 (12 validation months)**.
- Features: 26 (raw contemporaneous + 9 engineered; the engineered credit-score parse
  subsumes the raw column, so it is not double-fed).
- Tests: **58 passing** (`pytest -q`) — 39 core invariants + 19 edge-case, security and
  end-to-end tests (including a full pipeline run on a tiny organiser-style pack).

## Headline validation metrics (out-of-time)

| Target | Baseline RF AUC | Calibrated HGB AUC | LightGBM AUC | PR-AUC (HGB) | Brier (HGB) | ECE (HGB) |
|---|---|---|---|---|---|---|
| next_3m_delinquency_flag | 0.728 | **0.708** | 0.702 | 0.165 | 0.062 | 0.017 |
| next_6m_delinquency_flag | 0.760 | **0.785** | 0.773 | 0.330 | 0.088 | 0.025 |
| next_12m_default_flag | 0.888 | 0.960 | **0.962** | 0.844 | 0.018 | 0.024 |
| next_12m_prepayment_flag | 0.800 | 0.919 | **0.920** | 0.804 | 0.070 | 0.067 |
| exception_required | 0.755 | 0.854 | **0.860** | 0.659 | 0.076 | 0.025 |
| next_state (macro-F1) | 0.547 vs majority baseline 0.137 | | | | | |
| exception_type (macro-F1) | 0.845, weighted 0.927 | | | | | |

Ranking AUC and calibration tell different stories: the baseline RF sometimes ranks slightly
better than HGB on ROC-AUC (e.g. 3-month delinquency), but its Brier score is far worse
(0.149 vs 0.062) — the calibrated HGB is the decision-grade candidate precisely because its
probabilities are trustworthy, which is what the reviewer-facing outputs need.

Bootstrap CIs on validation ROC-AUC (400 resamples): default 0.960 [0.941–0.974];
delinquency 0.708 [0.685–0.728].

## Evaluation audit (the honest part)

K-fold loan-disjoint ablation (5 deterministic loan partitions), mean [min–max]:

| Target | Time-split AUC | Loan-disjoint AUC |
|---|---|---|
| next_12m_default_flag | 0.960 | **0.653 [0.598–0.711]** |
| next_3m_delinquency_flag | 0.708 | **0.574 [0.528–0.637]** |

The time-split number includes a legitimate panel-overlap bonus; the disjoint estimate
strips it out and stays clearly above the 0.5 coin-toss line — genuine cross-loan signal
exists and the memorisation effect is measured, not hidden. **Scale check (210k rows,
5.5 min):** disjoint default AUC 0.694 [0.671–0.716] vs time-split 0.878 — the gap
narrows as predicted, and both numbers ship together.

## Competing-risk curves (month 24, censoring-aware)
- Kaplan-Meier survival ≈ 0.60; cumulative incidence: default ≈ 0.09, prepayment ≈ 0.22,
  with Greenwood/Aalen-variance bands, versus naive row-rate baselines and the Markov
  cohort projection. The identity CIF_default + CIF_prepay = 1 − S(t) holds to machine
  precision and is regression-tested.

## Scenarios (mean projected default rate, bootstrap 5–95)
| Scenario | Mean default | 5th | 95th | Mean prepayment | 5th | 95th |
|---|---|---|---|---|---|---|
| Base | 0.084 | 0.065 | 0.104 | 0.279 | 0.258 | 0.301 |
| Adverse Credit | 0.090 | 0.075 | 0.109 | 0.237 | 0.214 | 0.255 |
| High Prepayment | 0.083 | 0.069 | 0.101 | **0.343** | 0.315 | 0.373 |

`scenario_drivers.csv` reports both default and prepayment attribution in percentage points.

## Data intelligence highlights
- 14 deterministic rules: duplicate rows, negative balances, DPD/status breaks, future
  `last_updated_at`, prepaid loans with positive balances — all named in the review queue.
- Servicer updates reconciled; balance/status conflicts and stale records flagged.
- Batch KPI per reporting month, Cramér's V associations, lift-based association rules,
  numeric/categorical PSI drift.

## Explainability & responsible AI
- Global permutation importance + TreeSHAP local explanations (`shap_tree`, verified).
- FP/FN analysis with segment concentration in `error_analysis.json` / `error_examples.csv`.
- Segment parity monitoring in `fairness_report.csv` (descriptive only — the model card
  keeps full bias/fairness review as a pre-production gate).
- **Counterfactuals** (`counterfactuals.csv`): documented what-if interventions re-scored
  for the highest-risk records with probability deltas.
- **Calibration by segment** (`calibration_by_segment.csv`): ECE per credit band / vintage /
  state.

## Monitoring, tracking & copilot
- `dashboard.html`: offline control-tower dashboard (batch quality, drift, survival
  curves, scenarios, review queue, copilot narrative, audit trail) — auto-opens in the
  browser after a run (disable with `--no-browser`).
- `runs_log.jsonl`: one tracking entry per run (optional W&B behind `tracking.enabled`).
- **Governance is honestly labelled:** every audit entry carries `used_llm` and a
  `scenario` tag. With an API key present, the run records a **live grounded call**
  (`used_llm: true`) and a **live ungrounded call rejected by policy** (`used_llm: true`,
  `rejected_by_human`); without a key the same entries are recorded as offline simulations
  with `used_llm: false` — never a fabricated model call.
