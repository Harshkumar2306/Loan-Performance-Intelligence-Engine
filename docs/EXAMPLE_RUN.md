# Example run - synthetic demo pack (2026-08-28)

> **Not a performance claim.** These numbers come from the built-in synthetic generator
> (`make_demo_data.py --rows 16000`), which exists to exercise the pipeline end-to-end.
> The organiser's real data pack will produce the submission metrics. Regenerate any time:
> `python scripts/make_demo_data.py --rows 16000 && python scripts/run_pipeline.py`.
> Every number below was read from the checked-in `outputs/` artefacts of that exact run.

## Setup
- Data: 12,393 train rows / 3,631 test rows, 1,142 loans, 30 reporting months (24 train months).
- Split: strict reporting-month holdout, cutoff 2024-07-01 (6 validation months).
- Features: 26 (raw contemporaneous + 9 engineered; the engineered credit-score parse
  subsumes the raw column, so it is not double-fed).
- Tests: 58 passing (`pytest -q`), including the competing-risk identity, the
  train/serve-consistency regression, the scenario-cache equivalence checks and a full
  end-to-end pipeline run on a tiny synthetic pack.

## Headline validation metrics (out-of-time, threshold-free)

| Target | Baseline RF AUC | Calibrated HGB AUC | LightGBM AUC | HGB PR-AUC | HGB Brier | HGB ECE |
|---|---|---|---|---|---|---|
| next_3m_delinquency_flag | 0.7674 | **0.8008** | 0.8011 | 0.2639 | 0.0586 | 0.0106 |
| next_6m_delinquency_flag | 0.8120 | **0.8801** | 0.8738 | 0.4989 | 0.0682 | 0.0138 |
| next_12m_default_flag | 0.9350 | **0.9999** | 0.9998 | 0.9986 | 0.0041 | 0.0191 |
| next_12m_prepayment_flag | 0.8613 | **0.9910** | 0.9916 | 0.9557 | 0.0270 | 0.0560 |
| exception_required | 0.8177 | **0.9109** | 0.9186 | 0.7430 | 0.0645 | 0.0336 |
| next_state (macro-F1) | 0.6049 vs majority baseline 0.1301 (weighted 0.9393) | | | | | |
| exception_type (macro-F1) | 0.8873, weighted 0.9573 | | | | | |

Operating points are chosen on validation only (recall-at-fixed-precision scan, with a
documented best-F1 fallback when the precision target is unreachable) and reported in
`metrics.json → threshold_metrics` and the generated model card.

## Evaluation audit (the honest part)

K-fold loan-disjoint ablation (5 deterministic partition variants), reported as mean [min-max]:

| Target | Time-split AUC | Loan-disjoint AUC (mean [range]) |
|---|---|---|
| next_12m_default_flag | 0.9999 | **0.675 [0.595-0.726]** |
| next_3m_delinquency_flag | 0.8008 | **0.585 [0.570-0.617]** |

The time-split default AUC is inflated by panel-overlap loan memorisation. The disjoint
estimate removes it and stays clearly above the 0.5 coin-toss line - the features carry
real cross-loan signal (credit, LTV, rate drive the default hazard in the generator), and
the memorisation effect is measured, not hidden. **Scale check (210k-row pack, verified
this run):** the gap narrows further - time-split 0.935 vs disjoint **0.715 [0.705-0.732]**
for default, and 0.688 [0.674-0.696] for delinquency - and the full pipeline completes in
~5 minutes. Both numbers ship together.

## Competing-risk curves (month 24, censoring-aware)
- Kaplan-Meier survival 0.547; cumulative incidence: default 0.084, prepayment 0.369,
  with Greenwood/Aalen-variance bands, versus naive row-rate baselines (default 0.039,
  prepayment 0.154) and the Markov cohort projection (default 0.122, prepayment 0.477).
  The identity CIF_default + CIF_prepay = 1 − S(t) holds to machine precision and is
  regression-tested.

## Scenarios (mean projected rates, bootstrap 5-95)
| Scenario | Default mean [5th-95th] | Prepayment mean [5th-95th] |
|---|---|---|
| Base | 0.0833 [0.0627-0.1058] | 0.2406 [0.2167-0.2663] |
| Adverse Credit | 0.0898 [0.0740-0.1104] | 0.1955 [0.1790-0.2187] |
| High Prepayment | 0.0829 [0.0637-0.1031] | 0.3029 [0.2712-0.3324] |

Driver attribution (`scenario_drivers.csv`): Adverse Credit +0.45pp default (entirely the
credit shock) and −4.26pp prepayment (the −1.0pp uplift); High Prepayment 0.0pp default and
+5.99pp prepayment (the +6.0pp uplift). Segment impacts per credit band, state, servicer and
vintage live in `scenario_report.csv`.

## Data intelligence highlights
- Rule breaches found: duplicate rows, negative balances, DPD/status breaks, future
  `last_updated_at`, prepaid loans with positive balances - all named in the review queue.
- 137 servicer updates reconciled; balance/status conflicts and stale records flagged.
- Batch KPI per reporting month, Cramér's V associations, numeric/categorical PSI drift.

## Explainability & responsible AI
- Global permutation importance + TreeSHAP local explanations (`shap_tree`, verified -
  all 375 local attribution rows are model-derived). Top default drivers: dti_numeric,
  interest_rate, original_balance, current_status, credit_score_numeric.
- FP/FN analysis with segment concentration in `error_analysis.json` / `error_examples.csv`.
- Segment parity monitoring in `fairness_report.csv` (descriptive only - the model card
  keeps full bias/fairness review as a pre-production gate).
- **Counterfactuals** (`counterfactuals.csv`): for the highest-risk records, each documented
  intervention (cure DPD, credit +40, LTV −10, rate −1, 20% paydown) is re-scored with its
  probability delta - e.g. L000718 falls from 0.9776 to 0.7785 under a 20% principal paydown.
- **Calibration by segment** (`calibration_by_segment.csv`): ECE per credit band / vintage /
  state, plus bootstrap 95% CIs on validation ROC-AUC in `metrics.json`
  (delinquency 0.8008 [0.7775, 0.8266]; 6-month 0.8801 [0.8651, 0.8972]).

## Monitoring & tracking
- `dashboard.html`: offline drift-monitoring dashboard (batch quality, drift, survival
  curves, scenarios, review queue, audit trail) - self-contained HTML/CSS/SVG with no
  JavaScript, so it renders in any modern browser without a server.
- `runs_log.jsonl`: one tracking entry per run (W&B logging available behind
  `tracking.enabled`); `experiments_log.jsonl`: hyperparameter trials from the
  `run_experiments.py` agentic experiment runner.

## Review queue & copilot
- 50 ranked anomalies with named drivers (rules + deviating features), recommended action
  and documented confidence blend.
- **Governance is honestly labelled:** every audit entry carries `used_llm` and a `scenario`
  tag. In this checked-in (offline) run the audit log records the grounded template and a
  simulated rejection, both honestly tagged `used_llm: false` with the model field set to
  `grounded-template-v1` / `ungrounded-call-template` - an offline run never claims a model
  call that did not happen. With an API key present the same code path records a
  `live_grounded_call` (the model writes the reviewer note from retrieved record facts) and
  a `live_ungrounded_call` (a bare-score verdict that the policy rejects for lacking
  grounding) - both `used_llm: true`. `scripts/copilot_demo.py` demonstrates the full loop
  with one command.
