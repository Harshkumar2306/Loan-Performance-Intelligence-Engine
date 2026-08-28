# Model Card - Loan Performance Intelligence Engine

> The runnable pipeline also generates `outputs/model_card.md` with the actual run's
> metrics injected from `metrics.json`. This document states the standing methodology.

## Intended use
Rank **monthly loan observations** for a human reviewer: expected delinquency/default/prepayment
risk, data-quality exceptions and record anomalies. It supports investigation; it must not
autonomously change servicing, approve/deny credit, or make a consumer-impacting decision.

## Data & units
Curated organiser-supplied panel data: one loan-month per row, joined to one origination record
per loan. The demonstration generator is synthetic and is a functional test only. Reporting month
is the temporal anchor. Public-source links in the challenge brief are not scraped or redistributed.

## Feature and leakage policy
Approved features are contemporaneously observable: balance, age, term, rate, DPD, borrower/property
bands, state, servicer, status, document fields, plus documented engineered features that use only
current + historical rows of the same loan. `next_*`, exception outcomes, post-event flags
(`default_flag`, `prepayment_flag`, `loss_severity_band`, `modification_flag`) and identifiers are
categorically excluded, and `assert_no_leakage` fails the run if any banned column reaches the model.

## Methods
A class-balanced random forest is the benchmark. The production candidate is HistGradientBoosting
inside preprocessing with sigmoid calibration. Evaluation is on a final contiguous reporting-month
holdout - never a random split. Multiclass next-state and exception-type models include majority-class
baselines. Monthly state transitions estimate observed next-state frequencies; last observed loan
months are right-censored. Kaplan-Meier and Aalen-Johansen competing-risk curves treat the competing
event and the end of observation as censoring, with Greenwood and Aalen-variance bands.

## Metrics and operating point
ROC-AUC, PR-AUC, F1, precision, recall, event rate, Brier score, ECE (reliability diagram), and
recall-at-fixed-precision per binary outcome; macro/weighted F1 per multiclass outcome; per-month
AUC/Brier for temporal stability. Operational thresholds are selected on validation only, against a
documented review-capacity / precision target.

## Evaluation audit (panel-overlap memorisation)
In panel data the same loan legitimately appears on both sides of a time split. A model can then
memorise loan identity through quasi-unique static features, inflating time-split AUC. LPIE quantifies
this with a **K-fold loan-disjoint ablation** (deterministic 70/30 loan partitions, repeated 5× and
reported as mean [min-max]): the gap between the two numbers is the memorisation share. On the
16k-row synthetic demo pack the disjoint default AUC is 0.675 [0.595–0.726] vs 0.9999 time-split, and
delinquency 0.585 [0.570–0.617] vs 0.801 — **above the 0.5 coin-toss line**, i.e. genuine cross-loan
signal exists and only the magnitude of the overlap bonus is being audited. On a 210k-row pack the gap
narrows further: 0.715 [0.705–0.732] vs 0.935. **Report both numbers; never present the time-split AUC
as pure skill.**

## Uncertainty, limitations & monitoring
Predicted probabilities are statistical estimates, not certainty. Calibration degrades under macro
regime change; sparse defaults make tail metrics unstable; source systems may be stale or conflicting.
`confidence = 0.7 * model_confidence + 0.3 * data_confidence` (documented blend of distance-from-coin-flip
and data quality). The pipeline ships a **drift-monitoring dashboard** (`outputs/dashboard.html`) with
batch quality, PSI drift, survival curves, scenario bounds and the review queue, and a per-run
experiment log (`runs_log.jsonl`). Monitor PSI, missingness, rule breaches, ECE/Brier and segment
performance monthly; retrain only after documented validation and approval.

## Smart LLM usage & audit honesty
Every copilot audit entry carries `used_llm` and a `scenario` tag. Offline runs record the grounded
template (`offline_grounded_template`) and a rejected ungrounded output
(`offline_governance_simulation`) - both with `used_llm: false`, because no model was called. Live
runs (API key + `llm.base_url` present) record real grounded and rejected ungrounded calls with
`used_llm: true`. No entry ever claims a model call that did not happen.

## Fairness and privacy
`fairness_report.csv` provides descriptive segment parity (event rate, predicted rate, FPR, FNR) - a
monitoring artefact, not a legal fairness assessment. Do not use protected characteristics or proxy
variables to make adverse consumer decisions. Before production: legal/compliance review, access
controls, retention controls, fairness analysis and adverse-action policy review are required.
