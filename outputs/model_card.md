# Model Card — Loan Performance Intelligence Engine (generated)

*Generated 2026-08-30 from the actual pipeline run. Values are never pre-filled.*

## Intended use
Rank monthly loan observations for a human reviewer. Not for autonomous servicing, credit
approval, or any consumer-impacting decision.

## Data & validation
- Time split: strict reporting-month out-of-time split; no future rows in training (cutoff 2024-07-01).
- Fit rows: 8,493  |  Validation rows: 3,900 across 6 months.
- Loans spanning the boundary (expected in panel data): 1,067.
- Feature count: 26 (see `features_used.json`).

## Metrics (out-of-time validation, threshold-free)

| Target | Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| next_3m_delinquency_flag | baseline RF | 0.7666 | 0.2316 | 0.1561 |
| next_3m_delinquency_flag | calibrated HGB | 0.8047 | 0.257 | 0.0585 |
| next_6m_delinquency_flag | baseline RF | 0.8135 | 0.33 | 0.1618 |
| next_6m_delinquency_flag | calibrated HGB | 0.8774 | 0.4771 | 0.069 |
| next_12m_default_flag | baseline RF | 0.9351 | 0.5046 | 0.1043 |
| next_12m_default_flag | calibrated HGB | 0.9999 | 0.9974 | 0.0045 |
| next_12m_prepayment_flag | baseline RF | 0.8589 | 0.5292 | 0.1801 |
| next_12m_prepayment_flag | calibrated HGB | 0.9927 | 0.9645 | 0.0249 |
| exception_required | baseline RF | 0.8207 | 0.5586 | 0.1825 |
| exception_required | calibrated HGB | 0.9112 | 0.7335 | 0.0659 |
| next_state | random forest (multiclass) | macro-F1 0.6106 | weighted-F1 0.9399 | classes ['30_dpd', '60_dpd', '90_dpd', 'current', 'default', 'prepaid'] |
| exception_type | random forest (multiclass) | macro-F1 0.8854 | weighted-F1 0.956 | classes ['data_break', 'missing_docs', 'none', 'source_conflict'] |

## Operating point (threshold chosen on validation only)

Per-target thresholds come from the recall-at-fixed-precision scan, falling back to the
validation best-F1 threshold when the precision target is unreachable (e.g. a strongly
compressed probability range on a low-event-rate target). Both models are evaluated at
the SAME threshold.

| Target | Threshold | Method | Model | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| next_3m_delinquency_flag | 0.1288 | best_validation_f1_fallback | baseline RF | 0.1561 | 0.0847 | 1.0 |
| next_3m_delinquency_flag | 0.1288 | best_validation_f1_fallback | calibrated HGB | 0.3261 | 0.2605 | 0.4358 |
| next_6m_delinquency_flag | 0.4366 | recall_at_precision_0.6 | baseline RF | 0.3227 | 0.1976 | 0.8793 |
| next_6m_delinquency_flag | 0.4366 | recall_at_precision_0.6 | calibrated HGB | 0.4 | 0.6141 | 0.2966 |
| next_12m_default_flag | 0.1864 | recall_at_precision_0.6 | baseline RF | 0.1077 | 0.0569 | 1.0 |
| next_12m_default_flag | 0.1864 | recall_at_precision_0.6 | calibrated HGB | 0.8848 | 0.7935 | 1.0 |
| next_12m_prepayment_flag | 0.8133 | recall_at_precision_0.6 | baseline RF | 0.0 | 0.0 | 0.0 |
| next_12m_prepayment_flag | 0.8133 | recall_at_precision_0.6 | calibrated HGB | 0.5951 | 1.0 | 0.4236 |
| exception_required | 0.7109 | recall_at_precision_0.6 | baseline RF | 0.2409 | 0.8495 | 0.1403 |
| exception_required | 0.7109 | recall_at_precision_0.6 | calibrated HGB | 0.4354 | 0.8462 | 0.2931 |

## Evaluation audit (panel-overlap memorisation)

| Target | Time-split ROC-AUC | Loan-disjoint ROC-AUC | Interpretation |
|---|---|---|---|
| next_12m_default_flag | 0.9999 | 0.6699 (range 0.5721–0.7493) | The time-split AUC includes legitimate panel overlap; this loan-disjoint estimate removes it and quantifies genuine cross-loan signal. Values near 0.5 mean the signal is carried by the loan's own history; values well above 0.5 mean features generalise across loans. Both numbers are shipped on purpose. |
| next_3m_delinquency_flag | 0.8047 | 0.5979 (range 0.5643–0.6215) | The time-split AUC includes legitimate panel overlap; this loan-disjoint estimate removes it and quantifies genuine cross-loan signal. Values near 0.5 mean the signal is carried by the loan's own history; values well above 0.5 mean features generalise across loans. Both numbers are shipped on purpose. |

## Bootstrap confidence intervals (validation ROC-AUC, 2.5–97.5 percentile)

| Target | Point | CI low | CI high |
|---|---|---|---|
| next_3m_delinquency_flag | 0.8047 | 0.7804 | 0.8292 |
| next_6m_delinquency_flag | 0.8774 | 0.8637 | 0.8942 |
| next_12m_default_flag | 0.9999 | 0.9998 | 1.0 |
| next_12m_prepayment_flag | 0.9927 | 0.9884 | 0.9958 |
| exception_required | 0.9112 | 0.8921 | 0.9275 |

## Calibration by segment (ECE per credit band / vintage / state)

Worst-calibrated segments first; the full table is in `calibration_by_segment.csv`.

| Target | Segment | Value | n | ECE |
|---|---|---|---|---|
| next_12m_prepayment_flag | credit_score_band | 740+ | 830 | 0.0863 |
| next_12m_prepayment_flag | state | WA | 429 | 0.0804 |
| next_12m_prepayment_flag | state | GA | 400 | 0.0787 |
| next_12m_prepayment_flag | credit_score_band | 680-739 | 744 | 0.0721 |
| next_12m_prepayment_flag | state | CA | 440 | 0.0719 |
| next_12m_prepayment_flag | state | TX | 352 | 0.0676 |
| exception_required | state | FL | 576 | 0.0618 |
| next_6m_delinquency_flag | state | TX | 443 | 0.058 |
| next_12m_prepayment_flag | vintage_year | 2023 | 2930 | 0.0577 |
| next_12m_prepayment_flag | state | IL | 433 | 0.0567 |
| next_12m_prepayment_flag | state | NY | 417 | 0.0566 |
| next_12m_prepayment_flag | state | FL | 459 | 0.0523 |

## Calibration & recall-at-fixed-precision

| Target | ECE | Recall@precision≥0.6 | Threshold |
|---|---|---|---|
| next_3m_delinquency_flag | 0.008 | *precision target 0.6 unreachable; max precision 0.0 at 0.5* | — |
| next_6m_delinquency_flag | 0.012 | 0.2966 | 0.4366 |
| next_12m_default_flag | 0.0184 | 1.0 | 0.1864 |
| next_12m_prepayment_flag | 0.0577 | 0.4236 | 0.8133 |
| exception_required | 0.0324 | 0.2931 | 0.7109 |

## Limitations
- Probabilities are statistical estimates; calibration degrades under regime change.
- Sparse default tails make extreme-decile metrics unstable.
- The synthetic demo pack is a functional test only, not evidence of real-world performance.

## Controls
- Targets and post-event flags are excluded by a hard schema gate (`assert_no_leakage`).
- No LLM participates in prediction; the copilot is read-only and advisory.
- Fairness/bias testing, legal review and approval thresholds are required before any production use.