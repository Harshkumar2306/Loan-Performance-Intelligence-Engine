# Model Card — Loan Performance Intelligence Engine (generated)

*Generated 2026-08-30 from the actual pipeline run. Values are never pre-filled.*

## Intended use
Rank monthly loan observations for a human reviewer. Not for autonomous servicing, credit
approval, or any consumer-impacting decision.

## Data & validation
- Time split: strict reporting-month out-of-time split; no future rows in training (cutoff 2023-07-01).
- Fit rows: 1,286  |  Validation rows: 11,107 across 18 months.
- Loans spanning the boundary (expected in panel data): 533.
- Feature count: 26 (see `features_used.json`).

## Metrics (out-of-time validation, threshold-free)

| Target | Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| next_3m_delinquency_flag | baseline RF | 0.6204 | 0.1073 | 0.113 |
| next_3m_delinquency_flag | calibrated HGB | 0.5999 | 0.1135 | 0.0663 |
| next_6m_delinquency_flag | baseline RF | 0.6889 | 0.223 | 0.1579 |
| next_6m_delinquency_flag | calibrated HGB | 0.6572 | 0.2171 | 0.1033 |
| next_12m_default_flag | baseline RF | 0.7814 | 0.2766 | 0.0936 |
| next_12m_default_flag | calibrated HGB | 0.7489 | 0.284 | 0.0491 |
| next_12m_prepayment_flag | baseline RF | 0.7097 | 0.4162 | 0.1887 |
| next_12m_prepayment_flag | calibrated HGB | 0.7152 | 0.4167 | 0.1479 |
| exception_required | baseline RF | 0.7091 | 0.4613 | 0.1678 |
| exception_required | calibrated HGB | 0.7132 | 0.4405 | 0.1084 |
| next_state | random forest (multiclass) | macro-F1 0.506 | weighted-F1 0.9294 | classes ['30_dpd', '60_dpd', '90_dpd', 'current', 'default', 'prepaid'] |
| exception_type | random forest (multiclass) | macro-F1 0.7334 | weighted-F1 0.8791 | classes ['data_break', 'missing_docs', 'none', 'source_conflict'] |

## Operating point (threshold chosen on validation only)

Per-target thresholds come from the recall-at-fixed-precision scan, falling back to the
validation best-F1 threshold when the precision target is unreachable (e.g. a strongly
compressed probability range on a low-event-rate target). Both models are evaluated at
the SAME threshold.

| Target | Threshold | Method | Model | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| next_3m_delinquency_flag | 0.1576 | best_validation_f1_fallback | baseline RF | 0.1392 | 0.075 | 0.9748 |
| next_3m_delinquency_flag | 0.1576 | best_validation_f1_fallback | calibrated HGB | 0.165 | 0.1548 | 0.1766 |
| next_6m_delinquency_flag | 0.1292 | best_validation_f1_fallback | baseline RF | 0.2092 | 0.1168 | 1.0 |
| next_6m_delinquency_flag | 0.1292 | best_validation_f1_fallback | calibrated HGB | 0.2772 | 0.1884 | 0.5242 |
| next_12m_default_flag | 0.3645 | best_validation_f1_fallback | baseline RF | 0.2747 | 0.1764 | 0.6213 |
| next_12m_default_flag | 0.3645 | best_validation_f1_fallback | calibrated HGB | 0.3191 | 0.3559 | 0.2892 |
| next_12m_prepayment_flag | 0.3557 | best_validation_f1_fallback | baseline RF | 0.3826 | 0.245 | 0.8723 |
| next_12m_prepayment_flag | 0.3557 | best_validation_f1_fallback | calibrated HGB | 0.43 | 0.3728 | 0.5078 |
| exception_required | 0.4997 | recall_at_precision_0.6 | baseline RF | 0.4287 | 0.584 | 0.3387 |
| exception_required | 0.4997 | recall_at_precision_0.6 | calibrated HGB | 0.3607 | 0.7032 | 0.2426 |

## Evaluation audit (panel-overlap memorisation)

| Target | Time-split ROC-AUC | Loan-disjoint ROC-AUC | Interpretation |
|---|---|---|---|
| next_12m_default_flag | 0.7489 | 0.6215 (range 0.4922–0.7243) | The time-split AUC includes legitimate panel overlap; this loan-disjoint estimate removes it and quantifies genuine cross-loan signal. Values near 0.5 mean the signal is carried by the loan's own history; values well above 0.5 mean features generalise across loans. Both numbers are shipped on purpose. |
| next_3m_delinquency_flag | 0.5999 | 0.5399 (range 0.4184–0.6687) | The time-split AUC includes legitimate panel overlap; this loan-disjoint estimate removes it and quantifies genuine cross-loan signal. Values near 0.5 mean the signal is carried by the loan's own history; values well above 0.5 mean features generalise across loans. Both numbers are shipped on purpose. |

## Bootstrap confidence intervals (validation ROC-AUC, 2.5–97.5 percentile)

| Target | Point | CI low | CI high |
|---|---|---|---|
| next_3m_delinquency_flag | 0.5999 | 0.5784 | 0.6205 |
| next_6m_delinquency_flag | 0.6572 | 0.6415 | 0.6735 |
| next_12m_default_flag | 0.7489 | 0.7237 | 0.7721 |
| next_12m_prepayment_flag | 0.7152 | 0.7017 | 0.7295 |
| exception_required | 0.7132 | 0.6965 | 0.7286 |

## Calibration by segment (ECE per credit band / vintage / state)

Worst-calibrated segments first; the full table is in `calibration_by_segment.csv`.

| Target | Segment | Value | n | ECE |
|---|---|---|---|---|
| next_12m_prepayment_flag | state | TX | 1117 | 0.1666 |
| next_12m_prepayment_flag | credit_score_band | 680-739 | 2354 | 0.1219 |
| next_12m_prepayment_flag | credit_score_band | 740+ | 2694 | 0.1055 |
| next_6m_delinquency_flag | state | IL | 1623 | 0.0878 |
| exception_required | state | IL | 1717 | 0.0874 |
| next_12m_prepayment_flag | state | IL | 1405 | 0.0866 |
| exception_required | credit_score_band | 680-739 | 2853 | 0.0855 |
| next_6m_delinquency_flag | credit_score_band | <620 | 1780 | 0.0777 |
| exception_required | state | NY | 1605 | 0.0745 |
| next_6m_delinquency_flag | state | GA | 1498 | 0.0714 |
| next_12m_prepayment_flag | credit_score_band | 620-679 | 2477 | 0.0713 |
| exception_required | state | WA | 1587 | 0.0705 |

## Calibration & recall-at-fixed-precision

| Target | ECE | Recall@precision≥0.6 | Threshold |
|---|---|---|---|
| next_3m_delinquency_flag | 0.0259 | *precision target 0.6 unreachable; max precision 0.0476 at 0.5* | — |
| next_6m_delinquency_flag | 0.0454 | *precision target 0.6 unreachable; max precision 0.3952 at 0.5* | — |
| next_12m_default_flag | 0.0255 | *precision target 0.6 unreachable; max precision 0.4858 at 0.5* | — |
| next_12m_prepayment_flag | 0.0702 | *precision target 0.6 unreachable; max precision 0.4583 at 0.5* | — |
| exception_required | 0.0633 | 0.2426 | 0.4997 |

## Limitations
- Probabilities are statistical estimates; calibration degrades under regime change.
- Sparse default tails make extreme-decile metrics unstable.
- The synthetic demo pack is a functional test only, not evidence of real-world performance.

## Controls
- Targets and post-event flags are excluded by a hard schema gate (`assert_no_leakage`).
- No LLM participates in prediction; the copilot is read-only and advisory.
- Fairness/bias testing, legal review and approval thresholds are required before any production use.