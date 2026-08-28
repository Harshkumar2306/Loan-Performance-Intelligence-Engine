# Explainability Report

*Generated 2026-08-30 by `run_pipeline.py`.*

## Global permutation importance (validation, average-precision scoring)

| Target | Feature | Importance ± std |
|---|---|---|
| exception_required | document_status | 0.262401 ± 0.008047 |
| exception_required | dti_numeric | 0.128406 ± 0.007408 |
| exception_required | original_balance | 0.122827 ± 0.002541 |
| exception_required | interest_rate | 0.079204 ± 0.00292 |
| exception_required | current_balance | 0.073675 ± 0.004123 |
| exception_required | ltv_numeric | 0.055117 ± 0.002499 |
| exception_required | state | 0.034237 ± 0.004575 |
| exception_required | loan_purpose | 0.025378 ± 0.004742 |
| exception_required | credit_score_numeric | 0.024281 ± 0.000291 |
| exception_required | remaining_term_months | 0.018424 ± 0.001819 |
| exception_required | occupancy_type | 0.015921 ± 0.000687 |
| exception_required | credit_score_band | 0.011896 ± 0.002012 |
| exception_required | property_type | 0.011051 ± 0.001092 |
| exception_required | servicer_name | 0.008648 ± 0.001062 |
| exception_required | current_status | 0.00402 ± 0.003041 |
| exception_required | dti_band | 0.003784 ± 0.000518 |
| exception_required | balance_ratio | 0.003536 ± 0.002596 |
| exception_required | ltv_band | 0.003015 ± 0.000683 |
| exception_required | term_ratio | 0.000884 ± 0.000582 |
| exception_required | loan_age_months | 0.000351 ± 0.001049 |
| exception_required | days_past_due | 0.000314 ± 0.000432 |
| exception_required | source_system | 0.0 ± 0.0 |
| exception_required | vintage_year | 0.0 ± 0.0 |
| exception_required | ever_dpd | -0.000236 ± 0.000599 |
| exception_required | dpd_velocity | -0.001883 ± 0.000385 |
| exception_required | paydown_rate | -0.003 ± 0.000302 |
| next_12m_default_flag | dti_numeric | 0.29681 ± 0.029768 |
| next_12m_default_flag | interest_rate | 0.286287 ± 0.006825 |
| next_12m_default_flag | original_balance | 0.261429 ± 0.005232 |
| next_12m_default_flag | current_status | 0.198222 ± 0.014994 |

## Error analysis (false positives / false negatives, validation period)

### next_3m_delinquency_flag

- False positives: 0 (rate 0.0)
- False negatives: 257 (rate 0.0701)
- Threshold: 0.5
- False Negatives Segment Concentration: {"credit_score_band": {"620-679": 0.1004, "<620": 0.0754, "680-739": 0.0676}, "state": {"TX": 0.1084, "IL": 0.0899, "WA": 0.0781}, "servicer_name": {"Atlas": 0.0834, "Beacon": 0.0676, "Cedar": 0.0609}, "ltv_band": {"100+": 0.1429, "81-100": 0.0849, "61-80": 0.0653}}

### next_6m_delinquency_flag

- False positives: 71 (rate 0.0194)
- False negatives: 268 (rate 0.0731)
- Threshold: 0.4366
- False Positives Segment Concentration: {"credit_score_band": {"<620": 0.0285, "740+": 0.0211, "620-679": 0.0166}, "state": {"IL": 0.0414, "WA": 0.0223, "FL": 0.0189}, "servicer_name": {"Beacon": 0.021, "Atlas": 0.0199, "Cedar": 0.0171}, "ltv_band": {"61-80": 0.0261, "100+": 0.026, "81-100": 0.018}}
- False Negatives Segment Concentration: {"credit_score_band": {"620-679": 0.1004, "680-739": 0.0805, "<620": 0.0804}, "state": {"WA": 0.0967, "TX": 0.0903, "CA": 0.0859}, "servicer_name": {"Atlas": 0.0943, "Beacon": 0.0661, "Cedar": 0.0617}, "ltv_band": {"81-100": 0.0892, "<=60": 0.0766, "100+": 0.0649}}

### next_12m_default_flag

- False positives: 38 (rate 0.0104)
- False negatives: 0 (rate 0.0)
- Threshold: 0.1864
- False Positives Segment Concentration: {"credit_score_band": {"<620": 0.0218, "620-679": 0.0188, "680-739": 0.0043}, "state": {"FL": 0.0246, "NY": 0.0144, "TX": 0.009}, "servicer_name": {"Atlas": 0.0154, "Beacon": 0.0105, "Cedar": 0.0057}, "ltv_band": {"100+": 0.039, "81-100": 0.0127, "61-80": 0.0125}}

### next_12m_prepayment_flag

- False positives: 0 (rate 0.0)
- False negatives: 200 (rate 0.0683)
- Threshold: 0.8133
- False Negatives Segment Concentration: {"credit_score_band": {"740+": 0.0988, "680-739": 0.0793, "<620": 0.0516}, "state": {"CA": 0.0977, "NY": 0.0767, "FL": 0.0654}, "servicer_name": {"Cedar": 0.0805, "Atlas": 0.067, "Beacon": 0.058}, "ltv_band": {"61-80": 0.0787, "81-100": 0.062, "<=60": 0.0558}}

### exception_required

- False positives: 30 (rate 0.0077)
- False negatives: 398 (rate 0.1021)
- Threshold: 0.7109
- False Positives Segment Concentration: {"credit_score_band": {"<620": 0.0131, "620-679": 0.0102, "740+": 0.0056}, "state": {"WA": 0.0125, "IL": 0.01, "FL": 0.0087}, "servicer_name": {"Beacon": 0.0099, "Atlas": 0.0068, "Cedar": 0.0062}, "ltv_band": {"61-80": 0.0082, "81-100": 0.0078, "<=60": 0.0072}}
- False Negatives Segment Concentration: {"credit_score_band": {"680-739": 0.1168, "740+": 0.1069, "620-679": 0.0942}, "state": {"FL": 0.125, "IL": 0.112, "CA": 0.1098}, "servicer_name": {"Atlas": 0.1117, "Cedar": 0.1062, "Beacon": 0.0902}, "ltv_band": {"<=60": 0.1178, "61-80": 0.1014, "81-100": 0.0971}}

## Local explanations

See `local_explanations.csv` — per-record, per-target attributions with an explicit
`method` column (`shap_tree` when TreeSHAP is available, otherwise an honestly labelled
`deviation_rank` heuristic). See `validation_predictions.csv` and `error_examples.csv`
for the underlying records.

## Uncertainty

`confidence = 0.7 * model_confidence + 0.3 * data_confidence`, where
`model_confidence = 2*|p-0.5|` and `data_confidence = data_quality_score/100`.
The formula is auditable in `explainability.model_confidence`.
## Counterfactual explanations

`counterfactuals.csv` holds what-if re-scores for the top-priority records: each documented intervention (cure DPD, credit +40, LTV −10, rate −1, 20% paydown) is applied and the probability delta reported, plus the combined cure. This answers "what would change the model's mind" for a reviewer.

## Segment parity (descriptive monitoring)

| Target | Segment | Value | n | Event rate | Predicted rate | FPR | FNR |
|---|---|---|---|---|---|---|---|
| next_12m_default_flag | credit_score_band | 620-679 | 906 | 0.0673 | 0.0861 | 0.0188 | 0.0 |
| next_12m_default_flag | credit_score_band | 680-739 | 932 | 0.03 | 0.0343 | 0.0043 | 0.0 |
| next_12m_default_flag | credit_score_band | 740+ | 1231 | 0.0187 | 0.0219 | 0.0032 | 0.0 |
| next_12m_default_flag | credit_score_band | <620 | 597 | 0.057 | 0.0787 | 0.0218 | 0.0 |
| next_12m_default_flag | state | CA | 547 | 0.0402 | 0.0475 | 0.0073 | 0.0 |
| next_12m_default_flag | state | FL | 529 | 0.0302 | 0.0548 | 0.0246 | 0.0 |
| next_12m_default_flag | state | GA | 499 | 0.024 | 0.0321 | 0.008 | 0.0 |
| next_12m_default_flag | state | IL | 556 | 0.0683 | 0.0719 | 0.0036 | 0.0 |
| next_12m_default_flag | state | NY | 554 | 0.0108 | 0.0253 | 0.0144 | 0.0 |
| next_12m_default_flag | state | TX | 443 | 0.0474 | 0.0564 | 0.009 | 0.0 |
| next_12m_default_flag | state | WA | 538 | 0.0576 | 0.0632 | 0.0056 | 0.0 |
| next_3m_delinquency_flag | credit_score_band | 620-679 | 906 | 0.1004 | 0.0 | 0.0 | 0.1004 |
| next_3m_delinquency_flag | credit_score_band | 680-739 | 932 | 0.0676 | 0.0 | 0.0 | 0.0676 |
| next_3m_delinquency_flag | credit_score_band | 740+ | 1231 | 0.0471 | 0.0 | 0.0 | 0.0471 |
| next_3m_delinquency_flag | credit_score_band | <620 | 597 | 0.0754 | 0.0 | 0.0 | 0.0754 |
| next_3m_delinquency_flag | state | CA | 547 | 0.075 | 0.0 | 0.0 | 0.075 |
| next_3m_delinquency_flag | state | FL | 529 | 0.0586 | 0.0 | 0.0 | 0.0586 |
| next_3m_delinquency_flag | state | GA | 499 | 0.0601 | 0.0 | 0.0 | 0.0601 |
| next_3m_delinquency_flag | state | IL | 556 | 0.0899 | 0.0 | 0.0 | 0.0899 |
| next_3m_delinquency_flag | state | NY | 554 | 0.0271 | 0.0 | 0.0 | 0.0271 |
| next_3m_delinquency_flag | state | TX | 443 | 0.1084 | 0.0 | 0.0 | 0.1084 |
| next_3m_delinquency_flag | state | WA | 538 | 0.0781 | 0.0 | 0.0 | 0.0781 |

*Descriptive parity monitoring only; full bias/fairness review is a documented pre-production gate (see the model card).*
