# Explainability Report

*Generated 2026-08-30 by `run_pipeline.py`.*

## Global permutation importance (validation, average-precision scoring)

| Target | Feature | Importance ± std |
|---|---|---|
| exception_required | document_status | 0.149186 ± 0.007761 |
| exception_required | interest_rate | 0.055926 ± 0.003887 |
| exception_required | dti_numeric | 0.040781 ± 0.003636 |
| exception_required | original_balance | 0.036172 ± 0.004851 |
| exception_required | credit_score_numeric | 0.033217 ± 0.004361 |
| exception_required | state | 0.027748 ± 0.00336 |
| exception_required | current_balance | 0.02697 ± 0.002942 |
| exception_required | loan_purpose | 0.019008 ± 0.007686 |
| exception_required | ltv_numeric | 0.01124 ± 0.005879 |
| exception_required | credit_score_band | 0.007402 ± 0.000101 |
| exception_required | dti_band | 0.006327 ± 0.002205 |
| exception_required | ever_dpd | 0.005751 ± 0.000911 |
| exception_required | property_type | 0.004079 ± 0.001493 |
| exception_required | occupancy_type | 0.003254 ± 0.000881 |
| exception_required | balance_ratio | 0.003101 ± 0.00157 |
| exception_required | servicer_name | 0.002137 ± 0.001194 |
| exception_required | current_status | 0.001542 ± 0.000976 |
| exception_required | days_past_due | 0.000669 ± 0.000356 |
| exception_required | ltv_band | 0.000563 ± 0.001399 |
| exception_required | source_system | 0.0 ± 0.0 |
| exception_required | dpd_velocity | 0.0 ± 0.0 |
| exception_required | vintage_year | 0.0 ± 0.0 |
| exception_required | paydown_rate | -0.0003 ± 0.00472 |
| exception_required | loan_age_months | -0.000397 ± 0.000357 |
| exception_required | term_ratio | -0.000864 ± 0.000474 |
| exception_required | remaining_term_months | -0.002247 ± 0.007188 |
| next_12m_default_flag | dti_numeric | 0.10874 ± 0.004694 |
| next_12m_default_flag | credit_score_numeric | 0.066089 ± 0.016165 |
| next_12m_default_flag | ltv_numeric | 0.065276 ± 0.010001 |
| next_12m_default_flag | interest_rate | 0.063577 ± 0.015311 |

## Error analysis (false positives / false negatives, validation period)

### next_3m_delinquency_flag

- False positives: 20 (rate 0.0019)
- False negatives: 752 (rate 0.0707)
- Threshold: 0.5
- False Positives Segment Concentration: {"credit_score_band": {"<620": 0.0039, "740+": 0.0026, "680-739": 0.0015}, "state": {"WA": 0.0058, "CA": 0.0032, "TX": 0.0023}, "servicer_name": {"Atlas": 0.0027, "Beacon": 0.0026, "Cedar": 0.0003}, "ltv_band": {"81-100": 0.0026, "<=60": 0.0017, "61-80": 0.0017}}
- False Negatives Segment Concentration: {"credit_score_band": {"<620": 0.0921, "620-679": 0.0895, "680-739": 0.0713}, "state": {"IL": 0.0986, "TX": 0.0912, "GA": 0.0701}, "servicer_name": {"Beacon": 0.0733, "Atlas": 0.0716, "Cedar": 0.0671}, "ltv_band": {"100+": 0.1198, "81-100": 0.078, "61-80": 0.0731}}

### next_6m_delinquency_flag

- False positives: 176 (rate 0.0166)
- False negatives: 1123 (rate 0.1056)
- Threshold: 0.5
- False Positives Segment Concentration: {"credit_score_band": {"<620": 0.023, "620-679": 0.017, "680-739": 0.0164}, "state": {"IL": 0.0567, "WA": 0.0161, "TX": 0.0131}, "servicer_name": {"Cedar": 0.021, "Atlas": 0.0149, "Beacon": 0.0138}, "ltv_band": {"81-100": 0.025, "<=60": 0.0166, "61-80": 0.0131}}
- False Negatives Segment Concentration: {"credit_score_band": {"<620": 0.1444, "620-679": 0.1409, "680-739": 0.1029}, "state": {"TX": 0.1337, "IL": 0.1183, "WA": 0.113}, "servicer_name": {"Atlas": 0.1143, "Beacon": 0.1112, "Cedar": 0.0917}, "ltv_band": {"100+": 0.2107, "81-100": 0.1157, "61-80": 0.1067}}

### next_12m_default_flag

- False positives: 127 (rate 0.0119)
- False negatives: 461 (rate 0.0434)
- Threshold: 0.5
- False Positives Segment Concentration: {"credit_score_band": {"<620": 0.0376, "620-679": 0.0196, "680-739": 0.0022}, "state": {"FL": 0.0185, "NY": 0.0172, "CA": 0.0171}, "servicer_name": {"Atlas": 0.0223, "Cedar": 0.0089, "Beacon": 0.0058}, "ltv_band": {"100+": 0.062, "81-100": 0.0254, "61-80": 0.0078}}
- False Negatives Segment Concentration: {"credit_score_band": {"<620": 0.0753, "620-679": 0.0601, "680-739": 0.0404}, "state": {"IL": 0.0616, "WA": 0.0482, "TX": 0.0464}, "servicer_name": {"Atlas": 0.0491, "Beacon": 0.0418, "Cedar": 0.0397}, "ltv_band": {"61-80": 0.0468, "81-100": 0.0455, "100+": 0.0372}}

### next_12m_prepayment_flag

- False positives: 695 (rate 0.0751)
- False negatives: 1198 (rate 0.1295)
- Threshold: 0.5
- False Positives Segment Concentration: {"credit_score_band": {"740+": 0.1262, "680-739": 0.1083, "620-679": 0.0371}, "state": {"TX": 0.1746, "GA": 0.0747, "IL": 0.0733}, "servicer_name": {"Cedar": 0.0898, "Atlas": 0.0696, "Beacon": 0.0664}, "ltv_band": {"81-100": 0.093, "<=60": 0.0858, "61-80": 0.0637}}
- False Negatives Segment Concentration: {"credit_score_band": {"740+": 0.1752, "680-739": 0.1194, "<620": 0.1169}, "state": {"GA": 0.1517, "IL": 0.1502, "WA": 0.1464}, "servicer_name": {"Cedar": 0.1404, "Beacon": 0.1311, "Atlas": 0.1163}, "ltv_band": {"61-80": 0.1355, "<=60": 0.1345, "81-100": 0.1159}}

### exception_required

- False positives: 165 (rate 0.0149)
- False negatives: 1221 (rate 0.1099)
- Threshold: 0.4997
- False Positives Segment Concentration: {"credit_score_band": {"620-679": 0.0224, "<620": 0.0202, "680-739": 0.0126}, "state": {"IL": 0.0315, "GA": 0.0241, "WA": 0.0195}, "servicer_name": {"Beacon": 0.0247, "Cedar": 0.01, "Atlas": 0.0087}, "ltv_band": {"61-80": 0.0205, "<=60": 0.0117, "81-100": 0.0074}}
- False Negatives Segment Concentration: {"credit_score_band": {"740+": 0.1177, "<620": 0.1088, "680-739": 0.1073}, "state": {"CA": 0.1223, "NY": 0.1121, "FL": 0.1118}, "servicer_name": {"Cedar": 0.1186, "Atlas": 0.1145, "Beacon": 0.0979}, "ltv_band": {"<=60": 0.1242, "61-80": 0.11, "81-100": 0.1057}}

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
| next_12m_default_flag | credit_score_band | 620-679 | 2647 | 0.0782 | 0.0378 | 0.0196 | 0.0601 |
| next_12m_default_flag | credit_score_band | 680-739 | 2750 | 0.0462 | 0.008 | 0.0022 | 0.0404 |
| next_12m_default_flag | credit_score_band | 740+ | 3456 | 0.02 | 0.0041 | 0.0006 | 0.0165 |
| next_12m_default_flag | credit_score_band | <620 | 1780 | 0.1 | 0.0624 | 0.0376 | 0.0753 |
| next_12m_default_flag | state | CA | 1580 | 0.0468 | 0.0228 | 0.0171 | 0.0411 |
| next_12m_default_flag | state | FL | 1511 | 0.0602 | 0.0324 | 0.0185 | 0.0463 |
| next_12m_default_flag | state | GA | 1498 | 0.0447 | 0.014 | 0.0073 | 0.0381 |
| next_12m_default_flag | state | IL | 1623 | 0.0795 | 0.0345 | 0.0166 | 0.0616 |
| next_12m_default_flag | state | NY | 1570 | 0.0229 | 0.0185 | 0.0172 | 0.0217 |
| next_12m_default_flag | state | TX | 1294 | 0.0719 | 0.0286 | 0.0031 | 0.0464 |
| next_12m_default_flag | state | WA | 1557 | 0.0584 | 0.0122 | 0.0019 | 0.0482 |
| next_3m_delinquency_flag | credit_score_band | 620-679 | 2647 | 0.0895 | 0.0 | 0.0 | 0.0895 |
| next_3m_delinquency_flag | credit_score_band | 680-739 | 2750 | 0.0713 | 0.0015 | 0.0015 | 0.0713 |
| next_3m_delinquency_flag | credit_score_band | 740+ | 3456 | 0.0451 | 0.0029 | 0.0026 | 0.0448 |
| next_3m_delinquency_flag | credit_score_band | <620 | 1780 | 0.0921 | 0.0039 | 0.0039 | 0.0921 |
| next_3m_delinquency_flag | state | CA | 1580 | 0.069 | 0.0032 | 0.0032 | 0.069 |
| next_3m_delinquency_flag | state | FL | 1511 | 0.0609 | 0.002 | 0.002 | 0.0609 |
| next_3m_delinquency_flag | state | GA | 1498 | 0.0701 | 0.0 | 0.0 | 0.0701 |
| next_3m_delinquency_flag | state | IL | 1623 | 0.0986 | 0.0 | 0.0 | 0.0986 |
| next_3m_delinquency_flag | state | NY | 1570 | 0.0382 | 0.0 | 0.0 | 0.0382 |
| next_3m_delinquency_flag | state | TX | 1294 | 0.092 | 0.0031 | 0.0023 | 0.0912 |
| next_3m_delinquency_flag | state | WA | 1557 | 0.0694 | 0.0058 | 0.0058 | 0.0694 |

*Descriptive parity monitoring only; full bias/fairness review is a documented pre-production gate (see the model card).*
