# Data Intelligence Report

*Generated 2026-08-30 by `run_pipeline.py`.*

- **Rows profiled:** 12,393  |  **Columns:** 43  |  **Reporting months:** 24
- **Rows below 60/100 data quality:** 1.2%

## Deterministic rule breaches

| Rule | Breaches |
|---|---|
| prepaid_positive_balance | 1,889 |
| dpd_status_break | 241 |
| balance_exceeds_original | 103 |
| missing_core | 56 |
| duplicate_row | 48 |
| negative_balance | 47 |
| future_update | 35 |
| dpd_impossible | 13 |
| default_no_dpd | 1 |
| invalid_term | 0 |
| term_implausible | 0 |
| invalid_date_order | 0 |
| age_date_break | 0 |
| negative_rate | 0 |

## Outlier flags (robust z > 5)

| Column | Flags |
|---|---|
| remaining_term_months | 4,974 |
| paydown_rate | 747 |
| balance_ratio | 158 |
| loan_age_months | 0 |
| original_balance | 0 |
| current_balance | 0 |
| interest_rate | 0 |
| credit_score | 0 |
| ltv | 0 |
| dti | 0 |
| credit_score_numeric | 0 |
| ltv_numeric | 0 |
| dti_numeric | 0 |
| term_ratio | 0 |

## Source reconciliation (servicer updates)

| Conflict | Rows |
|---|---|
| balance_source_conflict | 1,495 |
| status_source_conflict | 11,595 |
| stale_record | 0 |

## Highest numeric correlations

| Pair | |corr| |
|---|---|
| month_index__loan_age_months | 1.0 |
| ltv__ltv_numeric | 1.0 |
| dti__dti_numeric | 1.0 |
| credit_score__credit_score_numeric | 1.0 |
| remaining_term_months__term_months | 0.997 |
| original_balance__current_balance | 0.986 |
| month_index__term_ratio | 0.887 |
| loan_age_months__term_ratio | 0.887 |
| next_3m_delinquency_flag__next_6m_delinquency_flag | 0.757 |
| balance_ratio__paydown_rate | 0.623 |

## Strongest categorical associations (Cramér's V)

| Pair | V | p |
|---|---|---|
| current_status__next_state | 0.5778 | 0.0 |
| document_status__exception_type | 0.5774 | 0.0 |
| next_state__credit_score_band | 0.0894 | 0.0 |
| current_status__credit_score_band | 0.0837 | 0.0 |
| state__loan_purpose | 0.0833 | 0.0 |
| credit_score_band__state | 0.0761 | 0.0 |
| loan_purpose__property_type | 0.0677 | 0.0 |
| dti_band__state | 0.0656 | 0.0 |
| state__occupancy_type | 0.0651 | 0.0 |
| ltv_band__occupancy_type | 0.0648 | 0.0 |
| state__property_type | 0.0641 | 0.0 |
| servicer_name__credit_score_band | 0.0584 | 0.0 |

## Top association rules (by lift)

| Rule | Support | Confidence | Lift |
|---|---|---|---|
| `servicer_name=Atlas -> ltv_band=100+` | 0.0112 | 0.0362 | 1.523 |
| `servicer_name=Cedar -> next_state=90_dpd` | 0.0015 | 0.0044 | 1.361 |
| `servicer_name=Cedar -> current_status=90 DPD` | 0.0012 | 0.0037 | 1.334 |
| `servicer_name=Beacon -> dti_band=<=20` | 0.034 | 0.0943 | 1.208 |
| `servicer_name=Atlas -> credit_score_band=620-679` | 0.0923 | 0.2984 | 1.194 |
| `servicer_name=Beacon -> exception_type=data_break` | 0.0016 | 0.0043 | 1.173 |
| `servicer_name=Atlas -> current_status=60 DPD` | 0.0017 | 0.0055 | 1.171 |
| `servicer_name=Atlas -> state=WA` | 0.051 | 0.1648 | 1.16 |
| `servicer_name=Beacon -> state=FL` | 0.0597 | 0.1658 | 1.144 |
| `servicer_name=Cedar -> state=IL` | 0.0584 | 0.1765 | 1.142 |
| `servicer_name=Beacon -> state=CA` | 0.0612 | 0.1701 | 1.132 |
| `servicer_name=Atlas -> next_state=60_dpd` | 0.0019 | 0.0061 | 1.127 |

## Highly dependent fields (|corr| or V > 0.9)

credit_score__credit_score_numeric, dti__dti_numeric, ltv__ltv_numeric, month_index__loan_age_months, original_balance__current_balance, remaining_term_months__term_months

## Train vs test drift (PSI)

| Column | PSI |
|---|---|
| loan_age_months | 8.47 |
| remaining_term_months | 7.8514 |
| term_ratio | 7.6973 |
| balance_ratio | 1.0741 |
| paydown_rate | 0.0605 |
| ltv | 0.0457 |
| ltv_numeric | 0.0457 |
| current_balance | 0.0428 |
| days_past_due | 0.0399 |
| current_status | 0.2087 |

## Missingness patterns (top 8)

| Pattern | Rows |
|---|---|
| `complete_row` | 9,859 |
| `next_12m_prepayment_flag` | 1,878 |
| `next_3m_delinquency_flag+next_6m_delinquency_flag+next_12m_default_flag` | 480 |
| `next_state` | 98 |
| `current_balance+current_status+balance_ratio` | 22 |
| `current_balance+balance_ratio` | 21 |
| `next_12m_prepayment_flag+next_state` | 17 |
| `current_balance+next_12m_prepayment_flag+balance_ratio` | 9 |

## Data-quality score distribution

```
{
  "count": 12393.0,
  "mean": 83.16,
  "std": 7.35,
  "min": 39.0,
  "25%": 80.0,
  "50%": 82.0,
  "75%": 90.0,
  "max": 90.0
}
```