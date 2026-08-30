# Scenario & Stress Report

*Generated 2026-08-30 by `run_pipeline.py`.*

Scenarios are transparent feature shocks followed by the same fitted models, so every
number is auditable. Segment rows live in `scenario_report.csv`.

## Scenario-level projected default and prepayment rates (bootstrap 5–95 bounds)

| Scenario | Mean default | Default 5th | Default 95th | Mean prepay | Prepay 5th | Prepay 95th | Sims |
|---|---|---|---|---|---|---|---|
| Base | 0.09514 | 0.08016 | 0.11004 | 0.26468 | 0.2465 | 0.28214 | 30 |
| Adverse Credit | 0.12211 | 0.10344 | 0.13704 | 0.21373 | 0.19786 | 0.23094 | 30 |
| High Prepayment | 0.09397 | 0.08064 | 0.1068 | 0.32602 | 0.30732 | 0.34408 | 30 |

## Driver attribution (one-feature-at-a-time counterfactuals)

| Scenario | Δ default (pp) | Credit shock (pp) | Prepay-uplift effect on default (pp) | Δ prepayment (pp) | Uplift applied (pp) |
|---|---|---|---|---|---|
| Adverse Credit | 2.8524 | 2.8524 | 0.0 | -4.6984 | -1.0 |
| High Prepayment | 0.0 | 0.0 | 0.0 | 6.0 | 6.0 |

## Segment-level default-rate impacts (top 12 largest moves vs Base)

| Scenario | Segment | Value | Default rate | Delinquency | Prepayment |
|---|---|---|---|---|---|
| Adverse Credit | credit_score_band | <620 | 0.3025 | 0.1205 | 0.0856 |
| Adverse Credit | state | FL | 0.1485 | 0.0817 | 0.1541 |
| Adverse Credit | state | GA | 0.1185 | 0.0777 | 0.2405 |
| Adverse Credit | servicer_name | Beacon | 0.1127 | 0.0763 | 0.2030 |
| Adverse Credit | servicer_name | Cedar | 0.1332 | 0.0853 | 0.2369 |
| Adverse Credit | vintage_year | 2023 | 0.1230 | 0.0820 | 0.2184 |
| Adverse Credit | state | CA | 0.1337 | 0.0899 | 0.2088 |
| Adverse Credit | state | TX | 0.1291 | 0.0814 | 0.3050 |
| Adverse Credit | state | IL | 0.1472 | 0.1088 | 0.1968 |
| Adverse Credit | state | NY | 0.0970 | 0.0611 | 0.2251 |
| Adverse Credit | servicer_name | Atlas | 0.1241 | 0.0851 | 0.2163 |
| Adverse Credit | state | WA | 0.0869 | 0.0723 | 0.2179 |

## Segment-level prepayment-rate impacts (top 12 largest moves vs Base)

| Scenario | Segment | Value | Base prepay | Scenario prepay | Δ (pp) |
|---|---|---|---|---|---|
| High Prepayment | state | NY | 0.2767 | 0.3367 | 6.00 |
| High Prepayment | state | TX | 0.3549 | 0.4149 | 6.00 |
| High Prepayment | servicer_name | Cedar | 0.2798 | 0.3398 | 6.00 |
| High Prepayment | credit_score_band | <620 | 0.1204 | 0.1804 | 6.00 |
| High Prepayment | credit_score_band | 620-679 | 0.2073 | 0.2673 | 6.00 |
| High Prepayment | credit_score_band | 740+ | 0.3568 | 0.4168 | 6.00 |
| High Prepayment | state | GA | 0.2911 | 0.3511 | 6.00 |
| High Prepayment | state | IL | 0.2467 | 0.3067 | 6.00 |
| High Prepayment | state | WA | 0.2658 | 0.3258 | 6.00 |
| High Prepayment | servicer_name | Beacon | 0.2557 | 0.3157 | 6.00 |
| High Prepayment | vintage_year | 2023 | 0.2654 | 0.3254 | 6.00 |
| High Prepayment | state | CA | 0.2460 | 0.3060 | 6.00 |

See `scenario_drivers.csv` for the full attribution and `scenario_mc.csv` for bootstrap bounds.
## Stress sensitivity by feature cluster

`stress_sensitivity_clusters.csv` splits the stress impact by credit band × LTV band, showing which segments absorb the scenario instead of only the portfolio average.
