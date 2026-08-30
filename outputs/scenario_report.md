# Scenario & Stress Report

*Generated 2026-08-30 by `run_pipeline.py`.*

Scenarios are transparent feature shocks followed by the same fitted models, so every
number is auditable. Segment rows live in `scenario_report.csv`.

## Scenario-level projected default and prepayment rates (bootstrap 5–95 bounds)

| Scenario | Mean default | Default 5th | Default 95th | Mean prepay | Prepay 5th | Prepay 95th | Sims |
|---|---|---|---|---|---|---|---|
| Base | 0.08326 | 0.06368 | 0.10574 | 0.24338 | 0.21985 | 0.26954 | 30 |
| Adverse Credit | 0.08963 | 0.07246 | 0.10841 | 0.20225 | 0.18661 | 0.22439 | 30 |
| High Prepayment | 0.08314 | 0.06516 | 0.10564 | 0.30459 | 0.2744 | 0.33672 | 30 |

## Driver attribution (one-feature-at-a-time counterfactuals)

| Scenario | Δ default (pp) | Credit shock (pp) | Prepay-uplift effect on default (pp) | Δ prepayment (pp) | Uplift applied (pp) |
|---|---|---|---|---|---|
| Adverse Credit | 0.3863 | 0.3863 | 0.0 | -3.9194 | -1.0 |
| High Prepayment | 0.0 | 0.0 | 0.0 | 5.9916 | 6.0 |

## Segment-level default-rate impacts (top 12 largest moves vs Base)

| Scenario | Segment | Value | Default rate | Delinquency | Prepayment |
|---|---|---|---|---|---|
| Adverse Credit | credit_score_band | <620 | 0.1772 | 0.1259 | 0.1091 |
| Adverse Credit | state | FL | 0.1119 | 0.0830 | 0.1356 |
| Adverse Credit | state | GA | 0.0852 | 0.0773 | 0.2655 |
| Adverse Credit | servicer_name | Atlas | 0.0903 | 0.0787 | 0.2035 |
| Adverse Credit | state | WA | 0.0700 | 0.0736 | 0.2180 |
| Adverse Credit | state | TX | 0.1355 | 0.0927 | 0.2182 |
| Adverse Credit | credit_score_band | 680-739 | 0.0736 | 0.0767 | 0.1932 |
| Adverse Credit | vintage_year | 2023 | 0.0900 | 0.0798 | 0.2065 |
| Adverse Credit | servicer_name | Beacon | 0.0942 | 0.0753 | 0.1990 |
| Adverse Credit | state | IL | 0.1191 | 0.0981 | 0.2068 |
| Adverse Credit | credit_score_band | 620-679 | 0.1181 | 0.0901 | 0.1391 |
| Adverse Credit | servicer_name | Cedar | 0.0854 | 0.0857 | 0.2173 |

## Segment-level prepayment-rate impacts (top 12 largest moves vs Base)

| Scenario | Segment | Value | Base prepay | Scenario prepay | Δ (pp) |
|---|---|---|---|---|---|
| High Prepayment | credit_score_band | <620 | 0.1466 | 0.2066 | 6.00 |
| High Prepayment | state | CA | 0.2597 | 0.3197 | 6.00 |
| High Prepayment | state | FL | 0.1693 | 0.2293 | 6.00 |
| High Prepayment | state | IL | 0.2394 | 0.2994 | 6.00 |
| High Prepayment | credit_score_band | 620-679 | 0.1678 | 0.2278 | 6.00 |
| High Prepayment | servicer_name | Beacon | 0.2394 | 0.2993 | 6.00 |
| High Prepayment | state | NY | 0.2283 | 0.2882 | 5.99 |
| High Prepayment | vintage_year | 2023 | 0.2457 | 0.3056 | 5.99 |
| High Prepayment | servicer_name | Cedar | 0.2610 | 0.3210 | 5.99 |
| High Prepayment | state | WA | 0.2672 | 0.3271 | 5.99 |
| High Prepayment | credit_score_band | 680-739 | 0.2470 | 0.3069 | 5.99 |
| High Prepayment | servicer_name | Atlas | 0.2365 | 0.2964 | 5.99 |

See `scenario_drivers.csv` for the full attribution and `scenario_mc.csv` for bootstrap bounds.
## Stress sensitivity by feature cluster

`stress_sensitivity_clusters.csv` splits the stress impact by credit band × LTV band, showing which segments absorb the scenario instead of only the portfolio average.
