RECOMMENDATION ONLY — human review required.

**Executive Summary:**
Loan L000747 is currently in a severe 90-day delinquency status with an outstanding balance of $214,000, presenting a high probability of near-term default and significant data quality concerns that require immediate manual verification.

**Key Findings & Evidence:**
*   **Severe Delinquency:** The account is 90 days past due, which aligns with the "90 DPD" status and triggers the validation rule that positive days past due imply a delinquent state.
*   **Anomaly Detection:** The system flagged an anomaly score of 0.71 due to a "dpd_status_break," suggesting a potential inconsistency or irregularity in how the delinquency status was recorded or transitioned.
*   **Data Quality Concerns:** The record has a low data quality score of 61.0, indicating that the underlying information may be incomplete, inconsistent, or unreliable, which complicates automated risk assessment.
*   **Primary Driver:** The main factor driving the risk profile is the extended period of non-payment (days past due).

**Risk & Actuarial Context:**
*   **Default Probability:** The model estimates a 31% probability of default within the next 12 months.
*   **Delinquency Probability:** There is a 67% probability of continued or worsened delinquency within the next 3 months.
*   **Model Confidence:** The confidence level for these predictions is moderate at 0.62, further necessitating human oversight due to the lower data quality score.

**Recommended Action:**
*   **Immediate Manual Review:** A loan officer should immediately review the account to verify the accuracy of the 90-day delinquency status and investigate the cause of the "dpd_status_break" anomaly.
*   **Data Validation:** Prioritize cleaning and validating the data associated with this loan to improve the data quality score and ensure future risk models have accurate inputs.
*   **Collection Strategy:** Given the high 3-month delinquency probability, initiate or escalate collection efforts in accordance with the bank's policies for 90+ day delinquent accounts.
*   **Documentation:** Document all findings and actions taken in the loan file to maintain an audit trail and support potential loss mitigation or charge-off decisions.