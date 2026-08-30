RECOMMENDATION ONLY — human review required.

**Executive Summary:**
Loan L000747 is in a critical 90-day delinquency status with a substantial outstanding balance of $214,000, presenting a high likelihood of near-term default and significant data integrity concerns that require immediate manual verification.

**Key Findings & Evidence:**
*   **Severe Delinquency Status:** The account is currently 90 days past due, which aligns with the "90 DPD" status label and triggers the validation rule that any positive days past due implies a delinquent state.
*   **High Financial Exposure:** The current unpaid principal balance stands at $214,000, representing a significant loss exposure if the loan proceeds to default.
*   **Anomaly Detection Trigger:** The system flagged an anomaly score of 0.71 due to a "dpd_status_break," indicating a potential inconsistency or irregularity in how the delinquency status is being recorded or transitioning.
*   **Data Quality Concerns:** The record has a low data quality score of 61.0, suggesting that some underlying data points may be incomplete, inconsistent, or unreliable, which complicates automated risk assessment.
*   **Primary Risk Driver:** The "days_past_due" metric is identified as the top driver for the current risk profile, emphasizing the severity of the payment default.

**Risk & Actuarial Context:**
*   **Default Probability:** The model estimates a 31% probability of default within the next 12 months.
*   **Delinquency Probability:** There is a 67% probability of continued or worsened delinquency within the next 3 months.
*   **Confidence Level:** The model’s confidence in these predictions is moderate at 0.62, largely due to the lower data quality score, which introduces uncertainty into the actuarial projections.

**Recommended Action:**
*   **Immediate Manual Review:** A senior loan officer should immediately review the account to verify the accuracy of the 90-day delinquency status and the $214,000 balance, given the low data quality score.
*   **Investigate Anomaly:** Investigate the "dpd_status_break" anomaly to determine if there is a data entry error, a system glitch, or a genuine irregularity in the borrower’s payment history.
*   **Contact Borrower:** Initiate contact with the borrower to assess their current financial situation, intent to pay, and any potential reasons for the prolonged delinquency.
*   **Escalation Decision:** Based on the findings, decide whether to escalate the account for collection efforts, modify the loan terms, or proceed with foreclosure procedures, keeping in mind the high probability of near-term delinquency.