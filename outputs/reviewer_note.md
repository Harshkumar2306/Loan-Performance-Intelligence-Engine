RECOMMENDATION ONLY — human review required. Loan L000080 was prioritised because: negative_balance; outlier_remaining_term_months; outlier_balance_ratio. Projected 12-month default probability=0.0005; data-quality score=47.7; model confidence=0.819. Verify the source records, status and supporting documents before deciding whether to reconcile, monitor or escalate.

Field definitions cited:
- `loan_id`: unique loan identifier.
- `days_past_due`: days past due at the reporting month.
- `current_balance`: unpaid principal at the reporting month.
- `current_status`: performing status or delinquency stage.

Applicable validation rules:
- current_balance >= 0
