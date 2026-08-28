# Five-minute demo script

**0:00-0:15 - The problem:** "Loan analytics fails if data cannot be trusted. LPIE is ML-first: it decides what needs a reviewer, not what an LLM should guess. Let's run the engine live." (Run `python scripts/run_pipeline.py` in your terminal).

**0:15-1:00 - Execution & The Control Tower:** While the pipeline runs (~50s), explain that it's applying 14 rules, fitting multi-horizon models, and running scenarios. Once it finishes, it automatically opens `outputs/dashboard.html` in your browser. "This is our zero-dependency offline control tower." Show the **Data Quality & Drift** section.

**1:00-1:50 - Predict & Calibrate:** Scroll to the **Task 2 • Out-of-Time Model Validation Performance** section on the dashboard. Show the baseline RF vs calibrated HGB (and LightGBM) ROC-AUC, PR-AUC, Brier and ECE. Point out the operating points and recall-at-fixed-precision in `outputs/model_card.md`.

**1:50-2:25 - Time matters:** Scroll to the **Task 3 • Competing-Risk Survival Curves** section on the dashboard. Explain that final monthly observations are right-censored. Show the Kaplan-Meier / Aalen-Johansen curves with confidence bands vs the naive baseline.

**2:25-3:15 - Review queue:** Scroll to the **Task 4 • Prioritised Anomaly & Exception Reviewer Queue**. Narrate the anomaly score, named rule breaches, deviating features, default probability, and the recommended action. "This is an *escalation*, not an automated decision."

**3:15-4:00 - Stress:** Scroll to the **Task 5 • Macroeconomic Scenario Stress Simulation** chart. Compare Base vs Adverse Credit vs High Prepayment for BOTH projected default and prepayment rates, and note the Monte Carlo 5-95 bounds.

**4:00-4:35 - Copilot governance:** Scroll to the **Task 7 • Grounded LLM Reviewer Copilot Analysis**. Highlight the advisory prefix ("RECOMMENDATION ONLY"), grounded facts, and cited dictionary rules. Note that the full AI interaction is recorded in `llm_audit_log.jsonl`.

**4:35-5:00 - Evaluation honesty:** Open `metrics.json → evaluation_audit` (or point to the dashboard audit table): "The time-split AUC is an upper bound; the loan-disjoint ablation shows how much is memorisation - we report both." Close: "A reproducible, offline control tower delivering prediction, evidence, quality, and controls."
