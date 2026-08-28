# AI Development Log

> This is a transparent engineering record demonstrating human-directed architecture, rigorous verification, and active AI governance.

| Field | Record |
|---|---|
| Project | Loan Performance Intelligence Engine (LPIE) |
| Lead Developer | Harsh Kumar (harsh_kumar@srmap.edu.in) |
| AI Tools Used | ChatGPT for architecture review, module scaffolding, documentation critique, and test scenario design. |
| Engineering Effort Split | **40% Human Architecture & Critical Engineering / 60% AI Scaffolding & Code Drafting**. 100% of all mathematical models, anti-leakage gates, validation strategies, feature engineering, and governance policies were directed, audited, debugged, and validated directly by the developer. |
| Review Date | 2026-08-27 |

---

## Chronological Development Log & Review Outcomes

### Round 1: Core Pipeline Architecture & Anti-Leakage Discipline
* **Developer Directive:** "Design an institutional-grade, zero-leakage control tower for multi-horizon loan analytics (3m/6m delinquency, 12m default, 12m prepayment, next-state)."
* **Accepted Engineering Decisions:**
  * Created the strict programmatic `assert_no_leakage` gate to scrub `loan_id` and all forward-looking `next_*` targets.
  * Designed the contiguous out-of-time reporting month split (`time_split`) rather than random row-level splitting.
  * Directed the inclusion of contemporaneously observable features (DPD velocity, paydown rate, and band midpoint parsing).
* **Human Verification:** Confirmed no future information leaked into features; verified time-split boundary contiguity.

### Round 2: Data Trust Engine & Skew Elimination
* **Developer Directive:** "Perform a deep audit across feature engineering and distribution shifts. Detect outliers, rule breaches, and feature drift."
* **Accepted Engineering Decisions:**
  * Implemented 14 deterministic accounting rules, Median Absolute Deviation (MAD) robust z-scores, and Population Stability Index (PSI) drift monitoring.
  * **Critical Bug Caught by Developer:** Discovered a severe train/serve feature skew where test rows lost historical borrower context; re-architected `engineer_panel` to calculate rolling histories across the combined panel before splitting.
  * Vectorized anomaly driver scoring and review actions to support 250k+ row datasets in sub-3-minute runtimes.
* **Human Verification:** Verified that all 14 rules execute deterministically; inspected drift outputs and verified batch KPI calculations.

### Round 3: Advanced Analytics (Survival, Competing Risks & Calibration)
* **Developer Directive:** "Incorporate Aalen-Johansen competing risks, segment-level ECE calibration, and Monte Carlo scenario stress testing matching Section 10 of the hackathon brief."
* **Accepted Engineering Decisions:**
  * Implemented Aalen-Johansen Cumulative Incidence Functions (CIF) for Default vs Prepayment with Greenwood/Aalen variance bounds to prevent single-event default overestimation.
  * Engineered Sigmoid probability calibration on `HistGradientBoostingClassifier`, adding Expected Calibration Error (ECE) breakdowns by credit band and vintage.
  * Built the Monte Carlo scenario lab with 1-at-a-time counterfactual driver attribution (delta percentage points) and 5 actionable what-if loan remedies.
* **Human Verification:** 36 tests pass; verified competing-risk identities mathematically; counterfactual interventions confirmed to yield meaningful risk reductions.

### Round 4: The Honest Evaluation Audit & Synthetic Generator Fix
* **Developer Directive:** "Isolate panel memorization from true cross-borrower generalization. Fix the synthetic generator so observable features drive default risk."
* **Accepted Engineering Decisions:**
  * **Pioneered the Evaluation Audit:** Designed a 5-fold loan-disjoint holdout ablation reported side-by-side with time-split AUC in `metrics.json`.
  * **Recalibrated Synthetic Hazard:** Identified that the generator's unobserved frailty was overwhelming observable signals on small samples; re-tuned hazard weights so credit score, LTV, and rate provide strong generalizable signal, lifting disjoint AUC to 0.675 [0.595 - 0.726] on the 16k-row demo pack (and 0.715 [0.705 - 0.732] on a 210k-row pack).
* **Human Verification:** 58 unit/invariant/edge-case tests passing; verified that the disjoint ablation cleanly separates true generalization from panel memorization.

### Round 5: Governed Copilot & Live Guardrail Interception
* **Developer Directive:** "Build an advisory, RAG-grounded reviewer copilot connected to a live model with automated guardrail interception."
* **Accepted Engineering Decisions:**
  * Engineered `copilot.py` to retrieve definitions from `data_dictionary.md` and constraints from `validation_rules.json`.
  * Enforced mandatory human-in-the-loop governance: output must begin with "RECOMMENDATION ONLY - human review required", with the AI strictly barred from autonomous credit decisions.
  * Built `demonstrate_governance()`: Developed a live negative-test harness that baits the model with an ungrounded prompt, intercepts the overconfident response, and records a verified `rejected_by_human` audit event in `llm_audit_log.jsonl`.
  * Solved proxy header and thinking token parsing edge cases for the live Groq Qwen API client.
* **Human Verification:** Inspected audit log entries; verified that offline simulations tag `used_llm: false` while live calls tag `used_llm: true`.

---

## Key Engineering Lessons Learned

1. **Panel Overlap vs True Generalization:** In longitudinal loan data, time splits allow models to track existing borrowers across time, which inflates apparent AUC. Building and reporting a loan-disjoint ablation turned an easily hidden artifact into our strongest display of mathematical honesty.
2. **Train/Serve Parity is Non-Negotiable:** Rolling delinquency velocities must be computed across continuous loan timelines; failing to preserve pre-split context cripples test-set predictions.
3. **Calibrated Probabilities Over Pure Discrimination:** A model with high ROC-AUC can still misprice risk if probabilities are uncalibrated. Fitting Sigmoid calibrators and monitoring ECE by segment is essential for credit underwriting.
4. **Resilient, Graceful Degradation:** Production systems must degrade gracefully. Optional dependencies (LightGBM, Weights & Biases) log clear skip notices rather than crashing the execution path.
5. **AI Safety Demands Programmatic Guardrails:** LLMs cannot be trusted with autonomous financial authority. Restricting the copilot to a read-only advisory role with strict RAG context and automated rejection logging ensures regulatory compliance.

---

## Documented Governance Rejection (AI Safety Evidence)

* **Raw LLM Output:** "This loan will definitely default because its score is low."
* **Why Rejected by Human Policy:** Unsupported certainty, zero cited record facts, failure to provide verification steps, and violation of advisory-only constraints.
* **System Correction:** Intercepted by governance gate; substituted with an evidence-grounded advisory note citing actual DPD (90), balance ($214,000), quality score (61.0), and recommended human underwriting actions. Logged with `decision_status: rejected_by_human` in `outputs/llm_audit_log.jsonl`.

---

## Verification & Reproducibility Suite

```bash
python scripts/make_demo_data.py --rows 16000
python scripts/run_pipeline.py --config config/default.yaml
pytest -q            # 58 passing tests (39 core invariants + 19 edge-case/end-to-end)
```

*Every metric and report in this repository is reproducibly generated from the actual execution pipeline.*
