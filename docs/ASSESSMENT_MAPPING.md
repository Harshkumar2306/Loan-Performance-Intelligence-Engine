# Assessment coverage map

| Judging criterion | Points | Evidence in LPIE |
|---|---|---|
| Data Intelligence & Profiling | 15 | `profiling.py`: 14 deterministic rules, robust-z outliers, numeric **and categorical PSI drift**, Cramér's V associations, missingness patterns, source reconciliation, documented row quality score, per-month batch KPI (`batch_quality_kpi.csv`), human-readable `data_intelligence_report.md`. |
| Predictive Modelling | 20 | `modeling.py`: strict contiguous time split, baseline RF vs calibrated HGB vs optional LightGBM, class-imbalance handling, ROC-AUC/PR-AUC/F1/precision/recall/Brier, **recall-at-fixed-precision**, ECE reliability tables, per-month stability metrics, majority baselines for multiclass. |
| Time-to-Event / Transition | 15 | `intelligence.py`: observed transition matrix, Markov projection, **Kaplan-Meier + Aalen-Johansen competing-risk curves with Greenwood/Aalen-variance CIs**, explicit censoring treatment, naive-baseline comparison (`km_curves.csv`). |
| Anomaly & Exception Intelligence | 10 | Rule + Isolation Forest hybrid with named drivers; supervised `exception_required` and multiclass `exception_type` models; 50-row reviewer queue; validation-side anomalies with true labels. |
| Scenario & Stress Simulation | 10 | Base / Adverse Credit / High Prepayment feature shocks + rescoring; segment impacts (credit band, state, servicer, vintage); one-feature-at-a-time attribution; **Monte Carlo bootstrap 5-95 bounds**. |
| Explainability & Responsible AI | 10 | Global permutation importance, TreeSHAP local explanations (labelled fallback), **FP/FN error analysis with segment concentration**, **segment parity (fairness) monitoring**, documented confidence blend, generated model card, evaluation audit. |
| Smart LLM Usage | 10 | `copilot.py`: grounded dictionary/rule retrieval, facts-only context, optional OpenAI-compatible call, mandatory advisory prefix, JSONL audit log, human-decision status, documented rejected-output example. ML is never replaced by the LLM. |
| ML Engineering & Reproducibility | 5 | Typed modular package, versioned config, one-command pipeline, demo generator, 58-test suite, clean CSV/JSON artefacts, per-run JSONL experiment log (`runs_log.jsonl`), **self-contained monitoring dashboard** (`dashboard.html`), train/serve-consistent feature engineering. |
| Agentic Coding Evidence | 5 | `docs/AI_DEVELOPMENT_LOG.md` with representative prompts, accepted/rejected outputs, human review process and validation commands. |

## Evaluation honesty (why this beats a "big AUC")

A loan that spans the out-of-time boundary appears in both fit and validation. Any flexible
model can then partially memorise loan identity and inflate the time-split AUC. LPIE reports
a **loan-disjoint ablation** in `metrics.json → evaluation_audit` next to the headline metrics,
so judges see both the upper bound (time-split) and the memorisation-free estimate
(disjoint loans). The model card documents this as a known failure mode with its mitigation.

**Empirical evidence:** the K-fold loan-disjoint ablation reports mean [min-max] across five
deterministic loan partitions - default 0.675 [0.595-0.726], delinquency 0.585 [0.570-0.617] vs
time-split 0.9999 / 0.8008 on the 16k-row demo pack. The disjoint estimates sit clearly above the
0.5 coin-toss line, so the audit demonstrates real cross-loan signal rather than merely exposing the
memorisation bonus. On a 210k-row pack the pipeline completes in ~5 minutes with a narrowed gap
(default 0.715 [0.705-0.732] vs time-split 0.935).

## Covered advanced features (from the brief - all 15 bullets)

1. Competing-risk survival curves · 2. Monte Carlo portfolio simulation ·
3. drift monitoring dashboard (`dashboard.html`) · 4. segment-level scenario curves ·
5. **model calibration by vintage / credit band** (`calibration_by_segment.csv`) ·
6. **W&B experiment tracking** (`runs_log.jsonl` + optional `tracking.enabled`) ·
7. RAG-style grounding over the data dictionary and validation rules (copilot citations) ·
8. agentic experiment runner (`scripts/run_experiments.py`) ·
9. automated feature-store style pipeline (deterministic, leakage-gated feature engineering) ·
10. bias / fairness monitoring (`fairness_report.csv`) ·
11. counterfactual explanations (`counterfactuals.csv`) ·
12. stress sensitivity by feature cluster (`stress_sensitivity_clusters.csv`) ·
13. model confidence intervals (bootstrap CIs on validation metrics) ·
14. human-in-the-loop review queue with a mandatory human-decision field ·
15. synthetic-data stress testing (realistic pack generator + scenario lab).

## Remaining steps once the organiser data pack arrives

1. Align exact field names/types and the submission template; commit the schema contract.
2. Run the pipeline on the real pack and let `outputs/model_card.md` pick up the real metrics.
3. Tune `review.*` thresholds and the precision target against the stated reviewer capacity.
4. Manually audit 20 queue entries; record accepted/rejected recommendations in the audit log.
5. Record the five-minute screen capture following `DEMO_SCRIPT.md`.
