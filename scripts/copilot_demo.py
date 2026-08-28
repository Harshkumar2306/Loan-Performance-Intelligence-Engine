"""One-command LIVE copilot governance demo (requires an OpenAI-compatible API key).

Demonstrates the full LLM-assisted loop end-to-end with a real model:
  1. Grounded call  — facts-only context -> advisory reviewer note (accepted).
  2. Ungrounded call — the model is given ONLY a bare score -> its output is rejected
     because it cannot cite record facts, and the grounded note is substituted.

Run:  export OPENAI_API_KEY=...   # or set llm.base_url in config
      python scripts/copilot_demo.py --config config/default.yaml

Every step lands in outputs/llm_audit_demo.jsonl with used_llm: true, so the
governance loop is evidenced by real, timestamped model output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpi_engine.copilot import demonstrate_governance, log_audit, parse_rules

DEMO_RECORD = pd.Series({
    "loan_id": "L000747", "days_past_due": 90, "current_balance": 214_000,
    "current_status": "90 DPD", "anomaly_score": 0.71, "anomaly_reason": "dpd_status_break",
    "next_12m_default_prob": 0.31, "next_3m_delinquency_prob": 0.67,
    "data_quality_score": 61.0, "top_drivers": "days_past_due", "confidence": 0.62,
})


def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(exist_ok=True)
    rules = parse_rules(None)
    dictionary_path = Path(cfg["paths"]["data_dir"]) / cfg["paths"]["dictionary"]
    dictionary = dictionary_path.read_text() if dictionary_path.exists() else ""

    print("Copilot governance demo — live model calls")
    print("=" * 70)
    note, entries = demonstrate_governance(DEMO_RECORD, dictionary, rules, cfg.get("llm"))
    audit_path = out / "llm_audit_demo.jsonl"
    audit_path.write_text("")
    for entry in entries:
        log_audit(entry, audit_path)
        print(f"[{entry['scenario']}] used_llm={entry['used_llm']} -> {entry['decision_status']}")
        print(f"  model: {entry['model']}")
        print(f"  output: {str(entry['output'])[:300]}")
        print("-" * 70)
    (out / "reviewer_note_demo.md").write_text(note)
    if any(e["used_llm"] for e in entries):
        print(f"\nLIVE demonstration complete — audit trail: {audit_path}")
    else:
        print("\nNo API key found (OPENAI_API_KEY / llm.base_url). Entries were recorded as an "
              "offline simulation with used_llm: false. Set the key and re-run for the live demo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    main(parser.parse_args().config)
