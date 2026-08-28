"""Agentic experiment runner: small hyperparameter search over the production candidate.

Each trial trains on the same out-of-time fit period, scores the same contiguous
validation period, and appends one JSONL entry to `outputs/experiments_log.jsonl`
(via `lpi_engine.tracking.log_trial`) — the evidence trail behind the final model
choices in the AI Development Log. Human review of the log is the control loop.

Run:  python scripts/run_experiments.py --config config/default.yaml --trials 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpi_engine.data import engineer_panel, feature_columns, merge_attributes, normalise, read_pack, time_split
from lpi_engine.modeling import fit_binary
from lpi_engine.tracking import log_trial


def main(config_path: str, trials: int):
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(exist_ok=True)
    pack = read_pack(cfg["paths"])
    train = normalise(pack["train"])
    if "static" in pack:
        train = merge_attributes(train, normalise(pack["static"]))
    train, _ = engineer_panel(train, None)
    fit, valid, split = time_split(train, cfg["split"]["validation_months"])
    features = feature_columns(train)

    target = "next_12m_default_flag"
    tr = fit.dropna(subset=[target])
    va = valid.dropna(subset=[target])
    ytr, yva = tr[target].astype(int), va[target].astype(int)

    rng = np.random.default_rng(cfg["seed"])
    print(f"Experiment runner: {trials} trials on {target} "
          f"({len(tr):,} fit rows / {len(va):,} validation rows)")

    for i in range(trials):
        trial_cfg = dict(cfg["models"])
        trial_cfg["max_iter"] = int(rng.choice([100, 160, 240, 320]))
        trial_cfg["max_leaf_nodes"] = int(rng.choice([16, 24, 31, 48]))
        trial_cfg["min_samples_leaf"] = int(rng.choice([20, 40, 80]))
        _, metrics, _ = fit_binary(tr[features], ytr, va[features], yva, cfg["seed"], trial_cfg)
        roc_auc = metrics["calibrated_hgb"].get("roc_auc")
        trial = {
            "kind": "experiment_trial",
            "trial": i + 1,
            "target": target,
            "params": {k: trial_cfg[k] for k in ("max_iter", "max_leaf_nodes", "min_samples_leaf")},
            "roc_auc": roc_auc,
            "pr_auc": metrics["calibrated_hgb"].get("pr_auc"),
            "brier": metrics["calibrated_hgb"].get("brier"),
            "ece": metrics["calibration"].get("ece"),
        }
        log_trial(trial, out / "experiments_log.jsonl")
        print(f"  trial {i + 1:>2}: {trial['params']} -> AUC {roc_auc}")

    print(f"Trial log: {out / 'experiments_log.jsonl'} — review each trial before adopting parameters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--trials", type=int, default=8)
    main(parser.parse_args().config, parser.parse_args().trials)
