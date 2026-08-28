"""Experiment tracking: an always-on JSONL runs log plus optional Weights & Biases logging.

The brief lists "MLflow or Weights & Biases experiment tracking" as an advanced feature.
LPIE ships both layers:
* `runs_log.jsonl` — zero-dependency, always written, one JSONL entry per run/trial.
* W&B — logged when `tracking.enabled: true` and `wandb` is installed and authenticated;
  skipped gracefully (with the reason recorded) otherwise, so reproducibility never
  depends on a third-party account.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

RUN_LOG_FIELDS = [
    "timestamp_utc", "kind", "config", "seed", "train_rows", "test_rows", "n_features",
    "split_cutoff", "metrics", "evaluation_audit", "note", "tracking",
]


def _flatten_metrics(metrics: dict, prefix: str = "") -> dict:
    """Flatten nested metric dicts into dotted keys suitable for a tracking backend."""
    flat = {}
    for key, value in metrics.items():
        full_key = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, full_key + "/"))
        elif isinstance(value, (int, float, str, bool)) or value is None:
            flat[full_key] = value
        else:
            flat[full_key] = json.dumps(value, default=str)
    return flat


def log_run(run: dict, runs_path: Path, tracking_cfg: dict | None = None) -> dict:
    """Append one entry to the JSONL runs log, then optionally push it to W&B.

    Returns a small dict describing what was logged (and why W&B was skipped, if it was).
    """
    entry = {k: run.get(k) for k in RUN_LOG_FIELDS}
    # `run.get` yields None for missing keys, so a plain setdefault would leave the
    # timestamp as null — every JSONL line must carry a real UTC timestamp.
    if not entry.get("timestamp_utc"):
        entry["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    entry["tracking"] = {"jsonl": "logged"}
    result = {"jsonl": True, "wandb": "not_configured"}

    wandb_enabled = bool(tracking_cfg and tracking_cfg.get("enabled"))
    if wandb_enabled:
        try:
            import wandb  # type: ignore

            run_name = tracking_cfg.get("name") or f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
            wandb.init(project=tracking_cfg.get("project", "lpi-engine"),
                       entity=tracking_cfg.get("entity") or None,
                       name=run_name,
                       config=_flatten_metrics({k: v for k, v in run.items() if k != "metrics"}) or None,
                       reinit=True)
            wandb.log(_flatten_metrics(run.get("metrics", {})))
            wandb.finish()
            entry["tracking"]["wandb"] = {"status": "logged", "project": tracking_cfg.get("project"),
                                          "name": run_name}
            result["wandb"] = "logged"
        except ImportError:
            entry["tracking"]["wandb"] = {"status": "skipped", "reason": "wandb not installed"}
            result["wandb"] = "skipped: wandb not installed"
        except Exception as exc:  # auth failure, offline mode, etc. — never break the run
            entry["tracking"]["wandb"] = {"status": "skipped", "reason": type(exc).__name__}
            result["wandb"] = f"skipped: {type(exc).__name__}"

    with runs_path.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return result


def log_trial(trial: dict, trials_path: Path) -> None:
    """Append one hyperparameter trial to the experiment log."""
    trial.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    with trials_path.open("a") as f:
        f.write(json.dumps(trial, default=str) + "\n")
