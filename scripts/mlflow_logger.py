#!/usr/bin/env python3
"""
Post-training MLflow logger for existing runs.

Behavior:
- If --run-dir is provided → use it
- If --run-dir is omitted → auto-detect latest run in models/runs/mlx/

Usage:
    python scripts/mlflow_logger.py \
        --experiment-name v1_small_sem_norm_patch

    python scripts/mlflow_logger.py \
        --run-dir models/runs/mlx/2026-02-05_21-59-37 \
        --experiment-name v1_small_sem_norm_patch
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import dagshub
import matplotlib.pyplot as plt
import mlflow


# ===============================
# ARGUMENTS
# ===============================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a specific run directory. If omitted, latest run is used.",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("models/runs/mlx"),
        help="Root directory containing run folders (default: models/runs/mlx)",
    )
    p.add_argument("--experiment-name", required=True, type=str)
    return p.parse_args()


# ===============================
# RUN DISCOVERY
# ===============================
def find_latest_run(runs_root: Path) -> Path:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    candidates = [p for p in runs_root.iterdir() if p.is_dir() and p.name != "latest"]
    if not candidates:
        raise RuntimeError(f"No runs found in {runs_root}")

    # Pick newest by mtime (most reliable in practice)
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ===============================
# CORE LOGIC
# ===============================
def log_run_dir_to_mlflow(run_dir: Path, experiment_name: str) -> None:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    # ===============================
    # INIT DAGSHUB + MLFLOW
    # ===============================
    dagshub.init(
        repo_owner="Gautam-Galada",
        repo_name="compression-layer",
        mlflow=True,
    )

    mlflow.set_tracking_uri("https://dagshub.com/Gautam-Galada/compression-layer.mlflow")
    mlflow.set_experiment(experiment_name)

    # ===============================
    # LOAD FILES
    # ===============================
    run_json = run_dir / "run.json"
    train_log = run_dir / "train.log"

    if not run_json.exists():
        raise FileNotFoundError(f"Missing run.json in {run_dir}")
    if not train_log.exists():
        raise FileNotFoundError(f"Missing train.log in {run_dir}")

    with run_json.open("r", encoding="utf-8") as f:
        run_cfg = json.load(f)

    log_text = train_log.read_text(encoding="utf-8", errors="replace")

    # ===============================
    # REGEX PARSERS
    # ===============================
    train_pat = re.compile(
        r"Iter (\d+): Train loss ([0-9.]+).*Tokens/sec ([0-9.]+).*Peak mem ([0-9.]+) GB"
    )
    val_pat = re.compile(r"Iter (\d+): Val loss ([0-9.]+)")

    train_steps, train_loss, tokens_sec, peak_mem = [], [], [], []
    for m in train_pat.finditer(log_text):
        train_steps.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        tokens_sec.append(float(m.group(3)))
        peak_mem.append(float(m.group(4)))

    val_steps, val_loss = [], []
    for m in val_pat.finditer(log_text):
        val_steps.append(int(m.group(1)))
        val_loss.append(float(m.group(2)))

    # ===============================
    # MLFLOW RUN
    # ===============================
    run_name = run_cfg.get("started_at", run_dir.name)

    with mlflow.start_run(run_name=run_name):
        # ---------------------------
        # PARAMS
        # ---------------------------
        mlflow.log_params(
            {
                "model": run_cfg.get("model"),
                "git_sha": run_cfg.get("git_sha"),
                "data_dir": run_cfg.get("data_dir"),
                "lora_rank": run_cfg.get("lora_rank"),
                "lora_alpha": run_cfg.get("lora_alpha"),
                "batch_size": run_cfg.get("batch_size"),
                "learning_rate": run_cfg.get("learning_rate"),
                "iters": run_cfg.get("iters"),
            }
        )

        # ---------------------------
        # METRICS
        # ---------------------------
        for i, step in enumerate(train_steps):
            mlflow.log_metric("train_loss", train_loss[i], step=step)
            mlflow.log_metric("tokens_per_sec", tokens_sec[i], step=step)
            mlflow.log_metric("peak_mem_gb", peak_mem[i], step=step)

        for i, step in enumerate(val_steps):
            mlflow.log_metric("val_loss", val_loss[i], step=step)

        # ---------------------------
        # ARTIFACTS
        # ---------------------------
        mlflow.log_artifact(run_json)
        mlflow.log_artifact(train_log)

        adapter_dir = run_dir / "adapter"
        if adapter_dir.exists() and adapter_dir.is_dir():
            mlflow.log_artifacts(adapter_dir, artifact_path="weights")

        # ---------------------------
        # PLOTS
        # ---------------------------
        if train_steps and val_steps:
            plt.figure()
            plt.plot(train_steps, train_loss, label="Train Loss")
            plt.plot(val_steps, val_loss, label="Val Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Training vs Validation Loss")

            plot_path = run_dir / "loss_curve.png"
            plt.savefig(plot_path)
            plt.close()

            mlflow.log_artifact(plot_path)


# ===============================
# CLI ENTRYPOINT
# ===============================
def main() -> None:
    args = parse_args()

    run_dir: Path | None = args.run_dir
    if run_dir is None:
        run_dir = find_latest_run(args.runs_root)
        print(f"📌 Auto-selected latest run: {run_dir}")

    log_run_dir_to_mlflow(run_dir, args.experiment_name)
    print("✅ Existing training run successfully logged to MLflow")


if __name__ == "__main__":
    main()
