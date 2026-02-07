# tests/test_mlflow_logger.py
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest


class MlflowRecorder:
    def __init__(self):
        self.tracking_uri = None
        self.experiment = None
        self.run_name = None
        self.params = {}
        self.metrics = []  # (key, value, step)
        self.artifacts = []  # (path, artifact_path)
        self.log_artifacts_calls = []  # (dir_path, artifact_path)

    def set_tracking_uri(self, uri: str):
        self.tracking_uri = uri

    def set_experiment(self, name: str):
        self.experiment = name

    @contextmanager
    def start_run(self, run_name: str = None, **_kwargs):
        self.run_name = run_name
        yield

    def log_params(self, d: dict):
        self.params.update(d)

    def log_metric(self, key: str, value: float, step: int | None = None):
        self.metrics.append((key, value, step))

    def log_artifact(self, path: Path, artifact_path: str | None = None):
        self.artifacts.append((Path(path), artifact_path))

    def log_artifacts(self, dir_path: Path, artifact_path: str | None = None):
        self.log_artifacts_calls.append((Path(dir_path), artifact_path))


def _write_run_dir(
    base: Path,
    name: str,
    *,
    with_adapter: bool = True,
    with_matching_logs: bool = True,
) -> Path:
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "started_at": name,
                "model": "mistral",
                "git_sha": "abc123",
                "data_dir": "data/x",
                "lora_rank": 8,
                "lora_alpha": 16,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "iters": 100,
            }
        ),
        encoding="utf-8",
    )

    if with_matching_logs:
        # Must match your regex:
        # Iter (\d+): Train loss ([0-9.]+).*Tokens/sec ([0-9.]+).*Peak mem ([0-9.]+) GB
        # Iter (\d+): Val loss ([0-9.]+)
        (run_dir / "train.log").write_text(
            "\n".join(
                [
                    "Iter 10: Train loss 1.23 | Tokens/sec 456.7 | Peak mem 12.3 GB",
                    "Iter 20: Train loss 1.10 | Tokens/sec 470.0 | Peak mem 12.4 GB",
                    "Iter 10: Val loss 1.50",
                    "Iter 20: Val loss 1.40",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        (run_dir / "train.log").write_text("no matching lines\n", encoding="utf-8")

    if with_adapter:
        adapter = run_dir / "adapter"
        adapter.mkdir()
        (adapter / "adapter.safetensors").write_bytes(b"fake")
        (adapter / "config.json").write_text("{}", encoding="utf-8")

    return run_dir


@pytest.fixture
def mlflow_recorder(monkeypatch) -> MlflowRecorder:
    """
    Patch scripts/mlflow_logger.py's `mlflow` and `dagshub.init` so tests never touch
    any real tracking server (DagsHub / MLflow).
    """
    rec = MlflowRecorder()

    # IMPORTANT: import using the package-style path that matches your repo layout.
    # This assumes scripts/ is a python package or is importable via PYTHONPATH.
    #
    # If scripts/ is NOT a package, run tests with:
    #   PYTHONPATH=. pytest
    import scripts.mlflow_logger as m

    # Patch symbols inside the module under test
    monkeypatch.setattr(m, "mlflow", rec, raising=True)
    monkeypatch.setattr(m.dagshub, "init", lambda **kwargs: None, raising=True)

    # Ensure non-interactive matplotlib backend
    monkeypatch.setenv("MPLBACKEND", "Agg")

    return rec


def test_logs_params_metrics_and_artifacts(
    tmp_path: Path, mlflow_recorder: MlflowRecorder, monkeypatch
):
    import scripts.mlflow_logger as m

    runs_root = tmp_path / "runs" / "mlx"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_dir = _write_run_dir(runs_root, "2026-02-05_21-59-37", with_adapter=True)

    m.log_run_dir_to_mlflow(run_dir=run_dir, experiment_name="exp_test")

    assert mlflow_recorder.experiment == "exp_test"
    assert mlflow_recorder.tracking_uri is not None

    assert mlflow_recorder.params["model"] == "mistral"
    assert mlflow_recorder.params["lora_rank"] == 8
    assert mlflow_recorder.params["iters"] == 100

    # 2 train steps * 3 train metrics = 6 + 2 val metrics = 8
    assert len(mlflow_recorder.metrics) == 8
    assert ("train_loss", 1.23, 10) in mlflow_recorder.metrics
    assert ("tokens_per_sec", 470.0, 20) in mlflow_recorder.metrics
    assert ("val_loss", 1.40, 20) in mlflow_recorder.metrics

    artifact_names = {p.name for (p, ap) in mlflow_recorder.artifacts}
    assert "run.json" in artifact_names
    assert "train.log" in artifact_names
    assert "loss_curve.png" in artifact_names

    assert mlflow_recorder.log_artifacts_calls == [(run_dir / "adapter", "weights")]


def test_no_adapter_dir_does_not_log_weights(tmp_path: Path, mlflow_recorder: MlflowRecorder):
    import scripts.mlflow_logger as m

    runs_root = tmp_path / "runs" / "mlx"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_dir = _write_run_dir(runs_root, "2026-02-05_21-59-37", with_adapter=False)

    m.log_run_dir_to_mlflow(run_dir=run_dir, experiment_name="exp_test")
    assert mlflow_recorder.log_artifacts_calls == []


def test_nonmatching_log_produces_no_loss_plot(tmp_path: Path, mlflow_recorder: MlflowRecorder):
    import scripts.mlflow_logger as m

    runs_root = tmp_path / "runs" / "mlx"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_dir = _write_run_dir(
        runs_root,
        "2026-02-05_21-59-37",
        with_adapter=False,
        with_matching_logs=False,
    )

    m.log_run_dir_to_mlflow(run_dir=run_dir, experiment_name="exp_test")

    artifact_names = {p.name for (p, ap) in mlflow_recorder.artifacts}
    assert "run.json" in artifact_names
    assert "train.log" in artifact_names
    assert "loss_curve.png" not in artifact_names


def test_find_latest_run_ignores_latest_and_uses_mtime(
    tmp_path: Path, mlflow_recorder: MlflowRecorder
):
    """
    Validates the new behavior: find_latest_run() ignores 'latest' and picks by mtime.
    """
    import scripts.mlflow_logger as m

    runs_root = tmp_path / "runs" / "mlx"
    runs_root.mkdir(parents=True, exist_ok=True)

    # Create two runs
    older = _write_run_dir(runs_root, "2026-02-04_00-39-58", with_adapter=False)
    newer = _write_run_dir(runs_root, "2026-02-06_11-52-46", with_adapter=False)

    # Create a 'latest' directory that should be ignored
    (runs_root / "latest").mkdir()

    # Force mtimes so 'newer' is newest
    older_ts = 1_000_000_000
    newer_ts = 1_000_000_100
    (older).touch()
    (newer).touch()
    older.chmod(older.stat().st_mode)  # no-op; keeps lint happy

    # Set mtimes explicitly
    import os

    os.utime(older, (older_ts, older_ts))
    os.utime(newer, (newer_ts, newer_ts))

    latest = m.find_latest_run(runs_root)
    assert latest.name == newer.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
