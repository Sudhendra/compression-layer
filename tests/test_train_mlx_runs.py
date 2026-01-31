import json
from pathlib import Path

import pytest

from src.training.train_mlx import (
    MLXTrainingConfig,
    prepare_run_paths,
    update_latest_symlink,
)


def test_prepare_run_paths_writes_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    config = MLXTrainingConfig(data_dir=data_dir)

    monkeypatch.setattr("src.training.train_mlx.get_git_sha", lambda: "abc123")

    runs_root = tmp_path / "runs"
    paths = prepare_run_paths(config, runs_root)

    assert paths.run_dir.exists()
    assert paths.run_dir.parent == runs_root
    assert paths.meta_path.exists()
    assert paths.meta_path == paths.run_dir / "run.json"

    metadata = json.loads(paths.meta_path.read_text(encoding="utf-8"))
    assert metadata["started_at"].endswith("Z")
    assert metadata["git_sha"] == "abc123"
    assert metadata["data_dir"] == str(config.data_dir)
    assert metadata["model"] == config.model
    assert metadata["lora_rank"] == config.lora_rank
    assert metadata["lora_alpha"] == config.lora_alpha
    assert metadata["iters"] == config.iters
    assert metadata["batch_size"] == config.batch_size
    assert metadata["learning_rate"] == config.learning_rate


def test_update_latest_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-01-30_12-00-00"
    run_dir.mkdir()

    update_latest_symlink(tmp_path, run_dir)

    latest_path = tmp_path / "latest"
    assert latest_path.is_symlink()
    assert latest_path.resolve() == run_dir.resolve()


def test_update_latest_symlink_rejects_existing_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-01-30_12-00-00"
    run_dir.mkdir()

    latest_path = tmp_path / "latest"
    latest_path.mkdir()

    with pytest.raises(ValueError, match="directory exists"):
        update_latest_symlink(tmp_path, run_dir)
