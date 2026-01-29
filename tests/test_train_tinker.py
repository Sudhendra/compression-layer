"""Tests for Tinker SDK training helpers."""

from pathlib import Path

import pytest

from src.training import train_tinker
from src.training.train_tinker import TinkerTrainingConfig


class FakeTrainingClient:
    """Minimal training client stub."""


class FakeTinkerSDKClient:
    """Fake SDK client used by tests."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def create_training_client(self, config: TinkerTrainingConfig) -> FakeTrainingClient:
        return FakeTrainingClient()


def test_train_on_tinker_records_run_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_tinker, "TinkerSDKClient", FakeTinkerSDKClient)
    config = TinkerTrainingConfig(base_model="qwen-test", epochs=1, steps=2)

    result = train_tinker.train_on_tinker(config, api_key="test", output_dir=tmp_path)

    assert result.run_id is not None
    assert (tmp_path / "runs" / f"{result.run_id}.json").exists()
