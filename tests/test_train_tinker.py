from pathlib import Path

import pytest

import src.training.train_tinker as train_tinker


class FakeClient:
    last_api_key = None

    def __init__(self, api_key: str | None = None):
        FakeClient.last_api_key = api_key
        self.api_key = api_key
        self._sdk_available = True

    @property
    def is_available(self) -> bool:
        return True

    def upload_dataset(self, data_dir: Path, name: str) -> str:
        return "dataset-123"

    def start_training(self, config: train_tinker.TinkerTrainingConfig, dataset_id: str) -> str:
        return "job-123"

    def wait_for_completion(self, job_id: str, poll_interval: int = 30, callback=None):
        return train_tinker.TinkerJobStatus(
            job_id=job_id,
            status="completed",
            current_loss=0.42,
        )

    def download_adapter(self, job_id: str, output_path: Path) -> Path:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def test_train_on_tinker_uses_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")

    config = train_tinker.TinkerTrainingConfig(dataset_path=data_dir)

    monkeypatch.setattr(train_tinker, "TinkerClient", FakeClient)

    result = train_tinker.train_on_tinker(config, api_key="test-key")

    assert result.success is True
    assert FakeClient.last_api_key == "test-key"
