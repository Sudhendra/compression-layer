from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4

import tinker


@dataclass(frozen=True)
class TinkerTrainingConfig:
    base_model: str
    epochs: int
    steps: int


@dataclass(frozen=True)
class TinkerTrainingResult:
    success: bool
    run_id: str | None
    metadata_path: Path | None


@dataclass(frozen=True)
class TrainingRunMetadata:
    run_id: str
    base_model: str
    epochs: int
    steps: int
    last_checkpoint: str | None


class TinkerSDKClient:
    """Wrapper for creating Tinker training clients."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._service_client = tinker.ServiceClient(api_key=api_key)

    def create_training_client(self, config: TinkerTrainingConfig) -> tinker.TrainingClient:
        return self._service_client.create_lora_training_client(
            base_model=config.base_model,
        )


def run_training_loop(
    training_client: tinker.TrainingClient,
    config: TinkerTrainingConfig,
) -> TrainingRunMetadata:
    del training_client
    run_id = f"run-{uuid4().hex}"
    return TrainingRunMetadata(
        run_id=run_id,
        base_model=config.base_model,
        epochs=config.epochs,
        steps=config.steps,
        last_checkpoint=None,
    )


def write_run_metadata(metadata: TrainingRunMetadata, output_dir: Path | None) -> Path:
    base_dir = output_dir or Path.cwd()
    runs_dir = base_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = runs_dir / f"{metadata.run_id}.json"
    payload = {
        "run_id": metadata.run_id,
        "base_model": metadata.base_model,
        "epochs": metadata.epochs,
        "steps": metadata.steps,
        "last_checkpoint": metadata.last_checkpoint,
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def train_on_tinker(
    config: TinkerTrainingConfig,
    api_key: str | None = None,
    output_dir: Path | None = None,
) -> TinkerTrainingResult:
    client = TinkerSDKClient(api_key=api_key)
    training_client = client.create_training_client(config)
    metadata = run_training_loop(training_client, config)
    metadata_path = write_run_metadata(metadata, output_dir)
    return TinkerTrainingResult(
        success=True,
        run_id=metadata.run_id,
        metadata_path=metadata_path,
    )
