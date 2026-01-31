"""Tinker cloud training for compression model.

This module provides functionality to train a compression model on Tinker's
cloud infrastructure. Tinker provides fast and cost-effective fine-tuning
for LLMs with full LoRA support.

Note: Requires TINKER_API_KEY environment variable to be set.

Usage:
    from src.training.train_tinker import train_on_tinker, TinkerTrainingConfig

    config = TinkerTrainingConfig(
        model="Qwen/Qwen3-8B",
        dataset_path=Path("data/training"),
    )
    result = train_on_tinker(config)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TinkerLoRAConfig:
    """LoRA configuration for Tinker training."""

    rank: int = 64
    alpha: int = 128
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TinkerTrainingConfig:
    """Configuration for Tinker cloud training."""

    # Model
    model: str = "Qwen/Qwen3-8B"

    # Data paths
    dataset_path: Path = field(default_factory=lambda: Path("data/training"))
    output_dir: Path = field(default_factory=lambda: Path("models/adapters/tinker"))

    # Dataset name for Tinker
    dataset_name: str | None = None  # Auto-generated if not provided

    # LoRA configuration
    lora: TinkerLoRAConfig = field(default_factory=TinkerLoRAConfig)

    # Training parameters
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048

    # Job settings
    wait_for_completion: bool = True
    poll_interval: int = 30  # seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "model": self.model,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "max_seq_length": self.max_seq_length,
            "lora": {
                "r": self.lora.rank,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "target_modules": self.lora.target_modules,
            },
        }


@dataclass
class TinkerJobStatus:
    """Status of a Tinker training job."""

    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = 0.0  # 0.0 to 1.0
    current_epoch: int = 0
    current_loss: float | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TinkerTrainingResult:
    """Result from a Tinker training run."""

    success: bool
    job_id: str | None = None
    adapter_path: Path | None = None
    final_loss: float | None = None
    total_epochs: int = 0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class TinkerClient:
    """
    Client for Tinker cloud training API.

    This is a mock/placeholder implementation. When Tinker SDK is available,
    replace with actual SDK calls.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize Tinker client.

        Args:
            api_key: Tinker API key (defaults to TINKER_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("TINKER_API_KEY", "")
        if not self.api_key:
            logger.warning("TINKER_API_KEY not set. Tinker operations will fail.")

        self._sdk_available = self._check_sdk()

    def _check_sdk(self) -> bool:
        """Check if Tinker SDK is available."""
        try:
            import importlib.util

            return importlib.util.find_spec("tinker") is not None
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        """Check if Tinker is properly configured."""
        return bool(self.api_key) and self._sdk_available

    def upload_dataset(self, data_dir: Path, name: str) -> str:
        """
        Upload dataset to Tinker.

        Args:
            data_dir: Path to directory with train.jsonl, valid.jsonl, test.jsonl
            name: Dataset name

        Returns:
            Dataset ID
        """
        if not self._sdk_available:
            raise RuntimeError("Tinker SDK not available. Install with: pip install tinker")

        tinker = __import__("tinker")  # Dynamic import to avoid static analysis errors

        client = tinker.Client(api_key=self.api_key)

        # Tinker expects the data directory path
        dataset_id: str = client.data.upload(str(data_dir), name=name)

        logger.info(f"Uploaded dataset '{name}' with ID: {dataset_id}")
        return dataset_id

    def start_training(
        self,
        config: TinkerTrainingConfig,
        dataset_id: str,
    ) -> str:
        """
        Start a training job on Tinker.

        Args:
            config: Training configuration
            dataset_id: ID of uploaded dataset

        Returns:
            Job ID
        """
        if not self._sdk_available:
            raise RuntimeError("Tinker SDK not available. Install with: pip install tinker")

        tinker = __import__("tinker")

        client = tinker.Client(api_key=self.api_key)

        # Start training job
        job = client.train(
            model=config.model,
            dataset=dataset_id,
            lora_rank=config.lora.rank,
            lora_alpha=config.lora.alpha,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
        )

        logger.info(f"Started training job: {job.id}")
        return str(job.id)

    def get_job_status(self, job_id: str) -> TinkerJobStatus:
        """
        Get status of a training job.

        Args:
            job_id: Training job ID

        Returns:
            Job status
        """
        if not self._sdk_available:
            raise RuntimeError("Tinker SDK not available")

        tinker = __import__("tinker")

        client = tinker.Client(api_key=self.api_key)
        job = client.get_job(job_id)

        return TinkerJobStatus(
            job_id=job_id,
            status=job.status,
            progress=getattr(job, "progress", 0.0),
            current_epoch=getattr(job, "current_epoch", 0),
            current_loss=getattr(job, "current_loss", None),
            error=getattr(job, "error", None),
            metrics=getattr(job, "metrics", {}),
        )

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        callback: Any | None = None,
    ) -> TinkerJobStatus:
        """
        Wait for a training job to complete.

        Args:
            job_id: Training job ID
            poll_interval: Seconds between status checks
            callback: Optional callback(status) called on each poll

        Returns:
            Final job status
        """
        while True:
            status = self.get_job_status(job_id)

            if callback:
                callback(status)

            if status.status in ("completed", "failed"):
                return status

            logger.info(
                f"Job {job_id}: {status.status} "
                f"({status.progress * 100:.1f}%, epoch {status.current_epoch})"
            )

            time.sleep(poll_interval)

    def download_adapter(self, job_id: str, output_path: Path) -> Path:
        """
        Download trained adapter from completed job.

        Args:
            job_id: Training job ID
            output_path: Where to save adapter

        Returns:
            Path to downloaded adapter
        """
        if not self._sdk_available:
            raise RuntimeError("Tinker SDK not available")

        tinker = __import__("tinker")

        client = tinker.Client(api_key=self.api_key)
        job = client.get_job(job_id)

        output_path.mkdir(parents=True, exist_ok=True)
        job.download_adapter(str(output_path))

        logger.info(f"Downloaded adapter to {output_path}")
        return output_path


def train_on_tinker(
    config: TinkerTrainingConfig,
    api_key: str | None = None,
) -> TinkerTrainingResult:
    """
    Run training on Tinker cloud.

    This function:
    1. Validates configuration and data
    2. Uploads dataset to Tinker
    3. Starts training job
    4. Optionally waits for completion
    5. Downloads adapter

    Args:
        config: Training configuration

    Returns:
        TinkerTrainingResult with success status and adapter path
    """
    client = TinkerClient(api_key=api_key)

    # Check availability
    if not client.is_available:
        return TinkerTrainingResult(
            success=False,
            error="Tinker not available. Ensure TINKER_API_KEY is set and tinker SDK is installed.",
        )

    # Validate data
    if not config.dataset_path.exists():
        return TinkerTrainingResult(
            success=False,
            error=f"Dataset path not found: {config.dataset_path}",
        )

    train_file = config.dataset_path / "train.jsonl"
    if not train_file.exists():
        return TinkerTrainingResult(
            success=False,
            error=f"Training file not found: {train_file}",
        )

    try:
        # Generate dataset name if not provided
        dataset_name = config.dataset_name or f"compression-{int(time.time())}"

        # Upload dataset
        logger.info(f"Uploading dataset from {config.dataset_path}...")
        dataset_id = client.upload_dataset(config.dataset_path, dataset_name)

        # Start training
        logger.info(f"Starting training job with model {config.model}...")
        job_id = client.start_training(config, dataset_id)

        if not config.wait_for_completion:
            return TinkerTrainingResult(
                success=True,
                job_id=job_id,
                error=None,
            )

        # Wait for completion
        logger.info("Waiting for training to complete...")
        final_status = client.wait_for_completion(
            job_id,
            poll_interval=config.poll_interval,
        )

        if final_status.status != "completed":
            return TinkerTrainingResult(
                success=False,
                job_id=job_id,
                error=final_status.error or f"Training failed with status: {final_status.status}",
            )

        # Download adapter
        logger.info("Downloading trained adapter...")
        adapter_path = client.download_adapter(job_id, config.output_dir)

        return TinkerTrainingResult(
            success=True,
            job_id=job_id,
            adapter_path=adapter_path,
            final_loss=final_status.current_loss,
            total_epochs=config.epochs,
            metrics=final_status.metrics,
        )

    except Exception as e:
        logger.exception("Tinker training failed")
        return TinkerTrainingResult(
            success=False,
            error=str(e),
        )


def load_config_from_yaml(config_path: Path) -> TinkerTrainingConfig:
    """
    Load Tinker training config from YAML file.

    Args:
        config_path: Path to configs/training.yaml

    Returns:
        TinkerTrainingConfig populated from YAML
    """
    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)

    cloud_config = yaml_config.get("cloud", {})
    lora_config = cloud_config.get("lora", {})
    training_config = cloud_config.get("training", {})

    return TinkerTrainingConfig(
        model=cloud_config.get("model", "Qwen/Qwen3-8B"),
        lora=TinkerLoRAConfig(
            rank=lora_config.get("rank", 64),
            alpha=lora_config.get("alpha", 128),
            target_modules=lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        ),
        epochs=training_config.get("epochs", 3),
        batch_size=training_config.get("batch_size", 4),
        learning_rate=training_config.get("learning_rate", 2e-4),
    )


def estimate_cost(config: TinkerTrainingConfig, num_examples: int) -> dict[str, Any]:
    """
    Estimate training cost on Tinker.

    Based on documentation:
    - Qwen3-8B: $0.40/1M tokens

    Args:
        config: Training configuration
        num_examples: Number of training examples

    Returns:
        Dictionary with cost estimates
    """
    # Rough estimates based on typical compression pair lengths
    AVG_TOKENS_PER_EXAMPLE = 500  # system + user + assistant

    total_tokens = num_examples * AVG_TOKENS_PER_EXAMPLE * config.epochs

    # Cost per model (rough estimates)
    cost_per_million = {
        "Qwen/Qwen3-8B": 0.40,
        "Qwen/Qwen3-4B": 0.20,
        "Qwen/Qwen3-30B-A3B": 0.45,  # MoE, efficient
    }

    rate = cost_per_million.get(config.model, 0.40)
    estimated_cost = (total_tokens / 1_000_000) * rate

    return {
        "total_tokens": total_tokens,
        "cost_per_million": rate,
        "estimated_cost_usd": estimated_cost,
        "model": config.model,
        "epochs": config.epochs,
        "examples": num_examples,
    }
