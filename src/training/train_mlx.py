"""Local MLX LoRA training for compression model.

This module provides functionality to train a compression model locally
using MLX on Apple Silicon (M-series Macs). It wraps the mlx_lm.lora
command with compression-specific defaults.

Usage:
    from src.training.train_mlx import train_local, MLXTrainingConfig

    config = MLXTrainingConfig(
        model="mlx-community/Qwen3-4B-Instruct-2507-8bit",
        data_dir=Path("data/training"),
        iters=500,
    )
    result = train_local(config)
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.training.run_storage import create_run_dir
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class MLXTrainingConfig:
    """Configuration for local MLX LoRA training."""

    # Model
    model: str = "mlx-community/Qwen3-4B-Instruct-2507-8bit"

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("data/training"))
    adapter_path: Path = field(default_factory=lambda: Path("models/adapters/mlx"))

    # LoRA parameters
    lora_rank: int = 8
    lora_alpha: int = 16  # Usually 2x rank
    lora_layers: int = 16  # Number of layers to fine-tune

    # Training parameters
    iters: int = 500
    batch_size: int = 2
    learning_rate: float = 1e-4
    grad_accumulation_steps: int = 4  # Effective batch size = batch_size * grad_accumulation

    # Memory optimization
    grad_checkpoint: bool = True  # Gradient checkpointing (memory saver; not model checkpointing)
    max_seq_length: int = 2048  # Max sequence length

    # Logging
    steps_per_report: int = 10
    steps_per_eval: int = 100
    save_every: int = 100

    # Masking
    mask_prompt: bool = True  # Only compute loss on assistant response

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML config."""
        return {
            "model": self.model,
            "data": str(self.data_dir),
            "adapter_path": str(self.adapter_path),
            "lora_layers": self.lora_layers,
            "batch_size": self.batch_size,
            "iters": self.iters,
            "learning_rate": self.learning_rate,
            "steps_per_report": self.steps_per_report,
            "steps_per_eval": self.steps_per_eval,
            "save_every": self.save_every,
            "grad_checkpoint": self.grad_checkpoint,
            "max_seq_length": self.max_seq_length,
            "lora_parameters": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": 0.0,
                "scale": self.lora_alpha / self.lora_rank,
            },
        }


@dataclass
class TrainingResult:
    """Result from a training run."""

    success: bool
    adapter_path: Path | None
    final_loss: float | None = None
    total_iters: int = 0
    error: str | None = None


@dataclass(frozen=True)
class RunPaths:
    """Paths for a specific MLX training run."""

    run_dir: Path
    meta_path: Path
    log_path: Path
    err_path: Path


def check_mlx_available() -> bool:
    """Check if mlx_lm is available."""
    try:
        result = subprocess.run(
            ["python", "-c", "import mlx_lm"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_git_sha() -> str:
    """Get the current git SHA."""
    try:
        settings = get_settings()
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=settings.project_root,
        )
    except Exception:
        return "unknown"

    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip() or "unknown"


def prepare_run_paths(config: MLXTrainingConfig, runs_root: Path) -> RunPaths:
    """Create run directory and write metadata for MLX training."""
    run_dir = create_run_dir(runs_root)
    meta_path = run_dir / "run.json"
    log_path = run_dir / "train.log"
    err_path = run_dir / "train.err"
    metadata = {
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_sha": get_git_sha(),
        "data_dir": str(config.data_dir),
        "model": config.model,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "iters": config.iters,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return RunPaths(
        run_dir=run_dir,
        meta_path=meta_path,
        log_path=log_path,
        err_path=err_path,
    )


def update_latest_symlink(runs_root: Path, run_dir: Path) -> None:
    """Update the latest symlink to point at the current run."""
    latest_path = runs_root / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        if latest_path.is_dir() and not latest_path.is_symlink():
            raise ValueError(f"Cannot replace latest path; directory exists at {latest_path}")
        latest_path.unlink()
    latest_path.symlink_to(run_dir, target_is_directory=True)


def write_lora_config(config: MLXTrainingConfig, config_path: Path) -> None:
    """Write MLX LoRA config to YAML file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_config = config.to_dict()

    with open(config_path, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    logger.info(f"Wrote training config to {config_path}")


def train_local(config: MLXTrainingConfig) -> TrainingResult:
    """
    Run local MLX LoRA training.

    This function:
    1. Validates that MLX is available
    2. Writes a YAML config file
    3. Invokes mlx_lm.lora with the config
    4. Returns the result with adapter path

    Args:
        config: Training configuration

    Returns:
        TrainingResult with success status and adapter path
    """
    # Check MLX availability
    if not check_mlx_available():
        return TrainingResult(
            success=False,
            adapter_path=None,
            error="mlx_lm not available. Install with: pip install mlx-lm",
        )

    # Validate data directory
    if not config.data_dir.exists():
        return TrainingResult(
            success=False,
            adapter_path=None,
            error=f"Data directory not found: {config.data_dir}",
        )

    train_file = config.data_dir / "train.jsonl"
    valid_file = config.data_dir / "valid.jsonl"

    if not train_file.exists():
        return TrainingResult(
            success=False,
            adapter_path=None,
            error=f"Training file not found: {train_file}",
        )

    settings = get_settings()
    runs_root = settings.models_dir / "runs" / "mlx"
    run_paths = prepare_run_paths(config, runs_root)
    config.adapter_path = run_paths.run_dir / "adapter"

    # Write MLX LoRA config
    config_path = config.adapter_path / "lora_config.yaml"
    write_lora_config(config, config_path)

    # Build command
    cmd = [
        "python",
        "-m",
        "mlx_lm.lora",
        "-c",
        str(config_path),
        "--model",
        config.model,
        "--train",
        "--data",
        str(config.data_dir),
        "--adapter-path",
        str(config.adapter_path),
        "--batch-size",
        str(config.batch_size),
        "--iters",
        str(config.iters),
        "--learning-rate",
        str(config.learning_rate),
        "--num-layers",
        str(config.lora_layers),
        "--steps-per-report",
        str(config.steps_per_report),
        "--steps-per-eval",
        str(config.steps_per_eval),
        "--save-every",
        str(config.save_every),
    ]

    # Add optional flags
    if config.grad_checkpoint:
        cmd.append("--grad-checkpoint")

    if config.mask_prompt:
        cmd.append("--mask-prompt")

    if valid_file.exists():
        # MLX expects valid.jsonl in the same directory
        pass  # Already handled by --data flag

    logger.info(f"Starting MLX training with command: {' '.join(cmd)}")

    # Create adapter directory
    config.adapter_path.mkdir(parents=True, exist_ok=True)

    try:
        run_paths.log_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            open(run_paths.log_path, "w", encoding="utf-8") as log_file,
            open(run_paths.err_path, "w", encoding="utf-8") as err_file,
        ):
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=err_file,
                text=True,
            )

        if result.returncode != 0:
            logger.error(
                "Training failed. See %s for stderr logs.",
                run_paths.err_path,
            )
            return TrainingResult(
                success=False,
                adapter_path=None,
                error=f"Training failed. See logs in {run_paths.run_dir}",
            )

        # Check for adapter files
        adapter_files = list(config.adapter_path.glob("*.safetensors"))
        if not adapter_files:
            return TrainingResult(
                success=False,
                adapter_path=config.adapter_path,
                error="Training completed but no adapter files found",
            )

        logger.info(
            "Training completed. Adapter saved to %s",
            config.adapter_path,
        )
        update_latest_symlink(runs_root, run_paths.run_dir)

        return TrainingResult(
            success=True,
            adapter_path=config.adapter_path,
            total_iters=config.iters,
        )

    except Exception as e:
        logger.exception("Training failed with exception")
        return TrainingResult(
            success=False,
            adapter_path=None,
            error=str(e),
        )


def evaluate_adapter(
    model: str,
    adapter_path: Path,
    data_dir: Path,
) -> float | None:
    """
    Evaluate a trained adapter on test set.

    Args:
        model: Base model name/path
        adapter_path: Path to adapter
        data_dir: Path to data directory (must contain test.jsonl)

    Returns:
        Test perplexity or None if evaluation failed
    """
    test_file = data_dir / "test.jsonl"
    if not test_file.exists():
        logger.warning(f"Test file not found: {test_file}")
        return None

    cmd = [
        "python",
        "-m",
        "mlx_lm.lora",
        "--model",
        model,
        "--adapter-path",
        str(adapter_path),
        "--data",
        str(data_dir),
        "--test",
        "--mask-prompt",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Evaluation failed: {result.stderr}")
            return None

        # Parse perplexity from output
        # MLX outputs something like "Test loss 2.229, Test ppl 9.290."
        for line in result.stdout.split("\n"):
            if "ppl" in line.lower():
                match = re.search(r"ppl\s*:?\s*([0-9]+(?:\.[0-9]+)?)", line, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass

        return None

    except Exception:
        logger.exception("Evaluation failed")
        return None


def fuse_adapter(
    model: str,
    adapter_path: Path,
    output_path: Path,
    export_gguf: bool = False,
) -> bool:
    """
    Fuse LoRA adapter into base model.

    Args:
        model: Base model name/path
        adapter_path: Path to adapter
        output_path: Path for fused model
        export_gguf: Whether to also export to GGUF format

    Returns:
        True if successful
    """
    cmd = [
        "python",
        "-m",
        "mlx_lm.fuse",
        "--model",
        model,
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(output_path),
    ]

    if export_gguf:
        cmd.append("--export-gguf")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Fusion failed: {result.stderr}")
            return False

        logger.info(f"Model fused and saved to {output_path}")
        return True

    except Exception:
        logger.exception("Fusion failed")
        return False


def load_config_from_yaml(config_path: Path) -> MLXTrainingConfig:
    """
    Load training config from YAML file.

    Args:
        config_path: Path to configs/training.yaml

    Returns:
        MLXTrainingConfig populated from YAML
    """
    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)

    local_config = yaml_config.get("local", {})
    lora_config = local_config.get("lora", {})
    training_config = local_config.get("training", {})

    return MLXTrainingConfig(
        model=local_config.get("model", "mlx-community/Qwen3-4B-Instruct-2507-8bit"),
        lora_rank=lora_config.get("rank", 8),
        lora_alpha=lora_config.get("alpha", 16),
        iters=training_config.get("iters", 500),
        batch_size=training_config.get("batch_size", 2),
        learning_rate=training_config.get("learning_rate", 1e-4),
    )
