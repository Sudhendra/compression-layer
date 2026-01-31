#!/usr/bin/env python3
"""CLI script for local MLX LoRA training.

Train a compression model locally on Apple Silicon using MLX.

Usage:
    # Train with default settings
    python scripts/train_local.py

    # Train with custom iterations
    python scripts/train_local.py --iters 1000

    # Train with custom model
    python scripts/train_local.py --model mlx-community/Qwen3-4B-Instruct-2507-8bit

    # Evaluate existing adapter
    python scripts/train_local.py --evaluate --adapter-path models/adapters/mlx

    # Fuse adapter into model
    python scripts/train_local.py --fuse --adapter-path models/adapters/mlx
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.training import (
    MLXTrainingConfig,
    check_mlx_available,
    evaluate_adapter,
    fuse_adapter,
    train_local,
)
from src.training.train_mlx import load_config_from_yaml
from src.utils.config import get_settings

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train compression model locally with MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python scripts/train_local.py

  # Train with more iterations
  python scripts/train_local.py --iters 1000 --batch-size 4

  # Use config file
  python scripts/train_local.py --config configs/training.yaml

  # Evaluate trained adapter
  python scripts/train_local.py --evaluate

  # Fuse adapter into model
  python scripts/train_local.py --fuse --output models/fused/compression-v1
        """,
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", default=True, help="Train model (default)")
    mode.add_argument("--evaluate", action="store_true", help="Evaluate existing adapter")
    mode.add_argument("--fuse", action="store_true", help="Fuse adapter into base model")

    # Paths
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training config YAML (default: configs/training.yaml)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to training data directory (default: data/training/)",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Path to save/load adapter (default: models/adapters/mlx/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for fused model (only with --fuse)",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-4B-Instruct-2507-8bit",
        help="Base model to fine-tune (default: mlx-community/Qwen3-4B-Instruct-2507-8bit)",
    )

    # Training parameters
    parser.add_argument(
        "--iters",
        type=int,
        default=500,
        help="Number of training iterations (default: 500)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune (default: 16)",
    )

    # Memory optimization
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing (slower but uses less memory)",
    )

    # Fuse options
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Export fused model to GGUF format (only with --fuse)",
    )

    return parser.parse_args()


def print_config(config: MLXTrainingConfig) -> None:
    """Print training configuration."""
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", config.model)
    table.add_row("Data Directory", str(config.data_dir))
    table.add_row("Adapter Path", str(config.adapter_path))
    table.add_row("Iterations", str(config.iters))
    table.add_row("Batch Size", str(config.batch_size))
    table.add_row("Learning Rate", f"{config.learning_rate:.0e}")
    table.add_row("LoRA Rank", str(config.lora_rank))
    table.add_row("LoRA Layers", str(config.lora_layers))
    table.add_row("Grad Checkpoint", "Yes" if config.grad_checkpoint else "No")

    console.print(table)


def count_examples(data_dir: Path) -> dict[str, int]:
    """Count examples in each split."""
    counts = {}
    for split in ["train", "valid", "test"]:
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            with open(path) as f:
                counts[split] = sum(1 for _ in f)
        else:
            counts[split] = 0
    return counts


def main() -> int:
    """Main entry point."""
    args = parse_args()

    settings = get_settings()

    # Check MLX availability
    if not check_mlx_available():
        console.print("[red]Error: MLX not available.[/red]\nInstall with: pip install mlx-lm")
        return 1

    # Load config from YAML if provided
    if args.config and args.config.exists():
        config = load_config_from_yaml(args.config)
    else:
        config = MLXTrainingConfig()

    # Override with CLI arguments
    config.model = args.model
    config.data_dir = args.data or settings.data_dir / "training"
    if args.adapter_path:
        config.adapter_path = args.adapter_path
    else:
        latest_run_adapter = settings.models_dir / "runs" / "mlx" / "latest" / "adapter"
        if latest_run_adapter.exists():
            config.adapter_path = latest_run_adapter
        else:
            config.adapter_path = settings.adapters_dir / "mlx"
    config.iters = args.iters
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.lora_rank = args.lora_rank
    config.lora_layers = args.lora_layers
    config.grad_checkpoint = args.grad_checkpoint

    # Handle different modes
    if args.evaluate:
        return run_evaluate(config.model, config.adapter_path, config.data_dir)
    elif args.fuse:
        output = args.output or (settings.models_dir / "fused" / "compression-v1")
        return run_fuse(config.model, config.adapter_path, output, args.export_gguf)
    else:
        return run_train(config)


def run_train(config: MLXTrainingConfig) -> int:
    """Run training."""
    settings = get_settings()
    console.print(
        Panel.fit(
            "[bold]MLX Local Training[/bold]\n\n"
            "Training compression model with LoRA on Apple Silicon",
            border_style="green",
        )
    )

    # Print configuration
    print_config(config)

    # Check data
    if not config.data_dir.exists():
        console.print(f"[red]Error: Data directory not found: {config.data_dir}[/red]")
        console.print("\nRun this first:")
        console.print("  python scripts/format_training_data.py")
        return 1

    counts = count_examples(config.data_dir)
    console.print(
        f"\n[dim]Data: {counts.get('train', 0)} train, "
        f"{counts.get('valid', 0)} valid, {counts.get('test', 0)} test[/dim]"
    )

    if counts.get("train", 0) == 0:
        console.print("[red]Error: No training examples found[/red]")
        return 1

    # Run training
    console.print("\n[bold]Starting training...[/bold]")
    console.print("[dim]This may take a while depending on iterations and hardware[/dim]\n")

    result = train_local(config)

    if not result.success:
        console.print(f"[red]Training failed: {result.error}[/red]")
        return 1

    # Success
    run_dir = result.adapter_path.parent if result.adapter_path else None
    latest_symlink = settings.models_dir / "runs" / "mlx" / "latest"
    log_hint = ""
    if run_dir is not None:
        log_hint = f"\nRun logs: {run_dir / 'train.log'}"

    console.print(
        Panel.fit(
            f"[green]✓ Training completed![/green]\n\n"
            f"Adapter saved to: {result.adapter_path}\n"
            f"Iterations: {result.total_iters}\n"
            f"Latest run: {latest_symlink}"
            f"{log_hint}",
            border_style="green",
        )
    )

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Evaluate: python scripts/train_local.py --evaluate")
    console.print("  2. Fuse: python scripts/train_local.py --fuse")
    console.print(
        f"  3. Generate: mlx_lm.generate --model {config.model} --adapter-path {config.adapter_path}"
    )

    return 0


def run_evaluate(model: str, adapter_path: Path, data_dir: Path) -> int:
    """Run evaluation."""
    console.print("[bold]Evaluating adapter...[/bold]\n")

    if not adapter_path.exists():
        console.print(f"[red]Error: Adapter not found: {adapter_path}[/red]")
        return 1

    ppl = evaluate_adapter(model, adapter_path, data_dir)

    if ppl is None:
        console.print("[red]Evaluation failed[/red]")
        return 1

    console.print(
        Panel.fit(
            f"[green]Test Perplexity: {ppl:.2f}[/green]",
            border_style="green",
        )
    )

    return 0


def run_fuse(model: str, adapter_path: Path, output: Path, export_gguf: bool) -> int:
    """Fuse adapter into model."""
    console.print("[bold]Fusing adapter into model...[/bold]\n")

    if not adapter_path.exists():
        console.print(f"[red]Error: Adapter not found: {adapter_path}[/red]")
        return 1

    success = fuse_adapter(model, adapter_path, output, export_gguf)

    if not success:
        console.print("[red]Fusion failed[/red]")
        return 1

    console.print(
        Panel.fit(
            f"[green]✓ Model fused![/green]\n\nOutput: {output}",
            border_style="green",
        )
    )

    if export_gguf:
        console.print(f"\nGGUF exported to: {output / 'ggml-model-f16.gguf'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
