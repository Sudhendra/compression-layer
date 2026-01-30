#!/usr/bin/env python3
"""CLI script to format validated compression pairs for training.

Converts validated pairs from data/validated/ into training-ready JSONL files
in chat format for MLX and Tinker training.

Usage:
    # Format with default 80/10/10 split
    python scripts/format_training_data.py

    # Custom split ratios
    python scripts/format_training_data.py --train-ratio 0.9 --valid-ratio 0.05 --test-ratio 0.05

    # Custom output directory
    python scripts/format_training_data.py --output data/training_v2

    # Custom random seed for reproducibility
    python scripts/format_training_data.py --seed 123
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
    COMPRESSION_SYSTEM_PROMPT,
    format_for_training,
    load_validated_pairs,
)
from src.utils.config import get_settings

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Format validated compression pairs for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Format with default settings (80/10/10 split)
  python scripts/format_training_data.py

  # Custom split ratios
  python scripts/format_training_data.py --train-ratio 0.85 --valid-ratio 0.1 --test-ratio 0.05

  # Preview without writing files
  python scripts/format_training_data.py --dry-run
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to validated pairs directory (default: data/validated/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for training files (default: data/training/)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt for chat format (uses default if not specified)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without writing files",
    )

    return parser.parse_args()


def print_preview(
    validated_dir: Path, train_ratio: float, valid_ratio: float, test_ratio: float
) -> None:
    """Print a preview of what the formatting will do."""
    pairs = load_validated_pairs(validated_dir)

    nl_count = len([p for p in pairs if p.domain == "nl"])
    code_count = len([p for p in pairs if p.domain == "code"])
    total = len(pairs)

    train_count = int(total * train_ratio)
    valid_count = int(total * valid_ratio)
    test_count = total - train_count - valid_count

    table = Table(title="Formatting Preview")
    table.add_column("Split", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Percentage", style="yellow", justify="right")

    table.add_row("Train", str(train_count), f"{train_ratio * 100:.0f}%")
    table.add_row("Valid", str(valid_count), f"{valid_ratio * 100:.0f}%")
    table.add_row("Test", str(test_count), f"{test_ratio * 100:.0f}%")
    table.add_row("", "", "")
    table.add_row("Total", str(total), "100%")

    console.print(table)

    console.print(f"\n[dim]Domain breakdown: {nl_count} NL, {code_count} code[/dim]")


def print_example(validated_dir: Path, system_prompt: str | None) -> None:
    """Print an example of the formatted output."""
    from src.training import pair_to_chat_example

    pairs = load_validated_pairs(validated_dir)
    if not pairs:
        return

    # Pick one NL and one code example if available
    nl_pairs = [p for p in pairs if p.domain == "nl"]
    code_pairs = [p for p in pairs if p.domain == "code"]

    examples = []
    if nl_pairs:
        examples.append(("NL Example", nl_pairs[0]))
    if code_pairs:
        examples.append(("Code Example", code_pairs[0]))

    for title, pair in examples[:1]:  # Just show one example
        example = pair_to_chat_example(pair, system_prompt)
        import json

        formatted = json.dumps(
            {"messages": [m.model_dump() for m in example.messages]},
            indent=2,
            ensure_ascii=False,
        )

        # Truncate long content for display
        if len(formatted) > 1000:
            formatted = formatted[:1000] + "\n... (truncated)"

        console.print(Panel(formatted, title=title, border_style="blue"))


def main() -> int:
    """Main entry point."""
    args = parse_args()

    settings = get_settings()

    # Resolve paths
    validated_dir = args.input or settings.validated_dir
    output_dir = args.output or (settings.data_dir / "training")

    # Validate ratios
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        console.print(f"[red]Error: Ratios must sum to 1.0, got {total_ratio}[/red]")
        return 1

    console.print(
        Panel.fit(
            f"[bold]Formatting Training Data[/bold]\n\n"
            f"Input: {validated_dir}\n"
            f"Output: {output_dir}\n"
            f"Split: {args.train_ratio:.0%} / {args.valid_ratio:.0%} / {args.test_ratio:.0%}\n"
            f"Seed: {args.seed}",
            border_style="green",
        )
    )

    # Preview mode
    if args.dry_run:
        console.print("\n[yellow]DRY RUN - No files will be written[/yellow]\n")
        print_preview(validated_dir, args.train_ratio, args.valid_ratio, args.test_ratio)
        console.print("\n[bold]Example output format:[/bold]")
        print_example(validated_dir, args.system_prompt)
        return 0

    # Check input directory exists
    if not validated_dir.exists():
        console.print(f"[red]Error: Input directory not found: {validated_dir}[/red]")
        return 1

    # Run formatting
    try:
        stats = format_for_training(
            validated_dir=validated_dir,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            system_prompt=args.system_prompt,
        )

        # Print results
        table = Table(title="Formatting Complete")
        table.add_column("Split", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_column("File", style="dim")

        table.add_row("Train", str(stats.train), str(output_dir / "train.jsonl"))
        table.add_row("Valid", str(stats.valid), str(output_dir / "valid.jsonl"))
        table.add_row("Test", str(stats.test), str(output_dir / "test.jsonl"))
        table.add_row("", "", "")
        table.add_row("[bold]Total[/bold]", f"[bold]{stats.total}[/bold]", "")

        console.print(table)

        console.print(
            f"\n[dim]Domain breakdown: {stats.nl_count} NL, {stats.code_count} code[/dim]"
        )

        # Print system prompt info
        prompt = args.system_prompt or COMPRESSION_SYSTEM_PROMPT
        console.print(f"\n[dim]System prompt: {prompt[:80]}...[/dim]")

        console.print(f"\n[green]✓ Training data ready at {output_dir}[/green]")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Formatting failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
