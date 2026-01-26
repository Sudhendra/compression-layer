#!/usr/bin/env python3
"""CLI script to validate compression pairs across multiple models.

Usage:
    # Validate seed pairs and save only passing ones
    python scripts/validate_batch.py --input data/seed/pairs.jsonl --output data/validated/pairs.jsonl

    # Validate with custom threshold
    python scripts/validate_batch.py --input data/seed/pairs.jsonl --output data/validated/pairs.jsonl --threshold 0.90

    # Validate specific models only
    python scripts/validate_batch.py --input data/seed/pairs.jsonl --output data/validated/pairs.jsonl --models claude gpt
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from src.utils.config import get_settings
from src.utils.tokenizers import count_tokens
from src.validation.harness import (
    BatchValidationStats,
    CompressionPair,
    ValidationHarness,
    ValidationResult,
)
from src.validation.models import ModelType

console = Console()


MODEL_SHORTCUTS = {
    "claude": ModelType.CLAUDE_SONNET,
    "gpt": ModelType.GPT4O_MINI,
    "gemini": ModelType.GEMINI_FLASH,
}


def load_pairs(path: Path) -> list[CompressionPair]:
    """Load compression pairs from a JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                pairs.append(CompressionPair(**data))
    return pairs


def save_validated_pairs(
    pairs: list[CompressionPair],
    results: list[ValidationResult],
    output_path: Path,
    save_all: bool = False,
) -> tuple[int, int]:
    """
    Save pairs that passed validation to a JSONL file.

    Returns:
        Tuple of (passed_count, failed_count)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    failed = 0

    with open(output_path, "w") as f:
        for pair, result in zip(pairs, results, strict=True):
            if result.passed or save_all:
                # Include validation metadata
                data = pair.model_dump()
                data["validation"] = {
                    "passed": result.passed,
                    "min_equivalence": result.min_equivalence,
                    "compression_ratio": result.compression_ratio,
                    "equivalence_scores": {
                        model.value: score for model, score in result.equivalence_scores.items()
                    },
                }
                f.write(json.dumps(data) + "\n")

            if result.passed:
                passed += 1
            else:
                failed += 1

    return passed, failed


def print_summary(stats: BatchValidationStats, pairs: list[CompressionPair]) -> None:
    """Print a detailed summary of validation results."""
    table = Table(title="Validation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Calculate additional stats
    total_verbose = sum(count_tokens(p.verbose) for p in pairs)
    total_compressed = sum(count_tokens(p.compressed) for p in pairs)
    token_reduction = (1 - total_compressed / total_verbose) * 100 if total_verbose > 0 else 0

    table.add_row("Total pairs", str(stats.total_pairs))
    table.add_row("Passed", f"[green]{stats.passed_pairs}[/green]")
    table.add_row("Failed", f"[red]{stats.failed_pairs}[/red]")
    table.add_row("Pass rate", f"{stats.pass_rate:.1%}")
    table.add_row("", "")
    table.add_row("Avg compression ratio", f"{stats.avg_compression_ratio:.2%}")
    table.add_row("Avg token reduction", f"{token_reduction:.1f}%")
    table.add_row("Avg equivalence (min)", f"{stats.avg_equivalence:.3f}")
    table.add_row("Min equivalence", f"{stats.min_equivalence:.3f}")

    console.print(table)

    # Per-model breakdown
    if stats.results:
        model_table = Table(title="Per-Model Scores")
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Avg Score", style="green")
        model_table.add_column("Min Score", style="yellow")

        # Aggregate per-model scores
        model_scores: dict[ModelType, list[float]] = {}
        for result in stats.results:
            for model, score in result.equivalence_scores.items():
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(score)

        for model, scores in model_scores.items():
            avg = sum(scores) / len(scores)
            min_score = min(scores)
            model_table.add_row(
                model.value,
                f"{avg:.3f}",
                f"{min_score:.3f}",
            )

        console.print(model_table)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate compression pairs across multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/output
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input JSONL file with compression pairs",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to output JSONL file for validated pairs",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save all pairs (including failed) with validation metadata",
    )

    # Validation options
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.85,
        help="Minimum equivalence threshold to pass (default: 0.85)",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        choices=list(MODEL_SHORTCUTS.keys()),
        default=["claude", "gpt", "gemini"],
        help="Models to validate against (default: all)",
    )

    # Processing options
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=5,
        help="Maximum concurrent validations (default: 5)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        help="Limit number of pairs to validate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be validated without making API calls",
    )

    args = parser.parse_args()

    # Load pairs
    if not args.input.exists():
        console.print(f"[red]Input file not found: {args.input}[/red]")
        return 1

    pairs = load_pairs(args.input)
    if not pairs:
        console.print("[red]No pairs found in input file![/red]")
        return 1

    # Apply limit
    if args.limit:
        pairs = pairs[: args.limit]

    console.print(f"[cyan]Loaded {len(pairs)} pairs for validation[/cyan]")

    # Dry run
    if args.dry_run:
        console.print("\n[yellow]Dry run - would validate these pairs:[/yellow]")
        for i, pair in enumerate(pairs[:5]):
            v_preview = pair.verbose[:50] + "..." if len(pair.verbose) > 50 else pair.verbose
            c_preview = (
                pair.compressed[:50] + "..." if len(pair.compressed) > 50 else pair.compressed
            )
            console.print(f"  {i + 1}. {pair.domain}: {v_preview!r} -> {c_preview!r}")
        if len(pairs) > 5:
            console.print(f"  ... and {len(pairs) - 5} more")
        console.print(f"\nModels: {', '.join(args.models)}")
        console.print(f"Threshold: {args.threshold}")
        return 0

    # Check API keys
    settings = get_settings()
    missing_keys = []
    for model_name in args.models:
        if model_name == "claude" and not settings.anthropic_api_key:
            missing_keys.append("ANTHROPIC_API_KEY")
        elif model_name == "gpt" and not settings.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        elif model_name == "gemini" and not settings.google_api_key:
            missing_keys.append("GOOGLE_API_KEY")

    if missing_keys:
        console.print(f"[red]Missing API keys: {', '.join(missing_keys)}[/red]")
        return 1

    # Initialize harness
    models = [MODEL_SHORTCUTS[m] for m in args.models]
    harness = ValidationHarness(
        models=models,
        equivalence_threshold=args.threshold,
    )

    # Validate with progress
    console.print(
        f"\n[cyan]Validating against {len(models)} models with threshold {args.threshold}...[/cyan]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Validating pairs...", total=len(pairs))

        results: list[ValidationResult] = []
        sem = asyncio.Semaphore(args.concurrency)

        async def validate_one(pair: CompressionPair) -> ValidationResult:
            async with sem:
                result = await harness.validate_pair(pair)
                progress.advance(task)
                return result

        results = await asyncio.gather(*[validate_one(p) for p in pairs])

    # Create stats
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    avg_ratio = sum(r.compression_ratio for r in results) / len(results)
    avg_equiv = sum(r.min_equivalence for r in results) / len(results)
    min_equiv = min(r.min_equivalence for r in results)

    stats = BatchValidationStats(
        total_pairs=len(pairs),
        passed_pairs=passed,
        failed_pairs=failed,
        avg_compression_ratio=avg_ratio,
        avg_equivalence=avg_equiv,
        min_equivalence=min_equiv,
        pass_rate=passed / len(pairs),
        results=results,
    )

    # Save results
    saved_passed, saved_failed = save_validated_pairs(
        pairs, results, args.output, save_all=args.save_all
    )

    if args.save_all:
        console.print(f"\n[green]Saved all {len(pairs)} pairs to {args.output}[/green]")
    else:
        console.print(f"\n[green]Saved {saved_passed} validated pairs to {args.output}[/green]")

    # Print summary
    print_summary(stats, pairs)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
