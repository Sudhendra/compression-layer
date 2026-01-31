#!/usr/bin/env python3
"""CLI script to validate compression pairs across multiple models.

Features:
- Incremental save: Results saved as completed, survives crashes
- Resume support: Skips already-validated pairs on restart
- Caching: API responses cached to avoid duplicate calls
- Per-provider concurrency: Lower limits for less stable APIs
- Cost guardrails: Stops if projected cost exceeds budget
- Domain-specific tasks: Auto-selects appropriate tasks for code vs NL

Usage:
    # Basic validation
    python scripts/validate_batch.py -i data/seed/pairs.jsonl -o data/validated/pairs.jsonl

    # Resume interrupted validation (automatically detects existing output)
    python scripts/validate_batch.py -i data/seed/pairs.jsonl -o data/validated/pairs.jsonl --resume

    # With cost limit
    python scripts/validate_batch.py -i data/seed/pairs.jsonl -o data/validated/pairs.jsonl --max-cost 10.0

    # Validate specific models only
    python scripts/validate_batch.py -i data/seed/pairs.jsonl -o data/validated/pairs.jsonl -m claude gpt
"""

import argparse
import asyncio
import hashlib
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
    TimeRemainingColumn,
)
from rich.table import Table

from src.utils.caching import SemanticCache
from src.utils.config import get_settings
from src.utils.costs import get_cost_tracker
from src.validation.harness import (
    CompressionPair,
    ValidationHarness,
    ValidationResult,
)
from src.validation.metrics import TaskType
from src.validation.models import ModelType

console = Console()

MODEL_SHORTCUTS = {
    "claude": ModelType.CLAUDE_SONNET,
    "gpt": ModelType.GPT4O_MINI,
    "gemini": ModelType.GEMINI_FLASH,
}


def make_pair_hash(pair: CompressionPair) -> str:
    """Create a unique hash for a compression pair."""
    content = f"{pair.domain}:{pair.verbose}:{pair.compressed}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_pairs(path: Path) -> list[CompressionPair]:
    """Load compression pairs from a JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                pairs.append(CompressionPair(**data))
    return pairs


def load_validated_hashes(path: Path) -> set[str]:
    """Load hashes of already-validated pairs from output file."""
    hashes = set()
    if not path.exists():
        return hashes

    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Reconstruct hash from validated pair
                    pair = CompressionPair(
                        verbose=data["verbose"],
                        compressed=data["compressed"],
                        domain=data["domain"],
                        metadata=data.get("metadata"),
                    )
                    hashes.add(make_pair_hash(pair))
                except (json.JSONDecodeError, KeyError):
                    continue
    return hashes


def estimate_validation_cost(
    num_pairs: int,
    num_models: int,
    num_tasks: int = 2,
    avg_tokens_per_call: int = 500,
) -> float:
    """
    Estimate total cost for validation.

    Each pair requires:
    - 2 calls per task (verbose + compressed)
    - num_tasks tasks
    - num_models models
    = 2 * num_tasks * num_models calls per pair
    """
    calls_per_pair = 2 * num_tasks * num_models
    total_calls = num_pairs * calls_per_pair

    # Estimate cost (using average of model costs)
    avg_input_cost = 0.0
    avg_output_cost = 0.0

    for model in MODEL_SHORTCUTS.values():
        from src.utils.costs import MODEL_PRICING

        input_rate, output_rate = MODEL_PRICING.get(model.value, (3.0, 15.0))
        avg_input_cost += input_rate
        avg_output_cost += output_rate

    avg_input_cost /= len(MODEL_SHORTCUTS)
    avg_output_cost /= len(MODEL_SHORTCUTS)

    # Estimate tokens
    input_tokens = total_calls * avg_tokens_per_call
    output_tokens = total_calls * (avg_tokens_per_call // 2)

    cost = (input_tokens / 1_000_000) * avg_input_cost + (
        output_tokens / 1_000_000
    ) * avg_output_cost

    return cost


def save_single_result(
    pair: CompressionPair,
    result: ValidationResult,
    output_path: Path,
    save_all: bool = False,
) -> bool:
    """
    Save a single validation result to the output file (append mode).

    Returns True if the result was saved.
    """
    if not result.passed and not save_all:
        return False

    data = pair.model_dump()
    data["validation"] = {
        "passed": result.passed,
        "min_equivalence": result.min_equivalence,
        "compression_ratio": result.compression_ratio,
        "equivalence_scores": {
            model.value: score for model, score in result.equivalence_scores.items()
        },
        "llm_judge_used": result.llm_judge_used,
    }

    # Include LLM judge scores if available
    if result.llm_judge_scores:
        data["validation"]["llm_judge_scores"] = {
            model.value: score for model, score in result.llm_judge_scores.items()
        }

    with open(output_path, "a") as f:
        f.write(json.dumps(data) + "\n")

    return True


def print_summary(
    total_pairs: int,
    passed: int,
    failed: int,
    skipped: int,
    pairs: list[CompressionPair],
    results: list[ValidationResult],
) -> None:
    """Print a detailed summary of validation results."""
    table = Table(title="Validation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Calculate stats from results
    if results:
        avg_ratio = sum(r.compression_ratio for r in results) / len(results)
        avg_equiv = sum(r.min_equivalence for r in results) / len(results)
        min_equiv = min(r.min_equivalence for r in results)
        token_reduction = (1 - avg_ratio) * 100
    else:
        avg_ratio = avg_equiv = min_equiv = token_reduction = 0.0

    table.add_row("Total pairs", str(total_pairs))
    table.add_row("Skipped (already done)", f"[dim]{skipped}[/dim]")
    table.add_row("Validated this run", str(len(results)))
    table.add_row("Passed", f"[green]{passed}[/green]")
    table.add_row("Failed", f"[red]{failed}[/red]")
    table.add_row("Pass rate", f"{passed / max(passed + failed, 1):.1%}")
    table.add_row("", "")
    table.add_row("Avg compression ratio", f"{avg_ratio:.2%}")
    table.add_row("Avg token reduction", f"{token_reduction:.1f}%")
    table.add_row("Avg equivalence (min)", f"{avg_equiv:.3f}")
    table.add_row("Min equivalence", f"{min_equiv:.3f}")

    console.print(table)

    # Per-model breakdown
    if results:
        model_table = Table(title="Per-Model Scores")
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Avg Score", style="green")
        model_table.add_column("Min Score", style="yellow")

        model_scores: dict[ModelType, list[float]] = {}
        for result in results:
            for model, score in result.equivalence_scores.items():
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(score)

        for model, scores in model_scores.items():
            avg = sum(scores) / len(scores)
            min_score = min(scores)
            model_table.add_row(model.value, f"{avg:.3f}", f"{min_score:.3f}")

        console.print(model_table)

    # Cost summary
    tracker = get_cost_tracker()
    today_cost = tracker.get_daily_spend()
    total_cost = tracker.get_total_spend()

    cost_table = Table(title="Cost Summary")
    cost_table.add_column("Metric", style="cyan")
    cost_table.add_column("Value", style="green")
    cost_table.add_row("Today's spend", f"${today_cost:.2f}")
    cost_table.add_row("Total spend (all time)", f"${total_cost:.2f}")
    console.print(cost_table)


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
        default=0.72,
        help="Minimum equivalence threshold to pass (default: 0.72)",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        choices=list(MODEL_SHORTCUTS.keys()),
        default=["claude", "gpt", "gemini"],
        help="Models to validate against (default: all)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["qa", "reasoning", "code_gen"],
        help="Tasks to use (default: auto-select based on domain)",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLM-as-judge for more accurate equivalence (costs more)",
    )

    # Processing options
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=3,
        help="Maximum concurrent pair validations (default: 3)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        help="Limit number of pairs to validate",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume from existing output file (skip already-validated pairs)",
    )

    # Cost controls
    parser.add_argument(
        "--max-cost",
        type=float,
        help="Maximum cost in USD before stopping (default: no limit)",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate cost, don't run validation",
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

    all_pairs = load_pairs(args.input)
    if not all_pairs:
        console.print("[red]No pairs found in input file![/red]")
        return 1

    # Apply limit
    if args.limit:
        all_pairs = all_pairs[: args.limit]

    console.print(f"[cyan]Loaded {len(all_pairs)} pairs from {args.input}[/cyan]")

    # Check for resume
    skipped = 0
    pairs_to_validate = all_pairs

    if args.resume and args.output.exists():
        validated_hashes = load_validated_hashes(args.output)
        pairs_to_validate = [p for p in all_pairs if make_pair_hash(p) not in validated_hashes]
        skipped = len(all_pairs) - len(pairs_to_validate)

        if skipped > 0:
            console.print(
                f"[yellow]Resuming: {skipped} pairs already validated, {len(pairs_to_validate)} remaining[/yellow]"
            )

        if not pairs_to_validate:
            console.print("[green]All pairs already validated![/green]")
            return 0

    # Estimate cost
    num_tasks = len(args.tasks) if args.tasks else 2  # Default: 2 tasks
    estimated_cost = estimate_validation_cost(len(pairs_to_validate), len(args.models), num_tasks)

    console.print(f"[cyan]Estimated cost: ${estimated_cost:.2f}[/cyan]")

    if args.estimate_only:
        console.print(
            f"\n[yellow]Estimation only:[/yellow]\n"
            f"  Pairs to validate: {len(pairs_to_validate)}\n"
            f"  Models: {', '.join(args.models)}\n"
            f"  Tasks per pair: {num_tasks}\n"
            f"  API calls: {len(pairs_to_validate) * 2 * num_tasks * len(args.models)}\n"
            f"  Estimated cost: ${estimated_cost:.2f}"
        )
        return 0

    if args.max_cost and estimated_cost > args.max_cost:
        console.print(
            f"[red]Estimated cost (${estimated_cost:.2f}) exceeds max-cost (${args.max_cost:.2f})[/red]\n"
            f"Use --limit to reduce pairs or increase --max-cost"
        )
        return 1

    # Dry run
    if args.dry_run:
        console.print("\n[yellow]Dry run - would validate these pairs:[/yellow]")
        for i, pair in enumerate(pairs_to_validate[:5]):
            v_preview = pair.verbose[:50] + "..." if len(pair.verbose) > 50 else pair.verbose
            console.print(f"  {i + 1}. [{pair.domain}] {v_preview!r}")
        if len(pairs_to_validate) > 5:
            console.print(f"  ... and {len(pairs_to_validate) - 5} more")
        console.print(f"\nModels: {', '.join(args.models)}")
        console.print(f"Threshold: {args.threshold}")
        console.print(f"Estimated cost: ${estimated_cost:.2f}")
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

    # Initialize cache for validation responses
    cache_dir = settings.cache_dir / "validation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = SemanticCache(cache_dir)

    # Parse tasks if specified
    tasks = None
    if args.tasks:
        task_map = {
            "qa": TaskType.QA,
            "reasoning": TaskType.REASONING,
            "code_gen": TaskType.CODE_GEN,
        }
        tasks = [task_map[t] for t in args.tasks]

    # Initialize harness with cache
    models = [MODEL_SHORTCUTS[m] for m in args.models]
    harness = ValidationHarness(
        models=models,
        equivalence_threshold=args.threshold,
        tasks=tasks,
        cache=cache,
        use_llm_judge=args.use_llm_judge,
    )

    if args.use_llm_judge:
        console.print(
            "[yellow]Using LLM-as-judge for equivalence (more accurate but costs more)[/yellow]"
        )

    # Prepare output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Clear output file if not resuming
    if not (args.resume and args.output.exists()):
        args.output.write_text("")

    # Track results
    results: list[ValidationResult] = []
    passed = 0
    failed = 0
    cost_tracker = get_cost_tracker()
    initial_cost = cost_tracker.get_total_spend()
    stop_requested = False

    # Validate with progress
    console.print(
        f"\n[cyan]Validating {len(pairs_to_validate)} pairs against {len(models)} models "
        f"(threshold: {args.threshold})...[/cyan]"
    )

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Validating pairs...", total=len(pairs_to_validate))

        async def validate_one(
            pair: CompressionPair,
        ) -> tuple[CompressionPair, ValidationResult] | None:
            nonlocal passed, failed, stop_requested

            # Check if stop was requested before starting
            if stop_requested:
                progress.advance(task)
                return None

            async with sem:
                # Check again after acquiring semaphore
                if stop_requested:
                    progress.advance(task)
                    return None

                try:
                    result = await harness.validate_pair(pair)

                    # Thread-safe update and save
                    async with lock:
                        results.append(result)

                        if result.passed:
                            passed += 1
                        else:
                            failed += 1

                        # Save immediately (incremental) - uses file append
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None,
                            save_single_result,
                            pair,
                            result,
                            args.output,
                            args.save_all,
                        )

                        # Check cost limit
                        if args.max_cost and not stop_requested:
                            current_cost = cost_tracker.get_total_spend() - initial_cost
                            if current_cost > args.max_cost:
                                console.print(
                                    f"\n[red]Cost limit reached: ${current_cost:.2f} > ${args.max_cost:.2f}[/red]"
                                )
                                stop_requested = True

                    progress.advance(task)
                    return pair, result

                except Exception as e:
                    console.print(f"\n[red]Error validating pair: {e}[/red]")
                    progress.advance(task)
                    return None

        # Process pairs in batches to allow early stopping
        batch_size = args.concurrency * 2  # Process 2x concurrency at a time
        for i in range(0, len(pairs_to_validate), batch_size):
            if stop_requested:
                console.print(
                    f"[yellow]Stopping early due to cost limit. Processed {i} pairs.[/yellow]"
                )
                break

            batch = pairs_to_validate[i : i + batch_size]
            # Launch tasks for this batch and allow early stop within the batch
            tasks = [asyncio.create_task(validate_one(p)) for p in batch]

            for completed in asyncio.as_completed(tasks):
                # Wait for the next task in this batch to complete
                await completed

                # If a cost limit stop was requested, cancel remaining tasks in this batch
                if stop_requested:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    # Ensure all tasks in this batch have finished or been cancelled
                    await asyncio.gather(*tasks, return_exceptions=True)
                    break
    # Print summary
    console.print(f"\n[green]Saved results to {args.output}[/green]")
    print_summary(len(all_pairs), passed, failed, skipped, pairs_to_validate, results)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
