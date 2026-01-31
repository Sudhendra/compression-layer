#!/usr/bin/env python3
"""Validate synthetic compression pairs with multi-model evaluation.

Usage:
    python scripts/validate_synthetic.py \
      --input data/synthetic/code_v2.jsonl \
      --output data/validated/code_v2.jsonl \
      --threshold 0.80 \
      --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from src.generation.seed_generator import GeneratedPair
from src.validation.harness import CompressionPair, ValidationHarness
from src.validation.models import ModelType

console = Console()


def load_pairs(path: Path) -> list[GeneratedPair]:
    """Load generated pairs from JSONL."""
    pairs: list[GeneratedPair] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(GeneratedPair(**json.loads(line)))
    return pairs


def _write_lines(path: Path, lines: list[str], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for line in lines:
            f.write(line)


async def validate_batch(
    pairs: list[GeneratedPair],
    output_path: Path,
    threshold: float = 0.80,
    concurrency: int = 4,
    models: list[str] | None = None,
    harness: ValidationHarness | None = None,
) -> dict:
    """Validate pairs and save passing ones."""
    model_map = {
        "claude": ModelType.CLAUDE_SONNET,
        "gpt": ModelType.GPT4O_MINI,
        "gemini": ModelType.GEMINI_FLASH,
    }
    if models is None:
        models = ["claude", "gpt"]

    if harness is None:
        model_types = [model_map[m] for m in models]
        harness = ValidationHarness(
            models=model_types,
            equivalence_threshold=threshold,
        )

    passed = 0
    failed = 0

    sem = asyncio.Semaphore(concurrency)

    async def validate_one(pair: GeneratedPair) -> tuple[bool, GeneratedPair]:
        async with sem:
            compression_pair = CompressionPair(
                verbose=pair.verbose,
                compressed=pair.compressed,
                domain=pair.domain,
            )
            result = await harness.validate_pair(compression_pair)
            return result.min_equivalence >= threshold, pair

    await asyncio.to_thread(_write_lines, output_path, [], "w")

    with Progress() as progress:
        task = progress.add_task("Validating...", total=len(pairs))

        for i in range(0, len(pairs), max(1, concurrency * 2)):
            batch = pairs[i : i + max(1, concurrency * 2)]
            results = await asyncio.gather(*[validate_one(p) for p in batch])

            lines: list[str] = []
            for is_pass, pair in results:
                if is_pass:
                    passed += 1
                    lines.append(json.dumps(pair.model_dump()) + "\n")
                else:
                    failed += 1
                progress.advance(task)

            if lines:
                await asyncio.to_thread(_write_lines, output_path, lines, "a")

    return {"passed": passed, "failed": failed, "total": len(pairs)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=["claude", "gpt"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pairs = load_pairs(args.input)
    if args.limit:
        pairs = pairs[: args.limit]

    console.print(f"[bold]Validating {len(pairs)} pairs[/bold]")

    stats = asyncio.run(
        validate_batch(
            pairs,
            args.output,
            threshold=args.threshold,
            concurrency=args.concurrency,
            models=args.models,
        )
    )

    table = Table(title="Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Passed", str(stats["passed"]))
    table.add_row("Failed", str(stats["failed"]))
    table.add_row("Pass Rate", f"{stats['passed'] / stats['total'] * 100:.1f}%")
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
