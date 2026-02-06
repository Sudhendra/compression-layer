#!/usr/bin/env python3
"""Generate synthetic compression pairs using trained adapter.

Usage:
    python scripts/generate_synthetic.py --input data/raw/code.jsonl --domain code --limit 1000
    python scripts/generate_synthetic.py --input data/raw/nl_docs.jsonl --domain nl --limit 1000
    python scripts/generate_synthetic.py --input data/raw/code.jsonl --output data/synthetic/code_v2.jsonl --resume --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.generation.adapter_generator import AdapterGenerator
from src.generation.seed_generator import GeneratedPair

console = Console()


def load_corpus(path: Path, limit: int | None = None) -> list[str]:
    """Load text inputs from JSONL corpus."""
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get("text") or data.get("content") or data.get("code", "")
            if text and len(text) > 50:
                texts.append(text)
            if limit and len(texts) >= limit:
                break
    return texts


def count_existing(output_path: Path) -> int:
    """Count existing entries for resume support."""
    if not output_path.exists():
        return 0
    with open(output_path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


DomainType = Literal["nl", "code", "mixed"]


def write_pairs(
    texts: list[str],
    generator: AdapterGenerator,
    output_path: Path,
    domain: DomainType,
    batch_size: int | None,
    mode: str,
) -> int:
    """Write generated pairs in batches to avoid data loss."""
    if not texts:
        return 0

    if batch_size is None or batch_size <= 0:
        batch_size = len(texts)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode, encoding="utf-8") as f:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            compressions = generator.compress_batch(batch, show_progress=True)

            for text, compressed in zip(batch, compressions, strict=True):
                pair = GeneratedPair(
                    verbose=text,
                    compressed=compressed,
                    domain=domain,
                )
                f.write(json.dumps(pair.model_dump()) + "\n")

            f.flush()
            os.fsync(f.fileno())

    return len(texts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic compression pairs")
    parser.add_argument("--input", type=Path, required=True, help="Input corpus JSONL")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--domain", choices=["nl", "code"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Max examples to generate")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("models/runs/mlx/latest/adapter"),
    )
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-Instruct-2507-8bit")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Generate in batches and flush each batch (0 for full batch)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"data/synthetic/{args.domain}_pairs.jsonl")

    console.print("[bold]Synthetic Data Generation[/bold]")
    console.print(f"Input: {args.input}")
    console.print(f"Output: {args.output}")
    console.print(f"Domain: {args.domain}")
    console.print(f"Batch size: {args.batch_size}")

    texts = load_corpus(args.input, args.limit)
    console.print(f"Loaded {len(texts)} texts from corpus")

    start_idx = 0
    if args.resume:
        start_idx = count_existing(args.output)
        if start_idx > 0:
            console.print(f"[yellow]Resuming from index {start_idx}[/yellow]")
            texts = texts[start_idx:]

    if not texts:
        console.print("[green]Nothing to generate - already complete![/green]")
        return 0

    generator = AdapterGenerator(
        model=args.model,
        adapter_path=args.adapter,
    )

    mode = "a" if args.resume else "w"

    generated = write_pairs(
        texts,
        generator,
        args.output,
        domain=cast(DomainType, args.domain),
        batch_size=args.batch_size,
        mode=mode,
    )

    table = Table(title="Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Generated", str(generated))
    table.add_row("Output", str(args.output))
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
