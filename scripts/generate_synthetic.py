#!/usr/bin/env python3
"""Generate synthetic compression pairs using trained adapter.

Usage:
    python scripts/generate_synthetic.py --input data/raw/code.jsonl --domain code --limit 1000
    python scripts/generate_synthetic.py --input data/raw/nl_docs.jsonl --domain nl --limit 1000
    python scripts/generate_synthetic.py --input data/raw/code.jsonl --output data/synthetic/code_v2.jsonl --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"data/synthetic/{args.domain}_pairs.jsonl")

    console.print("[bold]Synthetic Data Generation[/bold]")
    console.print(f"Input: {args.input}")
    console.print(f"Output: {args.output}")
    console.print(f"Domain: {args.domain}")

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    with open(args.output, mode, encoding="utf-8") as f:
        compressions = generator.compress_batch(texts, show_progress=True)

        for text, compressed in zip(texts, compressions, strict=True):
            pair = GeneratedPair(
                verbose=text,
                compressed=compressed,
                domain=args.domain,
            )
            f.write(json.dumps(pair.model_dump()) + "\n")

    table = Table(title="Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Generated", str(len(texts)))
    table.add_row("Output", str(args.output))
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
