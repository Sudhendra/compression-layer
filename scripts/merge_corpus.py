#!/usr/bin/env python3
"""Merge multiple JSONL corpus files into unified format.

Usage:
    python scripts/merge_corpus.py \
      --inputs data/raw/code.jsonl data/raw/hf_thestack.jsonl \
      --output data/raw/code_merged.jsonl \
      --domain code \
      --dedupe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def compute_hash(text: str) -> str:
    """Compute hash for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def merge_corpus_files(
    input_files: list[Path],
    output_file: Path,
    domain: str,
    dedupe: bool = True,
    min_chars: int = 100,
    max_chars: int = 2500,
) -> dict:
    """Merge multiple JSONL files into one."""

    all_samples: list[dict] = []
    seen_hashes: set[str] = set()
    stats = {"total_read": 0, "duplicates": 0, "filtered": 0, "kept": 0}
    source_counts: dict[str, int] = {}

    for input_file in input_files:
        if not input_file.exists():
            console.print(f"[yellow]Warning: {input_file} not found, skipping[/yellow]")
            continue

        file_count = 0
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                stats["total_read"] += 1

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = item.get("text") or item.get("content") or item.get("code", "")
                if not text:
                    stats["filtered"] += 1
                    continue

                if len(text) < min_chars or len(text) > max_chars:
                    stats["filtered"] += 1
                    continue

                if dedupe:
                    text_hash = compute_hash(text)
                    if text_hash in seen_hashes:
                        stats["duplicates"] += 1
                        continue
                    seen_hashes.add(text_hash)

                normalized = {
                    "text": text,
                    "domain": domain,
                    "source": item.get("source", str(input_file.stem)),
                }

                if "language" in item:
                    normalized["language"] = item["language"]
                if "category" in item:
                    normalized["category"] = item["category"]

                all_samples.append(normalized)
                file_count += 1
                stats["kept"] += 1

        source_counts[input_file.name] = file_count
        console.print(f"  {input_file.name}: {file_count} samples")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    stats["source_counts"] = source_counts
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge corpus files")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", choices=["code", "nl"], required=True)
    parser.add_argument("--dedupe", action="store_true", default=True)
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=2500)
    args = parser.parse_args()

    console.print(f"[cyan]Merging {len(args.inputs)} files...[/cyan]")

    stats = merge_corpus_files(
        args.inputs,
        args.output,
        args.domain,
        dedupe=args.dedupe,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    table = Table(title="Merge Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total read", str(stats["total_read"]))
    table.add_row("Duplicates removed", str(stats["duplicates"]))
    table.add_row("Filtered (length)", str(stats["filtered"]))
    table.add_row("Final count", str(stats["kept"]))
    table.add_row("Output", str(args.output))
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
