#!/usr/bin/env python3
"""Download code samples from HuggingFace datasets.

Usage:
    python scripts/download_hf_code.py --dataset bigcode/the-stack --lang python --limit 2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()


def _content_key_for_dataset(dataset_name: str) -> str:
    if dataset_name == "bigcode/the-stack":
        return "content"
    if dataset_name == "codeparrot/github-code":
        return "code"
    return "content"


def _load_dataset_iter(
    dataset_name: str,
    language: str,
    split: str,
    load_dataset_fn: Callable,
) -> tuple[Iterable[dict], str]:
    if dataset_name == "bigcode/the-stack":
        ds = load_dataset_fn(
            dataset_name,
            data_dir=f"data/{language}",
            split=split,
            streaming=True,
        )
        return ds, "content"

    if dataset_name == "codeparrot/github-code":
        ds = load_dataset_fn(
            dataset_name,
            languages=[language],
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        return ds, "code"

    ds = load_dataset_fn(dataset_name, split=split, streaming=True)
    return ds, "content"


def _code_hash(code: str) -> str:
    return hashlib.md5(code.encode("utf-8")).hexdigest()[:16]


def extract_code_samples(
    dataset_name: str,
    language: str = "python",
    limit: int = 2000,
    min_chars: int = 150,
    max_chars: int = 2000,
    split: str = "train",
    dataset_iter: Iterable[dict] | None = None,
    load_dataset_fn: Callable | None = None,
) -> list[dict]:
    """Extract code samples from a HuggingFace dataset."""

    if load_dataset_fn is None:
        load_dataset_fn = load_dataset

    if dataset_iter is None:
        dataset_iter, content_key = _load_dataset_iter(
            dataset_name,
            language,
            split,
            load_dataset_fn,
        )
    else:
        content_key = _content_key_for_dataset(dataset_name)

    samples: list[dict] = []
    seen_hashes: set[str] = set()

    total_hint = limit * 3 if limit else None
    for item in track(dataset_iter, description="Processing...", total=total_hint):
        if limit and len(samples) >= limit:
            break

        code = item.get(content_key, "")
        if not code:
            continue

        if len(code) < min_chars or len(code) > max_chars:
            continue

        if not any(kw in code for kw in ["def ", "class ", "async def "]):
            continue

        code_hash = _code_hash(code[:500])
        if code_hash in seen_hashes:
            continue
        seen_hashes.add(code_hash)

        samples.append(
            {
                "text": code,
                "language": language,
                "source": dataset_name,
            }
        )

    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Download code from HuggingFace")
    parser.add_argument(
        "--dataset",
        default="bigcode/the-stack",
        choices=["bigcode/the-stack", "codeparrot/github-code"],
    )
    parser.add_argument("--lang", default="python")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/hf_code.jsonl"),
    )
    parser.add_argument("--min-chars", type=int, default=150)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    samples = extract_code_samples(
        args.dataset,
        language=args.lang,
        limit=args.limit,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        split=args.split,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    console.print(f"[green]Saved {len(samples)} samples to {args.output}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
