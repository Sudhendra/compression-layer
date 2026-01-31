#!/usr/bin/env python3
"""Download conversation samples from HuggingFace datasets.

Usage:
    python scripts/download_hf_conversations.py --dataset databricks/dolly-15k --limit 2000
    python scripts/download_hf_conversations.py --dataset OpenAssistant/oasst1 --limit 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()


def extract_dolly_samples(
    limit: int,
    min_chars: int,
    max_chars: int,
    dataset_iter: Iterable[dict] | None = None,
    dataset_name: str = "databricks/databricks-dolly-15k",
    load_dataset_fn: Callable | None = None,
) -> list[dict]:
    """Extract samples from databricks/dolly-15k."""
    if load_dataset_fn is None:
        load_dataset_fn = load_dataset

    if dataset_iter is None:
        dataset_iter = load_dataset_fn(dataset_name, split="train")

    samples: list[dict] = []
    for item in track(dataset_iter, description="Processing dolly-15k..."):
        if len(samples) >= limit:
            break

        text_parts: list[str] = []
        if item.get("context"):
            text_parts.append(item["context"])
        if item.get("instruction"):
            text_parts.append(item["instruction"])

        text = "\n\n".join(text_parts)
        if len(text) < min_chars or len(text) > max_chars:
            continue

        if text.count("```") > 2:
            continue

        samples.append(
            {
                "text": text,
                "source": "databricks/dolly-15k",
                "category": item.get("category", ""),
            }
        )

    return samples


def extract_oasst_samples(
    limit: int,
    min_chars: int,
    max_chars: int,
    dataset_iter: Iterable[dict] | None = None,
    dataset_name: str = "OpenAssistant/oasst1",
    load_dataset_fn: Callable | None = None,
) -> list[dict]:
    """Extract samples from OpenAssistant/oasst1."""
    if load_dataset_fn is None:
        load_dataset_fn = load_dataset

    if dataset_iter is None:
        dataset_iter = load_dataset_fn(dataset_name, split="train")

    samples: list[dict] = []
    seen: set[str] = set()

    for item in track(dataset_iter, description="Processing oasst1..."):
        if len(samples) >= limit:
            break

        if item.get("role") != "prompter":
            continue
        if item.get("lang") != "en":
            continue

        text = item.get("text", "")
        if len(text) < min_chars or len(text) > max_chars:
            continue

        text_hash = text[:200]
        if text_hash in seen:
            continue
        seen.add(text_hash)

        samples.append(
            {
                "text": text,
                "source": "OpenAssistant/oasst1",
                "message_id": item.get("message_id", ""),
            }
        )

    return samples


def extract_ultrachat_samples(
    limit: int,
    min_chars: int,
    max_chars: int,
    dataset_iter: Iterable[dict] | None = None,
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    split: str = "train_sft",
    load_dataset_fn: Callable | None = None,
) -> list[dict]:
    """Extract samples from HuggingFaceH4/ultrachat_200k."""
    if load_dataset_fn is None:
        load_dataset_fn = load_dataset

    if dataset_iter is None:
        dataset_iter = load_dataset_fn(dataset_name, split=split, streaming=True)

    samples: list[dict] = []
    seen: set[str] = set()

    for item in track(dataset_iter, description="Processing ultrachat..."):
        if len(samples) >= limit:
            break

        messages = item.get("messages", [])
        user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        text = "\n\n".join([t for t in user_msgs if t])
        if not text:
            text = item.get("prompt", "")

        if len(text) < min_chars or len(text) > max_chars:
            continue

        if text.count("```") > 2:
            continue

        text_hash = text[:200]
        if text_hash in seen:
            continue
        seen.add(text_hash)

        samples.append(
            {
                "text": text,
                "source": dataset_name,
                "split": split,
            }
        )

    return samples


def extract_openorca_samples(
    limit: int,
    min_chars: int,
    max_chars: int,
    dataset_iter: Iterable[dict] | None = None,
    dataset_name: str = "Open-Orca/OpenOrca",
    split: str = "train",
    load_dataset_fn: Callable | None = None,
) -> list[dict]:
    """Extract samples from Open-Orca/OpenOrca."""
    if load_dataset_fn is None:
        load_dataset_fn = load_dataset

    if dataset_iter is None:
        dataset_iter = load_dataset_fn(dataset_name, split=split, streaming=True)

    samples: list[dict] = []
    seen: set[str] = set()

    for item in track(dataset_iter, description="Processing openorca..."):
        if len(samples) >= limit:
            break

        system_prompt = item.get("system_prompt", "")
        question = item.get("question", "")
        text = question if not system_prompt else f"{system_prompt}\n\n{question}"

        if len(text) < min_chars or len(text) > max_chars:
            continue

        if text.count("```") > 2:
            continue

        text_hash = text[:200]
        if text_hash in seen:
            continue
        seen.add(text_hash)

        samples.append(
            {
                "text": text,
                "source": dataset_name,
                "split": split,
            }
        )

    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Download conversations from HuggingFace")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "databricks/databricks-dolly-15k",
            "OpenAssistant/oasst1",
            "HuggingFaceH4/ultrachat_200k",
            "Open-Orca/OpenOrca",
            "all",
        ],
    )
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()

    all_samples: list[dict] = []

    if args.dataset in ["databricks/databricks-dolly-15k", "all"]:
        samples = extract_dolly_samples(args.limit, args.min_chars, args.max_chars)
        all_samples.extend(samples)
        console.print(f"[green]Extracted {len(samples)} from dolly-15k[/green]")

    if args.dataset in ["OpenAssistant/oasst1", "all"]:
        samples = extract_oasst_samples(args.limit, args.min_chars, args.max_chars)
        all_samples.extend(samples)
        console.print(f"[green]Extracted {len(samples)} from oasst1[/green]")

    if args.dataset in ["HuggingFaceH4/ultrachat_200k", "all"]:
        samples = extract_ultrachat_samples(args.limit, args.min_chars, args.max_chars)
        all_samples.extend(samples)
        console.print(f"[green]Extracted {len(samples)} from ultrachat[/green]")

    if args.dataset in ["Open-Orca/OpenOrca", "all"]:
        samples = extract_openorca_samples(args.limit, args.min_chars, args.max_chars)
        all_samples.extend(samples)
        console.print(f"[green]Extracted {len(samples)} from openorca[/green]")

    if args.output is None:
        if args.dataset == "all":
            args.output = Path("data/raw/hf_conversations.jsonl")
        else:
            name = args.dataset.replace("/", "_")
            args.output = Path(f"data/raw/hf_{name}.jsonl")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    console.print(f"[green]Saved {len(all_samples)} total samples to {args.output}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
