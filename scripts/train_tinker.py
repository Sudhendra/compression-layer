#!/usr/bin/env python3
"""CLI script for Tinker SDK training and status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import train_on_tinker
from src.utils.config import get_settings, load_tinker_training_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or check status for Tinker SDK runs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Output directory for run metadata",
    )
    parser.add_argument(
        "--status",
        metavar="RUN_ID",
        help="Check status for a run ID",
    )
    return parser.parse_args(argv)


def print_status(run_id: str, output_dir: Path) -> int:
    metadata_path = output_dir / "runs" / f"{run_id}.json"
    if not metadata_path.exists():
        print(f"Run metadata not found: {metadata_path}", file=sys.stderr)
        return 1
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Invalid run metadata: {metadata_path}", file=sys.stderr)
        return 1
    print(f"Run {payload.get('run_id', run_id)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        return print_status(args.status, output_dir=args.output)
    config = load_tinker_training_config(args.config)
    settings = get_settings()
    result = train_on_tinker(config, api_key=settings.tinker_api_key, output_dir=args.output)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
