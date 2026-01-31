#!/usr/bin/env python3
"""CLI script to prepare code corpus from source files.

Usage:
    # Extract Python code from a directory
    python scripts/prepare_corpus.py --input data/raw/code/ --output data/raw/code.jsonl

    # With custom filters
    python scripts/prepare_corpus.py \
        --input data/raw/code/ \
        --output data/raw/code.jsonl \
        --min-lines 5 \
        --max-lines 50 \
        --exclude "test_*" "*_test.py"

    # Dry run to see what would be extracted
    python scripts/prepare_corpus.py --input data/raw/code/ --output data/raw/code.jsonl --dry-run
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.generation import CodeExtractionConfig, load_code_corpus, save_code_corpus

console = Console()


def print_summary(samples: list, output_path: Path) -> None:
    """Print a summary table of extraction results."""
    table = Table(title="Code Corpus Extraction Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Count by unit type
    functions = sum(1 for s in samples if s.unit_type == "function")
    classes = sum(1 for s in samples if s.unit_type == "class")
    methods = sum(1 for s in samples if s.unit_type == "method")

    # Calculate sizes
    total_chars = sum(len(s.code) for s in samples)
    avg_chars = total_chars // len(samples) if samples else 0
    total_lines = sum(s.end_line - s.start_line + 1 for s in samples)
    avg_lines = total_lines // len(samples) if samples else 0

    # Unique files
    unique_files = len(set(s.file_path for s in samples))

    table.add_row("Total samples", str(len(samples)))
    table.add_row("Functions", str(functions))
    table.add_row("Classes", str(classes))
    table.add_row("Methods", str(methods))
    table.add_row("", "")
    table.add_row("Unique source files", str(unique_files))
    table.add_row("Total characters", f"{total_chars:,}")
    table.add_row("Avg chars/sample", f"{avg_chars:,}")
    table.add_row("Avg lines/sample", str(avg_lines))
    table.add_row("", "")
    table.add_row("Output file", str(output_path))

    console.print(table)


def print_samples(samples: list, limit: int = 5) -> None:
    """Print sample previews."""
    console.print(
        f"\n[yellow]Sample previews (showing {min(limit, len(samples))} of {len(samples)}):[/yellow]\n"
    )

    for i, sample in enumerate(samples[:limit]):
        preview = sample.code[:100].replace("\n", "\\n")
        if len(sample.code) > 100:
            preview += "..."

        console.print(f"[cyan]{i + 1}. {sample.unit_type}: {sample.name}[/cyan]")
        console.print(f"   File: {sample.file_path}:{sample.start_line}")
        console.print(f"   Preview: {preview}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract code samples from source files for compression training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Source directory containing code files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSONL file path",
    )

    # Filter options
    parser.add_argument(
        "--min-lines",
        type=int,
        default=3,
        help="Minimum logical lines per sample (default: 3)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=100,
        help="Maximum logical lines per sample (default: 100)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=100,
        help="Minimum characters per sample (default: 100)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Maximum characters per sample (default: 3000)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help="Additional exclude patterns (e.g., 'test_*' '*_test.py')",
    )

    # Extraction options
    parser.add_argument(
        "--no-methods",
        action="store_true",
        help="Don't extract class methods separately",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test functions (test_*, *_test)",
    )
    parser.add_argument(
        "--include-dunders",
        action="store_true",
        help="Include all dunder methods",
    )
    parser.add_argument(
        "--include-trivial",
        action="store_true",
        help="Include trivial functions (pass, single return)",
    )

    # Output options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without saving",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Number of samples to preview (default: 5, 0 to disable)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        console.print(f"[red]Error: Input directory not found: {args.input}[/red]")
        return 1

    if not args.input.is_dir():
        console.print(f"[red]Error: Input must be a directory: {args.input}[/red]")
        return 1

    # Build config
    exclude_patterns = [
        "test_*.py",
        "*_test.py",
        "tests/*.py",
        "test/*.py",
        "**/tests/**/*.py",
        "**/test/**/*.py",
        "conftest.py",
        "setup.py",
        "setup.cfg",
    ]
    if args.exclude:
        exclude_patterns.extend(args.exclude)

    config = CodeExtractionConfig(
        languages=["python"],
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        exclude_patterns=exclude_patterns,
        skip_trivial=not args.include_trivial,
        skip_tests=not args.include_tests,
        skip_dunders=not args.include_dunders,
        include_methods=not args.no_methods,
    )

    console.print(f"[cyan]Extracting code from: {args.input}[/cyan]")
    console.print(f"[dim]Min lines: {config.min_lines}, Max lines: {config.max_lines}[/dim]")
    console.print(f"[dim]Min chars: {config.min_chars}, Max chars: {config.max_chars}[/dim]")

    # Extract code
    samples = load_code_corpus(args.input, config)

    if not samples:
        console.print("[yellow]No code samples found matching criteria.[/yellow]")
        return 0

    # Print summary
    print_summary(samples, args.output)

    # Show samples if requested
    if args.show_samples > 0:
        print_samples(samples, args.show_samples)

    # Save unless dry run
    if args.dry_run:
        console.print("\n[yellow]Dry run - no files written[/yellow]")
    else:
        count = save_code_corpus(samples, args.output)
        console.print(f"\n[green]Saved {count} samples to {args.output}[/green]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
