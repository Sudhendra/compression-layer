#!/usr/bin/env python3
"""CLI script to format markdown files to JSONL for NL corpus.

Usage:
    # Convert a directory of markdown files
    python scripts/format_markdown.py --input data/raw/docs/ --output data/raw/nl_docs.jsonl

    # Convert a single file
    python scripts/format_markdown.py --input README.md --output data/raw/readme.jsonl

    # Custom chunking and filters
    python scripts/format_markdown.py \
        --input data/raw/docs/ \
        --output data/raw/nl_docs.jsonl \
        --chunk-by section \
        --min-chars 100 \
        --max-chars 1500
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.generation import (
    MarkdownConfig,
    format_markdown_to_jsonl,
    process_markdown_directory,
    process_markdown_file,
)

console = Console()


def print_summary(results: list[dict], output_path: Path) -> None:
    """Print a summary table of formatting results."""
    table = Table(title="Markdown Formatting Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Calculate stats
    total_chars = sum(len(r["text"]) for r in results)
    avg_chars = total_chars // len(results) if results else 0
    unique_files = len(set(r.get("source_file", "") for r in results))

    # Length distribution
    short = sum(1 for r in results if len(r["text"]) < 200)
    medium = sum(1 for r in results if 200 <= len(r["text"]) < 1000)
    long = sum(1 for r in results if len(r["text"]) >= 1000)

    table.add_row("Total chunks", str(len(results)))
    table.add_row("Unique source files", str(unique_files))
    table.add_row("Total characters", f"{total_chars:,}")
    table.add_row("Avg chars/chunk", f"{avg_chars:,}")
    table.add_row("", "")
    table.add_row("Short (<200 chars)", str(short))
    table.add_row("Medium (200-1000)", str(medium))
    table.add_row("Long (>1000)", str(long))
    table.add_row("", "")
    table.add_row("Output file", str(output_path))

    console.print(table)


def print_samples(results: list[dict], limit: int = 5) -> None:
    """Print sample previews."""
    console.print(
        f"\n[yellow]Sample previews (showing {min(limit, len(results))} of {len(results)}):[/yellow]\n"
    )

    for i, result in enumerate(results[:limit]):
        text = result["text"]
        preview = text[:150].replace("\n", " ")
        if len(text) > 150:
            preview += "..."

        source = result.get("source_file", "unknown")
        console.print(f"[cyan]{i + 1}. From: {source}[/cyan]")
        console.print(f"   Length: {len(text)} chars")
        console.print(f"   Preview: {preview}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert markdown files to JSONL for NL corpus training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input markdown file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSONL file path",
    )

    # Chunking options
    parser.add_argument(
        "--chunk-by",
        choices=["paragraph", "section", "document"],
        default="paragraph",
        help="Chunking strategy (default: paragraph)",
    )

    # Filter options
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Minimum characters per chunk (default: 50)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum characters per chunk (default: 2000)",
    )

    # Content options
    parser.add_argument(
        "--keep-code",
        action="store_true",
        help="Keep code blocks in output (default: strip)",
    )
    parser.add_argument(
        "--keep-frontmatter",
        action="store_true",
        help="Keep YAML frontmatter (default: strip)",
    )
    parser.add_argument(
        "--include-headers",
        action="store_true",
        help="Include header lines in chunks",
    )

    # File patterns
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.md", "*.markdown"],
        help="File patterns to match (default: *.md *.markdown)",
    )

    # Output options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without saving",
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
        console.print(f"[red]Error: Input not found: {args.input}[/red]")
        return 1

    # Build config
    config = MarkdownConfig(
        chunk_by=args.chunk_by,
        strip_frontmatter=not args.keep_frontmatter,
        strip_code_blocks=not args.keep_code,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        include_headers=args.include_headers,
    )

    console.print(f"[cyan]Processing markdown from: {args.input}[/cyan]")
    console.print(f"[dim]Chunk by: {config.chunk_by}[/dim]")
    console.print(f"[dim]Min chars: {config.min_chars}, Max chars: {config.max_chars}[/dim]")
    console.print(f"[dim]Strip code blocks: {config.strip_code_blocks}[/dim]")

    # Process files
    if args.input.is_file():
        chunks = process_markdown_file(args.input, config)
        results = [{"text": chunk, "source_file": str(args.input)} for chunk in chunks]
    else:
        results = process_markdown_directory(args.input, config, args.patterns)

    if not results:
        console.print("[yellow]No text chunks extracted matching criteria.[/yellow]")
        return 0

    # Print summary
    print_summary(results, args.output)

    # Show samples if requested
    if args.show_samples > 0:
        print_samples(results, args.show_samples)

    # Save unless dry run
    if args.dry_run:
        console.print("\n[yellow]Dry run - no files written[/yellow]")
    else:
        count = format_markdown_to_jsonl(args.input, args.output, config, args.patterns)
        console.print(f"\n[green]Saved {count} chunks to {args.output}[/green]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
