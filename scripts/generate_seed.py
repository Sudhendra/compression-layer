#!/usr/bin/env python3
"""CLI script to generate seed compression pairs.

Usage:
    # Generate NL pairs from a JSONL file with 'text' field
    python scripts/generate_seed.py --input data/raw/texts.jsonl --output data/seed/nl_pairs.jsonl

    # Generate code pairs
    python scripts/generate_seed.py --input data/raw/code.jsonl --output data/seed/code_pairs.jsonl --domain code

    # Generate from inline texts
    python scripts/generate_seed.py --texts "User logged in at 9am" "Error occurred in database" --output data/seed/pairs.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.generation import GeneratedPair, SeedGenerator
from src.utils.config import GenerationConfig, get_settings
from src.utils.tokenizers import count_tokens

console = Console()


def load_inputs_from_jsonl(path: Path, text_field: str = "text") -> list[str]:
    """Load input texts from a JSONL file."""
    texts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if text_field in data:
                    texts.append(data[text_field])
    return texts


def load_inputs_from_txt(path: Path) -> list[str]:
    """Load input texts from a plain text file (one per line or paragraph-separated)."""
    content = path.read_text()
    # Try paragraph-separated first (double newline)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    # Fall back to line-separated
    return [line.strip() for line in content.split("\n") if line.strip()]


def print_summary(pairs: list[GeneratedPair], cached: int, generated: int) -> None:
    """Print a summary table of generation results."""
    table = Table(title="Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    total_verbose_tokens = sum(count_tokens(p.verbose) for p in pairs)
    total_compressed_tokens = sum(count_tokens(p.compressed) for p in pairs)
    avg_ratio = total_compressed_tokens / total_verbose_tokens if total_verbose_tokens > 0 else 0
    reduction_pct = (1 - avg_ratio) * 100

    table.add_row("Total pairs", str(len(pairs)))
    table.add_row("From cache", str(cached))
    table.add_row("Newly generated", str(generated))
    table.add_row("Total verbose tokens", f"{total_verbose_tokens:,}")
    table.add_row("Total compressed tokens", f"{total_compressed_tokens:,}")
    table.add_row("Avg compression ratio", f"{avg_ratio:.2%}")
    table.add_row("Avg token reduction", f"{reduction_pct:.1f}%")

    console.print(table)


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate seed compression pairs using Claude API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Path to input file (JSONL or TXT)",
    )
    input_group.add_argument(
        "--texts",
        "-t",
        nargs="+",
        help="Inline texts to compress",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--append",
        "-a",
        action="store_true",
        help="Append to existing output file",
    )

    # Generation options
    parser.add_argument(
        "--domain",
        "-d",
        choices=["nl", "code"],
        default="nl",
        help="Domain type (default: nl)",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="python",
        help="Programming language for code domain (default: python)",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field name in JSONL to use as input text (default: text)",
    )

    # Processing options
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=10,
        help="Maximum concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (regenerate all)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        help="Limit number of inputs to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without making API calls",
    )

    args = parser.parse_args()

    # Load inputs
    if args.texts:
        inputs = args.texts
    elif args.input.suffix == ".jsonl":
        inputs = load_inputs_from_jsonl(args.input, args.text_field)
    else:
        inputs = load_inputs_from_txt(args.input)

    if not inputs:
        console.print("[red]No inputs found![/red]")
        return 1

    # Apply limit
    if args.limit:
        inputs = inputs[: args.limit]

    console.print(f"[cyan]Loaded {len(inputs)} inputs for compression[/cyan]")

    # Dry run - just show inputs
    if args.dry_run:
        console.print("\n[yellow]Dry run - would process these inputs:[/yellow]")
        for i, text in enumerate(inputs[:5]):
            preview = text[:100] + "..." if len(text) > 100 else text
            console.print(f"  {i + 1}. {preview}")
        if len(inputs) > 5:
            console.print(f"  ... and {len(inputs) - 5} more")
        return 0

    # Check API key
    settings = get_settings()
    if not settings.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY not set in environment![/red]")
        return 1

    # Generate pairs with incremental saving
    config = GenerationConfig(concurrency=args.concurrency)
    generator = SeedGenerator(config)

    # Clear output file if not appending
    if not args.append and args.output.exists():
        args.output.unlink()

    result = await generator.generate_batch(
        inputs=inputs,
        domain=args.domain,
        language=args.language,
        concurrency=args.concurrency,
        output_path=args.output,  # Saves incrementally
    )

    console.print(f"[green]Saved {len(result.pairs)} pairs to {args.output}[/green]")

    # Print summary
    print_summary(result.pairs, result.cached_count, result.generated_count)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
