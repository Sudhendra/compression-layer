"""Generation module for compression pair synthesis."""

from .corpus_loader import (
    CodeExtractionConfig,
    CodeSample,
    load_code_corpus,
    load_nl_corpus,
    save_code_corpus,
)
from .md_formatter import (
    MarkdownConfig,
    format_markdown_to_jsonl,
    process_markdown,
    process_markdown_directory,
    process_markdown_file,
)
from .seed_generator import (
    GeneratedPair,
    GenerationResult,
    SeedGenerator,
    generate_seed_pairs,
)

__all__ = [
    # Corpus loader
    "CodeExtractionConfig",
    "CodeSample",
    "load_code_corpus",
    "load_nl_corpus",
    "save_code_corpus",
    # Markdown formatter
    "MarkdownConfig",
    "format_markdown_to_jsonl",
    "process_markdown",
    "process_markdown_directory",
    "process_markdown_file",
    # Seed generator
    "GeneratedPair",
    "GenerationResult",
    "SeedGenerator",
    "generate_seed_pairs",
]
