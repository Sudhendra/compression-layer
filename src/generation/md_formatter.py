"""Markdown to JSONL formatter for NL corpus preparation."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MarkdownConfig:
    """Configuration for markdown processing."""

    chunk_by: str = "paragraph"  # "paragraph" | "section" | "document"
    strip_frontmatter: bool = True
    strip_code_blocks: bool = True
    strip_html: bool = True
    min_chars: int = 50
    max_chars: int = 2000
    include_headers: bool = False  # Include header text in chunks


# Regex patterns
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HEADER_PATTERN = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\([^)]+\)")


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown."""
    return FRONTMATTER_PATTERN.sub("", content)


def _strip_code_blocks(content: str) -> str:
    """Remove fenced code blocks from markdown."""
    # Remove fenced code blocks
    content = CODE_BLOCK_PATTERN.sub("", content)
    # Remove inline code
    content = INLINE_CODE_PATTERN.sub("", content)
    return content


def _strip_html(content: str) -> str:
    """Remove HTML tags from markdown."""
    return HTML_TAG_PATTERN.sub("", content)


def _clean_markdown(content: str) -> str:
    """Clean markdown formatting, keeping plain text."""
    # Convert links to just their text
    content = LINK_PATTERN.sub(r"\1", content)
    # Remove images (or keep alt text)
    content = IMAGE_PATTERN.sub("", content)
    # Remove bold/italic markers
    content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
    content = re.sub(r"\*([^*]+)\*", r"\1", content)
    content = re.sub(r"__([^_]+)__", r"\1", content)
    content = re.sub(r"_([^_]+)_", r"\1", content)
    # Remove blockquote markers
    content = re.sub(r"^>\s*", "", content, flags=re.MULTILINE)
    # Remove list markers
    content = re.sub(r"^\s*[-*+]\s+", "", content, flags=re.MULTILINE)
    content = re.sub(r"^\s*\d+\.\s+", "", content, flags=re.MULTILINE)
    # Remove horizontal rules
    content = re.sub(r"^[-*_]{3,}\s*$", "", content, flags=re.MULTILINE)
    return content


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)
    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_by_paragraph(content: str, config: MarkdownConfig) -> list[str]:
    """Split content into paragraph chunks."""
    # Split by double newlines
    paragraphs = re.split(r"\n\s*\n", content)

    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Skip if it's just a header (unless configured to include)
        if not config.include_headers and re.match(r"^#{1,6}\s+", para):
            continue

        # Check length constraints
        if config.min_chars <= len(para) <= config.max_chars:
            chunks.append(para)
        elif len(para) > config.max_chars:
            # Split long paragraphs at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= config.max_chars:
                    current_chunk = (current_chunk + " " + sentence).strip()
                else:
                    if len(current_chunk) >= config.min_chars:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            if len(current_chunk) >= config.min_chars:
                chunks.append(current_chunk)

    return chunks


def _chunk_by_section(content: str, config: MarkdownConfig) -> list[str]:
    """Split content by headers into sections."""
    # Split by headers
    sections = re.split(r"(?=^#{1,6}\s+)", content, flags=re.MULTILINE)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Optionally remove the header line itself
        if not config.include_headers:
            section = HEADER_PATTERN.sub("", section).strip()

        if not section:
            continue

        if config.min_chars <= len(section) <= config.max_chars:
            chunks.append(section)
        elif len(section) > config.max_chars:
            # Fall back to paragraph chunking for long sections
            sub_chunks = _chunk_by_paragraph(section, config)
            chunks.extend(sub_chunks)

    return chunks


def _chunk_by_document(content: str, config: MarkdownConfig) -> list[str]:
    """Treat entire document as one chunk (if within limits)."""
    content = content.strip()
    if config.min_chars <= len(content) <= config.max_chars:
        return [content]
    # Fall back to section chunking if document is too long
    return _chunk_by_section(content, config)


def process_markdown(content: str, config: MarkdownConfig | None = None) -> list[str]:
    """
    Process markdown content into text chunks.

    Args:
        content: Raw markdown content
        config: Processing configuration

    Returns:
        List of text chunks
    """
    if config is None:
        config = MarkdownConfig()

    # Apply preprocessing
    if config.strip_frontmatter:
        content = _strip_frontmatter(content)

    if config.strip_code_blocks:
        content = _strip_code_blocks(content)

    if config.strip_html:
        content = _strip_html(content)

    # Clean markdown formatting
    content = _clean_markdown(content)
    content = _normalize_whitespace(content)

    if not content:
        return []

    # Chunk based on strategy
    if config.chunk_by == "paragraph":
        chunks = _chunk_by_paragraph(content, config)
    elif config.chunk_by == "section":
        chunks = _chunk_by_section(content, config)
    elif config.chunk_by == "document":
        chunks = _chunk_by_document(content, config)
    else:
        raise ValueError(f"Unknown chunk_by value: {config.chunk_by}")

    return chunks


def process_markdown_file(file_path: Path, config: MarkdownConfig | None = None) -> list[str]:
    """
    Process a single markdown file into text chunks.

    Args:
        file_path: Path to markdown file
        config: Processing configuration

    Returns:
        List of text chunks
    """
    content = file_path.read_text(encoding="utf-8")
    return process_markdown(content, config)


def process_markdown_directory(
    source_dir: Path,
    config: MarkdownConfig | None = None,
    file_patterns: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Process all markdown files in a directory.

    Args:
        source_dir: Directory containing markdown files
        config: Processing configuration
        file_patterns: Glob patterns for files (default: ["*.md", "*.markdown"])

    Returns:
        List of dicts with 'text' and 'source_file' keys
    """
    if config is None:
        config = MarkdownConfig()

    if file_patterns is None:
        file_patterns = ["*.md", "*.markdown"]

    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    results: list[dict[str, str]] = []

    for pattern in file_patterns:
        for md_file in source_dir.rglob(pattern):
            chunks = process_markdown_file(md_file, config)
            for chunk in chunks:
                results.append({"text": chunk, "source_file": str(md_file)})

    return results


def format_markdown_to_jsonl(
    input_path: Path,
    output_path: Path,
    config: MarkdownConfig | None = None,
    file_patterns: list[str] | None = None,
) -> int:
    """
    Convert markdown files to JSONL format.

    Args:
        input_path: Single markdown file or directory
        output_path: Output JSONL file path
        config: Processing configuration
        file_patterns: Glob patterns (for directory input)

    Returns:
        Number of chunks written
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.is_file():
        chunks = process_markdown_file(input_path, config)
        results = [{"text": chunk, "source_file": str(input_path)} for chunk in chunks]
    else:
        results = process_markdown_directory(input_path, config, file_patterns)

    # Write to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    return len(results)
