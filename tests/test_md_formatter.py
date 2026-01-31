"""Tests for md_formatter module."""

import json
import tempfile
from pathlib import Path

import pytest

from src.generation.md_formatter import (
    MarkdownConfig,
    _chunk_by_paragraph,
    _clean_markdown,
    _normalize_whitespace,
    _strip_code_blocks,
    _strip_frontmatter,
    format_markdown_to_jsonl,
    process_markdown,
    process_markdown_directory,
    process_markdown_file,
)


class TestStripFunctions:
    """Tests for stripping functions."""

    def test_strip_frontmatter(self):
        content = """---
title: My Document
date: 2024-01-01
---

This is the actual content."""

        result = _strip_frontmatter(content)
        assert "title:" not in result
        assert "This is the actual content." in result

    def test_strip_frontmatter_no_frontmatter(self):
        content = "Just regular content without frontmatter."
        result = _strip_frontmatter(content)
        assert result == content

    def test_strip_code_blocks_fenced(self):
        content = """Some text before.

```python
def hello():
    print("world")
```

Some text after."""

        result = _strip_code_blocks(content)
        assert "def hello" not in result
        assert "Some text before" in result
        assert "Some text after" in result

    def test_strip_code_blocks_inline(self):
        content = "Use `print()` to output text and `return` to exit."
        result = _strip_code_blocks(content)
        assert "`print()`" not in result
        assert "`return`" not in result


class TestCleanMarkdown:
    """Tests for markdown cleaning."""

    def test_clean_links(self):
        content = "Check out [this link](https://example.com) for more info."
        result = _clean_markdown(content)
        assert "this link" in result
        assert "https://example.com" not in result
        assert "[" not in result

    def test_clean_images(self):
        content = "Here is an image: ![alt text](image.png)"
        result = _clean_markdown(content)
        assert "![" not in result
        assert "image.png" not in result

    def test_clean_bold_italic(self):
        content = "This is **bold** and *italic* and __also bold__ and _also italic_."
        result = _clean_markdown(content)
        assert "**" not in result
        assert "*" not in result
        assert "__" not in result
        assert "_" not in result
        assert "bold" in result
        assert "italic" in result

    def test_clean_blockquotes(self):
        content = "> This is a quote.\n> Second line of quote."
        result = _clean_markdown(content)
        assert ">" not in result
        assert "This is a quote" in result

    def test_clean_list_markers(self):
        content = """- Item one
- Item two
* Item three
1. Numbered item"""
        result = _clean_markdown(content)
        assert "- " not in result
        assert "* " not in result
        assert "1. " not in result
        assert "Item one" in result


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_multiple_spaces(self):
        content = "Too    many     spaces    here."
        result = _normalize_whitespace(content)
        assert "    " not in result
        assert result == "Too many spaces here."

    def test_multiple_newlines(self):
        content = "First paragraph.\n\n\n\n\nSecond paragraph."
        result = _normalize_whitespace(content)
        assert "\n\n\n" not in result


class TestChunkByParagraph:
    """Tests for paragraph chunking."""

    def test_basic_paragraphs(self):
        content = """First paragraph with enough content to pass the minimum.

Second paragraph that also has sufficient content length.

Third paragraph completes our test with more text."""

        config = MarkdownConfig(min_chars=20, max_chars=500)
        chunks = _chunk_by_paragraph(content, config)

        assert len(chunks) == 3
        assert "First paragraph" in chunks[0]
        assert "Second paragraph" in chunks[1]

    def test_skip_short_paragraphs(self):
        content = """Short.

This paragraph is long enough to pass the minimum character filter requirement."""

        config = MarkdownConfig(min_chars=30, max_chars=500)
        chunks = _chunk_by_paragraph(content, config)

        assert len(chunks) == 1
        assert "Short" not in chunks[0]

    def test_split_long_paragraphs(self):
        content = "This is a very long paragraph. " * 50  # Creates long text

        config = MarkdownConfig(min_chars=20, max_chars=200)
        chunks = _chunk_by_paragraph(content, config)

        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_skip_headers(self):
        content = """## This is a Header

This is the paragraph content that follows the header."""

        config = MarkdownConfig(min_chars=20, max_chars=500, include_headers=False)
        chunks = _chunk_by_paragraph(content, config)

        assert not any(c.startswith("##") for c in chunks)


class TestProcessMarkdown:
    """Tests for process_markdown function."""

    def test_full_processing(self):
        content = """---
title: Test
---

# Introduction

This is the introduction paragraph with enough content.

```python
code_to_strip()
```

## Section Two

Another paragraph in section two with sufficient text."""

        config = MarkdownConfig(
            chunk_by="paragraph",
            strip_frontmatter=True,
            strip_code_blocks=True,
            min_chars=20,
            max_chars=500,
        )
        chunks = process_markdown(content, config)

        assert len(chunks) >= 2
        assert not any("title:" in c for c in chunks)
        assert not any("code_to_strip" in c for c in chunks)

    def test_empty_content(self):
        content = ""
        chunks = process_markdown(content)
        assert chunks == []

    def test_only_code_blocks(self):
        content = """```python
only_code()
```"""
        config = MarkdownConfig(strip_code_blocks=True)
        chunks = process_markdown(content, config)
        assert chunks == []


class TestProcessMarkdownFile:
    """Tests for file processing."""

    def test_process_single_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""# Test Document

This is the first paragraph of our test markdown file.

This is the second paragraph with more content to test.""")
            f.flush()

            config = MarkdownConfig(min_chars=20)
            chunks = process_markdown_file(Path(f.name), config)

            assert len(chunks) >= 2


class TestProcessMarkdownDirectory:
    """Tests for directory processing."""

    def test_process_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first markdown file
            (Path(tmpdir) / "doc1.md").write_text("This is the first document with enough content.")
            # Create second markdown file
            (Path(tmpdir) / "doc2.md").write_text(
                "This is the second document with sufficient text."
            )

            config = MarkdownConfig(min_chars=20)
            results = process_markdown_directory(Path(tmpdir), config)

            assert len(results) == 2
            assert all("text" in r for r in results)
            assert all("source_file" in r for r in results)

    def test_process_nested_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (Path(tmpdir) / "root.md").write_text("Root document with enough content here.")
            (subdir / "nested.md").write_text("Nested document also has sufficient text.")

            config = MarkdownConfig(min_chars=20)
            results = process_markdown_directory(Path(tmpdir), config)

            assert len(results) == 2

    def test_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError):
            process_markdown_directory(Path("/nonexistent/path"))


class TestFormatMarkdownToJsonl:
    """Tests for JSONL output."""

    def test_format_single_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("""First paragraph with enough content to pass filters.

Second paragraph that also has sufficient length.""")
            f.flush()
            input_path = Path(f.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)

        config = MarkdownConfig(min_chars=20)
        count = format_markdown_to_jsonl(input_path, output_path, config)

        assert count >= 2

        # Verify JSONL format
        with open(output_path) as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                assert "text" in data
                assert "source_file" in data

    def test_format_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "doc.md").write_text("Document content with enough text for testing.")

            output_path = Path(tmpdir) / "output.jsonl"
            config = MarkdownConfig(min_chars=20)

            count = format_markdown_to_jsonl(Path(tmpdir), output_path, config)

            assert count >= 1
            assert output_path.exists()

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.md"
            input_file.write_text("Content with enough text for the test.")

            output_path = Path(tmpdir) / "nested" / "deep" / "output.jsonl"
            config = MarkdownConfig(min_chars=20)

            format_markdown_to_jsonl(input_file, output_path, config)

            assert output_path.exists()
