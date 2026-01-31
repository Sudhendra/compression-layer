"""Corpus loaders for extracting code and NL data from source files."""

import ast
import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel


class CodeSample(BaseModel):
    """A code sample extracted from source files."""

    code: str
    language: str
    unit_type: str  # "function", "class", "method"
    name: str
    file_path: str
    start_line: int
    end_line: int
    metadata: dict[str, str] = {}


@dataclass
class CodeExtractionConfig:
    """Configuration for code extraction."""

    languages: list[str] = field(default_factory=lambda: ["python"])
    min_lines: int = 3
    max_lines: int = 100
    min_chars: int = 100
    max_chars: int = 3000
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
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
    )
    skip_trivial: bool = True
    skip_tests: bool = True
    skip_dunders: bool = True
    include_methods: bool = True


# Dunder methods worth keeping (have meaningful logic)
USEFUL_DUNDERS = {
    "__call__",
    "__getitem__",
    "__setitem__",
    "__iter__",
    "__next__",
    "__enter__",
    "__exit__",
}


def _is_trivial_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if a function is trivial (pass, ..., single return, simple assignment)."""
    # Get body excluding docstrings
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # Skip docstring

    if not body:
        return True

    if len(body) == 1:
        stmt = body[0]
        # pass statement
        if isinstance(stmt, ast.Pass):
            return True
        # Ellipsis (...)
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is ...
        ):
            return True
        # raise NotImplementedError
        if isinstance(stmt, ast.Raise):
            return True
        # Single return None or return with no value
        if isinstance(stmt, ast.Return) and (
            stmt.value is None or isinstance(stmt.value, ast.Constant)
        ):
            return True

    return False


def _is_trivial_init(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if __init__ only does self.x = x assignments."""
    if node.name != "__init__":
        return False

    body = node.body
    # Skip docstring
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]

    if not body:
        return True

    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            return False
        # Check if it's self.x = x pattern
        if len(stmt.targets) != 1:
            return False
        target = stmt.targets[0]
        if not isinstance(target, ast.Attribute):
            return False
        if not isinstance(target.value, ast.Name) or target.value.id != "self":
            return False

    return True


def _get_logical_lines(code: str) -> int:
    """Count logical lines (non-empty, non-comment)."""
    lines = code.strip().split("\n")
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def _should_skip_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    config: CodeExtractionConfig,
) -> bool:
    """Determine if a function should be skipped based on config."""
    name = node.name

    # Skip test functions
    if config.skip_tests and (name.startswith("test_") or name.endswith("_test")):
        return True

    # Skip most dunder methods
    if (
        config.skip_dunders
        and name.startswith("__")
        and name.endswith("__")
        and name not in USEFUL_DUNDERS
    ):
        return True

    # Skip trivial functions
    if config.skip_trivial:
        if _is_trivial_function(node):
            return True
        if _is_trivial_init(node):
            return True

    return False


def _extract_node_source(
    source: str, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
) -> str:
    """Extract source code for an AST node."""
    lines = source.split("\n")
    # ast line numbers are 1-indexed
    start = node.lineno - 1
    end = node.end_lineno
    return "\n".join(lines[start:end])


def extract_python_code(
    file_path: Path,
    config: CodeExtractionConfig,
) -> list[CodeSample]:
    """
    Extract functions and classes from a Python file.

    Args:
        file_path: Path to the Python file
        config: Extraction configuration

    Returns:
        List of CodeSample objects
    """
    samples: list[CodeSample] = []

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return samples

    for node in ast.walk(tree):
        # Extract top-level functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip if it's a method (has a parent class)
            # We'll handle methods when processing classes
            parent = getattr(node, "_parent", None)
            if parent is not None and isinstance(parent, ast.ClassDef):
                continue

            if _should_skip_function(node, config):
                continue

            code = _extract_node_source(source, node)
            logical_lines = _get_logical_lines(code)

            if logical_lines < config.min_lines or logical_lines > config.max_lines:
                continue
            if len(code) < config.min_chars or len(code) > config.max_chars:
                continue

            samples.append(
                CodeSample(
                    code=code,
                    language="python",
                    unit_type="function",
                    name=node.name,
                    file_path=str(file_path),
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                )
            )

        # Extract classes
        elif isinstance(node, ast.ClassDef):
            # Mark children with parent reference for method detection
            for child in ast.iter_child_nodes(node):
                child._parent = node  # type: ignore[attr-defined]

            class_code = _extract_node_source(source, node)
            class_logical_lines = _get_logical_lines(class_code)

            # Only include class if within size limits
            if (
                config.min_lines <= class_logical_lines <= config.max_lines
                and config.min_chars <= len(class_code) <= config.max_chars
            ):
                samples.append(
                    CodeSample(
                        code=class_code,
                        language="python",
                        unit_type="class",
                        name=node.name,
                        file_path=str(file_path),
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                    )
                )

            # Also extract individual methods if configured
            if config.include_methods:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if _should_skip_function(child, config):
                            continue

                        method_code = _extract_node_source(source, child)
                        method_logical_lines = _get_logical_lines(method_code)

                        if (
                            method_logical_lines < config.min_lines
                            or method_logical_lines > config.max_lines
                        ):
                            continue
                        if (
                            len(method_code) < config.min_chars
                            or len(method_code) > config.max_chars
                        ):
                            continue

                        samples.append(
                            CodeSample(
                                code=method_code,
                                language="python",
                                unit_type="method",
                                name=f"{node.name}.{child.name}",
                                file_path=str(file_path),
                                start_line=child.lineno,
                                end_line=child.end_lineno or child.lineno,
                                metadata={"class": node.name},
                            )
                        )

    return samples


def _matches_any_pattern(path: Path, patterns: list[str], base_dir: Path) -> bool:
    """Check if a path matches any of the exclude patterns."""
    rel_path = str(path.relative_to(base_dir))
    name = path.name

    for pattern in patterns:
        # Check against filename
        if fnmatch.fnmatch(name, pattern):
            return True
        # Check against relative path (for patterns like **/tests/**)
        if fnmatch.fnmatch(rel_path, pattern):
            return True

    return False


def load_code_corpus(
    source_dir: Path,
    config: CodeExtractionConfig | None = None,
) -> list[CodeSample]:
    """
    Load code samples from a directory of source files.

    Walks through the directory, parses Python files, and extracts
    functions and classes that meet the configured criteria.

    Args:
        source_dir: Root directory containing source files
        config: Extraction configuration (uses defaults if None)

    Returns:
        List of CodeSample objects

    Example:
        >>> samples = load_code_corpus(Path("data/raw/code/"))
        >>> print(f"Extracted {len(samples)} code samples")
    """
    if config is None:
        config = CodeExtractionConfig()

    samples: list[CodeSample] = []
    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Currently only Python is supported
    if "python" in config.languages:
        for py_file in source_dir.rglob("*.py"):
            # Skip excluded patterns
            if _matches_any_pattern(py_file, config.exclude_patterns, source_dir):
                continue

            file_samples = extract_python_code(py_file, config)
            samples.extend(file_samples)

    return samples


def load_nl_corpus(
    source_dir: Path,
    file_patterns: list[str] | None = None,
    text_field: str = "text",
    min_chars: int = 50,
    max_chars: int = 2000,
) -> list[str]:
    """
    Load natural language samples from JSONL files.

    Args:
        source_dir: Directory containing JSONL files
        file_patterns: Glob patterns for files to include (default: ["*.jsonl"])
        text_field: Field name in JSONL containing the text
        min_chars: Minimum character count
        max_chars: Maximum character count

    Returns:
        List of text strings
    """
    if file_patterns is None:
        file_patterns = ["*.jsonl"]

    texts: list[str] = []
    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for pattern in file_patterns:
        for jsonl_file in source_dir.rglob(pattern):
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if text_field in data:
                            text = data[text_field]
                            if min_chars <= len(text) <= max_chars:
                                texts.append(text)
                    except json.JSONDecodeError:
                        continue

    return texts


def save_code_corpus(samples: list[CodeSample], output_path: Path) -> int:
    """
    Save code samples to a JSONL file.

    Args:
        samples: List of CodeSample objects
        output_path: Path to output JSONL file

    Returns:
        Number of samples written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            # Convert to dict with 'text' field for compatibility with generate_seed.py
            data = {
                "text": sample.code,
                "language": sample.language,
                "unit_type": sample.unit_type,
                "name": sample.name,
                "file_path": sample.file_path,
                "start_line": sample.start_line,
                "end_line": sample.end_line,
            }
            if sample.metadata:
                data["metadata"] = sample.metadata
            f.write(json.dumps(data) + "\n")

    return len(samples)
