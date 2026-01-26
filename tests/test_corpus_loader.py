"""Tests for corpus_loader module."""

import json
import tempfile
from pathlib import Path

import pytest

from src.generation.corpus_loader import (
    CodeExtractionConfig,
    CodeSample,
    _get_logical_lines,
    _is_trivial_function,
    _is_trivial_init,
    extract_python_code,
    load_code_corpus,
    load_nl_corpus,
    save_code_corpus,
)


class TestCodeSample:
    """Tests for CodeSample model."""

    def test_create_sample(self):
        sample = CodeSample(
            code="def foo(): pass",
            language="python",
            unit_type="function",
            name="foo",
            file_path="/path/to/file.py",
            start_line=1,
            end_line=1,
        )
        assert sample.code == "def foo(): pass"
        assert sample.language == "python"
        assert sample.unit_type == "function"

    def test_sample_with_metadata(self):
        sample = CodeSample(
            code="def foo(): pass",
            language="python",
            unit_type="method",
            name="MyClass.foo",
            file_path="/path/to/file.py",
            start_line=1,
            end_line=1,
            metadata={"class": "MyClass"},
        )
        assert sample.metadata["class"] == "MyClass"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_logical_lines_simple(self):
        code = """def foo():
    x = 1
    return x"""
        assert _get_logical_lines(code) == 3

    def test_get_logical_lines_with_comments(self):
        code = """def foo():
    # This is a comment
    x = 1
    # Another comment
    return x"""
        assert _get_logical_lines(code) == 3  # Comments don't count

    def test_get_logical_lines_with_blank(self):
        code = """def foo():
    x = 1

    return x"""
        assert _get_logical_lines(code) == 3  # Blank lines don't count


class TestTrivialDetection:
    """Tests for trivial function detection."""

    def test_trivial_pass(self):
        import ast

        code = "def foo(): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        assert _is_trivial_function(func) is True

    def test_trivial_ellipsis(self):
        import ast

        code = "def foo(): ..."
        tree = ast.parse(code)
        func = tree.body[0]
        assert _is_trivial_function(func) is True

    def test_trivial_raise_not_implemented(self):
        import ast

        code = "def foo(): raise NotImplementedError()"
        tree = ast.parse(code)
        func = tree.body[0]
        assert _is_trivial_function(func) is True

    def test_trivial_return_none(self):
        import ast

        code = "def foo(): return None"
        tree = ast.parse(code)
        func = tree.body[0]
        assert _is_trivial_function(func) is True

    def test_non_trivial_function(self):
        import ast

        code = """def foo():
    x = 1
    return x + 1"""
        tree = ast.parse(code)
        func = tree.body[0]
        assert _is_trivial_function(func) is False

    def test_trivial_init_simple_assignment(self):
        import ast

        code = """def __init__(self, x, y):
    self.x = x
    self.y = y"""
        tree = ast.parse(code)
        func = tree.body[0]
        assert _is_trivial_init(func) is True

    def test_non_trivial_init(self):
        import ast

        code = """def __init__(self, x):
    self.x = x
    self.validate()"""
        tree = ast.parse(code)
        func = tree.body[0]
        assert _is_trivial_init(func) is False


class TestExtractPythonCode:
    """Tests for Python code extraction."""

    def test_extract_function(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''def calculate_sum(a, b):
    """Add two numbers."""
    result = a + b
    return result
''')
            f.flush()

            config = CodeExtractionConfig(min_lines=2, min_chars=20)
            samples = extract_python_code(Path(f.name), config)

            assert len(samples) == 1
            assert samples[0].unit_type == "function"
            assert samples[0].name == "calculate_sum"

    def test_extract_class(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''class MyClass:
    """A sample class."""
    
    def method_one(self):
        """First method."""
        return self.value * 2
    
    def method_two(self, x):
        """Second method."""
        return self.value + x
''')
            f.flush()

            config = CodeExtractionConfig(min_lines=2, min_chars=20, include_methods=True)
            samples = extract_python_code(Path(f.name), config)

            # Should get class + 2 methods
            unit_types = [s.unit_type for s in samples]
            assert "class" in unit_types
            assert unit_types.count("method") == 2

    def test_skip_trivial_function(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""def trivial():
    pass

def useful():
    x = compute_something()
    y = transform(x)
    return y
""")
            f.flush()

            config = CodeExtractionConfig(min_lines=2, min_chars=20, skip_trivial=True)
            samples = extract_python_code(Path(f.name), config)

            names = [s.name for s in samples]
            assert "trivial" not in names
            assert "useful" in names

    def test_skip_test_functions(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""def test_something():
    result = do_test()
    assert result == expected
    return result

def actual_function():
    x = compute()
    y = transform(x)
    return y
""")
            f.flush()

            config = CodeExtractionConfig(min_lines=2, min_chars=20, skip_tests=True)
            samples = extract_python_code(Path(f.name), config)

            names = [s.name for s in samples]
            assert "test_something" not in names
            assert "actual_function" in names

    def test_skip_dunder_methods(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""class MyClass:
    def __repr__(self):
        return f"MyClass({self.value})"
    
    def __call__(self, x):
        result = self.process(x)
        return result
    
    def regular_method(self):
        x = self.compute()
        return x
""")
            f.flush()

            config = CodeExtractionConfig(
                min_lines=2, min_chars=20, skip_dunders=True, include_methods=True
            )
            samples = extract_python_code(Path(f.name), config)

            names = [s.name for s in samples]
            # __repr__ should be skipped, __call__ kept (useful dunder)
            assert "MyClass.__repr__" not in names
            assert "MyClass.__call__" in names
            assert "MyClass.regular_method" in names

    def test_syntax_error_handling(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def broken( invalid syntax")
            f.flush()

            config = CodeExtractionConfig()
            samples = extract_python_code(Path(f.name), config)

            assert samples == []  # Should return empty, not raise


class TestLoadCodeCorpus:
    """Tests for load_code_corpus function."""

    def test_load_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python file
            py_file = Path(tmpdir) / "module.py"
            py_file.write_text('''def function_one():
    """First function."""
    x = compute_value()
    return x

def function_two(a, b):
    """Second function."""
    result = a + b
    return result
''')

            config = CodeExtractionConfig(min_lines=2, min_chars=20)
            samples = load_code_corpus(Path(tmpdir), config)

            assert len(samples) == 2
            assert all(s.language == "python" for s in samples)

    def test_exclude_test_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create regular file
            regular = Path(tmpdir) / "module.py"
            regular.write_text("""def real_function():
    x = compute()
    y = transform(x)
    return y
""")

            # Create test file
            test_file = Path(tmpdir) / "test_module.py"
            test_file.write_text("""def test_function():
    result = test()
    assert result
    return result
""")

            config = CodeExtractionConfig(min_lines=2, min_chars=20)
            samples = load_code_corpus(Path(tmpdir), config)

            files = [s.file_path for s in samples]
            assert not any("test_module" in f for f in files)

    def test_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError):
            load_code_corpus(Path("/nonexistent/path"))


class TestLoadNLCorpus:
    """Tests for load_nl_corpus function."""

    def test_load_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "texts.jsonl"
            with open(jsonl_file, "w") as f:
                f.write(
                    json.dumps({"text": "This is the first sample text that is long enough."})
                    + "\n"
                )
                f.write(
                    json.dumps({"text": "This is the second sample text that is also long."}) + "\n"
                )

            texts = load_nl_corpus(Path(tmpdir), min_chars=10)

            assert len(texts) == 2
            assert "first sample" in texts[0]

    def test_filter_by_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "texts.jsonl"
            with open(jsonl_file, "w") as f:
                f.write(json.dumps({"text": "Short."}) + "\n")
                f.write(
                    json.dumps({"text": "This is a much longer text that should pass the filter."})
                    + "\n"
                )

            texts = load_nl_corpus(Path(tmpdir), min_chars=20)

            assert len(texts) == 1
            assert "longer text" in texts[0]

    def test_custom_text_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_file = Path(tmpdir) / "data.jsonl"
            with open(jsonl_file, "w") as f:
                f.write(json.dumps({"content": "This text is in a custom field name."}) + "\n")

            texts = load_nl_corpus(Path(tmpdir), text_field="content", min_chars=10)

            assert len(texts) == 1
            assert "custom field" in texts[0]


class TestSaveCodeCorpus:
    """Tests for save_code_corpus function."""

    def test_save_and_reload(self):
        samples = [
            CodeSample(
                code="def foo(): return 1",
                language="python",
                unit_type="function",
                name="foo",
                file_path="/test.py",
                start_line=1,
                end_line=1,
            ),
            CodeSample(
                code="class Bar: pass",
                language="python",
                unit_type="class",
                name="Bar",
                file_path="/test.py",
                start_line=3,
                end_line=3,
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = Path(f.name)

        count = save_code_corpus(samples, output_path)
        assert count == 2

        # Verify the saved content
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 2

            data = json.loads(lines[0])
            assert data["text"] == "def foo(): return 1"
            assert data["language"] == "python"
            assert data["unit_type"] == "function"
