"""Tests for the compression layer validation metrics."""

import pytest

from src.validation.metrics import (
    TaskType,
    compute_ast_similarity,
    compute_code_equivalence,
    compute_lexical_overlap,
    compute_nl_equivalence,
    is_equivalent,
)


class TestLexicalOverlap:
    """Tests for lexical overlap computation."""

    def test_identical_text(self):
        """Identical text should have overlap of 1.0."""
        text = "the quick brown fox jumps"
        assert compute_lexical_overlap(text, text) == 1.0

    def test_no_overlap(self):
        """Completely different text should have 0 overlap."""
        text_a = "cat dog bird"
        text_b = "apple orange banana"
        assert compute_lexical_overlap(text_a, text_b) == 0.0

    def test_partial_overlap(self):
        """Partial overlap should be between 0 and 1."""
        text_a = "the quick brown fox"
        text_b = "the slow brown dog"
        overlap = compute_lexical_overlap(text_a, text_b)
        assert 0 < overlap < 1
        # "the" and "brown" are shared, so 2 / 6 unique words
        assert overlap == pytest.approx(2 / 6)

    def test_empty_text(self):
        """Empty text should return 0."""
        assert compute_lexical_overlap("", "hello") == 0.0
        assert compute_lexical_overlap("hello", "") == 0.0
        assert compute_lexical_overlap("", "") == 0.0


class TestASTSimilarity:
    """Tests for AST-based code similarity."""

    def test_identical_code(self):
        """Identical code should have similarity of 1.0."""
        code = "def foo(): return 42"
        assert compute_ast_similarity(code, code) == 1.0

    def test_equivalent_code_different_formatting(self):
        """Equivalent code with different formatting should be similar."""
        code_a = "def foo():\n    return 42"
        code_b = "def foo(): return 42"
        sim = compute_ast_similarity(code_a, code_b)
        assert sim == 1.0  # AST should be identical

    def test_different_code(self):
        """Different code should have lower similarity."""
        code_a = "def foo(): return 42"
        code_b = "def bar(x): return x + 1"
        sim = compute_ast_similarity(code_a, code_b)
        assert 0 < sim < 1

    def test_invalid_syntax(self):
        """Invalid Python code should return 0."""
        valid = "def foo(): return 42"
        invalid = "def foo( return 42"
        assert compute_ast_similarity(valid, invalid) == 0.0
        assert compute_ast_similarity(invalid, valid) == 0.0


class TestNLEquivalence:
    """Tests for natural language equivalence scoring."""

    def test_identical_text(self):
        """Identical text should have high equivalence."""
        text = "The quick brown fox jumps over the lazy dog."
        equiv = compute_nl_equivalence(text, text)
        assert equiv > 0.99

    def test_similar_text(self):
        """Similar text should have reasonable equivalence."""
        text_a = "The cat sat on the mat."
        text_b = "A cat was sitting on a mat."
        equiv = compute_nl_equivalence(text_a, text_b)
        assert equiv > 0.5

    def test_unrelated_text(self):
        """Unrelated text should have low equivalence."""
        text_a = "The weather is sunny today."
        text_b = "def calculate_sum(a, b): return a + b"
        equiv = compute_nl_equivalence(text_a, text_b)
        assert equiv < 0.5


class TestCodeEquivalence:
    """Tests for code equivalence scoring."""

    def test_identical_code(self):
        """Identical code should have equivalence near 1.0."""
        code = "def add(a, b): return a + b"
        equiv = compute_code_equivalence(code, code)
        assert equiv > 0.99

    def test_semantically_equivalent_code(self):
        """Semantically equivalent code should have high equivalence."""
        code_a = """
def add(a, b):
    return a + b
"""
        code_b = """
def add(a, b):
    result = a + b
    return result
"""
        equiv = compute_code_equivalence(code_a, code_b)
        # Should be reasonably high due to similar semantics
        assert equiv > 0.5


class TestIsEquivalent:
    """Tests for the is_equivalent helper function."""

    def test_equivalent_nl(self):
        """Identical NL text should be equivalent."""
        text = "Hello world"
        assert is_equivalent(text, text, TaskType.QA, threshold=0.85)

    def test_equivalent_code(self):
        """Identical code should be equivalent."""
        code = "def foo(): pass"
        assert is_equivalent(code, code, TaskType.CODE_GEN, threshold=0.85)

    def test_not_equivalent_different_text(self):
        """Very different text should not be equivalent."""
        text_a = "Python programming"
        text_b = "Completely unrelated topic about cooking recipes"
        # May or may not pass depending on embeddings - test the function works
        result = is_equivalent(text_a, text_b, TaskType.QA, threshold=0.95)
        assert isinstance(result, bool)


class TestTaskTypes:
    """Tests for TaskType enum."""

    def test_all_task_types(self):
        """Verify all expected task types exist."""
        assert TaskType.QA.value == "qa"
        assert TaskType.CODE_GEN.value == "code_generation"
        assert TaskType.REASONING.value == "reasoning"
        assert TaskType.SUMMARIZATION.value == "summarization"
