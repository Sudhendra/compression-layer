"""Domain classifier for routing inputs to appropriate compression strategies.

Classifies text as 'nl' (natural language), 'code', or 'mixed' based on
structural patterns and heuristics.
"""

import re
from dataclasses import dataclass
from enum import Enum


class Domain(str, Enum):
    """Content domain types."""

    NL = "nl"
    CODE = "code"
    MIXED = "mixed"


@dataclass
class ClassificationResult:
    """Result of domain classification."""

    domain: Domain
    confidence: float
    code_ratio: float
    indicators: list[str]


# Patterns that strongly indicate code
CODE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Python
    ("python_def", re.compile(r"\bdef\s+\w+\s*\(")),
    ("python_class", re.compile(r"\bclass\s+\w+\s*[:\(]")),
    ("python_import", re.compile(r"^\s*(import|from)\s+\w+", re.MULTILINE)),
    ("python_decorator", re.compile(r"^\s*@\w+", re.MULTILINE)),
    # JavaScript/TypeScript
    ("js_function", re.compile(r"\bfunction\s+\w+\s*\(")),
    ("js_const_let", re.compile(r"\b(const|let|var)\s+\w+\s*=")),
    ("js_arrow", re.compile(r"=>\s*[{\(]?")),
    ("js_async", re.compile(r"\basync\s+(function|def|\()")),
    # General code patterns
    ("curly_blocks", re.compile(r"\{\s*\n.*\n\s*\}", re.DOTALL)),
    ("semicolon_lines", re.compile(r";\s*$", re.MULTILINE)),
    ("type_annotation", re.compile(r":\s*(int|str|float|bool|list|dict|Any|None)\b")),
    ("return_statement", re.compile(r"^\s*return\s+", re.MULTILINE)),
    ("if_else_block", re.compile(r"^\s*(if|else|elif|else if)\s*[\(:]", re.MULTILINE)),
]

# Patterns that indicate natural language
NL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("sentence_end", re.compile(r"[.!?]\s+[A-Z]")),
    ("paragraph", re.compile(r"\n\n[A-Z]")),
    ("articles", re.compile(r"\b(the|a|an)\s+\w+", re.IGNORECASE)),
    ("conjunctions", re.compile(r"\b(and|but|or|however|therefore)\b", re.IGNORECASE)),
    ("questions", re.compile(r"\?$", re.MULTILINE)),
]


class DomainClassifier:
    """Classify text content into domains for compression routing."""

    def __init__(
        self,
        code_threshold: float = 0.3,
        high_code_threshold: float = 0.7,
    ):
        """
        Initialize the domain classifier.

        Args:
            code_threshold: Minimum code ratio to classify as 'mixed'
            high_code_threshold: Minimum code ratio to classify as 'code'
        """
        self.code_threshold = code_threshold
        self.high_code_threshold = high_code_threshold

    def classify(self, text: str) -> Domain:
        """
        Classify text into a domain.

        Args:
            text: Input text to classify

        Returns:
            Domain enum (NL, CODE, or MIXED)
        """
        result = self.classify_detailed(text)
        return result.domain

    def classify_detailed(self, text: str) -> ClassificationResult:
        """
        Classify text with detailed analysis.

        Args:
            text: Input text to classify

        Returns:
            ClassificationResult with domain, confidence, and indicators
        """
        if not text or not text.strip():
            return ClassificationResult(
                domain=Domain.NL,
                confidence=0.0,
                code_ratio=0.0,
                indicators=[],
            )

        lines = text.strip().split("\n")
        total_lines = len(lines)

        # Count lines matching code patterns
        code_indicators: list[str] = []
        code_line_indices: set[int] = set()

        for pattern_name, pattern in CODE_PATTERNS:
            for i, line in enumerate(lines):
                if pattern.search(line):
                    code_line_indices.add(i)
                    if pattern_name not in code_indicators:
                        code_indicators.append(pattern_name)

        # Also check for code blocks (multi-line patterns)
        full_text_code_matches = sum(1 for _, pattern in CODE_PATTERNS if pattern.search(text))

        # Calculate code ratio based on lines
        code_lines = len(code_line_indices)
        code_ratio = code_lines / max(total_lines, 1)

        # Adjust based on full-text pattern density
        pattern_density = full_text_code_matches / len(CODE_PATTERNS)
        adjusted_ratio = 0.7 * code_ratio + 0.3 * pattern_density

        # Check for NL indicators
        nl_matches = sum(1 for _, pattern in NL_PATTERNS if pattern.search(text))
        has_strong_nl = nl_matches >= 3

        # Determine domain
        if adjusted_ratio > self.high_code_threshold:
            domain = Domain.CODE
            confidence = min(adjusted_ratio + 0.1, 1.0)
        elif adjusted_ratio > self.code_threshold:
            domain = Domain.MIXED
            confidence = 0.6 + (adjusted_ratio - self.code_threshold) * 0.5
        else:
            domain = Domain.NL
            confidence = 1.0 - adjusted_ratio if has_strong_nl else 0.7

        return ClassificationResult(
            domain=domain,
            confidence=round(confidence, 3),
            code_ratio=round(adjusted_ratio, 3),
            indicators=code_indicators[:5],  # Top 5 indicators
        )

    def is_code(self, text: str) -> bool:
        """Quick check if text is primarily code."""
        return self.classify(text) == Domain.CODE

    def is_mixed(self, text: str) -> bool:
        """Quick check if text contains both code and NL."""
        return self.classify(text) == Domain.MIXED
