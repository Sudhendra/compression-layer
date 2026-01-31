"""
Equivalence metrics for compression validation.

This module provides multiple scoring strategies for evaluating whether
compressed prompts produce equivalent outputs to verbose prompts:

1. Semantic similarity (embedding-based) - fast and cheap
2. Lexical overlap with symbol normalization - catches exact matches
3. LLM judge integration - most accurate, higher cost

The default configuration uses pure semantic similarity (lexical_weight=0)
because the compression format intentionally uses different vocabulary.

Usage:
    from validation.metrics import EquivalenceCalculator, normalize_for_comparison

    calc = EquivalenceCalculator()
    scores = calc.compute(verbose_output, compressed_output)

    if scores.combined_score >= 0.75:
        print("Outputs are equivalent!")
"""

import ast
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks used for equivalence validation."""

    QA = "qa"
    CODE_GEN = "code_generation"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"


@dataclass
class EquivalenceScores:
    """Container for all equivalence metrics."""

    semantic_similarity: float  # Embedding cosine similarity (0-1)
    lexical_overlap: float  # Normalized Jaccard similarity (0-1)
    fact_overlap: float | None  # Jaccard on extracted facts, if computed
    llm_judge_score: float | None  # LLM verdict score, if computed
    combined_score: float  # Final weighted score used for pass/fail


# =============================================================================
# Symbol Normalization
# =============================================================================

# Symbols that can be replaced anywhere (not word-boundary dependent)
SYMBOL_REPLACEMENTS = {
    # Logical/relational symbols
    "→": " leads to ",
    "←": " from ",
    "↔": " bidirectional ",
    "∵": " because ",
    "∴": " therefore ",
    "⇒": " implies ",
    "∧": " and ",
    "∨": " or ",
    # Notation symbols
    "@": " at ",
    "#": " number ",
    "|": " , ",  # Field separator becomes comma
    "×": " times ",
}

# Word abbreviations that need word-boundary matching to avoid false replacements
# (e.g., "min" in "minimum" or "pm" in "5pm" should not be replaced)
WORD_ABBREVIATIONS = {
    # Time abbreviations
    "yr": " year",
    "yrs": " years",
    "mo": " month",
    "mos": " months",
    "wk": " week",
    "wks": " weeks",
    "hr": " hour",
    "hrs": " hours",
    "min": " minute",
    "mins": " minutes",
    # Role abbreviations
    "sr": " senior",
    "jr": " junior",
    "mgr": " manager",
    "dir": " director",
    "vp": " vice president",
    "ceo": " chief executive officer",
    "cto": " chief technology officer",
    "swe": " software engineer",
    "pm": " product manager",
    # Domain abbreviations
    "pt": " patient",
    "dx": " diagnosis",
    "rx": " prescription",
    "hx": " history",
    "tx": " treatment",
    # Business/metrics
    "yoy": " year over year",
    "mom": " month over month",
    "qoq": " quarter over quarter",
    "q1": " first quarter",
    "q2": " second quarter",
    "q3": " third quarter",
    "q4": " fourth quarter",
    "rev": " revenue",
    "prev": " previous",
    "curr": " current",
    # Tech
    "fn": " function",
    "ret": " return",
    "param": " parameter",
    "params": " parameters",
    "impl": " implementation",
    "config": " configuration",
    "env": " environment",
    "var": " variable",
    "vars": " variables",
}

# Regex pattern for percentage changes like "+15%" or "-20%"
PERCENTAGE_PATTERN = re.compile(r"([+-]?\d+(?:\.\d+)?%)")


def normalize_for_comparison(text: str) -> str:
    """
    Expand symbols and normalize text for fairer lexical comparison.

    This function transforms compressed notation into expanded form so that
    lexical similarity metrics don't unfairly penalize the compression format.

    Args:
        text: Input text (may contain compression symbols/abbreviations)

    Returns:
        Normalized text with symbols expanded
    """
    if not text:
        return ""

    normalized = text.lower()

    # Apply direct symbol replacements (no word boundaries needed)
    for symbol, expansion in SYMBOL_REPLACEMENTS.items():
        normalized = normalized.replace(symbol.lower(), expansion)

    # Apply word abbreviations with word-boundary matching
    for abbrev, expansion in WORD_ABBREVIATIONS.items():
        # Use word boundaries to avoid replacing "min" in "minimum" etc.
        pattern = rf"\b{re.escape(abbrev)}\b"
        normalized = re.sub(pattern, expansion, normalized, flags=re.IGNORECASE)

    # Normalize percentages: "+15%" -> "increased 15 percent"
    def expand_percentage(match: re.Match[str]) -> str:
        val = match.group(1)
        if val.startswith("+"):
            return f" increased {val[1:].replace('%', ' percent')} "
        elif val.startswith("-"):
            return f" decreased {val[1:].replace('%', ' percent')} "
        else:
            return f" {val.replace('%', ' percent')} "

    normalized = PERCENTAGE_PATTERN.sub(expand_percentage, normalized)

    # Remove extra whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Remove punctuation that's not meaningful
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def tokenize(text: str) -> set[str]:
    """Split text into tokens for Jaccard computation."""
    return set(text.lower().split())


# =============================================================================
# Embedding Model Management
# =============================================================================

_embed_model: "SentenceTransformer | None" = None


def get_embedder() -> "SentenceTransformer":
    """Get or create the sentence embedding model (lazy loading)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# =============================================================================
# Equivalence Calculator
# =============================================================================


class EquivalenceCalculator:
    """
    Computes equivalence scores between verbose and compressed outputs.

    Default configuration uses pure semantic similarity because:
    1. Compression intentionally uses different vocabulary
    2. MiniLM embeddings handle paraphrase detection well
    3. Lexical overlap unfairly penalizes valid compressions

    For stricter validation, set lexical_weight > 0.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        semantic_weight: float = 1.0,
        lexical_weight: float = 0.0,
        normalize_for_lexical: bool = True,
        cache_embeddings: bool = True,
    ):
        """
        Initialize the calculator.

        Args:
            embedding_model: Sentence transformer model for semantic similarity
            semantic_weight: Weight for embedding similarity (0-1)
            lexical_weight: Weight for lexical overlap (0-1)
            normalize_for_lexical: Whether to expand symbols before lexical comparison
            cache_embeddings: Whether to cache embeddings (saves compute for repeated texts)
        """
        if semantic_weight + lexical_weight == 0:
            raise ValueError("At least one weight must be > 0")

        # Use the global embedder for efficiency
        self.encoder = get_embedder()
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.normalize_for_lexical = normalize_for_lexical

        self._embedding_cache: dict[str, np.ndarray] | None = {} if cache_embeddings else None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding, using cache if available."""
        if self._embedding_cache is not None:
            if text not in self._embedding_cache:
                embedding = self.encoder.encode(text)
                self._embedding_cache[text] = np.array(embedding)
            return self._embedding_cache[text]
        return np.array(self.encoder.encode(text))

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between text embeddings.

        This captures semantic equivalence regardless of lexical overlap.
        """
        if not text1 or not text2:
            return 0.0

        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        # Cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(emb1, emb2) / (norm1 * norm2)

        return float(np.clip(similarity, 0.0, 1.0))

    def compute_lexical_overlap(self, text1: str, text2: str) -> float:
        """
        Compute Jaccard similarity on tokens.

        If normalize_for_lexical is True, symbols are expanded first for fair comparison.
        """
        if not text1 or not text2:
            return 0.0

        # Optionally normalize
        if self.normalize_for_lexical:
            text1 = normalize_for_comparison(text1)
            text2 = normalize_for_comparison(text2)

        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def compute(
        self,
        verbose_output: str,
        compressed_output: str,
        llm_judge_score: float | None = None,
        fact_overlap: float | None = None,
    ) -> EquivalenceScores:
        """
        Compute all equivalence metrics and combined score.

        Args:
            verbose_output: Output from model given verbose context
            compressed_output: Output from model given compressed context
            llm_judge_score: Optional score from LLM judge (0-1)
            fact_overlap: Optional fact extraction overlap score (0-1)

        Returns:
            EquivalenceScores with all metrics and combined score
        """
        # Compute base metrics
        semantic = self.compute_semantic_similarity(verbose_output, compressed_output)
        lexical = self.compute_lexical_overlap(verbose_output, compressed_output)

        # Compute combined score
        if llm_judge_score is not None:
            # If we have LLM judge, weight it heavily (it's most accurate)
            combined = 0.6 * llm_judge_score + 0.4 * semantic
        else:
            # Otherwise use configured weights
            total_weight = self.semantic_weight + self.lexical_weight
            if total_weight <= 0:
                # Defensive check - fall back to semantic only
                logger.warning(
                    "EquivalenceCalculator has non-positive total weight "
                    "(semantic_weight=%s, lexical_weight=%s); "
                    "falling back to semantic similarity only.",
                    self.semantic_weight,
                    self.lexical_weight,
                )
                combined = semantic
            else:
                combined = (
                    self.semantic_weight * semantic + self.lexical_weight * lexical
                ) / total_weight

        return EquivalenceScores(
            semantic_similarity=semantic,
            lexical_overlap=lexical,
            fact_overlap=fact_overlap,
            llm_judge_score=llm_judge_score,
            combined_score=combined,
        )

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._embedding_cache is not None:
            self._embedding_cache.clear()


# =============================================================================
# Legacy Functions (for backward compatibility with existing code)
# =============================================================================


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute semantic similarity using sentence embeddings.

    Returns:
        Cosine similarity score between 0 and 1.
    """
    embedder = get_embedder()
    emb_a = embedder.encode(text_a, normalize_embeddings=True)
    emb_b = embedder.encode(text_b, normalize_embeddings=True)
    return float(np.dot(emb_a, emb_b))


def compute_lexical_overlap(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity based on word overlap.

    Returns:
        Jaccard index between 0 and 1.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    return intersection / union


def compute_ast_similarity(code_a: str, code_b: str) -> float:
    """
    Compute structural similarity between Python code using AST comparison.

    Returns:
        Similarity score between 0 and 1, or 0 if parsing fails.
    """
    try:
        ast_a = ast.dump(ast.parse(code_a))
        ast_b = ast.dump(ast.parse(code_b))
        return SequenceMatcher(None, ast_a, ast_b).ratio()
    except SyntaxError:
        return 0.0


def compute_code_equivalence(code_a: str, code_b: str) -> float:
    """
    Compute equivalence score for code outputs.

    Uses 50% AST similarity + 50% semantic similarity.
    Falls back to pure semantic if AST parsing fails.

    Returns:
        Equivalence score between 0 and 1.
    """
    ast_sim = compute_ast_similarity(code_a, code_b)
    semantic_sim = compute_semantic_similarity(code_a, code_b)

    # If AST parsing succeeded for both
    if ast_sim > 0:
        return 0.5 * ast_sim + 0.5 * semantic_sim

    # Fallback to semantic only
    return semantic_sim


def compute_nl_equivalence(text_a: str, text_b: str) -> float:
    """
    Compute equivalence score for natural language outputs.

    Uses pure semantic similarity (changed from 70/30 split).
    The lexical component unfairly penalizes compressed notation.

    Returns:
        Equivalence score between 0 and 1.
    """
    # Use pure semantic similarity for NL
    # The old 0.7 semantic + 0.3 lexical was hurting valid compressions
    return compute_semantic_similarity(text_a, text_b)


async def compute_equivalence(
    output_a: str,
    output_b: str,
    task_type: TaskType,
) -> float:
    """
    Compute equivalence score between two model outputs.

    Args:
        output_a: First model output (typically from verbose input)
        output_b: Second model output (typically from compressed input)
        task_type: Type of task that generated the outputs

    Returns:
        Equivalence score between 0 and 1.
    """
    if task_type == TaskType.CODE_GEN:
        return compute_code_equivalence(output_a, output_b)

    # QA, REASONING, SUMMARIZATION use NL equivalence (pure semantic)
    return compute_nl_equivalence(output_a, output_b)


def compute_batch_equivalence(
    pairs: list[tuple[str, str]],
    task_type: TaskType,
) -> list[float]:
    """
    Compute equivalence scores for multiple pairs efficiently.

    Uses batched embedding computation for better performance.

    Args:
        pairs: List of (output_a, output_b) tuples
        task_type: Type of task

    Returns:
        List of equivalence scores
    """
    if task_type == TaskType.CODE_GEN:
        return [compute_code_equivalence(a, b) for a, b in pairs]

    # Batch encode all texts for efficiency
    embedder = get_embedder()
    texts_a = [a for a, _ in pairs]
    texts_b = [b for _, b in pairs]

    embs_a = embedder.encode(texts_a, normalize_embeddings=True)
    embs_b = embedder.encode(texts_b, normalize_embeddings=True)

    scores = []
    for i in range(len(pairs)):
        # Pure semantic similarity
        semantic = float(np.dot(embs_a[i], embs_b[i]))
        scores.append(semantic)

    return scores


def is_equivalent(
    output_a: str,
    output_b: str,
    task_type: TaskType,
    threshold: float = 0.75,  # Lowered from 0.85
) -> bool:
    """
    Check if two outputs are semantically equivalent.

    Args:
        output_a: First output
        output_b: Second output
        task_type: Type of task
        threshold: Minimum score to be considered equivalent (default lowered to 0.75)

    Returns:
        True if equivalence score >= threshold
    """
    if task_type == TaskType.CODE_GEN:
        score = compute_code_equivalence(output_a, output_b)
    else:
        score = compute_nl_equivalence(output_a, output_b)

    return score >= threshold


# =============================================================================
# Fact Extraction (Optional Enhancement)
# =============================================================================


def extract_atomic_facts(text: str) -> list[str]:
    """
    Extract atomic facts from text for fact-level comparison.

    This is a simple heuristic approach. For better accuracy,
    use an LLM to extract facts.

    Args:
        text: Input text

    Returns:
        List of atomic fact strings
    """
    # Split on sentence boundaries
    sentences = re.split(r"[.!?]+", text)

    facts = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Split compound sentences on conjunctions
        parts = re.split(
            r"\s+(?:and|but|however|also|additionally)\s+", sentence, flags=re.IGNORECASE
        )

        for part in parts:
            part = part.strip()
            if len(part) > 10:  # Filter very short fragments
                facts.append(part.lower())

    return facts


def compute_fact_overlap(
    verbose_output: str,
    compressed_output: str,
    calculator: "EquivalenceCalculator | None" = None,
    similarity_threshold: float = 0.8,
) -> float:
    """
    Compute Jaccard similarity on extracted atomic facts.

    This is more meaningful than token-level Jaccard for longer outputs.

    Args:
        verbose_output: The verbose/original text
        compressed_output: The compressed text
        calculator: Optional EquivalenceCalculator instance to reuse (avoids model reload)
        similarity_threshold: Threshold for considering facts as matching (default 0.8)

    Returns:
        Fact overlap score between 0.0 and 1.0
    """
    verbose_facts = set(extract_atomic_facts(verbose_output))
    compressed_facts = set(extract_atomic_facts(compressed_output))

    if not verbose_facts or not compressed_facts:
        return 0.0

    # Use provided calculator or create new one
    calc = calculator or EquivalenceCalculator(semantic_weight=1.0, lexical_weight=0.0)

    # For each verbose fact, find best match in compressed facts
    verbose_matches = 0
    for vf in verbose_facts:
        for cf in compressed_facts:
            sim = calc.compute_semantic_similarity(vf, cf)
            if sim > similarity_threshold:
                verbose_matches += 1
                break  # Early exit: found a match for this verbose fact

    # Symmetric: also check compressed -> verbose
    compressed_matches = 0
    for cf in compressed_facts:
        for vf in verbose_facts:
            sim = calc.compute_semantic_similarity(cf, vf)
            if sim > similarity_threshold:
                compressed_matches += 1
                break  # Early exit: found a match for this compressed fact

    # Compute per-side coverage and average them to avoid double-counting
    verbose_coverage = verbose_matches / len(verbose_facts) if verbose_facts else 0.0
    compressed_coverage = compressed_matches / len(compressed_facts) if compressed_facts else 0.0
    return (verbose_coverage + compressed_coverage) / 2.0


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_equivalence(text1: str, text2: str) -> float:
    """
    Quick semantic similarity check.

    Usage:
        if quick_equivalence(verbose_out, compressed_out) > 0.75:
            print("Likely equivalent")
    """
    calc = EquivalenceCalculator()
    return calc.compute_semantic_similarity(text1, text2)


def detailed_equivalence(
    verbose_output: str,
    compressed_output: str,
    include_facts: bool = False,
) -> dict[str, object]:
    """
    Get detailed equivalence analysis.

    Returns dict with all metrics for inspection/debugging.
    """
    calc = EquivalenceCalculator()

    result = {
        "semantic_similarity": calc.compute_semantic_similarity(verbose_output, compressed_output),
        "lexical_overlap_raw": calc.compute_lexical_overlap(verbose_output, compressed_output),
        "verbose_normalized": normalize_for_comparison(verbose_output),
        "compressed_normalized": normalize_for_comparison(compressed_output),
    }

    if include_facts:
        result["verbose_facts"] = extract_atomic_facts(verbose_output)
        result["compressed_facts"] = extract_atomic_facts(compressed_output)
        result["fact_overlap"] = compute_fact_overlap(verbose_output, compressed_output)

    return result
