"""Equivalence scoring metrics for compression validation."""

import ast
from difflib import SequenceMatcher
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class TaskType(Enum):
    """Types of tasks used for equivalence validation."""

    QA = "qa"
    CODE_GEN = "code_generation"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"


_embed_model: "SentenceTransformer | None" = None


def get_embedder() -> "SentenceTransformer":
    """Get or create the sentence embedding model (lazy loading)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


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

    Uses 70% semantic similarity + 30% lexical overlap.
    Per CLAUDE.md: NL: 0.7 x cosine(embeddings) + 0.3 x Jaccard

    Returns:
        Equivalence score between 0 and 1.
    """
    semantic = compute_semantic_similarity(text_a, text_b)
    lexical = compute_lexical_overlap(text_a, text_b)
    return 0.7 * semantic + 0.3 * lexical


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

    # QA, REASONING, SUMMARIZATION use NL equivalence
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
    for i, (text_a, text_b) in enumerate(pairs):
        semantic = float(np.dot(embs_a[i], embs_b[i]))
        lexical = compute_lexical_overlap(text_a, text_b)
        scores.append(0.7 * semantic + 0.3 * lexical)

    return scores


def is_equivalent(
    output_a: str,
    output_b: str,
    task_type: TaskType,
    threshold: float = 0.85,
) -> bool:
    """
    Check if two outputs are semantically equivalent.

    Args:
        output_a: First output
        output_b: Second output
        task_type: Type of task
        threshold: Minimum score to be considered equivalent

    Returns:
        True if equivalence score >= threshold
    """
    if task_type == TaskType.CODE_GEN:
        score = compute_code_equivalence(output_a, output_b)
    else:
        score = compute_nl_equivalence(output_a, output_b)

    return score >= threshold
