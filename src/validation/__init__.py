"""Validation module for cross-model equivalence testing."""

from .harness import (
    BatchValidationStats,
    CompressionPair,
    ValidationHarness,
    ValidationResult,
    validate_compression,
)
from .llm_judge import (
    EquivalenceVerdict,
    JudgeResult,
    LLMJudge,
    quick_judge,
)
from .metrics import (
    EquivalenceCalculator,
    EquivalenceScores,
    TaskType,
    compute_code_equivalence,
    compute_equivalence,
    compute_nl_equivalence,
    is_equivalent,
    normalize_for_comparison,
    quick_equivalence,
)
from .models import ModelClient, ModelType, model_type_from_string

__all__ = [
    # Harness
    "ValidationHarness",
    "ValidationResult",
    "BatchValidationStats",
    "CompressionPair",
    "validate_compression",
    # LLM Judge
    "LLMJudge",
    "JudgeResult",
    "EquivalenceVerdict",
    "quick_judge",
    # Metrics
    "EquivalenceCalculator",
    "EquivalenceScores",
    "TaskType",
    "compute_equivalence",
    "compute_nl_equivalence",
    "compute_code_equivalence",
    "is_equivalent",
    "normalize_for_comparison",
    "quick_equivalence",
    # Models
    "ModelClient",
    "ModelType",
    "model_type_from_string",
]
