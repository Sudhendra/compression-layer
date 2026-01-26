"""Validation module for cross-model equivalence testing."""

from .harness import (
    BatchValidationStats,
    CompressionPair,
    ValidationHarness,
    ValidationResult,
    validate_compression,
)
from .metrics import (
    TaskType,
    compute_code_equivalence,
    compute_equivalence,
    compute_nl_equivalence,
    is_equivalent,
)
from .models import ModelClient, ModelType, model_type_from_string

__all__ = [
    "ValidationHarness",
    "ValidationResult",
    "BatchValidationStats",
    "CompressionPair",
    "validate_compression",
    "TaskType",
    "compute_equivalence",
    "compute_nl_equivalence",
    "compute_code_equivalence",
    "is_equivalent",
    "ModelClient",
    "ModelType",
    "model_type_from_string",
]
