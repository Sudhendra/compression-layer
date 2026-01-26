"""Generation module for compression pair synthesis."""

from .seed_generator import (
    GeneratedPair,
    GenerationResult,
    SeedGenerator,
    generate_seed_pairs,
)

__all__ = [
    "GeneratedPair",
    "GenerationResult",
    "SeedGenerator",
    "generate_seed_pairs",
]
