"""Cross-model validation harness for compression pairs."""

import asyncio
from dataclasses import dataclass, field

from pydantic import BaseModel

from ..utils.caching import SemanticCache
from ..utils.tokenizers import count_tokens
from .metrics import TaskType, compute_equivalence
from .models import ModelClient, ModelType


class CompressionPair(BaseModel):
    """A pair of verbose and compressed text for validation."""

    verbose: str
    compressed: str
    domain: str  # "nl" | "code" | "mixed"
    metadata: dict[str, str] | None = None


@dataclass
class ValidationResult:
    """Result of validating a single compression pair."""

    verbose_tokens: int
    compressed_tokens: int
    compression_ratio: float
    equivalence_scores: dict[ModelType, float]
    min_equivalence: float
    passed: bool

    @property
    def token_reduction_percent(self) -> float:
        """Percentage of tokens reduced."""
        return (1 - self.compression_ratio) * 100


@dataclass
class BatchValidationStats:
    """Statistics from batch validation."""

    total_pairs: int
    passed_pairs: int
    failed_pairs: int
    avg_compression_ratio: float
    avg_equivalence: float
    min_equivalence: float
    pass_rate: float
    results: list[ValidationResult] = field(default_factory=list)


# Default task prompts for validation
DEFAULT_TASK_PROMPTS: dict[TaskType, str] = {
    TaskType.QA: """Based on the following context, answer the question concisely.

Context:
{context}

Question: What are the key points mentioned?
Answer:""",
    TaskType.REASONING: """Analyze the following information and explain the main implications.

Information:
{context}

Analysis:""",
    TaskType.CODE_GEN: """Based on the following specification, write the corresponding code.

Specification:
{context}

Code:""",
}


class ValidationHarness:
    """
    Cross-model validation harness for compression pairs.

    Validates that compressed text produces equivalent model outputs
    across multiple LLMs (Claude, GPT, Gemini).
    """

    def __init__(
        self,
        models: list[ModelType] | None = None,
        equivalence_threshold: float = 0.85,
        tasks: list[TaskType] | None = None,
        cache: SemanticCache | None = None,
    ):
        """
        Initialize the validation harness.

        Args:
            models: List of models to validate against (defaults to all)
            equivalence_threshold: Minimum equivalence score to pass
            tasks: Task types to test (defaults to QA + REASONING)
            cache: Optional cache for API responses
        """
        if models is None:
            models = [ModelType.CLAUDE_SONNET, ModelType.GPT4O_MINI, ModelType.GEMINI_FLASH]

        self.models = models
        self.threshold = equivalence_threshold
        self.tasks = tasks or [TaskType.QA, TaskType.REASONING]
        self.cache = cache

        # Initialize clients
        self.clients = {m: ModelClient(m, cache=cache) for m in models}

    async def validate_pair(
        self,
        pair: CompressionPair,
        task_prompts: dict[TaskType, str] | None = None,
    ) -> ValidationResult:
        """
        Validate a single compression pair across all models and tasks.

        Args:
            pair: The compression pair to validate
            task_prompts: Optional custom prompts (use {context} placeholder)

        Returns:
            ValidationResult with equivalence scores and pass/fail status
        """
        prompts = task_prompts or DEFAULT_TASK_PROMPTS
        scores: dict[ModelType, float] = {}

        async def eval_model(model_type: ModelType) -> tuple[ModelType, float]:
            """Evaluate equivalence for a single model across all tasks."""
            client = self.clients[model_type]
            task_scores: list[float] = []

            for task_type in self.tasks:
                prompt_template = prompts.get(task_type, prompts[TaskType.QA])

                # Get outputs for verbose and compressed inputs
                verbose_prompt = prompt_template.format(context=pair.verbose)
                compressed_prompt = prompt_template.format(context=pair.compressed)

                verbose_out = await client.complete(verbose_prompt)
                compressed_out = await client.complete(compressed_prompt)

                # Compute equivalence
                score = await compute_equivalence(verbose_out, compressed_out, task_type)
                task_scores.append(score)

            # Average across tasks
            return model_type, sum(task_scores) / len(task_scores)

        # Run all models in parallel
        results = await asyncio.gather(*[eval_model(m) for m in self.models])
        scores = dict(results)

        # Calculate metrics
        verbose_tokens = count_tokens(pair.verbose)
        compressed_tokens = count_tokens(pair.compressed)
        min_equiv = min(scores.values())

        return ValidationResult(
            verbose_tokens=verbose_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / verbose_tokens if verbose_tokens > 0 else 1.0,
            equivalence_scores=scores,
            min_equivalence=min_equiv,
            passed=min_equiv >= self.threshold,
        )

    async def validate_batch(
        self,
        pairs: list[CompressionPair],
        task_prompts: dict[TaskType, str] | None = None,
        concurrency: int = 10,
    ) -> BatchValidationStats:
        """
        Validate a batch of compression pairs with concurrency control.

        Args:
            pairs: List of compression pairs to validate
            task_prompts: Optional custom prompts
            concurrency: Maximum concurrent validations

        Returns:
            BatchValidationStats with aggregated results
        """
        sem = asyncio.Semaphore(concurrency)

        async def bounded(pair: CompressionPair) -> ValidationResult:
            async with sem:
                return await self.validate_pair(pair, task_prompts)

        results = await asyncio.gather(*[bounded(p) for p in pairs])

        # Aggregate statistics
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        avg_ratio = sum(r.compression_ratio for r in results) / len(results)
        avg_equiv = sum(r.min_equivalence for r in results) / len(results)
        min_equiv = min(r.min_equivalence for r in results)

        return BatchValidationStats(
            total_pairs=len(pairs),
            passed_pairs=passed,
            failed_pairs=failed,
            avg_compression_ratio=avg_ratio,
            avg_equivalence=avg_equiv,
            min_equivalence=min_equiv,
            pass_rate=passed / len(pairs),
            results=results,
        )

    async def quick_validate(
        self,
        verbose: str,
        compressed: str,
        domain: str = "nl",
    ) -> bool:
        """
        Quick validation check for a single pair.

        Args:
            verbose: Original verbose text
            compressed: Compressed version
            domain: Content domain ("nl", "code", "mixed")

        Returns:
            True if the pair passes validation
        """
        pair = CompressionPair(verbose=verbose, compressed=compressed, domain=domain)
        result = await self.validate_pair(pair)
        return result.passed


async def validate_compression(
    verbose: str,
    compressed: str,
    domain: str = "nl",
    threshold: float = 0.85,
) -> ValidationResult:
    """
    Convenience function to validate a single compression.

    Args:
        verbose: Original text
        compressed: Compressed text
        domain: Content domain
        threshold: Equivalence threshold

    Returns:
        ValidationResult
    """
    harness = ValidationHarness(equivalence_threshold=threshold)
    pair = CompressionPair(verbose=verbose, compressed=compressed, domain=domain)
    return await harness.validate_pair(pair)
