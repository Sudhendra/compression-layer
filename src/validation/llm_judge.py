"""
LLM-as-Judge validation for semantic equivalence.

This module provides LLM-based evaluation of whether two outputs (from verbose vs
compressed prompts) convey equivalent information. This is more accurate than
embedding similarity for the compression validation use case.

Usage:
    from validation.llm_judge import LLMJudge, EquivalenceVerdict

    judge = LLMJudge()
    result = await judge.evaluate(
        task_description="Answer the question based on the context",
        verbose_output="The capital of France is Paris.",
        compressed_output="Paris is France's capital."
    )

    if result.verdict == EquivalenceVerdict.EQUIVALENT:
        print("Outputs are equivalent!")

    score = judge.verdict_to_score(result)  # 0.0 - 1.0
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum

from ..utils.caching import SemanticCache
from .models import ModelClient, ModelType

logger = logging.getLogger(__name__)


class EquivalenceVerdict(Enum):
    """Possible verdicts from the LLM judge."""

    EQUIVALENT = "equivalent"
    PARTIAL = "partial"
    NOT_EQUIVALENT = "not_equivalent"


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""

    verdict: EquivalenceVerdict
    confidence: float  # 0.0 - 1.0
    reasoning: str
    missing_from_compressed: list[str]  # Facts in verbose output missing from compressed
    missing_from_verbose: list[str]  # Facts in compressed output missing from verbose


# The prompt template for the LLM judge
LLM_JUDGE_PROMPT = """You are evaluating whether two LLM outputs convey equivalent information.

## Context
A user provided context to an LLM in two forms:
1. VERBOSE: Full natural language
2. COMPRESSED: Dense notation with symbols and abbreviations

The LLM was asked to perform the same task using each version. You must judge if the outputs are informationally equivalent.

## Task Given to LLM
{task_description}

## Output from VERBOSE context
{verbose_output}

## Output from COMPRESSED context
{compressed_output}

## Evaluation Criteria

Two outputs are EQUIVALENT if:
- They reach the same conclusion/answer
- They reference the same key facts
- Any reasoning steps lead to compatible conclusions
- Stylistic differences (wording, formatting, structure) do NOT matter
- Minor phrasing variations are acceptable

Two outputs are PARTIAL if:
- Core conclusion/answer matches but some supporting details differ
- One output includes relevant information the other omits (but main point preserved)
- Minor factual discrepancies that don't change the fundamental meaning

Two outputs are NOT_EQUIVALENT if:
- Different conclusions or answers
- Contradictory facts on important points
- One output misses critical information that changes the meaning
- The outputs would lead to different decisions/actions

## Important Notes
- Focus on INFORMATION CONTENT, not style or formatting
- Ignore differences in punctuation, capitalization, or structure
- Two responses can use completely different words and still be equivalent
- Be generous with stylistic variation, strict with factual accuracy

## Response Format
Respond with ONLY a valid JSON object. No markdown code fences, no explanation outside the JSON:
{{
  "verdict": "equivalent" | "partial" | "not_equivalent",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation of your judgment>",
  "missing_from_compressed": ["<fact1>", "<fact2>"],
  "missing_from_verbose": ["<fact1>", "<fact2>"]
}}

If no facts are missing, use empty arrays: []
"""


class LLMJudge:
    """
    Uses an LLM to judge semantic equivalence between two outputs.

    This is more accurate than embedding similarity for compression validation
    because it can understand that different phrasings convey the same meaning.

    Cost tracking is handled via ModelClient which logs all API calls.
    """

    def __init__(
        self,
        judge_model: ModelType = ModelType.CLAUDE_SONNET,
        fallback_model: ModelType | None = ModelType.GPT4O_MINI,
        cache: SemanticCache | None = None,
    ):
        """
        Initialize the LLM judge.

        Args:
            judge_model: Primary model to use for judging
            fallback_model: Backup model if primary fails (optional)
            cache: Optional cache for API responses (saves cost on repeated evals)
        """
        # ModelClient already handles cost tracking via get_cost_tracker()
        self.primary_client = ModelClient(judge_model, cache=cache, operation="llm_judge")
        self.fallback_client = (
            ModelClient(fallback_model, cache=cache, operation="llm_judge_fallback")
            if fallback_model
            else None
        )
        self.judge_model = judge_model
        self.cache = cache

    async def evaluate(
        self,
        task_description: str,
        verbose_output: str,
        compressed_output: str,
        max_retries: int = 2,
    ) -> JudgeResult:
        """
        Evaluate equivalence between verbose and compressed outputs.

        Args:
            task_description: Description of the task the LLM was asked to perform
            verbose_output: Output when given verbose/full context
            compressed_output: Output when given compressed context
            max_retries: Number of retries on parse failure

        Returns:
            JudgeResult with verdict, confidence, and analysis
        """
        prompt = LLM_JUDGE_PROMPT.format(
            task_description=task_description,
            verbose_output=verbose_output,
            compressed_output=compressed_output,
        )

        last_error: Exception = Exception("No attempts made")

        # Try primary client with retries
        for attempt in range(max_retries + 1):
            try:
                response = await self.primary_client.complete(
                    prompt,
                    max_tokens=1024,
                    temperature=0.0,  # Deterministic judging
                )
                return self._parse_response(response)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Parse errors - retry or try fallback
                last_error = e
                logger.warning(f"Judge parse error (attempt {attempt + 1}): {e}")

            except Exception as e:
                # API/network errors - log and retry
                last_error = e
                logger.warning(f"Judge API error (attempt {attempt + 1}): {e}")

            # Try fallback if available on last attempt
            if attempt == max_retries and self.fallback_client:
                try:
                    logger.info("Trying fallback model for LLM judge")
                    response = await self.fallback_client.complete(
                        prompt, max_tokens=1024, temperature=0.0
                    )
                    return self._parse_response(response)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    last_error = fallback_error

        # Return conservative failure result (all retries exhausted)
        return JudgeResult(
            verdict=EquivalenceVerdict.PARTIAL,
            confidence=0.5,
            reasoning=f"Failed to parse judge response after {max_retries + 1} attempts: {last_error}",
            missing_from_compressed=[],
            missing_from_verbose=[],
        )

    def _parse_response(self, response: str) -> JudgeResult:
        """Parse the JSON response from the judge LLM."""
        # Clean up response - remove markdown code fences if present
        clean_response = response.strip()

        if clean_response.startswith("```"):
            # Remove opening fence
            lines = clean_response.split("\n")
            start_idx = 1 if lines[0].startswith("```") else 0
            # Find closing fence
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            clean_response = "\n".join(lines[start_idx:end_idx])

        # Remove any "json" language identifier
        if clean_response.startswith("json"):
            clean_response = clean_response[4:].strip()

        # Parse JSON
        data = json.loads(clean_response)

        # Validate and extract fields
        verdict_str = data["verdict"].lower()
        if verdict_str not in ["equivalent", "partial", "not_equivalent"]:
            raise ValueError(f"Invalid verdict: {verdict_str}")

        confidence = float(data["confidence"])
        if not 0.0 <= confidence <= 1.0:
            confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range

        return JudgeResult(
            verdict=EquivalenceVerdict(verdict_str),
            confidence=confidence,
            reasoning=str(data.get("reasoning", "")),
            missing_from_compressed=list(data.get("missing_from_compressed", [])),
            missing_from_verbose=list(data.get("missing_from_verbose", [])),
        )

    def verdict_to_score(self, result: JudgeResult) -> float:
        """
        Convert a JudgeResult to a numeric score for threshold comparison.

        The score combines the verdict category with the confidence level.

        Returns:
            Float between 0.0 and 1.0
        """
        base_scores = {
            EquivalenceVerdict.EQUIVALENT: 1.0,
            EquivalenceVerdict.PARTIAL: 0.65,  # Partial is closer to passing
            EquivalenceVerdict.NOT_EQUIVALENT: 0.25,
        }

        base = base_scores[result.verdict]

        # Modulate by confidence using continuous interpolation
        # At confidence=1.0: returns base score
        # At confidence=0.0: returns 0.5 (maximum uncertainty)
        # This provides a smooth, continuous function with no discontinuities
        return base * result.confidence + 0.5 * (1 - result.confidence)

    async def batch_evaluate(
        self,
        pairs: list[tuple[str, str, str]],  # (task_desc, verbose, compressed)
        concurrency: int = 5,
    ) -> list[JudgeResult]:
        """
        Evaluate multiple pairs concurrently.

        Args:
            pairs: List of (task_description, verbose_output, compressed_output) tuples
            concurrency: Max concurrent evaluations

        Returns:
            List of JudgeResult in same order as input pairs
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_with_limit(task_desc: str, verbose: str, compressed: str) -> JudgeResult:
            async with semaphore:
                return await self.evaluate(task_desc, verbose, compressed)

        tasks = [
            evaluate_with_limit(task_desc, verbose, compressed)
            for task_desc, verbose, compressed in pairs
        ]

        return await asyncio.gather(*tasks)


# Convenience function for quick evaluation
async def quick_judge(
    verbose_output: str, compressed_output: str, task_description: str = "General task"
) -> tuple[EquivalenceVerdict, float]:
    """
    Quick evaluation returning just verdict and score.

    Usage:
        verdict, score = await quick_judge(verbose_out, compressed_out)
    """
    judge = LLMJudge()
    result = await judge.evaluate(task_description, verbose_output, compressed_output)
    return result.verdict, judge.verdict_to_score(result)
