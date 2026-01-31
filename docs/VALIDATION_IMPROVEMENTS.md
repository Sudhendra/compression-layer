# Validation Improvements for LLM Compression Layer

## Executive Summary

The current validation pipeline is failing ~70% of compression pairs not because the compressions are bad, but because the **validation methodology is misaligned with the project goals**. This document outlines the problems and provides concrete implementation guidance.

---

## Problem Diagnosis

### Current State

- `data/validated/nl_pairs.jsonl` shows most `min_equivalence` scores clustered at 0.55–0.75
- Pairs with scores as high as 0.73 are marked `passed: false`
- Default threshold in `src/validation/harness.py` is 0.85

### Root Causes

| Issue | Impact | Priority |
|-------|--------|----------|
| Lexical similarity penalizes symbol notation | High | P0 |
| Temperature=0.4 introduces validation variance | High | P0 |
| Single metric for all task types | Medium | P1 |
| Threshold not empirically calibrated | Medium | P1 |
| MiniLM embeddings weak on symbol-dense text | Medium | P2 |

### What We're Actually Measuring vs. What We Should Measure

**Current approach:**
> "Does the model produce the same **output text** given compressed vs. verbose input?"

**Correct approach:**
> "Does the model extract the same **information** and reach the same **conclusions**?"

Two outputs can be semantically equivalent while having only 20% lexical overlap.

---

## Strategic Decision: Compression Format

### Options

| Path | Format | Innovation | Validation Ease |
|------|--------|------------|-----------------|
| A | Structured JSON | Low | High |
| B | Telegraphic notation (`\|`, `→`, `∴`) | High | Low |
| C | Hybrid (structured keys + telegraphic values) | Medium-High | Medium |

### Recommendation: Path C (Hybrid)

**Rationale:** The telegraphic notation tests the core research hypothesis (can LLMs reason over dense encodings?), but pure symbol-soup is hard to validate consistently. A hybrid approach preserves the innovation while enabling reliable validation.

**Example transformation:**

Verbose:
```
John Smith is a senior software engineer at Google who has been working on 
the search ranking team for the past three years. He previously worked at 
Microsoft for five years.
```

Pure telegraphic (current):
```
John Smith | sr SWE @ Google | search ranking team | 3yr | prev: Microsoft 5yr
```

Hybrid (recommended):
```
entity: John Smith | role: sr SWE | org: Google | team: search ranking | tenure: 3yr | prev: Microsoft 5yr
```

The hybrid format:
- Maintains compression benefits (dense values)
- Adds structural consistency (labeled fields)
- Enables field-coverage validation
- Still tests the core hypothesis

### If Keeping Pure Telegraphic Format

If you decide to keep the symbol-heavy format, enforce these constraints:

1. **Explicit enumeration**: Never truncate lists. Use `[N items]` placeholder if needed
2. **Symbol vocabulary**: Define and enforce a strict set of allowed symbols
3. **Abbreviation dictionary**: Maintain a reference that validators can use
4. **Consistency rules**: Same entity always compressed the same way

---

## Implementation Changes

### 1. Deterministic Validation (P0)

**File:** `src/validation/models.py`

```python
# BEFORE
async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
    # ... temperature defaults or set to 0.4

# AFTER
async def complete(
    self, 
    prompt: str, 
    max_tokens: int = 1024,
    temperature: float = 0.0  # Deterministic for validation
) -> str:
    match self.model_type:
        case ModelType.CLAUDE_SONNET:
            resp = await self._client.messages.create(
                model=self.model_type.value,
                max_tokens=max_tokens,
                temperature=temperature,  # Add this
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text
        
        case ModelType.GPT4O_MINI:
            resp = await self._client.chat.completions.create(
                model=self.model_type.value,
                max_tokens=max_tokens,
                temperature=temperature,  # Add this
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        
        case ModelType.GEMINI_FLASH:
            resp = await self._client.aio.models.generate_content(
                model=self.model_type.value,
                contents=prompt,
                generation_config={"temperature": temperature}  # Add this
            )
            return resp.text
```

### 2. LLM-as-Judge Validation (P0)

**New file:** `src/validation/llm_judge.py`

```python
"""
LLM-as-Judge validation for semantic equivalence.

This replaces/supplements embedding-based similarity with direct LLM evaluation
of whether two outputs convey equivalent information.
"""

from dataclasses import dataclass
from enum import Enum
from .models import ModelClient, ModelType

class EquivalenceVerdict(Enum):
    EQUIVALENT = "equivalent"
    PARTIAL = "partial"
    NOT_EQUIVALENT = "not_equivalent"

@dataclass
class JudgeResult:
    verdict: EquivalenceVerdict
    confidence: float  # 0.0 - 1.0
    reasoning: str
    missing_from_compressed: list[str]
    missing_from_verbose: list[str]

LLM_JUDGE_PROMPT = """You are evaluating whether two LLM outputs convey equivalent information.

## Context
A user provided context to an LLM in two forms:
1. VERBOSE: Full natural language
2. COMPRESSED: Dense notation with symbols and abbreviations

The LLM was asked to perform a task using each version. You must judge if the outputs are informationally equivalent.

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
- Stylistic differences (wording, structure) do NOT matter

Two outputs are PARTIAL if:
- Core conclusion matches but some supporting details differ
- One output includes relevant information the other omits
- Minor factual discrepancies that don't change the main point

Two outputs are NOT_EQUIVALENT if:
- Different conclusions or answers
- Contradictory facts
- One output misses critical information that changes meaning

## Response Format
Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
  "verdict": "equivalent" | "partial" | "not_equivalent",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your judgment",
  "missing_from_compressed": ["fact1", "fact2"],
  "missing_from_verbose": ["fact1", "fact2"]
}
"""

class LLMJudge:
    def __init__(self, judge_model: ModelType = ModelType.CLAUDE_SONNET):
        self.client = ModelClient(judge_model)
    
    async def evaluate(
        self,
        task_description: str,
        verbose_output: str,
        compressed_output: str
    ) -> JudgeResult:
        """Evaluate equivalence using LLM judgment."""
        
        prompt = LLM_JUDGE_PROMPT.format(
            task_description=task_description,
            verbose_output=verbose_output,
            compressed_output=compressed_output
        )
        
        response = await self.client.complete(prompt, temperature=0.0)
        
        # Parse JSON response
        import json
        try:
            # Strip any markdown code fences if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            
            data = json.loads(clean_response)
            
            return JudgeResult(
                verdict=EquivalenceVerdict(data["verdict"]),
                confidence=float(data["confidence"]),
                reasoning=data["reasoning"],
                missing_from_compressed=data.get("missing_from_compressed", []),
                missing_from_verbose=data.get("missing_from_verbose", [])
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback for malformed responses
            return JudgeResult(
                verdict=EquivalenceVerdict.NOT_EQUIVALENT,
                confidence=0.0,
                reasoning=f"Failed to parse judge response: {e}",
                missing_from_compressed=[],
                missing_from_verbose=[]
            )
    
    def verdict_to_score(self, result: JudgeResult) -> float:
        """Convert verdict to numeric score for threshold comparison."""
        base_scores = {
            EquivalenceVerdict.EQUIVALENT: 1.0,
            EquivalenceVerdict.PARTIAL: 0.7,
            EquivalenceVerdict.NOT_EQUIVALENT: 0.3
        }
        return base_scores[result.verdict] * result.confidence
```

### 3. Updated Metrics Module (P0)

**File:** `src/validation/metrics.py`

```python
"""
Equivalence metrics for compression validation.

Provides multiple scoring strategies:
1. Embedding similarity (fast, cheap)
2. Fact extraction overlap (medium cost)
3. LLM judge (expensive, most accurate)
"""

import re
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class EquivalenceScores:
    semantic_similarity: float  # Embedding cosine similarity
    fact_overlap: float | None  # Jaccard on extracted facts (if computed)
    llm_judge_score: float | None  # LLM verdict score (if computed)
    combined_score: float  # Weighted combination
    
# Symbol normalization for fair lexical comparison
SYMBOL_EXPANSIONS = {
    "→": " leads to ",
    "∵": " because ",
    "∴": " therefore ",
    "@": " at ",
    "#": " count ",
    "|": " , ",
    "+": " and ",
    "yr": " year",
    "mo": " month",
    "wk": " week",
    "pt": "patient",
    "sr": "senior",
    "SWE": "software engineer",
    "prev": "previously",
    "YoY": "year over year",
}

def normalize_for_comparison(text: str) -> str:
    """Expand symbols and normalize text for fairer comparison."""
    normalized = text.lower()
    for symbol, expansion in SYMBOL_EXPANSIONS.items():
        normalized = normalized.replace(symbol.lower(), expansion)
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

class EquivalenceCalculator:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        semantic_weight: float = 1.0,  # Changed: 100% semantic by default
        lexical_weight: float = 0.0,   # Changed: 0% lexical by default
    ):
        self.encoder = SentenceTransformer(embedding_model)
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between embeddings."""
        embeddings = self.encoder.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def compute_lexical_overlap(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity on normalized tokens."""
        # Normalize both texts (expand symbols)
        norm1 = normalize_for_comparison(text1)
        norm2 = normalize_for_comparison(text2)
        
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    def compute(
        self,
        verbose_output: str,
        compressed_output: str,
        llm_judge_score: float | None = None
    ) -> EquivalenceScores:
        """Compute combined equivalence score."""
        
        semantic = self.compute_semantic_similarity(verbose_output, compressed_output)
        lexical = self.compute_lexical_overlap(verbose_output, compressed_output)
        
        # Combined score (semantic-only by default)
        if llm_judge_score is not None:
            # If we have LLM judge, weight it heavily
            combined = 0.6 * llm_judge_score + 0.4 * semantic
        else:
            combined = (
                self.semantic_weight * semantic + 
                self.lexical_weight * lexical
            )
        
        return EquivalenceScores(
            semantic_similarity=semantic,
            fact_overlap=None,  # Computed separately if needed
            llm_judge_score=llm_judge_score,
            combined_score=combined
        )
```

### 4. Task-Specific Validation (P1)

**File:** `src/validation/task_validators.py`

```python
"""
Task-specific validation strategies.

Different tasks require different equivalence measures:
- QA: Answer correctness
- Reasoning: Conclusion + key steps
- Code: AST equivalence or test passing
- Summarization: Fact coverage
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    QA = "qa"
    REASONING = "reasoning"
    CODE_GEN = "code_generation"
    SUMMARIZATION = "summarization"

@dataclass
class TaskValidationResult:
    task_type: TaskType
    passed: bool
    score: float
    details: dict

class TaskValidator(ABC):
    @abstractmethod
    async def validate(
        self,
        verbose_output: str,
        compressed_output: str,
        task_context: dict
    ) -> TaskValidationResult:
        pass

class QAValidator(TaskValidator):
    """For QA tasks, check if answers match."""
    
    async def validate(
        self,
        verbose_output: str,
        compressed_output: str,
        task_context: dict
    ) -> TaskValidationResult:
        # Extract answers (assume last sentence or after "Answer:")
        verbose_answer = self._extract_answer(verbose_output)
        compressed_answer = self._extract_answer(compressed_output)
        
        # Semantic similarity of answers specifically
        from .metrics import EquivalenceCalculator
        calc = EquivalenceCalculator()
        similarity = calc.compute_semantic_similarity(verbose_answer, compressed_answer)
        
        return TaskValidationResult(
            task_type=TaskType.QA,
            passed=similarity >= 0.80,  # Lower threshold for answer-only
            score=similarity,
            details={
                "verbose_answer": verbose_answer,
                "compressed_answer": compressed_answer
            }
        )
    
    def _extract_answer(self, text: str) -> str:
        """Extract the answer portion from a response."""
        # Look for explicit answer markers
        markers = ["answer:", "the answer is", "result:"]
        text_lower = text.lower()
        
        for marker in markers:
            if marker in text_lower:
                idx = text_lower.index(marker) + len(marker)
                return text[idx:].strip().split('\n')[0]
        
        # Fallback: last sentence
        sentences = text.strip().split('.')
        return sentences[-1].strip() if sentences else text

class ReasoningValidator(TaskValidator):
    """For reasoning tasks, check conclusion AND key reasoning steps."""
    
    async def validate(
        self,
        verbose_output: str,
        compressed_output: str,
        task_context: dict
    ) -> TaskValidationResult:
        from .llm_judge import LLMJudge
        
        judge = LLMJudge()
        result = await judge.evaluate(
            task_description="Reasoning/analysis task - evaluate if conclusions and key reasoning steps match",
            verbose_output=verbose_output,
            compressed_output=compressed_output
        )
        
        score = judge.verdict_to_score(result)
        
        return TaskValidationResult(
            task_type=TaskType.REASONING,
            passed=score >= 0.70,  # Reasoning allows more variance
            score=score,
            details={
                "verdict": result.verdict.value,
                "reasoning": result.reasoning,
                "missing_from_compressed": result.missing_from_compressed
            }
        )

class CodeGenValidator(TaskValidator):
    """For code generation, compare AST structure or run tests."""
    
    async def validate(
        self,
        verbose_output: str,
        compressed_output: str,
        task_context: dict
    ) -> TaskValidationResult:
        import ast
        
        try:
            verbose_ast = ast.parse(verbose_output)
            compressed_ast = ast.parse(compressed_output)
            
            # Compare AST dumps (normalized)
            verbose_dump = ast.dump(verbose_ast, annotate_fields=False)
            compressed_dump = ast.dump(compressed_ast, annotate_fields=False)
            
            # Structural similarity
            if verbose_dump == compressed_dump:
                score = 1.0
            else:
                # Fallback to semantic similarity of code
                from .metrics import EquivalenceCalculator
                calc = EquivalenceCalculator()
                score = calc.compute_semantic_similarity(verbose_output, compressed_output)
            
            return TaskValidationResult(
                task_type=TaskType.CODE_GEN,
                passed=score >= 0.85,
                score=score,
                details={"ast_match": verbose_dump == compressed_dump}
            )
            
        except SyntaxError:
            # If code doesn't parse, fall back to text similarity
            from .metrics import EquivalenceCalculator
            calc = EquivalenceCalculator()
            score = calc.compute_semantic_similarity(verbose_output, compressed_output)
            
            return TaskValidationResult(
                task_type=TaskType.CODE_GEN,
                passed=score >= 0.85,
                score=score,
                details={"ast_match": False, "parse_error": True}
            )

# Factory function
def get_validator(task_type: TaskType) -> TaskValidator:
    validators = {
        TaskType.QA: QAValidator(),
        TaskType.REASONING: ReasoningValidator(),
        TaskType.CODE_GEN: CodeGenValidator(),
    }
    return validators.get(task_type, ReasoningValidator())  # Default to reasoning
```

### 5. Updated Validation Harness (P1)

**File:** `src/validation/harness.py` (key changes)

```python
# Add to imports
from .llm_judge import LLMJudge
from .task_validators import get_validator, TaskType
from .metrics import EquivalenceCalculator

class ValidationHarness:
    def __init__(
        self,
        models: list[ModelType],
        equivalence_threshold: float = 0.75,  # Lowered from 0.85
        tasks: list[TaskType] | None = None,
        use_llm_judge: bool = True,  # New option
    ):
        self.clients = {m: ModelClient(m) for m in models}
        self.threshold = equivalence_threshold
        self.tasks = tasks or [TaskType.QA, TaskType.REASONING]
        self.use_llm_judge = use_llm_judge
        self.llm_judge = LLMJudge() if use_llm_judge else None
        self.metrics = EquivalenceCalculator(
            semantic_weight=1.0,  # Pure semantic
            lexical_weight=0.0
        )
    
    async def validate_pair(
        self,
        pair: CompressionPair,
        task_prompts: dict[TaskType, str]
    ) -> ValidationResult:
        """Validate with improved methodology."""
        
        all_scores = {}
        
        for model_type, client in self.clients.items():
            model_scores = []
            
            for task_type in self.tasks:
                prompt_template = task_prompts[task_type]
                
                # Get outputs with temperature=0
                verbose_prompt = prompt_template.format(context=pair.verbose)
                compressed_prompt = prompt_template.format(context=pair.compressed)
                
                verbose_output = await client.complete(verbose_prompt, temperature=0.0)
                compressed_output = await client.complete(compressed_prompt, temperature=0.0)
                
                # Get LLM judge score if enabled
                llm_score = None
                if self.use_llm_judge:
                    judge_result = await self.llm_judge.evaluate(
                        task_description=f"{task_type.value} task",
                        verbose_output=verbose_output,
                        compressed_output=compressed_output
                    )
                    llm_score = self.llm_judge.verdict_to_score(judge_result)
                
                # Compute combined score
                scores = self.metrics.compute(
                    verbose_output,
                    compressed_output,
                    llm_judge_score=llm_score
                )
                
                model_scores.append(scores.combined_score)
            
            all_scores[model_type] = sum(model_scores) / len(model_scores)
        
        min_score = min(all_scores.values())
        
        return ValidationResult(
            verbose_tokens=count_tokens(pair.verbose),
            compressed_tokens=count_tokens(pair.compressed),
            compression_ratio=count_tokens(pair.compressed) / count_tokens(pair.verbose),
            equivalence_scores=all_scores,
            min_equivalence=min_score,
            passed=min_score >= self.threshold,
            task_type=self.tasks[0]  # Primary task
        )
```

---

## Calibration Process

### Step 1: Human Baseline (Required Before Tuning Thresholds)

1. Sample 50 compression pairs from `data/seed/nl_pairs.jsonl`
2. For each pair, run through validation to get model outputs
3. Have a human judge: "Are these outputs equivalent?" (Yes/Partial/No)
4. Record the `min_equivalence` score for each
5. Find the threshold that best separates human "Yes" from "No"

**Expected outcome:** The threshold will likely be 0.70-0.75, not 0.85

### Step 2: Domain-Specific Thresholds

After calibration, you may find different domains need different thresholds:

| Domain | Suggested Threshold | Rationale |
|--------|---------------------|-----------|
| NL (factual) | 0.75 | Facts must match |
| NL (conversational) | 0.70 | Style varies more |
| Code | 0.85 | Logic must match exactly |
| Mixed | 0.75 | Balance |

### Step 3: Iterative Refinement

1. Run validation with new methodology
2. Manually review failures at score 0.65-0.75
3. If most are actually equivalent, lower threshold
4. If many are truly different, investigate compression quality

---

## Cost Implications

| Validation Method | Cost per Pair | When to Use |
|-------------------|---------------|-------------|
| Embedding only | ~$0 | Quick filtering, dev iteration |
| Embedding + LLM judge | ~$0.01-0.02 | Final validation, production |
| Full multi-model + judge | ~$0.05-0.10 | Gold standard, calibration |

**Recommendation:** Use embedding-only for rapid iteration during development, then run LLM-judge validation as a final pass before training.

---

## Updated Compression Prompt (Optional)

If you decide to move to hybrid format, here's the updated prompt:

**File:** `src/generation/prompts/compress_nl.txt`

```
You are a semantic compression engine. Compress the input into minimal tokens while preserving ALL information.

FORMAT: Use labeled fields with dense values
- field: value | field: value | field: value
- Use standard abbreviations in values: sr (senior), yr (year), mo (month), prev (previous)
- Use symbols in values: → (leads to), + (and/with), @ (at/location)
- Enumerate lists explicitly or use [N items] placeholder

RULES:
1. Every fact in input MUST appear in output
2. No information loss - compression must be losslessly interpretable  
3. Consistent field names for same concepts
4. Expand any ambiguous abbreviations on first use

EXAMPLES:

Input: "John Smith is a senior software engineer at Google who has been working on the search ranking team for the past three years. He previously worked at Microsoft for five years."
Output: entity: John Smith | role: sr software engineer | org: Google | team: search ranking | tenure: 3yr | prev: Microsoft 5yr

Input: "The quarterly revenue increased by 15% compared to the same period last year, primarily driven by strong performance in the cloud services division which saw a 25% growth."
Output: metric: Q revenue | change: +15% YoY | driver: cloud services +25%

Input: "The patient presents with persistent headaches occurring three times per week for the past month, accompanied by sensitivity to light and occasional nausea."
Output: subject: patient | symptom: headaches 3x/wk × 1mo | associated: photosensitivity + nausea (intermittent)

INPUT:
{input}

OUTPUT:
```

---

## Quick Reference: What to Change Now

### Immediate (P0)
1. Set `temperature=0.0` in all validation model calls
2. Change metric weights to `semantic=1.0, lexical=0.0`
3. Lower threshold from 0.85 to 0.75

### Short-term (P1)
4. Implement `LLMJudge` class for more accurate equivalence
5. Add symbol normalization to lexical comparison
6. Run 50-pair human calibration

### Medium-term (P2)
7. Implement task-specific validators
8. Consider hybrid compression format
9. Upgrade embedding model (MPNet or E5)

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/validation/models.py` | Add `temperature` param, default to 0.0 |
| `src/validation/metrics.py` | Add symbol normalization, change default weights |
| `src/validation/harness.py` | Lower threshold, integrate LLM judge |
| `src/validation/llm_judge.py` | **New file** - LLM-as-judge implementation |
| `src/validation/task_validators.py` | **New file** - Task-specific validation |
| `configs/validation.yaml` | Update threshold, add `use_llm_judge` flag |

---

## Success Metrics

After implementing these changes, you should see:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Pass rate | ~30% | ~60-70% |
| Score distribution | 0.55-0.75 | 0.70-0.90 |
| False negatives | High | Low |
| Validation variance | High (temp=0.4) | Low (temp=0) |

If pass rate doesn't improve significantly, the issue is compression quality, not validation methodology.
