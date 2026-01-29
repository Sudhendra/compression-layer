# Universal LLM Compression Layer: Technical Implementation Plan

## Project Overview

**Objective**: Build a model-agnostic semantic compression layer that reduces token count for LLM inputs (memories, code, context) while preserving task performance across Claude, GPT, and Gemini.

**Target Metrics**:
- 40-60% token reduction
- >95% task equivalence retention
- <50ms compression latency (inference)
- Cross-model transfer rate >90%

---

## Architecture: Hybrid Local + Cloud

| Component | Platform | Use Case |
|-----------|----------|----------|
| Seed generation | Claude API | Initial compression pairs |
| Validation | Claude/GPT/Gemini APIs | Cross-model equivalence |
| Local iteration | M4 Pro + MLX | Fast experiments, inference |
| Production training | Tinker | Qwen3-8B fine-tuning |

---

## Phase 1: Infrastructure & Validation Harness

### 1.1 Project Structure

```
compression-layer/
├── src/
│   ├── validation/
│   │   ├── harness.py          # Core validation logic
│   │   ├── metrics.py          # Equivalence scoring
│   │   └── models.py           # API wrappers (Claude, GPT, Gemini)
│   ├── generation/
│   │   ├── seed_generator.py   # Initial pair generation
│   │   ├── code_compressor.py  # Code-specific compression
│   │   ├── nl_compressor.py    # NL-specific compression
│   │   └── prompts/
│   │       ├── compress_nl.txt
│   │       ├── compress_code.txt
│   │       └── validate_equivalence.txt
│   ├── training/
│   │   ├── train_tinker.py     # Tinker cloud training
│   │   ├── train_mlx.py        # Local MLX training
│   │   ├── dataset.py          # Dataset builder
│   │   └── config/
│   │       └── training.yaml
│   ├── inference/
│   │   ├── compressor.py       # Production inference (MLX)
│   │   ├── domain_classifier.py
│   │   └── token_budget.py
│   └── utils/
│       ├── tokenizers.py       # Multi-tokenizer utils
│       ├── caching.py          # Semantic cache
│       └── costs.py            # API cost tracking
├── data/
│   ├── raw/                    # Source corpora
│   ├── seed/                   # Initial generated pairs
│   ├── validated/              # Cross-model validated pairs
│   └── final/                  # Training-ready dataset
├── models/
│   └── adapters/               # LoRA adapters from Tinker
├── configs/
│   ├── generation.yaml
│   ├── validation.yaml
│   └── training.yaml
├── scripts/
│   ├── generate_seed.py
│   ├── validate_batch.py
│   ├── train_tinker.py
│   ├── train_local.py
│   └── evaluate.py
├── tests/
├── pyproject.toml
├── Makefile
└── README.md
```

### 1.2 Dependencies

```toml
# pyproject.toml
[project]
name = "llm-compression-layer"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # APIs
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    "google-genai>=1.0.0",
    "tinker-sdk>=0.1.0",
    
    # Local inference (MLX)
    "mlx>=0.20.0",
    "mlx-lm>=0.20.0",
    
    # Data & utilities
    "datasets>=3.0.0",
    "sentence-transformers>=3.0.0",
    "tiktoken>=0.7.0",
    "rich>=13.0.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "httpx>=0.27.0",
    "diskcache>=5.6.0",
    "numpy>=2.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.6.0",
    "mypy>=1.11.0",
]
```

### 1.3 Validation Harness Implementation

```python
# src/validation/harness.py
from dataclasses import dataclass
from enum import Enum
import asyncio
from .models import ModelClient, ModelType
from .metrics import compute_equivalence

class TaskType(Enum):
    QA = "qa"
    CODE_GEN = "code_generation"
    REASONING = "reasoning"

@dataclass
class ValidationResult:
    verbose_tokens: int
    compressed_tokens: int
    compression_ratio: float
    equivalence_scores: dict[ModelType, float]
    min_equivalence: float
    passed: bool

@dataclass
class CompressionPair:
    verbose: str
    compressed: str
    domain: str  # "nl" | "code" | "mixed"
    metadata: dict | None = None

class ValidationHarness:
    def __init__(
        self,
        models: list[ModelType],
        equivalence_threshold: float = 0.85,
        tasks: list[TaskType] | None = None
    ):
        self.clients = {m: ModelClient(m) for m in models}
        self.threshold = equivalence_threshold
        self.tasks = tasks or [TaskType.QA, TaskType.REASONING]
    
    async def validate_pair(
        self,
        pair: CompressionPair,
        task_prompts: dict[TaskType, str]
    ) -> ValidationResult:
        scores = {}
        
        async def eval_model(model_type: ModelType) -> tuple[ModelType, float]:
            client = self.clients[model_type]
            task_scores = []
            
            for task_type in self.tasks:
                prompt = task_prompts[task_type]
                verbose_out = await client.complete(prompt + pair.verbose)
                compressed_out = await client.complete(prompt + pair.compressed)
                score = await compute_equivalence(verbose_out, compressed_out, task_type)
                task_scores.append(score)
            
            return model_type, sum(task_scores) / len(task_scores)
        
        results = await asyncio.gather(*[eval_model(m) for m in self.clients])
        scores = dict(results)
        
        verbose_tokens = self._count_tokens(pair.verbose)
        compressed_tokens = self._count_tokens(pair.compressed)
        min_equiv = min(scores.values())
        
        return ValidationResult(
            verbose_tokens=verbose_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / verbose_tokens,
            equivalence_scores=scores,
            min_equivalence=min_equiv,
            passed=min_equiv >= self.threshold,
        )
    
    async def validate_batch(
        self,
        pairs: list[CompressionPair],
        task_prompts: dict[TaskType, str],
        concurrency: int = 10
    ) -> list[ValidationResult]:
        sem = asyncio.Semaphore(concurrency)
        async def bounded(pair):
            async with sem:
                return await self.validate_pair(pair, task_prompts)
        return await asyncio.gather(*[bounded(p) for p in pairs])
    
    def _count_tokens(self, text: str) -> int:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
```

```python
# src/validation/metrics.py
from sentence_transformers import SentenceTransformer
import numpy as np
from .harness import TaskType

_embed_model = None

def get_embedder():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model

async def compute_equivalence(output_a: str, output_b: str, task_type: TaskType) -> float:
    if task_type == TaskType.CODE_GEN:
        return compute_code_equivalence(output_a, output_b)
    
    embedder = get_embedder()
    emb_a = embedder.encode(output_a, normalize_embeddings=True)
    emb_b = embedder.encode(output_b, normalize_embeddings=True)
    cosine_sim = np.dot(emb_a, emb_b)
    lexical = compute_lexical_overlap(output_a, output_b)
    return 0.7 * cosine_sim + 0.3 * lexical

def compute_code_equivalence(code_a: str, code_b: str) -> float:
    import ast
    try:
        ast_a = ast.dump(ast.parse(code_a))
        ast_b = ast.dump(ast.parse(code_b))
        from difflib import SequenceMatcher
        ast_sim = SequenceMatcher(None, ast_a, ast_b).ratio()
    except SyntaxError:
        ast_sim = 0.0
    
    embedder = get_embedder()
    emb_a = embedder.encode(code_a, normalize_embeddings=True)
    emb_b = embedder.encode(code_b, normalize_embeddings=True)
    semantic_sim = np.dot(emb_a, emb_b)
    return 0.5 * ast_sim + 0.5 * semantic_sim

def compute_lexical_overlap(a: str, b: str) -> float:
    tokens_a, tokens_b = set(a.lower().split()), set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
```

```python
# src/validation/models.py
from enum import Enum

class ModelType(Enum):
    CLAUDE_SONNET = "claude-sonnet-4-20250514"
    GPT4O_MINI = "gpt-4o-mini"
    GEMINI_FLASH = "gemini-2.0-flash"

class ModelClient:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self._client = self._init_client()
    
    def _init_client(self):
        match self.model_type:
            case ModelType.CLAUDE_SONNET:
                from anthropic import AsyncAnthropic
                return AsyncAnthropic()
            case ModelType.GPT4O_MINI:
                from openai import AsyncOpenAI
                return AsyncOpenAI()
            case ModelType.GEMINI_FLASH:
                from google import genai
                return genai.Client()
    
    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        match self.model_type:
            case ModelType.CLAUDE_SONNET:
                resp = await self._client.messages.create(
                    model=self.model_type.value,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.content[0].text
            case ModelType.GPT4O_MINI:
                resp = await self._client.chat.completions.create(
                    model=self.model_type.value,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.choices[0].message.content
            case ModelType.GEMINI_FLASH:
                resp = await self._client.aio.models.generate_content(
                    model=self.model_type.value, contents=prompt
                )
                return resp.text
```

---

## Phase 2: Seed Data Generation

### 2.1 Compression Prompts

See `src/generation/prompts/compress_nl.txt` and `compress_code.txt` for full prompts.

### 2.2 Seed Generator

```python
# src/generation/seed_generator.py
import asyncio
from pathlib import Path
from dataclasses import dataclass
import json
from ..validation.models import ModelClient, ModelType
from ..utils.caching import SemanticCache

@dataclass
class GenerationConfig:
    nl_prompt_path: Path
    code_prompt_path: Path
    generator_model: ModelType
    output_dir: Path

class SeedGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.generator = ModelClient(config.generator_model)
        self.cache = SemanticCache(config.output_dir / "cache")
        self.nl_prompt = config.nl_prompt_path.read_text()
        self.code_prompt = config.code_prompt_path.read_text()
    
    async def generate_nl_pairs(self, inputs: list[str]) -> list[dict]:
        pairs = []
        for inp in inputs:
            cache_key = f"nl:{hash(inp)}"
            if cached := self.cache.get(cache_key):
                pairs.append(cached)
                continue
            
            prompt = self.nl_prompt.format(input=inp)
            compressed = await self.generator.complete(prompt)
            pair = {"verbose": inp, "compressed": compressed.strip(), "domain": "nl"}
            self.cache.set(cache_key, pair)
            pairs.append(pair)
        return pairs
    
    async def generate_code_pairs(self, inputs: list[str], language: str = "python") -> list[dict]:
        pairs = []
        for inp in inputs:
            cache_key = f"code:{language}:{hash(inp)}"
            if cached := self.cache.get(cache_key):
                pairs.append(cached)
                continue
            
            prompt = self.code_prompt.format(input=inp)
            compressed = await self.generator.complete(prompt)
            pair = {"verbose": inp, "compressed": compressed.strip(), "domain": "code", "language": language}
            self.cache.set(cache_key, pair)
            pairs.append(pair)
        return pairs
```

---

## Phase 3: Model Training

### 3.1 Training Options

| Platform | Model | Speed | Cost | When to Use |
|----------|-------|-------|------|-------------|
| **Tinker** | Qwen3-8B | Fast | ~$4-20 | Production training |
| **Tinker** | Qwen3-30B-A3B | Fast | ~$4-18 | MoE, cost-efficient |
| **Local MLX** | Qwen3-4B | Slow | Free | Quick experiments |

### 3.2 Tinker Training Script

```python
# src/training/train_tinker.py
import os
from pathlib import Path

from tinker import ServiceClient

from src.training.train_tinker import TinkerTrainingConfig, run_training_loop, write_run_metadata
from src.utils.config import load_tinker_training_config


def train_on_tinker(config_path: Path, output_dir: Path) -> Path:
    config = load_tinker_training_config(config_path)
    service_client = ServiceClient(api_key=os.environ["TINKER_API_KEY"])
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
    )
    metadata = run_training_loop(training_client, config)
    return write_run_metadata(metadata, output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default="configs/training.yaml")
    parser.add_argument("--output", type=Path, default="models/adapters/tinker")
    args = parser.parse_args()
    metadata_path = train_on_tinker(args.config, args.output)
    print(f"Run metadata: {metadata_path}")
```

### 3.3 Local MLX Training Script

```python
# src/training/train_mlx.py
"""Local training for quick iteration on M4 Pro."""
import subprocess
from pathlib import Path
import yaml

def train_local_mlx(config_path: Path, dataset_path: Path, output_dir: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # MLX-LM uses command line for fine-tuning
    cmd = [
        "mlx_lm.lora",
        "--model", config["local"]["model"],  # "mlx-community/Qwen3-4B-4bit"
        "--train",
        "--data", str(dataset_path),
        "--adapter-path", str(output_dir / "mlx_adapter"),
        "--batch-size", str(config["local"]["batch_size"]),
        "--iters", str(config["local"]["iters"]),
        "--learning-rate", str(config["local"]["lr"]),
        "--lora-rank", str(config["lora"]["r"]),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print(f"Adapter saved to: {output_dir / 'mlx_adapter'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default="configs/training.yaml")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="models/adapters")
    args = parser.parse_args()
    train_local_mlx(args.config, args.dataset, args.output)
```

### 3.4 Training Config

```yaml
# configs/training.yaml

# === TINKER (Production) ===
model:
  name: "Qwen/Qwen3-8B"
  # Alternatives:
  #   - Qwen/Qwen3-4B-Instruct-2507 (faster, cheaper)
  #   - Qwen/Qwen3-30B-A3B (MoE, good quality/cost ratio)

lora:
  r: 64
  alpha: 128
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  epochs: 3
  batch_size: 4
  lr: 2.0e-4

# === LOCAL MLX (Iteration) ===
local:
  model: "mlx-community/Qwen3-4B-4bit"
  batch_size: 2
  iters: 1000
  lr: 1.0e-4

# === DATA ===
data:
  train_path: "data/validated/train.jsonl"
  eval_path: "data/validated/test.jsonl"
  format: "text"

# === COST ESTIMATES (Tinker) ===
# Qwen3-8B: $0.40/1M tokens
# 10K pairs (~5M tokens) × 3 epochs = ~$6
# 50K pairs (~25M tokens) × 3 epochs = ~$30
# 5 training runs total: ~$30-150
```

---

## Phase 4: Production Inference

### 4.1 MLX Inference Service

```python
# src/inference/compressor.py
from pathlib import Path
from mlx_lm import load, generate
from .domain_classifier import DomainClassifier

class CompressionService:
    def __init__(self, model_path: str, adapter_path: Path | None = None):
        self.model, self.tokenizer = load(
            model_path,
            adapter_path=str(adapter_path) if adapter_path else None
        )
        self.domain_classifier = DomainClassifier()
    
    def compress(self, text: str, domain: str | None = None) -> str:
        if domain is None:
            domain = self.domain_classifier.classify(text)
        
        instruction = self._get_instruction(domain)
        prompt = f"""### Instruction:
{instruction}

### Input:
{text}

### Output:
"""
        output = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self._estimate_output_length(text),
            temp=0.3,
        )
        
        # Extract generated portion
        return output.split("### Output:")[-1].strip()
    
    def _get_instruction(self, domain: str) -> str:
        return {
            "nl": "Compress this text into minimal tokens preserving all semantic content:",
            "code": "Compress this code into minimal notation preserving logic and structure:",
            "mixed": "Compress this code and description into unified minimal notation:"
        }.get(domain, "Compress into minimal tokens:")
    
    def _estimate_output_length(self, text: str, ratio: float = 0.5) -> int:
        return max(50, int(len(text.split()) * ratio))
```

### 4.2 Domain Classifier

```python
# src/inference/domain_classifier.py
import re

class DomainClassifier:
    CODE_PATTERNS = [
        r'\bdef\s+\w+\s*\(', r'\bclass\s+\w+', r'\bimport\s+\w+',
        r'\bfunction\s+\w+', r'\bconst\s+\w+\s*=', r'=>', r'async\s+def',
    ]
    
    def __init__(self, code_threshold: float = 0.3):
        self.code_threshold = code_threshold
        self.code_re = [re.compile(p) for p in self.CODE_PATTERNS]
    
    def classify(self, text: str) -> str:
        lines = text.strip().split('\n')
        code_lines = sum(1 for line in lines if any(p.search(line) for p in self.code_re))
        code_ratio = code_lines / max(len(lines), 1)
        
        if code_ratio > 0.7:
            return "code"
        elif code_ratio > self.code_threshold:
            return "mixed"
        return "nl"
```

---

## Phase 5: Evaluation

```python
# scripts/evaluate.py
from dataclasses import dataclass
from rich.table import Table
from rich.console import Console
from src.validation.harness import ValidationHarness, CompressionPair, TaskType
from src.validation.models import ModelType
from src.inference.compressor import CompressionService

@dataclass
class BenchmarkResult:
    dataset: str
    avg_compression_ratio: float
    avg_equivalence: float
    cross_model_transfer: float
    latency_ms: float

async def run_benchmark(compressor: CompressionService, test_data_path, models):
    harness = ValidationHarness(models, equivalence_threshold=0.0)
    # ... (see full implementation in repo)
```

---

## Makefile

```makefile
# Makefile

.PHONY: setup train-local train-cloud validate evaluate

# Setup
setup:
	pip install -e ".[dev]"
	pip install mlx mlx-lm

# Local training (MLX on M4 Pro)
train-local:
	python scripts/train_local.py \
		--config configs/training.yaml \
		--dataset data/validated/train.jsonl

# Cloud training (Tinker)
train-cloud:
	python scripts/train_tinker.py \
		--config configs/training.yaml \
		--output models/adapters/tinker

# Validation
validate:
	python scripts/validate_batch.py \
		--input data/seed/pairs.jsonl \
		--output data/validated/pairs.jsonl

# Evaluation
evaluate:
	python scripts/evaluate.py \
		--adapter models/adapters/tinker_adapter \
		--test-data data/validated/test.jsonl
```

---

## Environment Setup

```bash
# .env
# Frontier models (validation)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# HuggingFace (model downloads)
HF_TOKEN=hf_...
HF_HUB_ENABLE_HF_TRANSFER=1

# Tinker (cloud training)
TINKER_API_KEY=tk_...

# Cost tracking
COST_LIMIT_DAILY_USD=50
COST_WARN_THRESHOLD_USD=30
```

---

## Execution Checklist

### Phase 1: Foundation
- [ ] Set up project structure
- [ ] Implement validation harness
- [ ] Test API wrappers for Claude, GPT, Gemini
- [ ] Implement semantic cache

### Phase 2: Seed Generation
- [ ] Collect source corpora (NL + code)
- [ ] Generate 1K seed pairs via Claude
- [ ] Run initial validation → filter to ~500 validated pairs

### Phase 3: Scale Generation
- [ ] Implement tiered generation pipeline
- [ ] Generate 10K pairs with local model bootstrapping
- [ ] Cross-model validation filtering → target 5K validated pairs

### Phase 4: Training V1
- [ ] Set up Tinker account, verify API access
- [ ] Build training dataset (JSONL)
- [ ] Train Qwen3-8B on Tinker
- [ ] Download adapter, test locally with MLX
- [ ] Evaluate on held-out test set

### Phase 5: Scale & Iterate
- [ ] Generate 50K+ pairs using V1 compressor
- [ ] Validate, filter, train V2 on expanded dataset
- [ ] Benchmark against baseline

### Phase 6: Production Integration
- [ ] Package adapter for MLX inference
- [ ] Integrate with existing memory system
- [ ] A/B test with real agent workloads

---

## Cost Projections

| Phase | API Calls | Est. Cost |
|-------|-----------|-----------|
| Seed generation (1K) | 2K calls | $20-40 |
| Initial validation | 6K calls | $60-100 |
| Scaled generation (10K) | 5K API + local | $50-80 |
| Final validation | 15K calls | $150-200 |
| **Tinker training (5 runs)** | - | $30-150 |
| **Total** | | **$310-570** |

Local inference on M4 Pro: Free.

---

## Success Criteria

| Metric | Target | Stretch |
|--------|--------|---------|
| Token reduction | 40% | 60% |
| Task equivalence | 95% | 98% |
| Cross-model transfer | 90% | 95% |
| Compression latency | <100ms | <50ms |
