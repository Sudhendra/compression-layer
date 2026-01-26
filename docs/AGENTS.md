# AGENTS.md

## Project Context

Universal LLM compression layer: reduces token count for memory/context injection while preserving semantic equivalence across Claude, GPT, Gemini.

## Architecture

```
compression-layer/
├── src/
│   ├── validation/     # Cross-model equivalence testing
│   ├── generation/     # Compression pair generation
│   ├── training/       # Unsloth fine-tuning pipeline
│   ├── inference/      # Production compressor service
│   └── utils/          # Tokenizers, caching, cost tracking
├── data/               # Corpora and generated datasets
├── models/             # Checkpoints, GGUF exports
├── configs/            # YAML configs
├── scripts/            # Entry points
└── tests/
```

## Core Principles

1. **Model-agnostic**: Compressions validate across 3+ frontier models
2. **Domain-aware**: Separate strategies for NL, code, mixed
3. **Cost-conscious**: Cheap models generate, expensive models validate
4. **Hybrid workflow**: MLX locally (iteration), Tinker cloud (production)

## Agent Guidelines

### Code Style
- Python 3.11+, type hints everywhere
- Async-first for API interactions
- Pydantic for configs/data models
- No classes where functions suffice

### Key Libraries
- **Local Training**: `mlx-lm` (Apple Silicon)
- **Cloud Training**: `tinker` (Thinking Machines)
- **APIs**: `anthropic`, `openai`, `google-genai`
- **Utils**: `tiktoken`, `sentence-transformers`, `rich`

### Error Handling
- Let exceptions propagate unless recoverable
- Use `rich` for CLI output
- Log API costs to `data/costs.log`

## Key Files

| File | Purpose |
|------|---------|
| `src/validation/harness.py` | Cross-model equivalence validation |
| `src/training/train_tinker.py` | Tinker cloud training |
| `src/training/train_mlx.py` | Local MLX training |
| `src/inference/compressor.py` | Production inference service |
| `configs/training.yaml` | Model & hyperparameters |

## Task-Specific Instructions

### Validation harness
- asyncio.gather with semaphore for rate limiting
- Cache API responses by content hash
- Code equivalence needs AST comparison

### Training with Tinker (production)
- Upload dataset via `tinker data upload`
- Use Qwen3-8B or Qwen3-30B-A3B (MoE, cost-efficient)
- Download adapter after training

### Training with MLX (local iteration)
- Use `mlx-community/Qwen3-4B-Instruct-4bit`
- `python -m mlx_lm.lora --train --data ./data`
- Good for quick experiments, ~150 tok/s on M4 Pro

### Inference
- Load MLX adapter: `--adapter-path ./adapters`
- Domain classifier is rule-based (fast)
- Batch where possible

## Common Patterns

### MLX Model Loading (Local)
```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-4B-Instruct-4bit")
response = generate(model, tokenizer, prompt="...", max_tokens=100)
```

### Tinker Training (Cloud)
```python
import tinker

client = tinker.Client()
job = client.train(
    model="Qwen/Qwen3-8B",
    dataset="compression-v1",
    lora_rank=64,
    epochs=3,
)
job.wait()
job.download_adapter("./models/adapter")
```

### API Client with Retry
```python
async def complete(self, prompt: str) -> str:
    for attempt in range(3):
        try:
            return await self._call_api(prompt)
        except RateLimitError:
            await asyncio.sleep(2 ** attempt)
    raise
```

### Batch Processing
```python
async def process_batch(items, fn, concurrency=10):
    sem = asyncio.Semaphore(concurrency)
    async def bounded(item):
        async with sem:
            return await fn(item)
    return await asyncio.gather(*[bounded(i) for i in items])
```

## Environment Variables

```bash
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
HF_TOKEN=              # For model downloads
TINKER_API_KEY=        # For cloud training
```

## Quick Commands

```bash
# Local inference (MLX)
python -m mlx_lm.generate --model mlx-community/Qwen3-4B-Instruct-4bit --prompt "..."

# Local training (MLX)
python -m mlx_lm.lora --model mlx-community/Qwen3-4B-Instruct-4bit --train --data ./data

# Cloud training (Tinker)
tinker train --model Qwen/Qwen3-8B --dataset compression-v1 --lora-rank 64

# Validate pairs
python scripts/validate_batch.py --input data/seed/pairs.jsonl
```

## Success Metrics

| Metric | Target |
|--------|--------|
| Token reduction | >40% |
| Task equivalence | >95% |
| Cross-model transfer | >90% |
| Inference latency | <100ms |
