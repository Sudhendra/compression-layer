# AGENTS.md

## Project Context

Universal LLM compression layer: reduces token count for memory/context injection while preserving semantic equivalence across Claude, GPT, Gemini.

**Repo**: https://github.com/Sudhendra/compression-layer

## Architecture

```
compression-layer/
├── .github/workflows/  # CI pipeline
├── src/
│   ├── validation/     # Cross-model equivalence testing
│   ├── generation/     # Compression pair generation
│   ├── training/       # Tinker + MLX training
│   ├── inference/      # Production compressor service
│   └── utils/          # Tokenizers, caching, cost tracking
├── data/               # Corpora and generated datasets (gitignored)
├── models/             # Checkpoints, GGUF exports (gitignored)
├── configs/            # YAML configs
├── docs/               # Project documentation
├── tests/              # Pytest test suite
└── scripts/            # Entry points
```

## Core Principles

1. **Model-agnostic**: Compressions validate across 3+ frontier models
2. **Domain-aware**: Separate strategies for NL, code, mixed
3. **Cost-conscious**: Cheap models generate, expensive models validate
4. **Hybrid workflow**: MLX locally (iteration), Tinker cloud (production)
5. **CI-gated**: All changes require passing lint, typecheck, and tests

## Git Workflow

### Branch Strategy
Each implementation phase gets its own branch:
```
main (protected - requires CI pass)
  └── phase-1-foundation    ✅ Complete (PR #1)
  └── phase-2-generation    
  └── phase-3-training      
  └── phase-4-inference     
  └── phase-5-evaluation    
```

### Per-Phase Workflow
```bash
# 1. Start from latest main
git checkout main && git pull origin main

# 2. Create phase branch
git checkout -b phase-X-name

# 3. Implement with atomic commits
git add . && git commit -m "feat: description"

# 4. Run CI checks locally BEFORE pushing
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/ --ignore-missing-imports
pytest tests/ -v

# 5. Push and create PR
git push -u origin phase-X-name
gh pr create --title "Phase X: Name" --body "## Summary\n..."

# 6. CI runs automatically on PR
# - lint job: ruff check + format
# - test job: mypy + pytest (Python 3.11 & 3.12)

# 7. After CI passes, merge to main
gh pr merge --squash
```

### Commit Message Format
- `feat:` — New feature
- `fix:` — Bug fix  
- `docs:` — Documentation
- `test:` — Adding tests
- `refactor:` — Code refactoring
- `chore:` — Maintenance

### What's Tracked vs Gitignored
| Tracked | Gitignored |
|---------|------------|
| `src/`, `tests/` | `data/` (all subdirs) |
| `configs/`, `docs/` | `models/`, `adapters/` |
| `.github/workflows/` | `.env`, `.venv/` |
| `pyproject.toml` | `*.log`, cache dirs |

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
- Use Qwen3-8B or Qwen3-30B-A3B (MoE, cost-efficient)
- Run `python scripts/train_tinker.py` (stores run metadata under `models/adapters/tinker/runs`)

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
import os
from pathlib import Path
from tinker import ServiceClient

from src.training.train_tinker import TinkerTrainingConfig, run_training_loop, write_run_metadata

service_client = ServiceClient(api_key=os.environ["TINKER_API_KEY"])
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
)
config = TinkerTrainingConfig(
    base_model="Qwen/Qwen3-8B",
    epochs=3,
    steps=300,
)
metadata = run_training_loop(training_client, config)
write_run_metadata(metadata, output_dir=Path("models/adapters/tinker"))
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
# === GIT & CI ===
# Run all CI checks locally
ruff check src/ tests/ && mypy src/ --ignore-missing-imports && pytest tests/ -v

# Create PR for current branch
gh pr create --title "Phase X: Description" --body "## Summary\n- bullet points"

# Check PR CI status
gh pr checks

# === LOCAL INFERENCE (MLX) ===
python -m mlx_lm.generate --model mlx-community/Qwen3-4B-Instruct-4bit --prompt "..."

# === LOCAL TRAINING (MLX) ===
python -m mlx_lm.lora --model mlx-community/Qwen3-4B-Instruct-4bit --train --data ./data

# === CLOUD TRAINING (Tinker) ===
python scripts/train_tinker.py --config configs/training.yaml --output models/adapters/tinker

# === VALIDATION ===
python scripts/validate_batch.py --input data/seed/pairs.jsonl
```

## Success Metrics

| Metric | Target |
|--------|--------|
| Token reduction | >40% |
| Task equivalence | >95% |
| Cross-model transfer | >90% |
| Inference latency | <100ms |
