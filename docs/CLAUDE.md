# CLAUDE.md

## Project Summary

Universal semantic compression layer for LLM inputs. Compresses memories, code, context before API calls while preserving reasoning equivalence across Claude/GPT/Gemini.

**Model**: Qwen3-8B (via Unsloth)  
**Training**: Unsloth + LoRA (2-5x faster, 70% less VRAM)  
**License**: Apache 2.0

## Build & Run

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Local (MLX on M4 Pro)
pip install -U mlx-lm

# Cloud (Tinker)
pip install tinker

# Tests
pytest tests/ -v
```

## Directory Purpose

- `src/validation/` — Equivalence testing across models
- `src/generation/` — Compression pair synthesis
- `src/training/` — Unsloth fine-tuning on Qwen3-8B
- `src/inference/` — Production compressor
- `data/raw/` — Source corpora (gitignored)
- `data/validated/` — Cross-model validated pairs
- `models/` — LoRA checkpoints, GGUF exports

## Code Conventions

- Async by default for I/O
- Type hints required
- Pydantic models for structured data
- `rich` for CLI output

## Critical Implementation Notes

### Training
- **Local (MLX)**: Quick iteration with Qwen3-4B on M4 Pro
- **Cloud (Tinker)**: Production runs with Qwen3-8B (~$10-50 total)
- Export adapters, run inference locally

### Inference (`src/inference/compressor.py`)
- Call `FastLanguageModel.for_inference(model)` — 2x faster
- Domain classifier is regex-based (no ML overhead)
- Use GGUF export + llama.cpp for lowest latency

### Validation (`src/validation/harness.py`)
- Compression passes only if min(equivalence) > 0.85 across ALL models
- NL: 0.7 × cosine(embeddings) + 0.3 × Jaccard
- Code: 0.5 × AST similarity + 0.5 × semantic

## API Keys Required

```bash
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
HF_TOKEN=           # For model downloads
TINKER_API_KEY=     # For cloud training
```

## Common Tasks

**Change base model**: Update `configs/training.yaml` → `model.name`

**Add compression domain**: 
1. Add prompt in `src/generation/prompts/`
2. Update `DomainClassifier` patterns
3. Add domain-specific equivalence in `metrics.py`

**Export for production**:
```python
model.save_pretrained_gguf("models/gguf", tokenizer, quantization_method="q4_k_m")
```

## Gotchas

- MLX requires macOS 15+ for best performance
- Use 4-bit models from `mlx-community/` for local work
- AST parsing fails on incomplete code — wrap in try/except
- Tinker dataset format: JSONL with `text` or `messages` field
