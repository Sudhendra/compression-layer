# TASKS.md

Prioritized implementation tasks.

## Phase 1: Foundation ✅ COMPLETE

### P0 — Core Infrastructure
- [x] Create directory structure per AGENTS.md
- [x] Set up pyproject.toml, install deps
- [x] Create `src/utils/config.py` — Pydantic settings
- [x] Create `src/utils/caching.py` — Disk-backed cache
- [x] Create `src/utils/tokenizers.py` — Multi-tokenizer utils
- [x] Create `src/utils/costs.py` — API cost tracking

### P0 — API Clients
- [x] `src/validation/models.py` — ModelType enum, ModelClient
- [x] Implement Anthropic/OpenAI/Gemini async clients
- [x] Add retry logic with exponential backoff

### P0 — Validation Harness
- [x] `src/validation/metrics.py` — Equivalence scoring
- [x] `src/validation/harness.py` — ValidationHarness class

### P1 — Testing
- [x] `tests/test_metrics.py`
- [x] `tests/test_harness.py` (mock API calls)
- [x] `tests/conftest.py` — Pytest fixtures

---

## Phase 2: Generation

### P0 — Prompts
- [x] `src/generation/prompts/compress_nl.txt` (already exists)
- [x] `src/generation/prompts/compress_code.txt` (already exists)

### P0 — Seed Generator
- [ ] `src/generation/seed_generator.py`
- [ ] Caching integration

### P1 — CLI Scripts
- [ ] `scripts/generate_seed.py`
- [ ] `scripts/validate_batch.py`

---

## Phase 3: Training

### P0 — Local Setup (MLX)
- [ ] Install MLX: `pip install -U mlx-lm`
- [ ] Download model: `huggingface-cli download mlx-community/Qwen3-4B-Instruct-4bit`
- [ ] Test inference locally

### P0 — Cloud Setup (Tinker)
- [ ] Get Tinker API key from https://tinker.thinkingmachines.ai
- [ ] Install: `pip install tinker`
- [ ] Verify: `tinker auth check`

### P0 — Training Scripts
- [ ] `src/training/train_mlx.py` — Local MLX training wrapper
- [ ] `src/training/train_tinker.py` — Tinker cloud training
- [ ] Dataset formatting for both platforms

### P1 — Dataset Builder
- [ ] `src/training/dataset.py` — Format for instruction tuning
- [ ] JSONL export compatible with MLX and Tinker

---

## Phase 4: Inference

### P0 — Compressor Service
- [ ] `src/inference/compressor.py`
  - [ ] Load adapter via MLX
  - [ ] Single/batch compression
  - [ ] Support both local and downloaded Tinker adapters

### P0 — Domain Classifier
- [ ] `src/inference/domain_classifier.py` — Regex-based

### P1 — Performance
- [ ] Benchmark inference speed on M4 Pro
- [ ] Optimize for <100ms latency

---

## Phase 5: Evaluation & Integration

### P0 — Benchmark Suite
- [ ] `scripts/evaluate.py`
- [ ] Cross-model validation
- [ ] Metrics reporting

### P1 — Integration
- [ ] Example: Memory store integration
- [ ] A/B test framework

---

## Stretch Goals

- [ ] Multi-language code support
- [ ] Streaming compression
- [ ] Adaptive compression by task type
