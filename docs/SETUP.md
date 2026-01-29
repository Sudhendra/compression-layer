# Setup Guide

## Overview

This project uses a **hybrid workflow**:
- **Tinker** (cloud) — Production training runs
- **MLX** (local M4 Pro) — Quick iteration, inference, validation

## Prerequisites

- **Local**: MacBook M4 Pro 24GB, macOS 15+
- **Cloud**: Tinker account (https://tinker.thinkingmachines.ai)
- **Python**: 3.11+

---

## Part 1: Local Environment (MLX)

### 1.1 Base Setup

```bash
# Clone project
git clone <repo-url>
cd compression-layer

# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install base deps
pip install -e ".[dev]"
```

### 1.2 Install MLX (Apple Silicon)

```bash
# MLX for local training/inference
pip install -U mlx-lm

# Verify
python -c "import mlx; print('MLX OK')"
```

### 1.3 Download Qwen Models (Local)

```bash
# Login to HuggingFace
pip install "huggingface_hub[cli]"
huggingface-cli login

# Download 4-bit Qwen for local use
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit

# Test inference
python -m mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --prompt "Compress this: The user is a software engineer at Google" \
  --max-tokens 100
```

### 1.4 Local Training (Small Scale)

For quick iteration with Qwen3-4B:

```bash
# Fine-tune with MLX LoRA (local)
python -m mlx_lm.lora \
  --model mlx-community/Qwen3-4B-Instruct-4bit \
  --train \
  --data ./data/validated \
  --iters 100 \
  --batch-size 2 \
  --lora-rank 8

# Test adapter
python -m mlx_lm.generate \
  --model mlx-community/Qwen3-4B-Instruct-4bit \
  --adapter-path ./adapters \
  --prompt "Compress: ..."
```

**Memory usage on M4 Pro 24GB:**
| Model | QLoRA Memory | Speed |
|-------|--------------|-------|
| Qwen3-4B | ~8GB | ~150 tok/s |
| Qwen2.5-7B | ~12GB | ~100 tok/s |

---

## Part 2: Cloud Training (Tinker)

### 2.1 Tinker Setup

```bash
# Install Tinker SDK
pip install tinker

# Set API key
export TINKER_API_KEY=your_key_here
```

Or add to `.env`:
```bash
TINKER_API_KEY=tk_...
```

### 2.2 Training with Tinker

```python
# src/training/train_tinker.py
import os
from tinker import ServiceClient
from pathlib import Path

from src.training.train_tinker import TinkerTrainingConfig, run_training_loop, write_run_metadata

# Initialize SDK clients
service_client = ServiceClient(api_key=os.environ["TINKER_API_KEY"])
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
)

# Configure training
config = TinkerTrainingConfig(
    base_model="Qwen/Qwen3-8B",
    epochs=3,
    steps=300,
)

# Run training loop and persist metadata
metadata = run_training_loop(training_client, config)
metadata_path = write_run_metadata(metadata, output_dir=Path("models/adapters/tinker"))
print(f"Run ID: {metadata.run_id}")
print(f"Run metadata: {metadata_path}")
```

### 2.3 Tinker CLI Workflow

```bash
# Start training (records run metadata under models/adapters/tinker/runs)
python scripts/train_tinker.py \
  --config configs/training.yaml \
  --output models/adapters/tinker

# Check status
python scripts/train_tinker.py \
  --status <run-id> \
  --output models/adapters/tinker

# Inspect run metadata
cat models/adapters/tinker/runs/<run-id>.json
```

### 2.4 Cost Estimation

| Model | Per 1M Tokens | 10K pairs (~5M tok) | 50K pairs (~25M tok) |
|-------|---------------|---------------------|----------------------|
| Qwen3-4B | $0.22 | ~$1.10 | ~$5.50 |
| Qwen3-8B | $0.40 | ~$2.00 | ~$10.00 |
| Qwen3-30B-A3B | $0.36 | ~$1.80 | ~$9.00 |

**5 training runs**: $10-50 total

---

## Part 3: API Keys

Create `.env`:

```bash
# Frontier models (validation)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# HuggingFace (model downloads)
HF_TOKEN=hf_...
HF_HUB_ENABLE_HF_TRANSFER=1

# Tinker (cloud training)
TINKER_API_KEY=tk_...
```

---

## Part 4: Recommended Workflow

### Phase 1-2: Local (MLX)
- Generate seed pairs via Claude API
- Run validation harness
- Quick experiments with Qwen3-4B locally

### Phase 3-4: Cloud (Tinker)
- Train Qwen3-8B on validated dataset
- Iterate on hyperparameters
- ~$20-50 total cost

### Phase 5-6: Local (MLX)
- Download trained adapter
- Run inference locally on M4 Pro
- Integrate with memory system

---

## Quick Commands

```bash
# Local inference
python -m mlx_lm.generate --model mlx-community/Qwen3-4B-Instruct-4bit --prompt "..."

# Local training (small scale)
python -m mlx_lm.lora --model mlx-community/Qwen3-4B-Instruct-4bit --train --data ./data

# Cloud training (production)
python scripts/train_tinker.py --config configs/training.yaml --output models/adapters/tinker

# Run validation
python scripts/validate_batch.py --input data/seed/pairs.jsonl
```

---

## Troubleshooting

### MLX "out of memory"
- Use smaller model: `Qwen3-4B` instead of `8B`
- Reduce batch size to 1
- Use 4-bit quantized models from `mlx-community/`

### Tinker job failed
- Check dataset format (JSONL with `text` or `messages` field)
- Verify API key in `.env` or shell: `TINKER_API_KEY`
- Inspect run metadata: `models/adapters/tinker/runs/<run-id>.json`
- Re-run status: `python scripts/train_tinker.py --status <run-id> --output models/adapters/tinker`

### Slow local inference
- Ensure using 4-bit model: `*-4bit`
- Increase wired memory limit (macOS 15+):
  ```bash
  sudo sysctl iogpu.wired_limit_mb=20000
  ```
