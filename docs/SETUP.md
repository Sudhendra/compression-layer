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

### 1.5 Run Outputs

Local MLX runs are stored per run under:

```
models/runs/mlx/<timestamp>
```

Each run directory captures run metadata (`run.json`), training logs (`train.log`,
`train.err`), and adapter outputs under `adapter/`.
The newest run is always accessible via the `models/runs/mlx/latest` symlink when
using `scripts/train_local.py`.

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
import tinker
from tinker import LoraConfig, TrainingConfig

# Initialize
client = tinker.Client()

# Configure training
config = TrainingConfig(
    model="Qwen/Qwen3-8B",
    dataset="path/to/data.jsonl",  # Upload via tinker CLI
    lora=LoraConfig(
        r=64,
        alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
    epochs=3,
    batch_size=4,
    learning_rate=2e-4,
)

# Start training
job = client.train(config)
print(f"Job ID: {job.id}")

# Monitor
job.wait()
print(f"Final loss: {job.metrics['loss']}")

# Download adapter
job.download_adapter("./models/tinker_adapter")
```

### 2.3 Tinker CLI Workflow

```bash
# Upload dataset
tinker data upload ./data/validated/pairs.jsonl --name compression-v1

# Start training
tinker train \
  --model Qwen/Qwen3-8B \
  --dataset compression-v1 \
  --lora-rank 64 \
  --epochs 3

# Check status
tinker jobs list

# Download results
tinker jobs download <job-id> --output ./models/
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
tinker train --model Qwen/Qwen3-8B --dataset compression-v1

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
- Verify API key: `tinker auth check`
- Check job logs: `tinker jobs logs <job-id>`

### Slow local inference
- Ensure using 4-bit model: `*-4bit`
- Increase wired memory limit (macOS 15+):
  ```bash
  sudo sysctl iogpu.wired_limit_mb=20000
  ```
