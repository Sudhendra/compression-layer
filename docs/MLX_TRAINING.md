# MLX Local Training Runbook

This guide walks through downloading the base model, running local MLX LoRA
training, and monitoring logs/checkpoints on Apple Silicon.

## Prerequisites

- Activate the project venv:

```bash
source .venv/bin/activate
```

- Ensure MLX is installed:

```bash
pip install -U mlx-lm
```

- Authenticate with Hugging Face if needed:

```bash
hf auth login
```

## Download the base model

Recommended local model (M4 Pro 24GB):

```bash
hf download mlx-community/Qwen3-4B-Instruct-2507-8bit
```

## Run training

Default run (uses config defaults):

```bash
python scripts/train_local.py --train
```

Override the model explicitly (optional):

```bash
python scripts/train_local.py --train --model mlx-community/Qwen3-4B-Instruct-2507-8bit
```

## Evaluate the latest adapter

If you trained with the default run storage, evaluation will use
`models/runs/mlx/latest/adapter` automatically.

```bash
python scripts/train_local.py --evaluate
```

To evaluate a specific run, pass the adapter path explicitly:

```bash
python scripts/train_local.py --evaluate --adapter-path models/runs/mlx/<timestamp>/adapter
```

## Where outputs are stored

Each run writes to a timestamped directory:

```
models/runs/mlx/<timestamp>/
```

Contents include:
- `run.json` (metadata)
- `train.log` (stdout)
- `train.err` (stderr)
- `adapter/` (LoRA checkpoints)

The latest run is available at:

```
models/runs/mlx/latest
```

## Monitor logs during training

Follow training output:

```bash
tail -f models/runs/mlx/latest/train.log
```

Follow stderr (warnings/errors):

```bash
tail -f models/runs/mlx/latest/train.err
```

## Quick sanity checks

- `train.log` updates every `steps_per_report` steps.
- `adapter/` gains `.safetensors` files at each `save_every` interval.
- If training fails, check `train.err` and the CLI error message for details.

## Notes on checkpointing

- **Gradient checkpointing** reduces memory usage during training. It does **not**
  control whether LoRA adapters are saved to disk.
- Adapter checkpoints are saved according to `save_every` and live under
  `models/runs/mlx/<timestamp>/adapter/`.
