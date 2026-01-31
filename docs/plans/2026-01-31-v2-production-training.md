# V2 Production Training Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Scale from 2K to 10K+ training pairs using the v1 adapter for synthetic data generation, then train a production-quality model on Tinker and release it.

**Architecture:** Use the trained MLX adapter (89% pass rate, 62% compression) to generate compressions at scale, validate with Claude+GPT, train on Tinker with Qwen3-8B, evaluate, and publish.

**Tech Stack:** MLX (local inference), Tinker (cloud training), Claude/GPT APIs (validation), HuggingFace (model hosting).

---

## Current State Assessment

### What We Have
- **V1 Adapter:** `models/runs/mlx/2026-01-30_17-14-36/adapter` (iter-500)
  - 89% pass rate (Claude + GPT evaluators)
  - 62% avg compression ratio (38% token savings)
  - 0.859 avg equivalence score
- **Training Data:** 2,199 examples (1,759 train / 219 valid / 221 test)
- **Raw Corpus:** ~3MB code + NL data in `data/raw/`
- **Evaluation System:** Task-equivalence metrics with multi-model validation

### What We Need
- **Target:** 10K+ validated training pairs
- **Production Model:** Qwen3-8B trained on Tinker
- **Release Artifacts:** HuggingFace model card, adapter weights, documentation

---

## Phase 1: Synthetic Data Generation (Scale to 10K)

### Task 1: Create adapter-based compression generator

**Files:**
- Create: `src/generation/adapter_generator.py`
- Test: `tests/test_adapter_generator.py`

**Step 1: Write the failing test**

```python
# tests/test_adapter_generator.py
import pytest
from pathlib import Path
from src.generation.adapter_generator import AdapterGenerator

def test_adapter_generator_compresses_text():
    """Adapter generator produces compressed output."""
    gen = AdapterGenerator(
        model="mlx-community/Qwen3-4B-Instruct-2507-8bit",
        adapter_path=Path("models/runs/mlx/latest/adapter"),
    )
    result = gen.compress("The quick brown fox jumps over the lazy dog.")
    assert len(result) < len("The quick brown fox jumps over the lazy dog.")
    assert len(result) > 0
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_adapter_generator.py::test_adapter_generator_compresses_text -v`
Expected: FAIL with "AdapterGenerator not found"

**Step 3: Write minimal implementation**

```python
# src/generation/adapter_generator.py
"""Generate compressions using trained MLX adapter."""

from pathlib import Path
from typing import Callable
import mlx_lm
from mlx_lm.sample_utils import make_sampler

class AdapterGenerator:
    """Generate compressions using a trained LoRA adapter."""
    
    def __init__(
        self,
        model: str = "mlx-community/Qwen3-4B-Instruct-2507-8bit",
        adapter_path: Path | None = None,
        system_prompt: str | None = None,
        temp: float = 0.2,
    ):
        self.model_name = model
        self.adapter_path = adapter_path
        self.system_prompt = system_prompt or (
            "You are a semantic compression engine. Compress the input into minimal tokens "
            "while preserving all information for equivalent LLM reasoning. Use dense notation: "
            "labeled fields, standard abbreviations, and symbols (→ | + @). Never lose information."
        )
        
        loaded = mlx_lm.load(model, adapter_path=str(adapter_path) if adapter_path else None)
        self.mlx_model, self.tokenizer = loaded[:2]
        self.sampler = make_sampler(temp=temp)
    
    def compress(self, text: str, max_tokens: int = 512) -> str:
        """Compress input text using the adapter."""
        import re
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Compress:\n{text}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        output = mlx_lm.generate(
            model=self.mlx_model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=self.sampler,
        )
        
        # Clean artifacts
        output = output.strip()
        output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
        output = output.replace("</tool_call>", "").strip()
        
        # Remove backtick wrappers
        if output.startswith("```") and output.endswith("```"):
            lines = output.split("\n")
            if len(lines) >= 2:
                output = "\n".join(lines[1:-1]).strip()
        
        return output
    
    def compress_batch(
        self,
        texts: list[str],
        max_tokens: int = 512,
        show_progress: bool = True,
    ) -> list[str]:
        """Compress multiple texts."""
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
        
        results = []
        if show_progress:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn()) as progress:
                task = progress.add_task("Compressing...", total=len(texts))
                for text in texts:
                    results.append(self.compress(text, max_tokens))
                    progress.advance(task)
        else:
            for text in texts:
                results.append(self.compress(text, max_tokens))
        
        return results
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_adapter_generator.py::test_adapter_generator_compresses_text -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/generation/adapter_generator.py tests/test_adapter_generator.py
git commit -m "feat: add adapter-based compression generator"
```

---

### Task 2: Create batch generation script

**Files:**
- Create: `scripts/generate_synthetic.py`
- Modify: `src/generation/__init__.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Generate synthetic compression pairs using trained adapter.

Usage:
    # Generate from raw corpus
    python scripts/generate_synthetic.py --input data/raw/code.jsonl --domain code --limit 1000
    
    # Generate from NL corpus
    python scripts/generate_synthetic.py --input data/raw/nl_docs.jsonl --domain nl --limit 1000
    
    # Resume from checkpoint
    python scripts/generate_synthetic.py --input data/raw/code.jsonl --output data/synthetic/code_v2.jsonl --resume
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.generation.adapter_generator import AdapterGenerator
from src.generation.seed_generator import GeneratedPair

console = Console()


def load_corpus(path: Path, limit: int | None = None) -> list[str]:
    """Load text inputs from JSONL corpus."""
    texts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Support different corpus formats
            text = data.get("text") or data.get("content") or data.get("code", "")
            if text and len(text) > 50:  # Skip very short entries
                texts.append(text)
            if limit and len(texts) >= limit:
                break
    return texts


def count_existing(output_path: Path) -> int:
    """Count existing entries for resume support."""
    if not output_path.exists():
        return 0
    with open(output_path) as f:
        return sum(1 for line in f if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic compression pairs")
    parser.add_argument("--input", type=Path, required=True, help="Input corpus JSONL")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--domain", choices=["nl", "code"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Max examples to generate")
    parser.add_argument("--adapter", type=Path, default=Path("models/runs/mlx/latest/adapter"))
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-Instruct-2507-8bit")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = Path(f"data/synthetic/{args.domain}_pairs.jsonl")
    
    console.print(f"[bold]Synthetic Data Generation[/bold]")
    console.print(f"Input: {args.input}")
    console.print(f"Output: {args.output}")
    console.print(f"Domain: {args.domain}")
    
    # Load corpus
    texts = load_corpus(args.input, args.limit)
    console.print(f"Loaded {len(texts)} texts from corpus")
    
    # Handle resume
    start_idx = 0
    if args.resume:
        start_idx = count_existing(args.output)
        if start_idx > 0:
            console.print(f"[yellow]Resuming from index {start_idx}[/yellow]")
            texts = texts[start_idx:]
    
    if not texts:
        console.print("[green]Nothing to generate - already complete![/green]")
        return 0
    
    # Initialize generator
    generator = AdapterGenerator(
        model=args.model,
        adapter_path=args.adapter,
    )
    
    # Generate compressions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    
    with open(args.output, mode) as f:
        compressions = generator.compress_batch(texts, show_progress=True)
        
        for text, compressed in zip(texts, compressions):
            pair = GeneratedPair(
                verbose=text,
                compressed=compressed,
                domain=args.domain,
            )
            f.write(json.dumps(pair.model_dump()) + "\n")
    
    # Summary
    table = Table(title="Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Generated", str(len(texts)))
    table.add_row("Output", str(args.output))
    console.print(table)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Test the script**

Run: `source .venv/bin/activate && python scripts/generate_synthetic.py --input data/raw/code.jsonl --domain code --limit 10`
Expected: Generates 10 pairs to data/synthetic/code_pairs.jsonl

**Step 3: Commit**

```bash
git add scripts/generate_synthetic.py src/generation/__init__.py
git commit -m "feat: add batch synthetic data generation script"
```

---

### Task 3: Generate 5K code pairs

**Step 1: Run generation**

```bash
source .venv/bin/activate && python scripts/generate_synthetic.py \
  --input data/raw/code.jsonl \
  --domain code \
  --limit 5000 \
  --output data/synthetic/code_v2.jsonl
```

Expected: ~5K pairs in data/synthetic/code_v2.jsonl (may take 2-4 hours on M4 Pro)

**Step 2: Verify output**

```bash
wc -l data/synthetic/code_v2.jsonl
head -3 data/synthetic/code_v2.jsonl | python -m json.tool
```

---

### Task 4: Generate 5K NL pairs

**Step 1: Run generation**

```bash
source .venv/bin/activate && python scripts/generate_synthetic.py \
  --input data/raw/nl_docs.jsonl \
  --domain nl \
  --limit 5000 \
  --output data/synthetic/nl_v2.jsonl
```

---

## Phase 2: Validation Pipeline

### Task 5: Create batch validation script for synthetic data

**Files:**
- Create: `scripts/validate_synthetic.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Validate synthetic compression pairs with multi-model evaluation.

Usage:
    # Validate code pairs
    python scripts/validate_synthetic.py \
      --input data/synthetic/code_v2.jsonl \
      --output data/validated/code_v2.jsonl \
      --threshold 0.80 \
      --concurrency 4
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from src.generation.seed_generator import GeneratedPair
from src.validation.harness import CompressionPair, ValidationHarness
from src.validation.models import ModelType

console = Console()


def load_pairs(path: Path) -> list[GeneratedPair]:
    """Load generated pairs from JSONL."""
    pairs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                pairs.append(GeneratedPair(**json.loads(line)))
    return pairs


async def validate_batch(
    pairs: list[GeneratedPair],
    output_path: Path,
    threshold: float = 0.80,
    concurrency: int = 4,
    models: list[str] = ["claude", "gpt"],
) -> dict:
    """Validate pairs and save passing ones."""
    model_map = {
        "claude": ModelType.CLAUDE_SONNET,
        "gpt": ModelType.GPT4O_MINI,
        "gemini": ModelType.GEMINI_FLASH,
    }
    model_types = [model_map[m] for m in models]
    
    harness = ValidationHarness(
        models=model_types,
        equivalence_threshold=threshold,
        use_llm_judge=False,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    passed = 0
    failed = 0
    
    sem = asyncio.Semaphore(concurrency)
    
    async def validate_one(pair: GeneratedPair, idx: int) -> bool:
        async with sem:
            compression_pair = CompressionPair(
                verbose=pair.verbose,
                compressed=pair.compressed,
                domain=pair.domain,
            )
            result = await harness.validate_pair(compression_pair)
            return result.min_equivalence >= threshold, pair, result
    
    with Progress() as progress:
        task = progress.add_task("Validating...", total=len(pairs))
        
        with open(output_path, "w") as f:
            for i in range(0, len(pairs), concurrency * 2):
                batch = pairs[i:i + concurrency * 2]
                results = await asyncio.gather(*[
                    validate_one(p, j) for j, p in enumerate(batch, i)
                ])
                
                for is_pass, pair, result in results:
                    if is_pass:
                        passed += 1
                        f.write(json.dumps(pair.model_dump()) + "\n")
                    else:
                        failed += 1
                    progress.advance(task)
    
    return {"passed": passed, "failed": failed, "total": len(pairs)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=["claude", "gpt"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    pairs = load_pairs(args.input)
    if args.limit:
        pairs = pairs[:args.limit]
    
    console.print(f"[bold]Validating {len(pairs)} pairs[/bold]")
    
    stats = asyncio.run(validate_batch(
        pairs,
        args.output,
        threshold=args.threshold,
        concurrency=args.concurrency,
        models=args.models,
    ))
    
    table = Table(title="Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Passed", str(stats["passed"]))
    table.add_row("Failed", str(stats["failed"]))
    table.add_row("Pass Rate", f"{stats['passed'] / stats['total'] * 100:.1f}%")
    console.print(table)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Commit**

```bash
git add scripts/validate_synthetic.py
git commit -m "feat: add batch validation script for synthetic data"
```

---

### Task 6: Validate synthetic pairs

**Step 1: Validate code pairs**

```bash
source .venv/bin/activate && python scripts/validate_synthetic.py \
  --input data/synthetic/code_v2.jsonl \
  --output data/validated/code_v2.jsonl \
  --threshold 0.80 \
  --concurrency 4 \
  --models claude gpt
```

**Step 2: Validate NL pairs**

```bash
source .venv/bin/activate && python scripts/validate_synthetic.py \
  --input data/synthetic/nl_v2.jsonl \
  --output data/validated/nl_v2.jsonl \
  --threshold 0.80 \
  --concurrency 4 \
  --models claude gpt
```

---

### Task 7: Merge and format training data

**Step 1: Merge validated pairs**

```bash
cat data/validated/code_v2.jsonl data/validated/nl_v2.jsonl > data/validated/all_v2.jsonl
wc -l data/validated/all_v2.jsonl
```

**Step 2: Format for training**

```bash
source .venv/bin/activate && python scripts/format_training_data.py \
  --input data/validated \
  --output data/training_v2 \
  --train-ratio 0.9 \
  --valid-ratio 0.05 \
  --test-ratio 0.05
```

---

## Phase 3: Tinker Production Training

### Task 8: Update Tinker training config

**Files:**
- Modify: `configs/training.yaml`

**Step 1: Update config**

```yaml
# configs/training.yaml

# === LOCAL (MLX on M4 Pro) ===
local:
  model: "mlx-community/Qwen3-4B-Instruct-2507-8bit"
  lora:
    rank: 8
    alpha: 16
  training:
    iters: 500
    batch_size: 2
    learning_rate: 1.0e-4

# === CLOUD (Tinker) - PRODUCTION ===
cloud:
  model: "Qwen/Qwen3-8B"
  lora:
    rank: 64
    alpha: 128
    dropout: 0.05
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
    learning_rate: 2.0e-4
    warmup_ratio: 0.03
    max_seq_length: 2048

# === DATA ===
data:
  train_path: "data/training_v2/train.jsonl"
  valid_path: "data/training_v2/valid.jsonl"
  test_path: "data/training_v2/test.jsonl"
```

**Step 2: Commit**

```bash
git add configs/training.yaml
git commit -m "chore: update training config for v2 production run"
```

---

### Task 9: Run Tinker training

**Step 1: Estimate cost**

```bash
source .venv/bin/activate && python -c "
from src.training.train_tinker import estimate_cost, TinkerTrainingConfig
config = TinkerTrainingConfig(model='Qwen/Qwen3-8B', epochs=3)
estimate = estimate_cost(config, num_examples=10000)
print(f'Estimated cost: \${estimate[\"estimated_cost_usd\"]:.2f}')
print(f'Total tokens: {estimate[\"total_tokens\"]:,}')
"
```

**Step 2: Run training**

```bash
source .venv/bin/activate && python scripts/train_tinker.py \
  --config configs/training.yaml \
  --data data/training_v2 \
  --output models/adapters/tinker_v2
```

---

### Task 10: Evaluate Tinker model

**Step 1: Download adapter** (if not automatic)

```bash
# Adapter should be at models/adapters/tinker_v2/
ls -la models/adapters/tinker_v2/
```

**Step 2: Convert for MLX inference** (if needed)

The Tinker adapter may need conversion for MLX. Check format and convert if necessary.

**Step 3: Run evaluation**

```bash
source .venv/bin/activate && python scripts/evaluate_adapter.py \
  --model Qwen/Qwen3-8B \
  --adapter-path models/adapters/tinker_v2 \
  --data data/training_v2/test.jsonl \
  --limit 100 \
  --models claude gpt \
  --equivalence-threshold 0.80 \
  --output models/eval/tinker_v2_eval.jsonl
```

**Step 4: Compare with v1**

| Metric | V1 (MLX 4B) | V2 (Tinker 8B) | Target |
|--------|-------------|----------------|--------|
| Pass Rate | 89% | ? | >90% |
| Compression | 62% | ? | 50-60% |
| Avg Equiv | 0.859 | ? | >0.85 |

---

## Phase 4: Release

### Task 11: Prepare release artifacts

**Files:**
- Create: `models/release/README.md` (model card)
- Create: `models/release/config.json`

**Step 1: Create model card**

```markdown
# Semantic Compression LoRA - Qwen3-8B

A LoRA adapter trained to compress verbose text into minimal tokens while preserving semantic meaning for LLM reasoning.

## Performance

| Metric | Value |
|--------|-------|
| Pass Rate (0.80 threshold) | X% |
| Avg Compression Ratio | X% |
| Avg Equivalence Score | X |
| Training Examples | 10K+ |

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("Qwen/Qwen3-8B", adapter_path="path/to/adapter")

messages = [
    {"role": "system", "content": "You are a semantic compression engine..."},
    {"role": "user", "content": "Compress:\n<your text>"},
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
output = generate(model, tokenizer, prompt=prompt, max_tokens=512)
```

## Training Details

- Base Model: Qwen/Qwen3-8B
- Training Platform: Tinker
- LoRA Rank: 64
- Training Examples: 10K+
- Epochs: 3

## License

MIT
```

**Step 2: Commit release artifacts**

```bash
git add models/release/
git commit -m "docs: add model card and release artifacts"
```

---

### Task 12: Upload to HuggingFace (optional)

**Step 1: Create HF repo**

```bash
huggingface-cli repo create semantic-compression-lora --type model
```

**Step 2: Upload adapter**

```bash
huggingface-cli upload semantic-compression-lora models/adapters/tinker_v2/
```

---

## Summary Checklist

### Phase 1: Synthetic Generation
- [ ] Create adapter generator module
- [ ] Create batch generation script
- [ ] Generate 5K code pairs (~2-4 hours)
- [ ] Generate 5K NL pairs (~2-4 hours)

### Phase 2: Validation
- [ ] Create batch validation script
- [ ] Validate code pairs (~$20-40 API cost)
- [ ] Validate NL pairs (~$20-40 API cost)
- [ ] Merge and format training data

### Phase 3: Tinker Training
- [ ] Update training config
- [ ] Run Tinker training (~$6-15)
- [ ] Download and evaluate adapter
- [ ] Compare with v1 baseline

### Phase 4: Release
- [ ] Create model card
- [ ] Package release artifacts
- [ ] Upload to HuggingFace (optional)

### Estimated Costs
| Item | Estimate |
|------|----------|
| Validation API calls | $40-80 |
| Tinker training | $10-20 |
| Evaluation | $5-10 |
| **Total** | **$55-110** |

### Estimated Time
| Task | Duration |
|------|----------|
| Code generation | 2-4 hours |
| NL generation | 2-4 hours |
| Validation | 4-8 hours |
| Tinker training | 1-2 hours |
| Evaluation | 1 hour |
| **Total** | **10-19 hours** |

---

## Notes & Risks

1. **Generation quality**: V1 adapter may produce some low-quality compressions - validation will filter these
2. **API rate limits**: May need to adjust concurrency for validation
3. **Tinker availability**: Ensure API key is valid and quota is available
4. **Model conversion**: Tinker adapter format may differ from MLX - may need conversion step
5. **Diminishing returns**: More data doesn't always mean better - monitor quality metrics

## Success Criteria

- [ ] 10K+ validated training pairs (>80% pass rate from generation)
- [ ] V2 model achieves >90% pass rate at 0.80 threshold
- [ ] V2 compression ratio between 50-65%
- [ ] V2 avg equivalence >0.85
- [ ] Model packaged and documented for release
