"""Evaluate compression adapters with task-equivalence metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rich.console import Console
from rich.table import Table

from src.inference.domain_classifier import DomainClassifier
from src.training.train_mlx import evaluate_adapter as evaluate_ppl
from src.utils.config import get_settings
from src.utils.costs import get_cost_tracker
from src.validation.harness import CompressionPair, ValidationHarness, ValidationResult
from src.validation.metrics import TaskType
from src.validation.models import ModelType

console = Console()


@dataclass(frozen=True)
class EvaluationExample:
    """Input example to evaluate with the adapter."""

    input_text: str
    domain: str


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics captured for a single evaluation example."""

    input_text: str
    compressed_text: str
    domain: str
    compression_ratio: float
    min_equivalence: float
    equivalence_scores: dict[str, float]
    passed: bool
    duration_ms: float


class HarnessProtocol(Protocol):
    async def validate_pair(self, pair: CompressionPair) -> ValidationResult: ...


def load_test_examples(path: Path, limit: int | None = None) -> list[EvaluationExample]:
    """Load evaluation examples from MLX chat-format JSONL."""
    examples: list[EvaluationExample] = []
    if not path.exists():
        raise FileNotFoundError(f"Test data not found: {path}")

    classifier = DomainClassifier()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            messages = payload.get("messages", [])
            user_text = None
            for message in messages:
                if message.get("role") == "user":
                    user_text = message.get("content", "")
                    break
            if not user_text:
                continue
            input_text = user_text.replace("Compress:\n", "").strip()
            if not input_text:
                continue
            domain = classifier.classify(input_text).value
            examples.append(EvaluationExample(input_text=input_text, domain=domain))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def create_generator(
    model: str, adapter_path: Path, system_prompt: str, temp: float = 0.2
) -> Callable[[str], str]:
    """Create a generator function for MLX model + adapter with proper chat template."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    loaded = mlx_lm.load(model, adapter_path=str(adapter_path))
    mlx_model, tokenizer = loaded[:2]
    sampler = make_sampler(temp=temp)

    def _generate(input_text: str) -> str:
        # Build proper chat format with system prompt
        messages = [{"role": "user", "content": f"Compress:\n{input_text}"}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        output = mlx_lm.generate(
            model=mlx_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=512,
            sampler=sampler,
        )
        # Strip Qwen3 thinking artifacts
        output = output.strip()

        # Remove <think>...</think> blocks (can appear at start or after other artifacts)
        import re

        output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

        # Remove </tool_call> artifacts (sometimes leaked)
        output = output.replace("</tool_call>", "").strip()

        # Remove leading/trailing backticks if they wrap the entire output
        if output.startswith("```") and output.endswith("```"):
            # Keep the content but remove wrapper backticks
            lines = output.split("\n")
            if len(lines) >= 2:
                # Remove first line (```lang) and last line (```)
                output = "\n".join(lines[1:-1]).strip()

        return output

    return _generate


async def run_evaluation(
    *,
    examples: list[EvaluationExample],
    generator: Callable[[str], str],
    harness: HarnessProtocol,
    output_path: Path,
    concurrency: int = 2,
    min_equivalence: float = 0.0,
) -> list[EvaluationResult]:
    """Evaluate adapter by generating compressions and scoring equivalence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    sem = asyncio.Semaphore(concurrency)
    results: list[EvaluationResult] = []
    lock = asyncio.Lock()
    completed = 0

    console.print(
        f"[cyan]Evaluating {len(examples)} examples with concurrency={concurrency}[/cyan]\n"
    )

    async def evaluate_one(example: EvaluationExample, idx: int) -> EvaluationResult:
        nonlocal completed
        async with sem:
            start = time.perf_counter()
            # Generator now handles chat template internally
            compressed = generator(example.input_text)
            gen_ms = (time.perf_counter() - start) * 1000

            pair = CompressionPair(
                verbose=example.input_text,
                compressed=compressed,
                domain=example.domain,
            )
            validation = await harness.validate_pair(pair)
            duration_ms = (time.perf_counter() - start) * 1000

            result = EvaluationResult(
                input_text=example.input_text,
                compressed_text=compressed,
                domain=example.domain,
                compression_ratio=validation.compression_ratio,
                min_equivalence=validation.min_equivalence,
                equivalence_scores={
                    model.value: score for model, score in validation.equivalence_scores.items()
                },
                passed=validation.min_equivalence >= min_equivalence,
                duration_ms=duration_ms,
            )

            async with lock:
                results.append(result)
                completed += 1
                output_path.open("a", encoding="utf-8").write(json.dumps(result.__dict__) + "\n")

                # Print progress for each completed example
                status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
                preview = example.input_text[:50].replace("\n", " ")
                if len(example.input_text) > 50:
                    preview += "..."
                console.print(
                    f"[{completed}/{len(examples)}] {status} "
                    f"ratio={result.compression_ratio:.1%} "
                    f"equiv={result.min_equivalence:.3f} "
                    f"gen={gen_ms:.0f}ms total={duration_ms:.0f}ms "
                    f"[dim]{preview}[/dim]"
                )

            return result

    await asyncio.gather(*(evaluate_one(example, i) for i, example in enumerate(examples)))
    console.print()  # blank line before summary
    return results


def print_summary(results: list[EvaluationResult]) -> None:
    if not results:
        console.print("[red]No evaluation results to summarize.[/red]")
        return

    avg_ratio = sum(r.compression_ratio for r in results) / len(results)
    avg_equiv = sum(r.min_equivalence for r in results) / len(results)
    min_equiv = min(r.min_equivalence for r in results)
    avg_latency = sum(r.duration_ms for r in results) / len(results)

    table = Table(title="Adapter Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Examples", str(len(results)))
    table.add_row("Avg compression ratio", f"{avg_ratio:.2%}")
    table.add_row("Avg equivalence", f"{avg_equiv:.3f}")
    table.add_row("Min equivalence", f"{min_equiv:.3f}")
    table.add_row("Avg latency", f"{avg_latency:.1f} ms")
    console.print(table)

    model_table = Table(title="Per-Model Equivalence")
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Avg", style="green")
    model_table.add_column("Min", style="yellow")
    per_model: dict[str, list[float]] = {}
    for result in results:
        for model_name, score in result.equivalence_scores.items():
            per_model.setdefault(model_name, []).append(score)

    for model_name, scores in per_model.items():
        model_table.add_row(
            model_name,
            f"{sum(scores) / len(scores):.3f}",
            f"{min(scores):.3f}",
        )
    console.print(model_table)

    cost_tracker = get_cost_tracker()
    cost_table = Table(title="Cost Summary")
    cost_table.add_column("Metric", style="cyan")
    cost_table.add_column("Value", style="green")
    cost_table.add_row("Today's spend", f"${cost_tracker.get_daily_spend():.2f}")
    cost_table.add_row("Total spend", f"${cost_tracker.get_total_spend():.2f}")
    console.print(cost_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate adapter quality with task-equivalence metrics.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-4B-Instruct-2507-8bit",
        help="Base model used for adapter.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Path to adapter to evaluate.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/training/test.jsonl"),
        help="Path to test JSONL (chat format).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/eval/adapter_eval.jsonl"),
        help="Output path for evaluation results.",
    )
    parser.add_argument(
        "--equivalence-threshold",
        type=float,
        default=0.72,
        help="Equivalence threshold for pass/fail reporting.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["claude", "gpt", "gemini"],
        default=["claude", "gpt", "gemini"],
        help="Models used to score equivalence.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["qa", "reasoning", "code_gen"],
        help="Tasks for equivalence scoring.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Concurrent evaluations.",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLM-as-judge in addition to embeddings.",
    )
    parser.add_argument(
        "--include-system-prompt",
        action="store_true",
        help="Prefix generation with system prompt.",
    )
    parser.add_argument(
        "--measure-ppl",
        action="store_true",
        help="Also compute MLX test perplexity.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = get_settings()

    if not args.adapter_path.exists():
        console.print(f"[red]Adapter path not found: {args.adapter_path}[/red]")
        return 1

    examples = load_test_examples(args.data, limit=args.limit)
    if not examples:
        console.print("[red]No examples found to evaluate.[/red]")
        return 1

    model_map = {
        "claude": ModelType.CLAUDE_SONNET,
        "gpt": ModelType.GPT4O_MINI,
        "gemini": ModelType.GEMINI_FLASH,
    }
    models = [model_map[m] for m in args.models]

    tasks = None
    if args.tasks:
        task_map = {
            "qa": TaskType.QA,
            "reasoning": TaskType.REASONING,
            "code_gen": TaskType.CODE_GEN,
        }
        tasks = [task_map[t] for t in args.tasks]

    cache_dir = settings.cache_dir / "evaluation"
    cache_dir.mkdir(parents=True, exist_ok=True)

    harness = ValidationHarness(
        models=models,
        equivalence_threshold=args.equivalence_threshold,
        tasks=tasks,
        cache=None,
        use_llm_judge=args.use_llm_judge,
    )

    if args.measure_ppl:
        ppl = evaluate_ppl(args.model, args.adapter_path, args.data.parent)
        if ppl is not None:
            console.print(f"[green]Test Perplexity: {ppl:.2f}[/green]")

    system_prompt = settings.data_dir / "training" / "system_prompt.txt"
    if system_prompt.exists():
        system_prompt_text = system_prompt.read_text(encoding="utf-8")
    else:
        system_prompt_text = (
            "You are a semantic compression engine. Compress the input into minimal tokens "
            "while preserving all information for equivalent LLM reasoning. Use dense notation: "
            "labeled fields, standard abbreviations, and symbols (→ | + @). Never lose information."
        )

    # Pass system prompt to generator (it handles chat template internally)
    generator = create_generator(
        args.model,
        args.adapter_path,
        system_prompt=system_prompt_text if args.include_system_prompt else "",
    )
    results = await run_evaluation(
        examples=examples,
        generator=generator,
        harness=harness,
        output_path=args.output,
        concurrency=args.concurrency,
        min_equivalence=args.equivalence_threshold,
    )

    print_summary(results)
    console.print(f"[green]Saved evaluation results to {args.output}[/green]")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
