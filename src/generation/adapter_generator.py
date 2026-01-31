"""Generate compressions using trained MLX adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


class AdapterGenerator:
    """Generate compressions using a trained LoRA adapter."""

    def __init__(
        self,
        model: str = "mlx-community/Qwen3-4B-Instruct-2507-8bit",
        adapter_path: Path | None = None,
        system_prompt: str | None = None,
        temp: float = 0.2,
        load_fn: Callable | None = None,
        generate_fn: Callable | None = None,
        sampler_factory: Callable | None = None,
    ) -> None:
        self.model_name = model
        self.adapter_path = adapter_path
        self.system_prompt = system_prompt or (
            "You are a semantic compression engine. Compress the input into minimal tokens "
            "while preserving all information for equivalent LLM reasoning. Use dense notation: "
            "labeled fields, standard abbreviations, and symbols (-> | + @). Never lose information."
        )

        if load_fn is None or generate_fn is None or sampler_factory is None:
            import mlx_lm
            from mlx_lm.sample_utils import make_sampler

            load_fn = load_fn or mlx_lm.load
            generate_fn = generate_fn or mlx_lm.generate
            sampler_factory = sampler_factory or make_sampler

        self._generate = generate_fn
        self._sampler = sampler_factory(temp)
        loaded = load_fn(model, adapter_path=str(adapter_path) if adapter_path else None)
        self.mlx_model, self.tokenizer = loaded[:2]

    def compress(self, text: str, max_tokens: int = 512) -> str:
        """Compress input text using the adapter."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Compress:\n{text}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        output = self._generate(
            model=self.mlx_model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=self._sampler,
        )

        output = output.strip()
        output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
        output = output.replace("</tool_call>", "").strip()

        if output.startswith("```") and output.endswith("```"):
            lines = output.split("\n")
            if len(lines) >= 2:
                output = "\n".join(lines[1:-1]).strip()
            else:
                output = output[3:-3].strip()

        return output

    def compress_batch(
        self,
        texts: list[str],
        max_tokens: int = 512,
        show_progress: bool = True,
    ) -> list[str]:
        """Compress multiple texts."""
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

        results: list[str] = []
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
            ) as progress:
                task = progress.add_task("Compressing...", total=len(texts))
                for text in texts:
                    results.append(self.compress(text, max_tokens))
                    progress.advance(task)
        else:
            for text in texts:
                results.append(self.compress(text, max_tokens))

        return results
