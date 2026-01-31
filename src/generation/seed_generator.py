"""Seed data generator for compression pairs using Claude API."""

import asyncio
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..utils.caching import SemanticCache, make_cache_key
from ..utils.config import GenerationConfig, get_settings
from ..utils.costs import get_cost_tracker
from ..validation.models import ModelClient, model_type_from_string

console = Console()


class GeneratedPair(BaseModel):
    """A generated compression pair."""

    verbose: str
    compressed: str
    domain: Literal["nl", "code", "mixed"]
    language: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Result of a generation batch."""

    pairs: list[GeneratedPair]
    total_input_tokens: int
    total_output_tokens: int
    cached_count: int
    generated_count: int


class SeedGenerator:
    """
    Generates compression pairs using Claude API.

    Uses caching to avoid regenerating identical inputs and tracks
    API costs throughout the generation process.
    """

    def __init__(
        self,
        config: GenerationConfig | None = None,
        cache: SemanticCache | None = None,
    ):
        """
        Initialize the seed generator.

        Args:
            config: Generation configuration (uses defaults if None)
            cache: Optional semantic cache for deduplication
        """
        self.config = config or GenerationConfig()
        self.settings = get_settings()

        # Initialize cache
        if cache is None:
            cache_dir = self.settings.cache_dir / "generation"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache = SemanticCache(cache_dir)
        self.cache = cache

        # Load prompts
        prompts_dir = Path(__file__).parent / "prompts"
        self.nl_prompt = (prompts_dir / "compress_nl.txt").read_text()
        self.code_prompt = (prompts_dir / "compress_code.txt").read_text()

        # Initialize model client
        primary_model = model_type_from_string(self.config.primary_model)
        self.client = ModelClient(
            primary_model,
            cache=None,  # We handle caching at the generator level
            max_retries=self.config.max_retries,
            operation="generation",
        )
        self.cost_tracker = get_cost_tracker()

    def _format_prompt(self, template: str, input_text: str) -> str:
        """
        Format a prompt template by replacing {input} placeholder.

        Uses simple string replacement instead of .format() to avoid
        issues with curly braces in example code within the prompts.
        """
        return template.replace("{input}", input_text)

    async def generate_nl_pair(
        self,
        input_text: str,
        use_cache: bool = True,
    ) -> GeneratedPair:
        """
        Generate a compressed version of natural language text.

        Args:
            input_text: Verbose natural language text to compress
            use_cache: Whether to check cache first

        Returns:
            GeneratedPair with original and compressed text
        """
        cache_key = make_cache_key("nl", input_text, model=self.config.primary_model)

        # Check cache
        if use_cache and (cached := self.cache.get(cache_key)):
            return GeneratedPair(**cached)

        # Generate compression
        prompt = self._format_prompt(self.nl_prompt, input_text)
        compressed = await self.client.complete(
            prompt,
            temperature=self.config.temperature,
        )

        pair = GeneratedPair(
            verbose=input_text,
            compressed=compressed.strip(),
            domain="nl",
        )

        # Cache result
        if use_cache:
            self.cache.set(cache_key, pair.model_dump())

        return pair

    async def generate_code_pair(
        self,
        input_code: str,
        language: str = "python",
        use_cache: bool = True,
    ) -> GeneratedPair:
        """
        Generate a compressed version of code.

        Args:
            input_code: Verbose code to compress
            language: Programming language
            use_cache: Whether to check cache first

        Returns:
            GeneratedPair with original and compressed code
        """
        cache_key = make_cache_key(
            "code",
            input_code,
            model=self.config.primary_model,
            lang=language,
        )

        # Check cache
        if use_cache and (cached := self.cache.get(cache_key)):
            return GeneratedPair(**cached)

        # Generate compression
        prompt = self._format_prompt(self.code_prompt, input_code)
        compressed = await self.client.complete(
            prompt,
            temperature=self.config.temperature,
        )

        pair = GeneratedPair(
            verbose=input_code,
            compressed=compressed.strip(),
            domain="code",
            language=language,
        )

        # Cache result
        if use_cache:
            self.cache.set(cache_key, pair.model_dump())

        return pair

    async def generate_batch(
        self,
        inputs: list[str],
        domain: Literal["nl", "code"] = "nl",
        language: str = "python",
        concurrency: int | None = None,
        show_progress: bool = True,
        output_path: Path | None = None,
    ) -> GenerationResult:
        """
        Generate compression pairs for a batch of inputs.

        Args:
            inputs: List of verbose texts to compress
            domain: Domain type ("nl" or "code")
            language: Programming language (for code domain)
            concurrency: Max concurrent requests (defaults to config)
            show_progress: Whether to show progress bar
            output_path: If provided, save pairs incrementally to this file

        Returns:
            GenerationResult with all pairs and statistics
        """
        concurrency = concurrency or self.config.concurrency
        sem = asyncio.Semaphore(concurrency)

        pairs: list[GeneratedPair] = []
        cached_count = 0
        generated_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        lock = asyncio.Lock()

        # Prepare output file for incremental writes
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        def save_pair_to_file(pair: GeneratedPair) -> None:
            """Save a single pair to the output file (append mode)."""
            if output_path:
                with open(output_path, "a") as f:
                    f.write(json.dumps(pair.model_dump()) + "\n")

        async def process_one(text: str, idx: int) -> GeneratedPair:
            nonlocal cached_count, generated_count
            async with sem:
                # Check cache first to count cached items
                if domain == "nl":
                    cache_key = make_cache_key("nl", text, model=self.config.primary_model)
                else:
                    cache_key = make_cache_key(
                        "code", text, model=self.config.primary_model, lang=language
                    )

                is_cached = self.cache.get(cache_key) is not None

                # Generate pair
                if domain == "nl":
                    pair = await self.generate_nl_pair(text)
                else:
                    pair = await self.generate_code_pair(text, language)

                # Thread-safe counter update and file write
                async with lock:
                    if is_cached:
                        cached_count += 1
                    else:
                        generated_count += 1

                    # Write to file incrementally
                    save_pair_to_file(pair)

                return pair

        if show_progress:
            from rich.progress import BarColumn, TaskProgressColumn, TimeRemainingColumn

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Generating {len(inputs)} {domain} pairs...",
                    total=len(inputs),
                )

                # Process in parallel with progress tracking
                async def process_with_progress(text: str, idx: int) -> GeneratedPair:
                    pair = await process_one(text, idx)
                    progress.advance(task)
                    return pair

                pairs = await asyncio.gather(
                    *[process_with_progress(text, i) for i, text in enumerate(inputs)]
                )
        else:
            # Process in parallel without progress bar
            pairs = await asyncio.gather(*[process_one(text, i) for i, text in enumerate(inputs)])

        return GenerationResult(
            pairs=list(pairs),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            cached_count=cached_count,
            generated_count=generated_count,
        )

    def save_pairs(
        self,
        pairs: list[GeneratedPair],
        output_path: Path,
        append: bool = False,
    ) -> None:
        """
        Save generated pairs to a JSONL file.

        Args:
            pairs: List of GeneratedPair objects
            output_path: Path to output file
            append: Whether to append to existing file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"

        with open(output_path, mode) as f:
            for pair in pairs:
                f.write(json.dumps(pair.model_dump()) + "\n")

        console.print(f"[green]Saved {len(pairs)} pairs to {output_path}[/green]")

    @staticmethod
    def load_pairs(input_path: Path) -> list[GeneratedPair]:
        """
        Load pairs from a JSONL file.

        Args:
            input_path: Path to JSONL file

        Returns:
            List of GeneratedPair objects
        """
        pairs = []
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    pairs.append(GeneratedPair(**json.loads(line)))
        return pairs


async def generate_seed_pairs(
    inputs: list[str],
    domain: Literal["nl", "code"] = "nl",
    language: str = "python",
    output_path: Path | None = None,
) -> list[GeneratedPair]:
    """
    Convenience function to generate and optionally save seed pairs.

    Args:
        inputs: List of verbose texts
        domain: Domain type
        language: Programming language (for code)
        output_path: Optional path to save results

    Returns:
        List of generated pairs
    """
    generator = SeedGenerator()
    result = await generator.generate_batch(inputs, domain=domain, language=language)

    if output_path:
        generator.save_pairs(result.pairs, output_path)

    return result.pairs
