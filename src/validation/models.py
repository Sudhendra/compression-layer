"""Model type definitions and async API clients for validation."""

import asyncio
import random
from enum import Enum
from typing import Any

from pydantic import BaseModel
from rich.console import Console

from ..utils.caching import SemanticCache, make_cache_key
from ..utils.config import get_settings
from ..utils.costs import get_cost_tracker

console = Console()

# Retryable error patterns (rate limits + server errors)
RETRYABLE_PATTERNS = [
    "rate",
    "429",
    "500",
    "502",
    "503",
    "520",
    "overloaded",
    "internal server error",
    "server error",
    "temporarily unavailable",
    "capacity",
]


class ModelType(Enum):
    """Supported models for cross-model validation."""

    CLAUDE_SONNET = "claude-sonnet-4-20250514"
    GPT4O_MINI = "gpt-4o-mini"
    GEMINI_FLASH = "gemini-2.0-flash"


class APIResponse(BaseModel):
    """Standardized API response."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str


class ModelClient:
    """
    Async client wrapper for multiple LLM providers.

    Supports Claude, GPT, and Gemini with retry logic and caching.
    """

    def __init__(
        self,
        model_type: ModelType,
        cache: SemanticCache | None = None,
        max_retries: int = 5,
        operation: str = "validation",
    ):
        self.model_type = model_type
        self.cache = cache
        self.max_retries = max_retries
        self.operation = operation
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate API client."""
        settings = get_settings()

        match self.model_type:
            case ModelType.CLAUDE_SONNET:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)

            case ModelType.GPT4O_MINI:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=settings.openai_api_key)

            case ModelType.GEMINI_FLASH:
                from google import genai

                self._client = genai.Client(api_key=settings.google_api_key)

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,  # Deterministic for validation
        use_cache: bool = True,
    ) -> str:
        """
        Generate a completion from the model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use cached responses

        Returns:
            Generated text response
        """
        # Build cache key
        cache_key = make_cache_key(
            self.model_type.value,
            prompt,
            max_tokens=max_tokens,
            temp=temperature,
        )

        # Check cache first
        if use_cache and self.cache and (cached := self.cache.get(cache_key)):
            return str(cached["text"])

        # Call API with retry
        response = await self._call_with_retry(prompt, max_tokens, temperature)

        # Cache the response
        if use_cache and self.cache:
            self.cache.set(cache_key, {"text": response.text})

        # Log cost
        tracker = get_cost_tracker()
        tracker.log_call(
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            operation=self.operation,
        )

        return response.text

    async def _call_with_retry(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> APIResponse:
        """
        Call API with exponential backoff retry for transient errors.

        Handles:
        - Rate limits (429)
        - Server errors (500, 502, 503, 520)
        - Overloaded/capacity errors

        Uses exponential backoff with jitter to avoid thundering herd.
        """
        last_error: Exception | None = None
        base_wait = 2.0  # Base wait time in seconds

        for attempt in range(self.max_retries):
            try:
                return await self._call_api(prompt, max_tokens, temperature)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if error is retryable
                is_retryable = any(pattern in error_str for pattern in RETRYABLE_PATTERNS)

                if is_retryable:
                    # Exponential backoff with jitter
                    wait_time = base_wait * (2**attempt) + random.uniform(0, 1)
                    max_wait = 60.0  # Cap at 60 seconds

                    wait_time = min(wait_time, max_wait)

                    console.print(
                        f"[yellow]Retryable error on {self.model_type.value} "
                        f"(attempt {attempt + 1}/{self.max_retries}): "
                        f"{type(e).__name__}. Waiting {wait_time:.1f}s...[/yellow]"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Non-retryable error, raise immediately
                    raise

        # All retries exhausted
        console.print(
            f"[red]Max retries ({self.max_retries}) exceeded for {self.model_type.value}[/red]"
        )
        raise last_error or Exception("Max retries exceeded")

    async def _call_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> APIResponse:
        """Make the actual API call based on model type."""
        match self.model_type:
            case ModelType.CLAUDE_SONNET:
                return await self._call_claude(prompt, max_tokens, temperature)
            case ModelType.GPT4O_MINI:
                return await self._call_openai(prompt, max_tokens, temperature)
            case ModelType.GEMINI_FLASH:
                return await self._call_gemini(prompt, max_tokens, temperature)

    async def _call_claude(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> APIResponse:
        """Call Claude API."""
        response = await self._client.messages.create(
            model=self.model_type.value,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return APIResponse(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model_type.value,
        )

    async def _call_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> APIResponse:
        """Call OpenAI API."""
        response = await self._client.chat.completions.create(
            model=self.model_type.value,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return APIResponse(
            text=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=self.model_type.value,
        )

    async def _call_gemini(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> APIResponse:
        """Call Gemini API."""
        from google.genai import types

        response = await self._client.aio.models.generate_content(
            model=self.model_type.value,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        # Gemini usage metadata
        usage = response.usage_metadata
        return APIResponse(
            text=response.text or "",
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            model=self.model_type.value,
        )


def model_type_from_string(model_name: str) -> ModelType:
    """Convert a model name string to ModelType enum."""
    for mt in ModelType:
        if mt.value == model_name:
            return mt
    raise ValueError(f"Unknown model: {model_name}")
