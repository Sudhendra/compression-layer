"""Model type definitions and async API clients for validation."""

import asyncio
from enum import Enum
from typing import Any

from pydantic import BaseModel

from ..utils.caching import SemanticCache, make_cache_key
from ..utils.config import get_settings
from ..utils.costs import get_cost_tracker


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
        max_retries: int = 3,
    ):
        self.model_type = model_type
        self.cache = cache
        self.max_retries = max_retries
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
        temperature: float = 0.4,
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
            operation="validation",
        )

        return response.text

    async def _call_with_retry(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> APIResponse:
        """Call API with exponential backoff retry."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return await self._call_api(prompt, max_tokens, temperature)
            except Exception as e:
                last_error = e
                # Check for rate limit errors
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise

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
