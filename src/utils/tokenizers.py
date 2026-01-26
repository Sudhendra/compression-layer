"""Multi-tokenizer utilities for token counting across different models."""

from enum import Enum
from functools import lru_cache

import tiktoken


class TokenizerType(Enum):
    """Supported tokenizer types for different model families."""

    OPENAI = "cl100k_base"  # GPT-4, GPT-4o, etc.
    CLAUDE = "cl100k_base"  # Claude uses similar tokenization
    GEMINI = "cl100k_base"  # Approximate with cl100k
    QWEN = "cl100k_base"  # Qwen uses its own, but cl100k is close approximation


@lru_cache(maxsize=4)
def get_tokenizer(tokenizer_type: TokenizerType = TokenizerType.OPENAI) -> tiktoken.Encoding:
    """Get a cached tokenizer instance."""
    return tiktoken.get_encoding(tokenizer_type.value)


def count_tokens(text: str, tokenizer_type: TokenizerType = TokenizerType.OPENAI) -> int:
    """
    Count tokens in text using the specified tokenizer.

    Args:
        text: Input text to tokenize
        tokenizer_type: Which tokenizer to use

    Returns:
        Number of tokens
    """
    enc = get_tokenizer(tokenizer_type)
    return len(enc.encode(text))


def count_tokens_multi(text: str) -> dict[str, int]:
    """
    Count tokens across all supported tokenizers.

    Returns:
        Dict mapping tokenizer name to token count
    """
    return {
        "openai": count_tokens(text, TokenizerType.OPENAI),
        "claude": count_tokens(text, TokenizerType.CLAUDE),
        "gemini": count_tokens(text, TokenizerType.GEMINI),
    }


def estimate_cost(
    text: str,
    model: str,
    is_input: bool = True,
) -> float:
    """
    Estimate API cost for text based on model pricing.

    Pricing as of 2024 (per 1M tokens):
    - Claude Sonnet: $3 input, $15 output
    - GPT-4o-mini: $0.15 input, $0.60 output
    - Gemini Flash: $0.075 input, $0.30 output
    """
    pricing = {
        # (input_per_1m, output_per_1m)
        "claude-sonnet-4-20250514": (3.0, 15.0),
        "gpt-4o-mini": (0.15, 0.60),
        "gemini-2.0-flash": (0.075, 0.30),
    }

    tokens = count_tokens(text)
    rate_input, rate_output = pricing.get(model, (1.0, 1.0))
    rate = rate_input if is_input else rate_output

    return (tokens / 1_000_000) * rate


def compression_ratio(original: str, compressed: str) -> float:
    """
    Calculate token compression ratio.

    Returns:
        Ratio of compressed to original tokens (lower is better).
        0.5 means 50% compression (half the tokens).
    """
    original_tokens = count_tokens(original)
    compressed_tokens = count_tokens(compressed)

    if original_tokens == 0:
        return 1.0

    return compressed_tokens / original_tokens


def token_reduction_percent(original: str, compressed: str) -> float:
    """
    Calculate percentage of tokens reduced.

    Returns:
        Percentage reduction (higher is better).
        50.0 means 50% fewer tokens.
    """
    return (1 - compression_ratio(original, compressed)) * 100
