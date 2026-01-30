"""Format validated compression pairs for training.

Converts validated pairs from data/validated/ into training-ready JSONL files
in chat format (messages with system/user/assistant roles) for MLX and Tinker.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# System prompt for compression task
COMPRESSION_SYSTEM_PROMPT = (
    "You are a semantic compression engine. Compress the input into minimal tokens "
    "while preserving all information for equivalent LLM reasoning. Use dense notation: "
    "labeled fields, standard abbreviations, and symbols (→ | + @). Never lose information."
)

# User prompt template
USER_PROMPT_TEMPLATE = "Compress:\n{verbose}"


class ValidatedPair(BaseModel):
    """A validated compression pair from the validation pipeline."""

    verbose: str
    compressed: str
    domain: str
    metadata: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    """A single message in chat format."""

    role: str
    content: str


class ChatExample(BaseModel):
    """A training example in chat format."""

    messages: list[ChatMessage]


@dataclass
class SplitStats:
    """Statistics from data splitting."""

    total: int
    train: int
    valid: int
    test: int
    nl_count: int
    code_count: int


def load_validated_pairs(validated_dir: Path) -> list[ValidatedPair]:
    """
    Load all validated pairs from the validated directory.

    Args:
        validated_dir: Path to data/validated/

    Returns:
        List of validated compression pairs
    """
    pairs: list[ValidatedPair] = []

    # Load NL pairs
    nl_path = validated_dir / "nl_pairs.jsonl"
    if nl_path.exists():
        with open(nl_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Only include pairs that passed validation
                    if data.get("validation", {}).get("passed", True):
                        pairs.append(ValidatedPair(**data))
        logger.info(f"Loaded {len([p for p in pairs if p.domain == 'nl'])} NL pairs from {nl_path}")

    # Load code pairs
    code_path = validated_dir / "code_pairs.jsonl"
    if code_path.exists():
        with open(code_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get("validation", {}).get("passed", True):
                        pairs.append(ValidatedPair(**data))
        logger.info(
            f"Loaded {len([p for p in pairs if p.domain == 'code'])} code pairs from {code_path}"
        )

    logger.info(f"Total validated pairs loaded: {len(pairs)}")
    return pairs


def pair_to_chat_example(pair: ValidatedPair, system_prompt: str | None = None) -> ChatExample:
    """
    Convert a validated pair to chat format.

    Args:
        pair: The validated compression pair
        system_prompt: Optional system prompt (uses default if not provided)

    Returns:
        ChatExample with messages in chat format
    """
    system = system_prompt or COMPRESSION_SYSTEM_PROMPT
    user_content = USER_PROMPT_TEMPLATE.format(verbose=pair.verbose)

    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user_content),
        ChatMessage(role="assistant", content=pair.compressed),
    ]

    return ChatExample(messages=messages)


def split_data(
    pairs: list[ValidatedPair],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_by_domain: bool = True,
) -> tuple[list[ValidatedPair], list[ValidatedPair], list[ValidatedPair], SplitStats]:
    """
    Split pairs into train/valid/test sets.

    Args:
        pairs: All validated pairs
        train_ratio: Fraction for training (default: 0.8)
        valid_ratio: Fraction for validation (default: 0.1)
        test_ratio: Fraction for testing (default: 0.1)
        seed: Random seed for reproducibility
        stratify_by_domain: Whether to stratify by domain (nl/code)

    Returns:
        Tuple of (train, valid, test) lists and statistics
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    random.seed(seed)

    if stratify_by_domain:
        # Split by domain to maintain proportions
        nl_pairs = [p for p in pairs if p.domain == "nl"]
        code_pairs = [p for p in pairs if p.domain == "code"]

        random.shuffle(nl_pairs)
        random.shuffle(code_pairs)

        def split_list(
            items: list[ValidatedPair],
        ) -> tuple[list[ValidatedPair], list[ValidatedPair], list[ValidatedPair]]:
            n = len(items)
            train_end = int(n * train_ratio)
            valid_end = train_end + int(n * valid_ratio)
            return items[:train_end], items[train_end:valid_end], items[valid_end:]

        nl_train, nl_valid, nl_test = split_list(nl_pairs)
        code_train, code_valid, code_test = split_list(code_pairs)

        train = nl_train + code_train
        valid = nl_valid + code_valid
        test = nl_test + code_test

        # Shuffle each split
        random.shuffle(train)
        random.shuffle(valid)
        random.shuffle(test)
    else:
        # Simple random split
        shuffled = pairs.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        valid_end = train_end + int(n * valid_ratio)

        train = shuffled[:train_end]
        valid = shuffled[train_end:valid_end]
        test = shuffled[valid_end:]

    stats = SplitStats(
        total=len(pairs),
        train=len(train),
        valid=len(valid),
        test=len(test),
        nl_count=len([p for p in pairs if p.domain == "nl"]),
        code_count=len([p for p in pairs if p.domain == "code"]),
    )

    return train, valid, test, stats


def write_chat_jsonl(
    pairs: list[ValidatedPair],
    output_path: Path,
    system_prompt: str | None = None,
) -> int:
    """
    Write pairs to JSONL file in chat format.

    Args:
        pairs: List of validated pairs
        output_path: Path to output JSONL file
        system_prompt: Optional custom system prompt

    Returns:
        Number of examples written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            example = pair_to_chat_example(pair, system_prompt)
            # Convert to dict for JSON serialization
            example_dict = {"messages": [m.model_dump() for m in example.messages]}
            f.write(json.dumps(example_dict, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(pairs)} examples to {output_path}")
    return len(pairs)


def format_for_training(
    validated_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    system_prompt: str | None = None,
) -> SplitStats:
    """
    Main function to format validated pairs for training.

    Loads validated pairs, splits into train/valid/test, and writes
    JSONL files in chat format.

    Args:
        validated_dir: Path to data/validated/
        output_dir: Path to output directory (data/training/)
        train_ratio: Fraction for training
        valid_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        system_prompt: Optional custom system prompt

    Returns:
        SplitStats with information about the splits
    """
    # Load all validated pairs
    pairs = load_validated_pairs(validated_dir)

    if not pairs:
        raise ValueError(f"No validated pairs found in {validated_dir}")

    # Split data
    train, valid, test, stats = split_data(
        pairs,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)

    write_chat_jsonl(train, output_dir / "train.jsonl", system_prompt)
    write_chat_jsonl(valid, output_dir / "valid.jsonl", system_prompt)
    write_chat_jsonl(test, output_dir / "test.jsonl", system_prompt)

    logger.info(
        f"Formatted {stats.total} pairs: "
        f"{stats.train} train, {stats.valid} valid, {stats.test} test"
    )
    logger.info(f"Domain breakdown: {stats.nl_count} NL, {stats.code_count} code")

    return stats


def write_completions_jsonl(
    pairs: list[ValidatedPair],
    output_path: Path,
    prompt_prefix: str = "Compress the following text while preserving semantic meaning:\n\n",
) -> int:
    """
    Write pairs to JSONL file in completions format (alternative to chat).

    Args:
        pairs: List of validated pairs
        output_path: Path to output JSONL file
        prompt_prefix: Prefix for the prompt

    Returns:
        Number of examples written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            example = {
                "prompt": f"{prompt_prefix}{pair.verbose}",
                "completion": pair.compressed,
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(pairs)} completions to {output_path}")
    return len(pairs)


def write_text_jsonl(
    pairs: list[ValidatedPair],
    output_path: Path,
    template: str = "### Input:\n{verbose}\n\n### Compressed:\n{compressed}",
) -> int:
    """
    Write pairs to JSONL file in text format (simplest format).

    Args:
        pairs: List of validated pairs
        output_path: Path to output JSONL file
        template: Template for combining verbose and compressed

    Returns:
        Number of examples written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            text = template.format(verbose=pair.verbose, compressed=pair.compressed)
            example = {"text": text}
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(pairs)} text examples to {output_path}")
    return len(pairs)
