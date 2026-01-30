"""Training module for compression model fine-tuning.

This module provides:
- Data formatting for training (format_data.py)
- Local MLX LoRA training (train_mlx.py)
- Tinker cloud training (train_tinker.py)
"""

from .format_data import (
    COMPRESSION_SYSTEM_PROMPT,
    ChatExample,
    ChatMessage,
    SplitStats,
    ValidatedPair,
    format_for_training,
    load_validated_pairs,
    pair_to_chat_example,
    split_data,
    write_chat_jsonl,
    write_completions_jsonl,
    write_text_jsonl,
)
from .tinker_sdk import TinkerSDKClient
from .train_mlx import (
    MLXTrainingConfig,
    TrainingResult,
    check_mlx_available,
    evaluate_adapter,
    fuse_adapter,
    train_local,
)
from .train_tinker import (
    TinkerClient,
    TinkerJobStatus,
    TinkerLoRAConfig,
    TinkerTrainingConfig,
    TinkerTrainingResult,
    estimate_cost,
    train_on_tinker,
)

__all__ = [
    # Format data
    "COMPRESSION_SYSTEM_PROMPT",
    "ChatExample",
    "ChatMessage",
    "SplitStats",
    "ValidatedPair",
    "format_for_training",
    "load_validated_pairs",
    "pair_to_chat_example",
    "split_data",
    "write_chat_jsonl",
    "write_completions_jsonl",
    "write_text_jsonl",
    # MLX training
    "MLXTrainingConfig",
    "TrainingResult",
    "check_mlx_available",
    "evaluate_adapter",
    "fuse_adapter",
    "train_local",
    # Tinker training
    "TinkerClient",
    "TinkerJobStatus",
    "TinkerLoRAConfig",
    "TinkerTrainingConfig",
    "TinkerTrainingResult",
    "estimate_cost",
    "train_on_tinker",
    "TinkerSDKClient",
]
