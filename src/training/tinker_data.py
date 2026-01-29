from __future__ import annotations

from typing import Protocol

from tinker import EncodedTextChunk
from tinker.types import Datum, ModelInput, TensorData


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


def split_prompt_completion(messages: list[str]) -> tuple[str, str]:
    if not messages:
        raise ValueError("messages must not be empty")
    if any(not message.strip() for message in messages):
        raise ValueError("messages must not be empty")

    prompt = "\n".join(messages[:-1])
    completion = messages[-1]
    return prompt, completion


def render_chat_example(messages: list[str], tokenizer: TokenizerLike) -> Datum:
    prompt, completion = split_prompt_completion(messages)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    return Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=tokens[:-1])]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=tokens[1:], dtype="int64"),
            "weights": TensorData(data=weights[1:], dtype="float32"),
        },
    )
