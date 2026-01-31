from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


@dataclass
class ModelInput:
    tokens: list[int]

    @classmethod
    def from_ints(cls, tokens: list[int]) -> ModelInput:
        return cls(tokens=tokens)


@dataclass
class Datum:
    model_input: ModelInput
    loss_fn_inputs: dict[str, list[int]]


def split_prompt_completion(messages: list[dict[str, str]]) -> tuple[str, str]:
    if not messages:
        raise ValueError("messages must not be empty")

    prompt = "\n".join(message["content"] for message in messages[:-1])
    completion = messages[-1]["content"]
    return prompt, completion


def render_chat_example(messages: list[dict[str, str]], tokenizer: TokenizerLike) -> Datum:
    prompt, completion = split_prompt_completion(messages)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    return Datum(
        model_input=ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
    )
