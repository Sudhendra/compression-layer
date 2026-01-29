"""Tests for tinker data helpers."""

import pytest

from src.training.tinker_data import render_chat_example, split_prompt_completion


class FakeTokenizer:
    """Simple tokenizer for tests."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = [ord(char) for char in text]
        if add_special_tokens:
            return [1, *tokens, 2]
        return tokens


DUMMY_MESSAGES = ["Hello", "World"]


def test_split_prompt_completion_requires_messages():
    with pytest.raises(ValueError):
        split_prompt_completion([])


def test_split_prompt_completion_rejects_blank_message():
    with pytest.raises(ValueError):
        split_prompt_completion(["Hello", " "])


def test_render_chat_example_masks_prompt_tokens():
    tokenizer = FakeTokenizer()
    datum = render_chat_example(DUMMY_MESSAGES, tokenizer)
    prompt, completion = split_prompt_completion(DUMMY_MESSAGES)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens

    model_tokens = datum.model_input.chunks[0].tokens
    target_tokens = datum.loss_fn_inputs["target_tokens"].data
    weights = datum.loss_fn_inputs["weights"].data

    assert model_tokens == all_tokens[:-1]
    assert target_tokens == all_tokens[1:]
    assert len(target_tokens) == len(weights)
    assert len(target_tokens) == len(model_tokens)
    assert weights == [0] * (len(prompt_tokens) - 1) + [1] * len(completion_tokens)
