from pathlib import Path

from src.generation.adapter_generator import AdapterGenerator


class DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        return "PROMPT"


def test_adapter_generator_compresses_with_injected_backend() -> None:
    calls = {}

    def load_fn(model, adapter_path=None):
        calls["load"] = (model, adapter_path)
        return "model", DummyTokenizer()

    def generate_fn(model, tokenizer, prompt, max_tokens, sampler):
        calls["generate"] = (model, prompt, max_tokens, sampler)
        return "```short```"

    def sampler_factory(temp):
        calls["sampler"] = temp
        return "sampler"

    gen = AdapterGenerator(
        model="dummy-model",
        adapter_path=Path("models/runs/mlx/latest/adapter"),
        load_fn=load_fn,
        generate_fn=generate_fn,
        sampler_factory=sampler_factory,
    )

    result = gen.compress("The quick brown fox jumps over the lazy dog.", max_tokens=123)

    assert result == "short"
    assert calls["load"][0] == "dummy-model"
    assert calls["generate"][1] == "PROMPT"
    assert calls["generate"][2] == 123
