"""Tests for adapter evaluation utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pytest

from src.evaluation.evaluate_adapter import (
    EvaluationExample,
    EvaluationResult,
    create_generator,
    load_test_examples,
    run_evaluation,
)
from src.validation.harness import CompressionPair, ValidationResult
from src.validation.models import ModelType


@pytest.fixture
def sample_test_file(tmp_path: Path) -> Path:
    content = (
        '{"messages": ['
        '{"role": "system", "content": "You are a semantic compression engine."},'
        '{"role": "user", "content": "Compress:\\nHello world"},'
        '{"role": "assistant", "content": "Hello | world"}'
        "]}\n"
        '{"messages": ['
        '{"role": "system", "content": "You are a semantic compression engine."},'
        '{"role": "user", "content": "Compress:\\nSecond example"},'
        '{"role": "assistant", "content": "Second | example"}'
        "]}"
    )
    path = tmp_path / "test.jsonl"
    path.write_text(content)
    return path


def test_load_test_examples(sample_test_file: Path) -> None:
    examples = load_test_examples(sample_test_file)
    assert len(examples) == 2
    assert examples[0].input_text == "Hello world"
    assert examples[1].input_text == "Second example"


def test_create_generator_uses_loaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        pass

    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return "formatted_prompt"

    class FakeSampler:
        pass

    def fake_load(model: str, adapter_path: str):
        calls["load"] = (model, adapter_path)
        return FakeModel(), FakeTokenizer()

    def fake_make_sampler(temp: float):
        calls["make_sampler"] = temp
        return FakeSampler()

    def fake_generate(*, model, tokenizer, prompt, max_tokens, sampler):
        calls["generate"] = (model, tokenizer, prompt, max_tokens, sampler)
        return "compressed"

    modules = cast(dict[str, object], __import__("sys").modules)
    monkeypatch.setitem(
        modules,
        "mlx_lm",
        type("FakeMlx", (), {"load": fake_load, "generate": fake_generate}),
    )
    monkeypatch.setitem(
        modules,
        "mlx_lm.sample_utils",
        type("FakeSampleUtils", (), {"make_sampler": fake_make_sampler}),
    )

    generator = create_generator("base-model", Path("/tmp/adapter"), system_prompt="System prompt")
    output = generator("input text")

    assert output == "compressed"
    assert calls["load"] == ("base-model", str(Path("/tmp/adapter")))
    # Generator now uses apply_chat_template, so prompt is the formatted result
    assert cast(tuple[object, object, str, int, object], calls["generate"])[2] == "formatted_prompt"


@dataclass
class FakeGenerator:
    response: str = "compressed"

    def __call__(self, prompt: str) -> str:
        return self.response


class HarnessProtocol(Protocol):
    async def validate_pair(self, pair: CompressionPair) -> ValidationResult: ...


@dataclass
class FakeHarness:
    response_score: float = 0.9

    async def validate_pair(self, pair: CompressionPair) -> ValidationResult:
        return ValidationResult(
            verbose_tokens=10,
            compressed_tokens=5,
            compression_ratio=0.5,
            equivalence_scores={ModelType.CLAUDE_SONNET: self.response_score},
            min_equivalence=self.response_score,
            passed=True,
        )


@pytest.mark.asyncio
async def test_run_evaluation_writes_results(tmp_path: Path) -> None:
    examples = [
        EvaluationExample(input_text="Example 1", domain="nl"),
        EvaluationExample(input_text="Example 2", domain="code"),
    ]
    output_path = tmp_path / "results.jsonl"

    results = await run_evaluation(
        examples=examples,
        generator=FakeGenerator(response="compressed output"),
        harness=cast(HarnessProtocol, FakeHarness(response_score=0.88)),
        output_path=output_path,
    )

    assert len(results) == 2
    assert isinstance(results[0], EvaluationResult)
    assert output_path.exists()
    output_lines = output_path.read_text().strip().splitlines()
    assert len(output_lines) == 2
