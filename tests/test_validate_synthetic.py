import asyncio
import json

from scripts.validate_synthetic import load_pairs, validate_batch
from src.generation.seed_generator import GeneratedPair


def test_load_pairs_reads_jsonl(tmp_path) -> None:
    input_path = tmp_path / "pairs.jsonl"
    payloads = [
        {"verbose": "a", "compressed": "b", "domain": "nl"},
        {"verbose": "c", "compressed": "d", "domain": "code"},
    ]
    input_path.write_text(
        "\n".join(json.dumps(item) for item in payloads) + "\n",
        encoding="utf-8",
    )

    pairs = load_pairs(input_path)

    assert len(pairs) == 2
    assert pairs[0].verbose == "a"


def test_validate_batch_writes_passing_pairs(tmp_path) -> None:
    output_path = tmp_path / "validated.jsonl"

    pairs = [
        GeneratedPair(verbose="a", compressed="b", domain="nl"),
        GeneratedPair(verbose="c", compressed="d", domain="nl"),
    ]

    class DummyResult:
        def __init__(self, score):
            self.min_equivalence = score

    class DummyHarness:
        def __init__(self):
            self.calls = 0

        async def validate_pair(self, pair):
            self.calls += 1
            return DummyResult(0.9 if self.calls == 1 else 0.5)

    stats = asyncio.run(
        validate_batch(
            pairs,
            output_path,
            threshold=0.8,
            concurrency=2,
            models=["claude"],
            harness=DummyHarness(),
        )
    )

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()

    assert stats["passed"] == 1
    assert len(lines) == 1
