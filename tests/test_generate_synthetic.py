import json

from scripts.generate_synthetic import count_existing, load_corpus


def test_load_corpus_filters_and_limits(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    payloads = [
        {"text": "short"},
        {"content": "This is long enough to keep."},
        {"code": "def foo():\n    return 1\n" * 5},
    ]
    input_path.write_text(
        "\n".join(json.dumps(item) for item in payloads) + "\n",
        encoding="utf-8",
    )

    texts = load_corpus(input_path, limit=1)

    assert len(texts) == 1
    assert "long enough" in texts[0] or "def foo" in texts[0]


def test_count_existing_counts_non_empty_lines(tmp_path) -> None:
    output_path = tmp_path / "output.jsonl"
    output_path.write_text("{}\n\n{}\n", encoding="utf-8")

    assert count_existing(output_path) == 2
