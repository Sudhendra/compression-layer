import json

from scripts.merge_corpus import merge_corpus_files


def test_merge_corpus_files_dedupes_and_normalizes(tmp_path) -> None:
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    output = tmp_path / "out.jsonl"

    file_a.write_text(
        "\n".join(
            [
                json.dumps({"text": "def foo(): return 1", "source": "a"}),
                json.dumps({"content": "def foo(): return 1", "source": "a"}),
                json.dumps({"text": "short"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    file_b.write_text(
        "\n".join(
            [
                json.dumps({"code": "class Bar: pass", "source": "b"}),
                json.dumps({"text": "def foo(): return 1", "source": "b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = merge_corpus_files(
        input_files=[file_a, file_b],
        output_file=output,
        domain="code",
        dedupe=True,
        min_chars=10,
        max_chars=2000,
    )

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    payloads = [json.loads(line) for line in lines]

    assert stats["kept"] == len(payloads)
    assert stats["duplicates"] == 2
    assert stats["filtered"] == 1
    assert all(item["domain"] == "code" for item in payloads)
