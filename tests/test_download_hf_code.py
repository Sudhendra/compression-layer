from scripts.download_hf_code import extract_code_samples


def test_extract_code_samples_filters_and_dedupes() -> None:
    items = [
        {"content": "def foo():\n    return 1\n" * 10},
        {"content": "def foo():\n    return 1\n" * 10},
        {"content": "print('hello')"},
        {"content": "class Bar:\n    pass\n" * 5},
        {"content": "def x():\n    pass"},
    ]

    samples = extract_code_samples(
        dataset_name="bigcode/the-stack",
        language="python",
        limit=10,
        min_chars=50,
        max_chars=2000,
        split="train",
        dataset_iter=items,
    )

    assert len(samples) == 2
    assert all(sample["source"] == "bigcode/the-stack" for sample in samples)
    assert all(sample["language"] == "python" for sample in samples)


def test_extract_code_samples_uses_dataset_specific_key() -> None:
    items = [
        {"code": "def from_code_key():\n    return True\n" * 5},
        {"code": "class FromCode:\n    pass\n" * 5},
    ]

    samples = extract_code_samples(
        dataset_name="codeparrot/github-code",
        language="python",
        limit=10,
        min_chars=50,
        max_chars=2000,
        split="train",
        dataset_iter=items,
    )

    assert len(samples) == 2


def test_extract_code_samples_trusts_remote_code_for_github_code() -> None:
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return []

    samples = extract_code_samples(
        dataset_name="codeparrot/github-code",
        language="python",
        limit=0,
        min_chars=50,
        max_chars=2000,
        split="train",
        load_dataset_fn=fake_load_dataset,
    )

    assert samples == []
    assert calls["kwargs"]["trust_remote_code"] is True
    assert calls["kwargs"]["languages"] == ["python"]
