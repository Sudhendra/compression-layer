from scripts.download_hf_conversations import (
    extract_dolly_samples,
    extract_oasst_samples,
    extract_openorca_samples,
    extract_ultrachat_samples,
)


def test_extract_dolly_samples_filters_and_formats() -> None:
    items = [
        {
            "instruction": "Summarize the doc",
            "context": "This is a long enough context to keep.",
            "category": "qa",
        },
        {"instruction": "short", "context": ""},
        {
            "instruction": "Use code",
            "context": "```\nprint('hi')\n```\n```\nprint('bye')\n```\n```\nprint('skip')\n```",
        },
    ]

    samples = extract_dolly_samples(
        limit=10,
        min_chars=20,
        max_chars=500,
        dataset_iter=items,
    )

    assert len(samples) == 1
    assert samples[0]["source"] == "databricks/dolly-15k"
    assert samples[0]["category"] == "qa"
    assert "Summarize" in samples[0]["text"]


def test_extract_oasst_samples_dedupes_and_filters() -> None:
    items = [
        {"role": "assistant", "lang": "en", "text": "Skip me"},
        {"role": "prompter", "lang": "de", "text": "Skip me too"},
        {"role": "prompter", "lang": "en", "text": "Keep this", "message_id": "1"},
        {"role": "prompter", "lang": "en", "text": "Keep this", "message_id": "2"},
        {"role": "prompter", "lang": "en", "text": "Also keep", "message_id": "3"},
    ]

    samples = extract_oasst_samples(
        limit=10,
        min_chars=5,
        max_chars=200,
        dataset_iter=items,
    )

    assert len(samples) == 2
    assert {sample["text"] for sample in samples} == {"Keep this", "Also keep"}


def test_extract_dolly_samples_uses_dataset_name() -> None:
    calls = {}

    def fake_load_dataset(name, split):
        calls["name"] = name
        calls["split"] = split
        return []

    samples = extract_dolly_samples(
        limit=0,
        min_chars=1,
        max_chars=10,
        dataset_iter=None,
        dataset_name="databricks/databricks-dolly-15k",
        load_dataset_fn=fake_load_dataset,
    )

    assert samples == []
    assert calls["name"] == "databricks/databricks-dolly-15k"
    assert calls["split"] == "train"


def test_extract_ultrachat_samples_joins_user_messages() -> None:
    items = [
        {
            "prompt": "Initial user prompt",
            "messages": [
                {"role": "user", "content": "Initial user prompt"},
                {"role": "assistant", "content": "Assistant reply"},
                {"role": "user", "content": "Follow-up question"},
            ],
        },
        {
            "prompt": "Initial user prompt",
            "messages": [{"role": "user", "content": "Initial user prompt"}],
        },
    ]

    samples = extract_ultrachat_samples(
        limit=10,
        min_chars=5,
        max_chars=200,
        dataset_iter=items,
    )

    assert len(samples) == 2
    assert any("Follow-up question" in sample["text"] for sample in samples)


def test_extract_openorca_samples_combines_system_prompt() -> None:
    items = [
        {
            "system_prompt": "You are helpful",
            "question": "Explain the concept",
            "response": "Sure",
        }
    ]

    samples = extract_openorca_samples(
        limit=10,
        min_chars=5,
        max_chars=200,
        dataset_iter=items,
    )

    assert len(samples) == 1
    assert "You are helpful" in samples[0]["text"]
    assert "Explain the concept" in samples[0]["text"]
