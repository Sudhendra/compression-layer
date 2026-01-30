"""Tests for the training data formatting module."""

import json
import tempfile
from pathlib import Path

from src.training.format_data import (
    COMPRESSION_SYSTEM_PROMPT,
    ChatExample,
    ChatMessage,
    ValidatedPair,
    format_for_training,
    load_validated_pairs,
    pair_to_chat_example,
    split_data,
    write_chat_jsonl,
)


class TestValidatedPair:
    """Tests for ValidatedPair model."""

    def test_create_pair(self) -> None:
        """Test creating a validated pair."""
        pair = ValidatedPair(
            verbose="This is a verbose text.",
            compressed="verbose_text",
            domain="nl",
        )
        assert pair.verbose == "This is a verbose text."
        assert pair.compressed == "verbose_text"
        assert pair.domain == "nl"
        assert pair.metadata is None
        assert pair.validation is None

    def test_create_pair_with_validation(self) -> None:
        """Test creating a pair with validation data."""
        pair = ValidatedPair(
            verbose="Hello world",
            compressed="hello_world",
            domain="nl",
            validation={"passed": True, "min_equivalence": 0.85},
        )
        assert pair.validation is not None
        assert pair.validation["passed"] is True
        assert pair.validation["min_equivalence"] == 0.85


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_create_message(self) -> None:
        """Test creating a chat message."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestChatExample:
    """Tests for ChatExample model."""

    def test_create_example(self) -> None:
        """Test creating a chat example."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
        ]
        example = ChatExample(messages=messages)
        assert len(example.messages) == 3
        assert example.messages[0].role == "system"


class TestPairToChatExample:
    """Tests for pair_to_chat_example function."""

    def test_convert_pair_default_system(self) -> None:
        """Test converting a pair with default system prompt."""
        pair = ValidatedPair(
            verbose="This is verbose text",
            compressed="compressed_text",
            domain="nl",
        )
        example = pair_to_chat_example(pair)

        assert len(example.messages) == 3
        assert example.messages[0].role == "system"
        assert example.messages[0].content == COMPRESSION_SYSTEM_PROMPT
        assert example.messages[1].role == "user"
        assert "Compress:" in example.messages[1].content
        assert "This is verbose text" in example.messages[1].content
        assert example.messages[2].role == "assistant"
        assert example.messages[2].content == "compressed_text"

    def test_convert_pair_custom_system(self) -> None:
        """Test converting a pair with custom system prompt."""
        pair = ValidatedPair(
            verbose="Hello",
            compressed="hi",
            domain="nl",
        )
        example = pair_to_chat_example(pair, system_prompt="Custom prompt")

        assert example.messages[0].content == "Custom prompt"


class TestSplitData:
    """Tests for split_data function."""

    def test_split_ratios(self) -> None:
        """Test that split ratios are respected."""
        pairs = [
            ValidatedPair(verbose=f"text{i}", compressed=f"c{i}", domain="nl") for i in range(100)
        ]

        train, valid, test, stats = split_data(
            pairs,
            train_ratio=0.8,
            valid_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )

        assert len(train) == 80
        assert len(valid) == 10
        assert len(test) == 10
        assert stats.total == 100
        assert stats.train == 80
        assert stats.valid == 10
        assert stats.test == 10

    def test_split_stratified(self) -> None:
        """Test stratified split by domain."""
        nl_pairs = [
            ValidatedPair(verbose=f"nl{i}", compressed=f"c{i}", domain="nl") for i in range(50)
        ]
        code_pairs = [
            ValidatedPair(verbose=f"code{i}", compressed=f"c{i}", domain="code") for i in range(50)
        ]
        pairs = nl_pairs + code_pairs

        train, valid, test, stats = split_data(
            pairs,
            train_ratio=0.8,
            valid_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            stratify_by_domain=True,
        )

        assert stats.nl_count == 50
        assert stats.code_count == 50

    def test_split_reproducible(self) -> None:
        """Test that splits are reproducible with same seed."""
        pairs = [
            ValidatedPair(verbose=f"text{i}", compressed=f"c{i}", domain="nl") for i in range(100)
        ]

        train1, _, _, _ = split_data(pairs, seed=42)
        train2, _, _, _ = split_data(pairs, seed=42)

        assert [p.verbose for p in train1] == [p.verbose for p in train2]


class TestWriteChatJsonl:
    """Tests for write_chat_jsonl function."""

    def test_write_jsonl(self) -> None:
        """Test writing pairs to JSONL file."""
        pairs = [
            ValidatedPair(verbose="Hello world", compressed="hello_world", domain="nl"),
            ValidatedPair(verbose="Goodbye", compressed="bye", domain="nl"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.jsonl"
            count = write_chat_jsonl(pairs, output_path)

            assert count == 2
            assert output_path.exists()

            # Verify content
            with open(output_path) as f:
                lines = f.readlines()
                assert len(lines) == 2

                # Parse first line
                first = json.loads(lines[0])
                assert "messages" in first
                assert len(first["messages"]) == 3
                assert first["messages"][0]["role"] == "system"
                assert first["messages"][2]["content"] == "hello_world"


class TestLoadValidatedPairs:
    """Tests for load_validated_pairs function."""

    def test_load_pairs(self) -> None:
        """Test loading validated pairs from JSONL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validated_dir = Path(tmpdir)

            # Create nl_pairs.jsonl
            nl_path = validated_dir / "nl_pairs.jsonl"
            with open(nl_path, "w") as f:
                for i in range(5):
                    pair = {
                        "verbose": f"text{i}",
                        "compressed": f"c{i}",
                        "domain": "nl",
                        "validation": {"passed": True},
                    }
                    f.write(json.dumps(pair) + "\n")

            # Create code_pairs.jsonl
            code_path = validated_dir / "code_pairs.jsonl"
            with open(code_path, "w") as f:
                for i in range(3):
                    pair = {
                        "verbose": f"code{i}",
                        "compressed": f"c{i}",
                        "domain": "code",
                        "validation": {"passed": True},
                    }
                    f.write(json.dumps(pair) + "\n")

            pairs = load_validated_pairs(validated_dir)
            assert len(pairs) == 8

    def test_filter_failed_pairs(self) -> None:
        """Test that failed pairs are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validated_dir = Path(tmpdir)

            nl_path = validated_dir / "nl_pairs.jsonl"
            with open(nl_path, "w") as f:
                # Passed pair
                f.write(
                    json.dumps(
                        {
                            "verbose": "text1",
                            "compressed": "c1",
                            "domain": "nl",
                            "validation": {"passed": True},
                        }
                    )
                    + "\n"
                )
                # Failed pair
                f.write(
                    json.dumps(
                        {
                            "verbose": "text2",
                            "compressed": "c2",
                            "domain": "nl",
                            "validation": {"passed": False},
                        }
                    )
                    + "\n"
                )

            pairs = load_validated_pairs(validated_dir)
            assert len(pairs) == 1
            assert pairs[0].verbose == "text1"


class TestFormatForTraining:
    """Tests for format_for_training function."""

    def test_full_pipeline(self) -> None:
        """Test the full formatting pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validated_dir = Path(tmpdir) / "validated"
            output_dir = Path(tmpdir) / "training"
            validated_dir.mkdir()

            # Create test data
            nl_path = validated_dir / "nl_pairs.jsonl"
            with open(nl_path, "w") as f:
                for i in range(20):
                    pair = {
                        "verbose": f"text{i}",
                        "compressed": f"c{i}",
                        "domain": "nl",
                        "validation": {"passed": True},
                    }
                    f.write(json.dumps(pair) + "\n")

            # Run formatting
            stats = format_for_training(
                validated_dir=validated_dir,
                output_dir=output_dir,
                train_ratio=0.8,
                valid_ratio=0.1,
                test_ratio=0.1,
                seed=42,
            )

            # Verify outputs
            assert (output_dir / "train.jsonl").exists()
            assert (output_dir / "valid.jsonl").exists()
            assert (output_dir / "test.jsonl").exists()

            assert stats.total == 20
            assert stats.train == 16
            assert stats.valid == 2
            assert stats.test == 2

            # Verify content format
            with open(output_dir / "train.jsonl") as f:
                first = json.loads(f.readline())
                assert "messages" in first
                assert len(first["messages"]) == 3
