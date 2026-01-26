"""Tests for the seed generator module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generation.seed_generator import (
    GeneratedPair,
    GenerationResult,
    SeedGenerator,
    generate_seed_pairs,
)


@pytest.fixture
def mock_model_client():
    """Create a mock ModelClient that returns compressed text."""
    with patch("src.generation.seed_generator.ModelClient") as mock:
        client_instance = AsyncMock()
        client_instance.complete = AsyncMock(return_value="compressed output")
        mock.return_value = client_instance
        yield mock


@pytest.fixture
def mock_cache():
    """Create a mock SemanticCache."""
    cache = MagicMock()
    cache.get.return_value = None  # Cache miss by default
    cache.set = MagicMock()
    return cache


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestGeneratedPair:
    """Tests for the GeneratedPair model."""

    def test_nl_pair_creation(self):
        pair = GeneratedPair(
            verbose="This is verbose text",
            compressed="verbose text",
            domain="nl",
        )
        assert pair.domain == "nl"
        assert pair.language is None

    def test_code_pair_creation(self):
        pair = GeneratedPair(
            verbose="def foo(): pass",
            compressed="fn:foo()",
            domain="code",
            language="python",
        )
        assert pair.domain == "code"
        assert pair.language == "python"

    def test_pair_serialization(self):
        pair = GeneratedPair(
            verbose="test input",
            compressed="test",
            domain="nl",
            metadata={"source": "test"},
        )
        data = pair.model_dump()
        assert data["verbose"] == "test input"
        assert data["compressed"] == "test"
        assert data["metadata"]["source"] == "test"

        # Round-trip
        reconstructed = GeneratedPair(**data)
        assert reconstructed == pair


class TestGenerationResult:
    """Tests for the GenerationResult model."""

    def test_result_creation(self):
        pairs = [
            GeneratedPair(verbose="a", compressed="a", domain="nl"),
            GeneratedPair(verbose="b", compressed="b", domain="nl"),
        ]
        result = GenerationResult(
            pairs=pairs,
            total_input_tokens=100,
            total_output_tokens=50,
            cached_count=1,
            generated_count=1,
        )
        assert len(result.pairs) == 2
        assert result.cached_count == 1
        assert result.generated_count == 1


class TestSeedGenerator:
    """Tests for the SeedGenerator class."""

    @pytest.mark.asyncio
    async def test_generate_nl_pair(self, mock_model_client, mock_cache):
        """Test generating a single NL compression pair."""
        with (
            patch("src.generation.seed_generator.SemanticCache", return_value=mock_cache),
            patch("src.generation.seed_generator.get_cost_tracker"),
        ):
            generator = SeedGenerator(cache=mock_cache)

            pair = await generator.generate_nl_pair("This is verbose text")

            assert pair.domain == "nl"
            assert pair.verbose == "This is verbose text"
            assert pair.compressed == "compressed output"
            mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_nl_pair_cached(self, mock_model_client, mock_cache):
        """Test that cached results are returned."""
        cached_data = {
            "verbose": "cached input",
            "compressed": "cached output",
            "domain": "nl",
            "language": None,
            "metadata": {},
        }
        mock_cache.get.return_value = cached_data

        with (
            patch("src.generation.seed_generator.SemanticCache", return_value=mock_cache),
            patch("src.generation.seed_generator.get_cost_tracker"),
        ):
            generator = SeedGenerator(cache=mock_cache)

            pair = await generator.generate_nl_pair("cached input")

            assert pair.compressed == "cached output"
            # Client should not be called if cached - just check the pair was from cache
            assert pair.verbose == "cached input"

    @pytest.mark.asyncio
    async def test_generate_code_pair(self, mock_model_client, mock_cache):
        """Test generating a code compression pair."""
        with (
            patch("src.generation.seed_generator.SemanticCache", return_value=mock_cache),
            patch("src.generation.seed_generator.get_cost_tracker"),
        ):
            generator = SeedGenerator(cache=mock_cache)

            pair = await generator.generate_code_pair(
                "def foo(): pass",
                language="python",
            )

            assert pair.domain == "code"
            assert pair.language == "python"
            assert pair.verbose == "def foo(): pass"

    @pytest.mark.asyncio
    async def test_generate_batch(self, mock_model_client, mock_cache):
        """Test batch generation."""
        with (
            patch("src.generation.seed_generator.SemanticCache", return_value=mock_cache),
            patch("src.generation.seed_generator.get_cost_tracker"),
        ):
            generator = SeedGenerator(cache=mock_cache)

            inputs = ["text 1", "text 2", "text 3"]
            result = await generator.generate_batch(
                inputs,
                domain="nl",
                show_progress=False,
            )

            assert len(result.pairs) == 3
            assert all(p.domain == "nl" for p in result.pairs)

    def test_save_pairs(self, temp_output_dir):
        """Test saving pairs to JSONL."""
        pairs = [
            GeneratedPair(verbose="a", compressed="a'", domain="nl"),
            GeneratedPair(verbose="b", compressed="b'", domain="nl"),
        ]

        output_path = temp_output_dir / "pairs.jsonl"

        with (
            patch("src.generation.seed_generator.SemanticCache"),
            patch("src.generation.seed_generator.get_cost_tracker"),
            patch("src.generation.seed_generator.ModelClient"),
        ):
            generator = SeedGenerator()
            generator.save_pairs(pairs, output_path)

        assert output_path.exists()

        # Verify contents
        loaded = []
        with open(output_path) as f:
            for line in f:
                loaded.append(json.loads(line))

        assert len(loaded) == 2
        assert loaded[0]["verbose"] == "a"
        assert loaded[1]["verbose"] == "b"

    def test_save_pairs_append(self, temp_output_dir):
        """Test appending pairs to existing file."""
        output_path = temp_output_dir / "pairs.jsonl"

        # Write initial pair
        with open(output_path, "w") as f:
            f.write(json.dumps({"verbose": "initial", "compressed": "i", "domain": "nl"}) + "\n")

        pairs = [GeneratedPair(verbose="new", compressed="n", domain="nl")]

        with (
            patch("src.generation.seed_generator.SemanticCache"),
            patch("src.generation.seed_generator.get_cost_tracker"),
            patch("src.generation.seed_generator.ModelClient"),
        ):
            generator = SeedGenerator()
            generator.save_pairs(pairs, output_path, append=True)

        # Verify both pairs exist
        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_load_pairs(self, temp_output_dir):
        """Test loading pairs from JSONL."""
        output_path = temp_output_dir / "pairs.jsonl"

        # Write test data
        with open(output_path, "w") as f:
            f.write(json.dumps({"verbose": "a", "compressed": "a'", "domain": "nl"}) + "\n")
            f.write(
                json.dumps(
                    {"verbose": "b", "compressed": "b'", "domain": "code", "language": "python"}
                )
                + "\n"
            )

        pairs = SeedGenerator.load_pairs(output_path)

        assert len(pairs) == 2
        assert pairs[0].verbose == "a"
        assert pairs[1].language == "python"


class TestGenerateSeedPairsFunction:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_generate_seed_pairs_basic(self, mock_model_client, mock_cache):
        """Test the convenience function."""
        with (
            patch("src.generation.seed_generator.SemanticCache", return_value=mock_cache),
            patch("src.generation.seed_generator.get_cost_tracker"),
        ):
            pairs = await generate_seed_pairs(
                inputs=["text 1", "text 2"],
                domain="nl",
            )

            assert len(pairs) == 2

    @pytest.mark.asyncio
    async def test_generate_seed_pairs_with_output(
        self, mock_model_client, mock_cache, temp_output_dir
    ):
        """Test saving output via convenience function."""
        output_path = temp_output_dir / "output.jsonl"

        with (
            patch("src.generation.seed_generator.SemanticCache", return_value=mock_cache),
            patch("src.generation.seed_generator.get_cost_tracker"),
        ):
            await generate_seed_pairs(
                inputs=["text 1"],
                domain="nl",
                output_path=output_path,
            )

            assert output_path.exists()
