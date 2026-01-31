"""Tests for the validation harness with mocked API calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.validation.harness import (
    BatchValidationStats,
    CompressionPair,
    ValidationHarness,
    ValidationResult,
)
from src.validation.metrics import TaskType
from src.validation.models import ModelType


@pytest.fixture
def mock_model_client():
    """Create a mock ModelClient that returns predictable responses."""
    mock = MagicMock()
    mock.complete = AsyncMock(return_value="This is a test response.")
    return mock


@pytest.fixture
def sample_pair():
    """Create a sample compression pair for testing."""
    return CompressionPair(
        verbose="The user named John Smith is a senior software engineer at Google.",
        compressed="John Smith | sr SWE @ Google",
        domain="nl",
    )


@pytest.fixture
def sample_code_pair():
    """Create a sample code compression pair for testing."""
    return CompressionPair(
        verbose="""
def calculate_total(items):
    total = 0
    for item in items:
        total += item['price']
    return total
""",
        compressed="fn:calculate_total(items) = sum(i.price for i in items)",
        domain="code",
    )


class TestCompressionPair:
    """Tests for CompressionPair model."""

    def test_create_pair(self, sample_pair):
        """Test creating a compression pair."""
        assert sample_pair.verbose.startswith("The user")
        assert sample_pair.compressed.startswith("John Smith")
        assert sample_pair.domain == "nl"

    def test_pair_with_metadata(self):
        """Test creating a pair with metadata."""
        pair = CompressionPair(
            verbose="test verbose",
            compressed="test compressed",
            domain="code",
            metadata={"source": "test", "language": "python"},
        )
        assert pair.metadata["language"] == "python"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_token_reduction_percent(self):
        """Test token reduction calculation."""
        result = ValidationResult(
            verbose_tokens=100,
            compressed_tokens=40,
            compression_ratio=0.4,
            equivalence_scores={ModelType.CLAUDE_SONNET: 0.9},
            min_equivalence=0.9,
            passed=True,
        )
        assert result.token_reduction_percent == 60.0

    def test_passed_result(self):
        """Test that passed is correctly set."""
        result = ValidationResult(
            verbose_tokens=100,
            compressed_tokens=50,
            compression_ratio=0.5,
            equivalence_scores={
                ModelType.CLAUDE_SONNET: 0.92,
                ModelType.GPT4O_MINI: 0.88,
            },
            min_equivalence=0.88,
            passed=True,
        )
        assert result.passed
        assert result.min_equivalence == 0.88


class TestValidationHarness:
    """Tests for ValidationHarness class."""

    @pytest.mark.asyncio
    async def test_harness_initialization(self):
        """Test harness initializes with default models."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
        ):
            mock_client_class.return_value = MagicMock()
            mock_calc_class.return_value = MagicMock()
            harness = ValidationHarness()

            assert len(harness.models) == 3
            assert ModelType.CLAUDE_SONNET in harness.models
            assert harness.threshold == 0.72  # Updated default threshold

    @pytest.mark.asyncio
    async def test_harness_custom_models(self):
        """Test harness with custom model list."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
        ):
            mock_client_class.return_value = MagicMock()
            mock_calc_class.return_value = MagicMock()
            harness = ValidationHarness(
                models=[ModelType.GPT4O_MINI],
                equivalence_threshold=0.90,
            )

            assert len(harness.models) == 1
            assert harness.threshold == 0.90

    @pytest.mark.asyncio
    async def test_validate_pair_mocked(self, sample_pair):
        """Test validation with mocked API calls."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
        ):
            # Setup mock client
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="Test response about John Smith")
            mock_client_class.return_value = mock_client

            # Setup mock calculator to return high similarity
            mock_calc = MagicMock()
            mock_calc.compute_semantic_similarity = MagicMock(return_value=0.95)
            mock_calc_class.return_value = mock_calc

            harness = ValidationHarness(
                models=[ModelType.CLAUDE_SONNET],
                tasks=[TaskType.QA],
            )

            result = await harness.validate_pair(sample_pair)

            assert isinstance(result, ValidationResult)
            assert result.verbose_tokens > 0
            assert result.compressed_tokens > 0
            assert result.compression_ratio < 1.0
            assert ModelType.CLAUDE_SONNET in result.equivalence_scores

    @pytest.mark.asyncio
    async def test_quick_validate_mocked(self):
        """Test quick_validate convenience method."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
        ):
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="Identical response")
            mock_client_class.return_value = mock_client

            # Setup mock calculator to return high similarity
            mock_calc = MagicMock()
            mock_calc.compute_semantic_similarity = MagicMock(return_value=0.95)
            mock_calc_class.return_value = mock_calc

            harness = ValidationHarness(
                models=[ModelType.GPT4O_MINI],
                tasks=[TaskType.QA],
            )

            # Since both calls return identical response, equivalence should be 1.0
            passed = await harness.quick_validate(
                verbose="Test verbose text",
                compressed="Test compressed",
                domain="nl",
            )

            assert isinstance(passed, bool)

    @pytest.mark.asyncio
    async def test_validate_batch_mocked(self, sample_pair):
        """Test batch validation."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
        ):
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="Batch response")
            mock_client_class.return_value = mock_client

            # Setup mock calculator to return high similarity
            mock_calc = MagicMock()
            mock_calc.compute_semantic_similarity = MagicMock(return_value=0.95)
            mock_calc_class.return_value = mock_calc

            harness = ValidationHarness(
                models=[ModelType.CLAUDE_SONNET],
                tasks=[TaskType.QA],
            )

            pairs = [sample_pair, sample_pair]
            stats = await harness.validate_batch(pairs, concurrency=2)

            assert isinstance(stats, BatchValidationStats)
            assert stats.total_pairs == 2
            assert len(stats.results) == 2


class TestBatchValidationStats:
    """Tests for BatchValidationStats."""

    def test_stats_calculation(self):
        """Test batch stats are correctly computed."""
        results = [
            ValidationResult(
                verbose_tokens=100,
                compressed_tokens=50,
                compression_ratio=0.5,
                equivalence_scores={ModelType.CLAUDE_SONNET: 0.9},
                min_equivalence=0.9,
                passed=True,
            ),
            ValidationResult(
                verbose_tokens=200,
                compressed_tokens=80,
                compression_ratio=0.4,
                equivalence_scores={ModelType.CLAUDE_SONNET: 0.8},
                min_equivalence=0.8,
                passed=False,  # Below 0.85 threshold
            ),
        ]

        stats = BatchValidationStats(
            total_pairs=2,
            passed_pairs=1,
            failed_pairs=1,
            avg_compression_ratio=0.45,
            avg_equivalence=0.85,
            min_equivalence=0.8,
            pass_rate=0.5,
            results=results,
        )

        assert stats.pass_rate == 0.5
        assert stats.avg_compression_ratio == 0.45


class TestIntegration:
    """Integration tests (can be skipped without API keys)."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires API keys")
    async def test_real_validation(self, sample_pair):
        """Test real validation with actual API calls."""
        harness = ValidationHarness(
            models=[ModelType.GPT4O_MINI],  # Cheapest option
            equivalence_threshold=0.85,
        )

        result = await harness.validate_pair(sample_pair)

        assert result.compression_ratio < 1.0
        print(f"Compression ratio: {result.compression_ratio:.2%}")
        print(f"Equivalence scores: {result.equivalence_scores}")
        print(f"Passed: {result.passed}")
