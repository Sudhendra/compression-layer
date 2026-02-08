"""
Unit Tests for Sanitization and Dataset Manager
Run with: python -m pytest tests/test_sanitization_manager.py -v
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

# ============================================================================
# IMPORTS - Fixed to match actual function signatures
# ============================================================================

try:
    from scripts.data_sanitization import (
        compute_compression_ratio_tokens,  # ✓ Fixed: was compute_compression_ratio
        extract_verbose_compressed,
        is_code_sample,
        rule_a_ratio_check,
        rule_b_orphaned_symbols,
        rule_c_negation_preservation,
        rule_d_semantic_symbol_usage_nl,
        sanitize_and_extract,  # ✓ Added: for integration tests
    )

    HAS_SANITIZE = True
except ImportError:
    HAS_SANITIZE = False

try:
    from scripts.dataset_manager import (
        count_samples,
        # get_config,  # ✓ Added: to create config objects
        load_state,
        # log_change,  # ✓ Added: for log tests
        revert_to_original,  # ✓ Added: for integration tests
        save_state,
        update_to_sanitized,  # ✓ Added: for integration tests
    )

    HAS_DATASET_MANAGER = True
except ImportError:
    HAS_DATASET_MANAGER = False


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temp directory, cleanup after test."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock config for testing."""
    return {
        "active_train": temp_dir / "train.jsonl",
        "original_backup": temp_dir / "train.original.jsonl",
        "sanitized_data": temp_dir / "sanitized_train.jsonl",
        "state_file": temp_dir / ".dataset_state.json",
        "log_file": temp_dir / ".dataset_changes.log",
    }


# ============================================================================
# TESTS: SANITIZATION - Helper Functions
# ============================================================================


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestCodeDetection:
    """Test code vs natural language detection."""

    def test_detects_python_code(self):
        assert is_code_sample("def hello():\n    return 'hi'")

    def test_detects_natural_language(self):
        assert not is_code_sample("The cat sat on the mat.")

    def test_detects_javascript(self):
        # Multi-line with multiple code signals
        js_code = """
        const x = () => {
            return true;
        }
        function test() {
            let y = 5;
        }
        """
        assert is_code_sample(js_code)


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestExtraction:
    """Test extracting verbose and compressed text."""

    def test_extracts_correctly(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Compress: Hello world"},
                {"role": "assistant", "content": "hi world"},
            ]
        }
        verbose, compressed, error = extract_verbose_compressed(
            sample, 0
        )  # ✓ Fixed: added sample_id
        assert verbose == "Hello world"
        assert compressed == "hi world"
        assert error is None

    def test_handles_missing_messages(self):
        sample = {"messages": []}
        verbose, compressed, error = extract_verbose_compressed(
            sample, 0
        )  # ✓ Fixed: added sample_id
        assert verbose is None
        assert compressed is None
        assert error is not None


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestCompressionRatio:
    """Test compression ratio calculation."""

    def test_basic_ratio(self):
        # Note: compute_compression_ratio_tokens uses actual tokenizer
        # This test assumes ratio > 1.0 means expansion
        ratio = compute_compression_ratio_tokens("one two three four", "1 2")
        assert ratio < 1.0  # Good compression


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestRuleA:
    """Test Rule A: compression ratio validation."""

    def test_passes_good_ratio(self):
        passed, _ = rule_a_ratio_check("one two three", "1 2")
        assert passed

    def test_fails_bad_ratio(self):
        passed, _ = rule_a_ratio_check("hi", "hello there friend")
        assert not passed


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestRuleB:
    """Test Rule B: orphaned symbols."""

    def test_passes_clean_text(self):
        passed, _ = rule_b_orphaned_symbols(
            "Paris @ France", is_code=False
        )  # ✓ Fixed: added is_code
        assert passed

    def test_fails_symbol_at_start(self):
        passed, _ = rule_b_orphaned_symbols("→ bad start", is_code=False)  # ✓ Fixed: added is_code
        assert not passed

    def test_allows_colon_at_end(self):
        passed, _ = rule_b_orphaned_symbols("function:", is_code=False)  # ✓ Fixed: added is_code
        assert passed

    def test_allows_decorator_for_code(self):
        passed, _ = rule_b_orphaned_symbols(
            "@classmethod", is_code=True
        )  # ✓ New: test code-aware behavior
        assert passed


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestRuleC:
    """Test Rule C: negation preservation."""

    def test_passes_no_negation(self):
        passed, _ = rule_c_negation_preservation("I like it", "like")
        assert passed

    def test_passes_preserved_negation(self):
        passed, _ = rule_c_negation_preservation("I do not like it", "not like")
        assert passed

    def test_fails_lost_negation(self):
        passed, _ = rule_c_negation_preservation("I never eat meat", "eat meat")
        assert not passed


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestRuleD:
    """Test Rule D: semantic symbol usage."""

    def test_passes_location_with_at(self):
        passed, _ = rule_d_semantic_symbol_usage_nl("Paris is located in France", "Paris @ France")
        assert passed

    def test_fails_location_without_at(self):
        passed, _ = rule_d_semantic_symbol_usage_nl("Tokyo is located in Japan", "Tokyo Japan")
        assert not passed


# ============================================================================
# TESTS: SANITIZATION - Integration (Main Flow)
# ============================================================================


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestSanitizeDataset:
    """Test main sanitization flow."""

    def test_sanitize_dataset_splits_correctly(self, temp_dir):
        """Test that sanitize_and_extract splits good/bad samples correctly."""

        # Create input file with 2 good + 2 bad samples
        input_file = temp_dir / "train.jsonl"
        sanitized_file = temp_dir / "sanitized.jsonl"
        unsanitized_file = temp_dir / "unsanitized.jsonl"

        good_sample_1 = {
            "messages": [
                {"role": "user", "content": "Compress: one two three four"},
                {"role": "assistant", "content": "1 2"},
            ]
        }

        good_sample_2 = {
            "messages": [
                {"role": "user", "content": "Compress: hello world"},
                {"role": "assistant", "content": "hi"},
            ]
        }

        # Bad sample: expansion (ratio > 1.0)
        bad_sample_1 = {
            "messages": [
                {"role": "user", "content": "Compress: hi"},
                {"role": "assistant", "content": "hello there my friend"},
            ]
        }

        # Bad sample: orphaned symbol
        bad_sample_2 = {
            "messages": [
                {"role": "user", "content": "Compress: test"},
                {"role": "assistant", "content": "→ bad"},
            ]
        }

        with open(input_file, "w", encoding="utf-8") as f:
            for sample in [good_sample_1, good_sample_2, bad_sample_1, bad_sample_2]:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Run sanitization
        stats = sanitize_and_extract(input_file, sanitized_file, unsanitized_file)

        # Assert
        assert stats["total_input"] == 4
        assert stats["passed_all"] == 2
        assert stats["failed_all"] == 2
        assert sanitized_file.exists()
        assert unsanitized_file.exists()

    def test_unicode_symbols_preserved(self, temp_dir):
        """Test that → ∵ @ symbols are NOT escaped in sanitized output."""

        input_file = temp_dir / "train.jsonl"
        sanitized_file = temp_dir / "sanitized.jsonl"
        unsanitized_file = temp_dir / "unsanitized.jsonl"

        # Sample that PASSES all validation rules
        # (avoid causation/location keywords to prevent Rule D failures)
        sample = {
            "messages": [
                {"role": "user", "content": "Compress: This implies that result"},
                {"role": "assistant", "content": "this → that"},
            ]
        }

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Run sanitization
        stats = sanitize_and_extract(input_file, sanitized_file, unsanitized_file)

        # Verify it passed validation
        assert stats["passed_all"] == 1, f"Sample failed: {stats['failed_samples']}"

        # Read back from sanitized file
        with open(sanitized_file, encoding="utf-8") as f:
            result_text = f.read()

        # Assert NOT escaped
        assert "→" in result_text, "Arrow symbol should be preserved, not escaped"
        assert "\\u2192" not in result_text, "Arrow should not be escaped as \\u2192"


# ============================================================================
# TESTS: DATASET MANAGER - File Operations
# ============================================================================


@pytest.mark.skipif(not HAS_DATASET_MANAGER, reason="dataset_manager not available")
class TestFileOperations:
    """Test file utilities."""

    def test_count_samples(self, temp_dir):
        test_file = temp_dir / "test.jsonl"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write('{"test": 1}\n')
            f.write('{"test": 2}\n')
            f.write('{"test": 3}\n')

        assert count_samples(test_file) == 3

    def test_count_nonexistent_file(self, temp_dir):
        count = count_samples(temp_dir / "missing.jsonl")
        assert count == 0

    def test_count_empty_file(self, temp_dir):
        test_file = temp_dir / "empty.jsonl"
        test_file.touch()
        assert count_samples(test_file) == 0


# ============================================================================
# TESTS: DATASET MANAGER - State Management
# ============================================================================


@pytest.mark.skipif(not HAS_DATASET_MANAGER, reason="dataset_manager not available")
class TestState:
    """Test state management."""

    def test_load_default_state(self, mock_config):
        state = load_state(mock_config)  # ✓ Fixed: pass config
        assert state["current"] == "original"
        assert state["change_count"] == 0
        assert state["last_action"] is None

    def test_save_and_load(self, mock_config):
        test_state = {
            "current": "sanitized",
            "last_action": "update",
            "last_change": "2024-01-01T00:00:00",
            "change_count": 1,
        }
        save_state(mock_config, test_state)  # ✓ Fixed: pass config
        loaded = load_state(mock_config)  # ✓ Fixed: pass config
        assert loaded == test_state


# ============================================================================
# TESTS: DATASET MANAGER - Integration (Main Flows)
# ============================================================================


@pytest.mark.skipif(not HAS_DATASET_MANAGER, reason="dataset_manager not available")
class TestUpdateToSanitized:
    """Test update_to_sanitized flow."""

    def test_update_creates_backup_first_time(self, mock_config):
        """Test that first update creates backup."""

        # Create original and sanitized files
        with open(mock_config["active_train"], "w") as f:
            f.write('{"original": 1}\n')
            f.write('{"original": 2}\n')

        with open(mock_config["sanitized_data"], "w") as f:
            f.write('{"sanitized": 1}\n')

        # Execute update
        result = update_to_sanitized(mock_config)

        # Assert
        assert result
        assert mock_config["original_backup"].exists()  # Backup created

        state = load_state(mock_config)
        assert state["current"] == "sanitized"
        assert state["last_action"] == "update"

    def test_update_blocks_consecutive_updates(self, mock_config):
        """Test that you can't update twice in a row."""

        # Setup files
        with open(mock_config["active_train"], "w") as f:
            f.write('{"original": 1}\n')

        with open(mock_config["sanitized_data"], "w") as f:
            f.write('{"sanitized": 1}\n')

        # First update
        update_to_sanitized(mock_config)

        # Second update (should be blocked)
        result = update_to_sanitized(mock_config)

        # Assert
        assert not result


@pytest.mark.skipif(not HAS_DATASET_MANAGER, reason="dataset_manager not available")
class TestRevertToOriginal:
    """Test revert_to_original flow."""

    def test_revert_restores_from_backup(self, mock_config):
        """Test that revert copies backup to active."""

        # Setup: Update first (creates backup)
        with open(mock_config["active_train"], "w") as f:
            f.write('{"original": 1}\n')

        with open(mock_config["sanitized_data"], "w") as f:
            f.write('{"sanitized": 1}\n')

        update_to_sanitized(mock_config)

        # Execute: Revert
        result = revert_to_original(mock_config)

        # Assert
        assert result

        state = load_state(mock_config)
        assert state["current"] == "original"
        assert state["last_action"] == "revert"

    def test_revert_blocks_consecutive_reverts(self, mock_config):
        """Test that you can't revert twice in a row."""

        # Setup: Update then revert
        with open(mock_config["active_train"], "w") as f:
            f.write('{"original": 1}\n')

        with open(mock_config["sanitized_data"], "w") as f:
            f.write('{"sanitized": 1}\n')

        update_to_sanitized(mock_config)
        revert_to_original(mock_config)  # First revert

        # Execute: Try again
        result = revert_to_original(mock_config)  # Second revert

        # Assert
        assert not result


@pytest.mark.skipif(not HAS_DATASET_MANAGER, reason="dataset_manager not available")
class TestLogUpdates:
    """Test log file updates."""

    def test_operations_append_to_log(self, mock_config):
        """Test that update/revert write to log file."""

        # Setup files
        with open(mock_config["active_train"], "w") as f:
            f.write('{"test": 1}\n')

        with open(mock_config["sanitized_data"], "w") as f:
            f.write('{"test": 1}\n')

        # Execute operations
        update_to_sanitized(mock_config)
        revert_to_original(mock_config)

        # Assert
        assert mock_config["log_file"].exists()
        log_lines = mock_config["log_file"].read_text().strip().split("\n")
        assert len(log_lines) == 2
        assert "UPDATE" in log_lines[0]
        assert "REVERT" in log_lines[1]


# ============================================================================
# EDGE CASES
# ============================================================================


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestEdgeCases:
    """Test edge cases and safety."""

    def test_empty_symbol_check(self):
        passed, _ = rule_b_orphaned_symbols("", is_code=False)  # ✓ Fixed: added is_code
        assert not passed

    def test_unicode_handling(self):
        passed, _ = rule_b_orphaned_symbols(
            "test → result", is_code=False
        )  # ✓ Fixed: added is_code
        assert passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
