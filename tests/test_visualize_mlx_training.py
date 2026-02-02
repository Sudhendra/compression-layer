import pytest

from scripts.visualize_mlx_training import (
    MLXLogParser,
    TrainingRunManager,
)

# ------------------------------------------------------------
# TrainingRunManager tests
# ------------------------------------------------------------


def test_get_latest_run(tmp_path):
    base = tmp_path / "runs"
    base.mkdir()

    (base / "2026-01-01_00-00-00").mkdir()
    (base / "2026-02-01_01-28-37").mkdir()

    manager = TrainingRunManager(base)
    timestamp, run_dir = manager.get_latest_run()

    assert timestamp == "2026-02-01_01-28-37"
    assert run_dir.name == timestamp


def test_get_run_invalid_timestamp(tmp_path):
    manager = TrainingRunManager(tmp_path)
    result = manager.get_run("bad-timestamp")

    assert result is None


def test_check_training_success(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Training complete\nSaved final weights")

    manager = TrainingRunManager(tmp_path)
    success, error = manager.check_training_status(log)

    assert success is True
    assert error is None


def test_check_training_failure_out_of_memory(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("CUDA out of memory")

    manager = TrainingRunManager(tmp_path)
    success, error = manager.check_training_status(log)

    assert success is False
    assert "Out of Memory" in error


# ------------------------------------------------------------
# MLXLogParser tests
# ------------------------------------------------------------


def test_parse_size_string():
    parser = MLXLogParser()

    assert parser.parse_size_string("1K") == 1_000
    assert parser.parse_size_string("2.5M") == 2_500_000
    assert parser.parse_size_string("1B") == 1_000_000_000
    assert parser.parse_size_string("42") == 42.0


def test_parse_log_basic_metrics():
    log_content = """
iters: 1000
Trainable parameters: 0.12% (7.34M/6.1B)
Iter 1 Train loss 2.345 Learning Rate 1e-4 It/sec 1.2 Tokens/sec 120 Trained Tokens 1000 Peak mem 12.3 GB
Iter 10 Val loss 1.987
Iter 100 Train loss 1.234 Learning Rate 1e-4 It/sec 1.3 Tokens/sec 130 Trained Tokens 10000 Peak mem 12.5 GB
""".strip()

    parser = MLXLogParser()
    metrics = parser.parse_log(log_content)

    # Metadata
    assert metrics.total_iters == 1000
    assert metrics.trainable_percentage == 0.12
    assert metrics.trainable_params == "7.34M/6.1B"

    # Iterations & losses
    assert metrics.iterations == [1, 100]
    assert len(metrics.train_losses) == 2
    assert metrics.train_losses[0] == pytest.approx(2.345, rel=1e-4)
    assert metrics.train_losses[-1] == pytest.approx(1.234, rel=1e-4)

    # Validation
    assert metrics.val_iterations == [10]
    assert metrics.val_losses == [pytest.approx(1.987, rel=1e-4)]

    # Performance metrics
    assert metrics.iterations_per_sec[-1] == pytest.approx(1.3, rel=1e-4)
    assert metrics.tokens_per_sec[-1] == pytest.approx(130, rel=1e-4)

    # Memory & tokens
    assert metrics.peak_memory[-1] == pytest.approx(12.5, rel=1e-4)
    assert metrics.trained_tokens[-1] == 10000


def test_parse_log_sets_timestamp_from_path(tmp_path):
    run_dir = tmp_path / "2026-02-01_01-28-37"
    run_dir.mkdir()
    log_path = run_dir / "train.log"

    log_path.write_text(
        "Iter 1 Train loss 1.0 Learning Rate 1e-4 It/sec 1.0 Tokens/sec 100 Trained Tokens 100 Peak mem 10.0 GB"
    )

    parser = MLXLogParser()
    metrics = parser.parse_log(log_path.read_text(), log_path)

    assert metrics.timestamp == "2026-02-01_01-28-37"
    assert metrics.log_path == log_path
