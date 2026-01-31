"""Tests for run storage helpers."""

from datetime import datetime, timedelta

from src.training import run_storage
from src.training.run_storage import create_run_dir


def test_create_run_dir_creates_expected_structure(tmp_path) -> None:
    run_dir = create_run_dir(tmp_path)

    assert run_dir.exists()
    assert run_dir.name


def test_create_run_dir_retries_on_collision(tmp_path, monkeypatch) -> None:
    first_time = datetime(2024, 1, 2, 3, 4, 5)
    second_time = first_time + timedelta(seconds=1)
    timestamp_format = "%Y-%m-%d_%H-%M-%S"

    class FixedDatetime:
        def __init__(self, values):
            self._values = iter(values)

        def now(self):
            return next(self._values)

    monkeypatch.setattr(
        run_storage,
        "datetime",
        FixedDatetime([first_time, second_time]),
    )
    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(run_storage.time, "sleep", fake_sleep)

    (tmp_path / first_time.strftime(timestamp_format)).mkdir()

    run_dir = create_run_dir(tmp_path)

    assert run_dir.exists()
    assert run_dir.name == second_time.strftime(timestamp_format)
    assert sleep_calls
