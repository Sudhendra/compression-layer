"""Run storage helpers for MLX training."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path


def create_run_dir(base_dir: Path) -> Path:
    """Create a timestamped run directory under the base directory."""
    max_attempts = 5
    for attempt in range(max_attempts):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = base_dir / timestamp
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(1)
            continue
        return run_dir
    raise RuntimeError("Failed to create a unique run directory.")
