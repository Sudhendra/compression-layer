#!/usr/bin/env python3
"""CLI wrapper for adapter evaluation."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate_adapter import main

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
