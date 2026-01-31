"""Tinker SDK adapter layer."""

from __future__ import annotations

import os


class TinkerSDKClient:
    """Lightweight wrapper for SDK availability."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("TINKER_API_KEY", "")
        self.is_available = bool(self.api_key)
