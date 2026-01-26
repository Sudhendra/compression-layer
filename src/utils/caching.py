"""Disk-backed semantic cache using diskcache."""

import hashlib
import json
from pathlib import Path
from typing import Any

from diskcache import Cache

from .config import get_settings


class SemanticCache:
    """
    Disk-backed cache for API responses and computed results.

    Keys are content hashes to enable semantic deduplication.
    """

    def __init__(self, cache_dir: Path | None = None):
        if cache_dir is None:
            cache_dir = get_settings().cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(cache_dir))

    def _hash_key(self, key: str) -> str:
        """Create a consistent hash for any string key."""
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def get(self, key: str) -> Any | None:
        """Get value from cache by key."""
        hashed = self._hash_key(key)
        value = self._cache.get(hashed)
        if value is not None:
            return json.loads(value)
        return None

    def set(self, key: str, value: Any, expire: int | None = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key (will be hashed)
            value: Value to store (must be JSON serializable)
            expire: Optional TTL in seconds
        """
        hashed = self._hash_key(key)
        self._cache.set(hashed, json.dumps(value), expire=expire)

    def delete(self, key: str) -> bool:
        """Delete a key from the cache. Returns True if key existed."""
        hashed = self._hash_key(key)
        return bool(self._cache.delete(hashed))

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def close(self) -> None:
        """Close the cache connection."""
        self._cache.close()

    def __enter__(self) -> "SemanticCache":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def content_hash(content: str) -> str:
    """
    Generate a hash key for content.

    Used to create cache keys for API responses based on input content.
    """
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def make_cache_key(prefix: str, content: str, **kwargs: Any) -> str:
    """
    Create a cache key from prefix, content, and optional parameters.

    Example:
        key = make_cache_key("claude", prompt, model="sonnet", temp=0.4)
    """
    parts = [prefix, content_hash(content)]
    if kwargs:
        # Sort for deterministic ordering
        params = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        parts.append(params)
    return ":".join(parts)
