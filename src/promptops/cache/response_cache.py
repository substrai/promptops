"""Prompt response cache with TTL and version-aware invalidation.

Provides a high-performance cache for LLM prompt responses, keyed by
prompt version and input hash. Features include configurable TTL,
automatic invalidation on version bumps, LRU eviction, and hit/miss
statistics tracking.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    key: str
    value: Any
    prompt_version: str
    created_at: float
    ttl_seconds: float
    access_count: int = 0
    last_accessed: float = 0.0

    @property
    def is_expired(self) -> bool:
        """Check if this entry has exceeded its TTL."""
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def remaining_ttl(self) -> float:
        """Get remaining TTL in seconds."""
        remaining = self.ttl_seconds - (time.time() - self.created_at)
        return max(0.0, remaining)


@dataclass
class CacheStats:
    """Cache hit/miss statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    expirations: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Hit rate as a fraction (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """Miss rate as a fraction (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.misses / self.total_requests

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
        self.expirations = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to a dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "expirations": self.expirations,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
        }


class ResponseCache:
    """LRU cache for prompt responses with TTL and version-aware invalidation.

    Cache keys are derived from (prompt_version + input_hash) to ensure
    that version bumps automatically invalidate stale entries.

    Args:
        max_size: Maximum number of entries in the cache.
        default_ttl: Default time-to-live in seconds for cache entries.
        current_version: The current prompt version string.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
        current_version: str = "1.0.0",
    ):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._current_version = current_version
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = threading.RLock()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def current_version(self) -> str:
        """Get the current prompt version."""
        return self._current_version

    @property
    def size(self) -> int:
        """Get the current number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    def set_version(self, version: str, invalidate: bool = True) -> int:
        """Update the prompt version, optionally invalidating old entries.

        Args:
            version: The new prompt version string.
            invalidate: Whether to invalidate entries from previous versions.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            old_version = self._current_version
            self._current_version = version

            if not invalidate or old_version == version:
                return 0

            # Remove entries with old version
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry.prompt_version != version
            ]
            for key in keys_to_remove:
                del self._cache[key]
                self._stats.invalidations += 1

            return len(keys_to_remove)

    def get(self, prompt_input: str, version: Optional[str] = None) -> Optional[Any]:
        """Retrieve a cached response.

        Args:
            prompt_input: The prompt input string to look up.
            version: Optional version override (defaults to current version).

        Returns:
            The cached response value, or None if not found/expired.
        """
        version = version or self._current_version
        key = self._make_key(prompt_input, version)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None

            # Version mismatch check
            if entry.prompt_version != self._current_version:
                del self._cache[key]
                self._stats.invalidations += 1
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._stats.hits += 1
            return entry.value

    def put(
        self,
        prompt_input: str,
        response: Any,
        ttl: Optional[float] = None,
        version: Optional[str] = None,
    ) -> str:
        """Store a response in the cache.

        Args:
            prompt_input: The prompt input string (used for key generation).
            response: The response value to cache.
            ttl: Optional TTL override in seconds.
            version: Optional version override.

        Returns:
            The cache key used for storage.
        """
        version = version or self._current_version
        ttl = ttl if ttl is not None else self._default_ttl
        key = self._make_key(prompt_input, version)

        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_lru()

            entry = CacheEntry(
                key=key,
                value=response,
                prompt_version=version,
                created_at=time.time(),
                ttl_seconds=ttl,
                last_accessed=time.time(),
            )
            self._cache[key] = entry
            self._cache.move_to_end(key)

        return key

    def invalidate(self, prompt_input: str, version: Optional[str] = None) -> bool:
        """Invalidate a specific cache entry.

        Args:
            prompt_input: The prompt input string.
            version: Optional version override.

        Returns:
            True if an entry was removed, False otherwise.
        """
        version = version or self._current_version
        key = self._make_key(prompt_input, version)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.invalidations += 1
                return True
            return False

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of expired entries removed.
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats.expirations += 1
            return len(expired_keys)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if self._cache:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

    def _make_key(self, prompt_input: str, version: str) -> str:
        """Generate a cache key from prompt version and input hash.

        Args:
            prompt_input: The prompt input string.
            version: The prompt version.

        Returns:
            A deterministic cache key string.
        """
        input_hash = hashlib.sha256(prompt_input.encode("utf-8")).hexdigest()[:16]
        return f"{version}:{input_hash}"

    def __contains__(self, prompt_input: str) -> bool:
        """Check if a prompt input is in the cache (without updating stats)."""
        key = self._make_key(prompt_input, self._current_version)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                return False
            return True

    def __len__(self) -> int:
        return self.size
