"""Tests for prompt response cache with TTL and version-aware invalidation."""

import time
from unittest.mock import patch

import pytest

from promptops.cache.response_cache import CacheEntry, CacheStats, ResponseCache


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_not_expired_within_ttl(self):
        """Test that entry is not expired within TTL."""
        entry = CacheEntry(
            key="test",
            value="response",
            prompt_version="1.0.0",
            created_at=time.time(),
            ttl_seconds=3600.0,
        )
        assert not entry.is_expired
        assert entry.remaining_ttl > 0

    def test_entry_expired_after_ttl(self):
        """Test that entry is expired after TTL."""
        entry = CacheEntry(
            key="test",
            value="response",
            prompt_version="1.0.0",
            created_at=time.time() - 100,
            ttl_seconds=50.0,
        )
        assert entry.is_expired
        assert entry.remaining_ttl == 0.0


class TestCacheStats:
    """Tests for CacheStats tracking."""

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == pytest.approx(0.75)
        assert stats.miss_rate == pytest.approx(0.25)
        assert stats.total_requests == 100

    def test_zero_requests_returns_zero_rate(self):
        """Test that zero requests gives 0.0 rates."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 0.0

    def test_reset_clears_all_stats(self):
        """Test that reset clears all statistics."""
        stats = CacheStats(hits=10, misses=5, evictions=2)
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0


class TestResponseCache:
    """Tests for the main ResponseCache."""

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = ResponseCache(max_size=100, default_ttl=3600)
        cache.put("hello world", "response_1")
        result = cache.get("hello world")
        assert result == "response_1"

    def test_get_miss_returns_none(self):
        """Test that cache miss returns None."""
        cache = ResponseCache()
        result = cache.get("nonexistent")
        assert result is None
        assert cache.stats.misses == 1

    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = ResponseCache(default_ttl=1.0)
        cache.put("prompt", "response")

        # Manually expire the entry
        key = cache._make_key("prompt", cache.current_version)
        cache._cache[key].created_at = time.time() - 2.0

        result = cache.get("prompt")
        assert result is None
        assert cache.stats.expirations == 1

    def test_version_aware_invalidation(self):
        """Test that version bump invalidates old entries."""
        cache = ResponseCache(current_version="1.0.0")
        cache.put("prompt", "old_response")

        invalidated = cache.set_version("2.0.0", invalidate=True)
        assert invalidated == 1

        result = cache.get("prompt")
        assert result is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ResponseCache(max_size=3)
        cache.put("a", "response_a")
        cache.put("b", "response_b")
        cache.put("c", "response_c")

        # Access 'a' to make it recently used
        cache.get("a")

        # Adding 'd' should evict 'b' (least recently used)
        cache.put("d", "response_d")

        assert cache.get("a") == "response_a"
        assert cache.get("b") is None  # evicted
        assert cache.get("c") == "response_c"
        assert cache.get("d") == "response_d"

    def test_invalidate_specific_entry(self):
        """Test invalidating a specific cache entry."""
        cache = ResponseCache()
        cache.put("prompt1", "response1")
        cache.put("prompt2", "response2")

        removed = cache.invalidate("prompt1")
        assert removed is True
        assert cache.get("prompt1") is None
        assert cache.get("prompt2") == "response2"

    def test_clear_removes_all_entries(self):
        """Test that clear removes all entries."""
        cache = ResponseCache()
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")

        count = cache.clear()
        assert count == 3
        assert cache.size == 0

    def test_cleanup_expired_entries(self):
        """Test cleanup of expired entries."""
        cache = ResponseCache(default_ttl=1.0)
        cache.put("fresh", "value1")
        cache.put("stale", "value2")

        # Manually expire one entry
        key = cache._make_key("stale", cache.current_version)
        cache._cache[key].created_at = time.time() - 2.0

        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.get("fresh") == "value1"

    def test_stats_tracking(self):
        """Test that hit/miss stats are tracked correctly."""
        cache = ResponseCache()
        cache.put("exists", "value")

        cache.get("exists")  # hit
        cache.get("missing")  # miss

        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
        assert cache.stats.hit_rate == pytest.approx(0.5)

    def test_contains_check(self):
        """Test __contains__ for membership testing."""
        cache = ResponseCache()
        cache.put("present", "value")

        assert "present" in cache
        assert "absent" not in cache

    def test_version_set_without_invalidation(self):
        """Test version update without invalidation."""
        cache = ResponseCache(current_version="1.0.0")
        cache.put("prompt", "response")

        invalidated = cache.set_version("2.0.0", invalidate=False)
        assert invalidated == 0
        assert cache.size == 1
