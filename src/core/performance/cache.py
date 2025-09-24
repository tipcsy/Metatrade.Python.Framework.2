"""
High-performance caching system.

This module provides comprehensive caching capabilities with multiple
strategies, automatic expiration, and performance monitoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import pickle
import threading
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"                # Least Recently Used
    LFU = "lfu"                # Least Frequently Used
    FIFO = "fifo"              # First In, First Out
    TTL = "ttl"                # Time To Live only
    ADAPTIVE = "adaptive"       # Adaptive based on access patterns


class CacheEntry(BaseModel, Generic[T]):
    """Cache entry with metadata."""

    key: str = Field(description="Cache key")
    value: Any = Field(description="Cached value")
    created_at: datetime = Field(description="Creation timestamp")
    last_accessed: datetime = Field(description="Last access timestamp")
    access_count: int = Field(default=1, description="Access count")
    ttl_seconds: Optional[int] = Field(default=None, description="Time to live in seconds")
    size_bytes: int = Field(default=0, description="Entry size in bytes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False

        age_seconds = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age_seconds > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    @property
    def seconds_since_access(self) -> float:
        """Get seconds since last access."""
        return (datetime.now(timezone.utc) - self.last_accessed).total_seconds()

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


class CacheStats(BaseModel):
    """Cache statistics."""

    total_requests: int = Field(default=0, description="Total cache requests")
    cache_hits: int = Field(default=0, description="Cache hits")
    cache_misses: int = Field(default=0, description="Cache misses")
    evictions: int = Field(default=0, description="Cache evictions")
    entries_count: int = Field(default=0, description="Current entries count")
    total_size_bytes: int = Field(default=0, description="Total cache size in bytes")
    average_access_time_ms: float = Field(default=0.0, description="Average access time in milliseconds")

    @property
    def hit_ratio(self) -> float:
        """Get cache hit ratio."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def miss_ratio(self) -> float:
        """Get cache miss ratio."""
        return 1.0 - self.hit_ratio


class Cache:
    """
    High-performance cache with multiple eviction strategies.

    Provides thread-safe caching with automatic expiration,
    size limits, and comprehensive statistics tracking.
    """

    def __init__(
        self,
        name: str = "default",
        max_size: int = 10000,
        max_memory_mb: int = 100,
        default_ttl_seconds: Optional[int] = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU
    ):
        """
        Initialize cache.

        Args:
            name: Cache name
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl_seconds: Default TTL in seconds
            strategy: Eviction strategy
        """
        self.name = name
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.strategy = strategy

        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

        # Strategy-specific data structures
        self._access_order: List[str] = []  # For LRU
        self._access_frequency: Dict[str, int] = {}  # For LFU
        self._insertion_order: List[str] = []  # For FIFO

        # Statistics
        self._stats = CacheStats()

        # Performance tracking
        self._access_times: List[float] = []

        logger.info(f"Cache '{name}' initialized with strategy {strategy.value}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        start_time = time.time()

        try:
            with self._lock:
                self._stats.total_requests += 1

                # Check if key exists
                if key not in self._entries:
                    self._stats.cache_misses += 1
                    return default

                entry = self._entries[key]

                # Check expiration
                if entry.is_expired:
                    self._remove_entry(key)
                    self._stats.cache_misses += 1
                    return default

                # Update access information
                entry.touch()
                self._update_access_tracking(key)

                self._stats.cache_hits += 1
                return entry.value

        finally:
            # Track access time
            access_time_ms = (time.time() - start_time) * 1000
            self._track_access_time(access_time_ms)

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live (uses default if None)
            metadata: Optional metadata

        Returns:
            True if set successfully
        """
        with self._lock:
            try:
                # Calculate value size
                size_bytes = self._calculate_size(value)
                ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                    ttl_seconds=ttl,
                    size_bytes=size_bytes,
                    metadata=metadata or {}
                )

                # Check if we need to evict entries
                self._ensure_capacity(size_bytes)

                # Remove existing entry if present
                if key in self._entries:
                    self._remove_entry(key)

                # Add new entry
                self._entries[key] = entry
                self._add_to_tracking(key)

                # Update statistics
                self._stats.entries_count = len(self._entries)
                self._stats.total_size_bytes = sum(e.size_bytes for e in self._entries.values())

                return True

            except Exception as e:
                logger.error(f"Error setting cache entry {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._access_order.clear()
            self._access_frequency.clear()
            self._insertion_order.clear()

            self._stats.entries_count = 0
            self._stats.total_size_bytes = 0

            logger.info(f"Cache '{self.name}' cleared")

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._entries:
                return False

            entry = self._entries[key]
            if entry.is_expired:
                self._remove_entry(key)
                return False

            return True

    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._entries.keys())

    def values(self) -> List[Any]:
        """Get all cache values."""
        with self._lock:
            return [entry.value for entry in self._entries.values()]

    def items(self) -> List[tuple[str, Any]]:
        """Get all cache items as key-value pairs."""
        with self._lock:
            return [(key, entry.value) for key, entry in self._entries.items()]

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            stats = self._stats.model_copy()
            stats.entries_count = len(self._entries)
            stats.total_size_bytes = sum(e.size_bytes for e in self._entries.values())

            # Calculate average access time
            if self._access_times:
                stats.average_access_time_ms = sum(self._access_times) / len(self._access_times)

            return stats

    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about cache entry."""
        with self._lock:
            if key not in self._entries:
                return None

            entry = self._entries[key]
            return {
                "key": entry.key,
                "created_at": entry.created_at.isoformat(),
                "last_accessed": entry.last_accessed.isoformat(),
                "access_count": entry.access_count,
                "ttl_seconds": entry.ttl_seconds,
                "size_bytes": entry.size_bytes,
                "age_seconds": entry.age_seconds,
                "seconds_since_access": entry.seconds_since_access,
                "is_expired": entry.is_expired,
                "metadata": entry.metadata
            }

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._entries.items()
                if entry.is_expired
            ]

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries from cache '{self.name}'")

            return len(expired_keys)

    def _ensure_capacity(self, new_entry_size: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Check size limit
        while len(self._entries) >= self.max_size and self._entries:
            self._evict_entry()

        # Check memory limit
        current_memory = sum(e.size_bytes for e in self._entries.values())
        while (current_memory + new_entry_size > self.max_memory_bytes and self._entries):
            evicted_size = self._evict_entry()
            current_memory -= evicted_size

    def _evict_entry(self) -> int:
        """Evict entry based on strategy and return size of evicted entry."""
        if not self._entries:
            return 0

        if self.strategy == CacheStrategy.LRU:
            key_to_evict = self._get_lru_key()
        elif self.strategy == CacheStrategy.LFU:
            key_to_evict = self._get_lfu_key()
        elif self.strategy == CacheStrategy.FIFO:
            key_to_evict = self._get_fifo_key()
        elif self.strategy == CacheStrategy.TTL:
            key_to_evict = self._get_oldest_key()
        else:  # ADAPTIVE
            key_to_evict = self._get_adaptive_key()

        if key_to_evict:
            entry = self._entries[key_to_evict]
            size = entry.size_bytes
            self._remove_entry(key_to_evict)
            self._stats.evictions += 1
            return size

        return 0

    def _get_lru_key(self) -> Optional[str]:
        """Get least recently used key."""
        if not self._access_order:
            return list(self._entries.keys())[0] if self._entries else None
        return self._access_order[0]

    def _get_lfu_key(self) -> Optional[str]:
        """Get least frequently used key."""
        if not self._access_frequency:
            return list(self._entries.keys())[0] if self._entries else None

        min_freq = min(self._access_frequency.values())
        for key, freq in self._access_frequency.items():
            if freq == min_freq:
                return key
        return None

    def _get_fifo_key(self) -> Optional[str]:
        """Get first in, first out key."""
        if not self._insertion_order:
            return list(self._entries.keys())[0] if self._entries else None
        return self._insertion_order[0]

    def _get_oldest_key(self) -> Optional[str]:
        """Get oldest entry by creation time."""
        if not self._entries:
            return None

        oldest_key = min(self._entries.keys(), key=lambda k: self._entries[k].created_at)
        return oldest_key

    def _get_adaptive_key(self) -> Optional[str]:
        """Get key using adaptive strategy."""
        # Simple adaptive: prefer LRU for frequently accessed items, FIFO for others
        if not self._entries:
            return None

        # Calculate average access count
        avg_access = sum(e.access_count for e in self._entries.values()) / len(self._entries)

        # Find entries with low access count
        low_access_keys = [
            key for key, entry in self._entries.items()
            if entry.access_count < avg_access * 0.5
        ]

        if low_access_keys:
            # Use FIFO for low-access items
            oldest_key = min(low_access_keys, key=lambda k: self._entries[k].created_at)
            return oldest_key
        else:
            # Use LRU for high-access items
            return self._get_lru_key()

    def _remove_entry(self, key: str) -> None:
        """Remove entry and update tracking structures."""
        if key in self._entries:
            del self._entries[key]

        if key in self._access_order:
            self._access_order.remove(key)

        if key in self._access_frequency:
            del self._access_frequency[key]

        if key in self._insertion_order:
            self._insertion_order.remove(key)

    def _add_to_tracking(self, key: str) -> None:
        """Add key to tracking structures."""
        # Add to insertion order
        self._insertion_order.append(key)

        # Add to access order
        self._access_order.append(key)

        # Initialize access frequency
        self._access_frequency[key] = 1

    def _update_access_tracking(self, key: str) -> None:
        """Update access tracking for key."""
        # Update LRU order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        # Update access frequency
        self._access_frequency[key] = self._access_frequency.get(key, 0) + 1

    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback size estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) for k, v in value.items())
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            else:
                return 64  # Default estimate

    def _track_access_time(self, access_time_ms: float) -> None:
        """Track cache access time for statistics."""
        self._access_times.append(access_time_ms)

        # Limit access time history
        if len(self._access_times) > 1000:
            self._access_times = self._access_times[-1000:]


class CacheManager:
    """
    Manager for multiple named caches.

    Provides centralized cache management with automatic cleanup
    and global statistics tracking.
    """

    def __init__(self):
        """Initialize cache manager."""
        self._caches: Dict[str, Cache] = {}
        self._lock = threading.RLock()

        # Global statistics
        self._global_stats = {
            "total_caches": 0,
            "total_requests": 0,
            "total_hits": 0,
            "total_misses": 0
        }

        # Start cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

        logger.info("Cache manager initialized")

    def get_cache(
        self,
        name: str,
        max_size: int = 10000,
        max_memory_mb: int = 100,
        default_ttl_seconds: Optional[int] = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU
    ) -> Cache:
        """
        Get or create named cache.

        Args:
            name: Cache name
            max_size: Maximum entries
            max_memory_mb: Maximum memory in MB
            default_ttl_seconds: Default TTL
            strategy: Eviction strategy

        Returns:
            Cache instance
        """
        with self._lock:
            if name not in self._caches:
                self._caches[name] = Cache(
                    name=name,
                    max_size=max_size,
                    max_memory_mb=max_memory_mb,
                    default_ttl_seconds=default_ttl_seconds,
                    strategy=strategy
                )
                self._global_stats["total_caches"] += 1
                logger.info(f"Created cache: {name}")

            return self._caches[name]

    def remove_cache(self, name: str) -> bool:
        """Remove named cache."""
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                self._global_stats["total_caches"] -= 1
                logger.info(f"Removed cache: {name}")
                return True
            return False

    def clear_all_caches(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            logger.info("Cleared all caches")

    def get_all_caches(self) -> Dict[str, Cache]:
        """Get all caches."""
        with self._lock:
            return self._caches.copy()

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global cache statistics."""
        with self._lock:
            stats = self._global_stats.copy()

            # Aggregate cache statistics
            total_entries = 0
            total_memory = 0
            total_requests = 0
            total_hits = 0

            for cache in self._caches.values():
                cache_stats = cache.get_stats()
                total_entries += cache_stats.entries_count
                total_memory += cache_stats.total_size_bytes
                total_requests += cache_stats.total_requests
                total_hits += cache_stats.cache_hits

            stats.update({
                "total_entries": total_entries,
                "total_memory_mb": total_memory / (1024 * 1024),
                "total_requests": total_requests,
                "total_hits": total_hits,
                "global_hit_ratio": total_hits / total_requests if total_requests > 0 else 0.0
            })

            return stats

    def get_cache_summaries(self) -> List[Dict[str, Any]]:
        """Get summary information for all caches."""
        with self._lock:
            summaries = []

            for name, cache in self._caches.items():
                stats = cache.get_stats()
                summaries.append({
                    "name": name,
                    "strategy": cache.strategy.value,
                    "entries": stats.entries_count,
                    "max_size": cache.max_size,
                    "memory_mb": stats.total_size_bytes / (1024 * 1024),
                    "max_memory_mb": cache.max_memory_bytes / (1024 * 1024),
                    "hit_ratio": stats.hit_ratio,
                    "total_requests": stats.total_requests
                })

            return summaries

    def cleanup_all_expired(self) -> Dict[str, int]:
        """Clean up expired entries in all caches."""
        with self._lock:
            cleanup_results = {}

            for name, cache in self._caches.items():
                expired_count = cache.cleanup_expired()
                if expired_count > 0:
                    cleanup_results[name] = expired_count

            return cleanup_results

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        async def cleanup_task():
            """Periodic cleanup of expired entries."""
            while True:
                try:
                    cleanup_results = self.cleanup_all_expired()
                    if cleanup_results:
                        total_cleaned = sum(cleanup_results.values())
                        logger.debug(f"Cleaned up {total_cleaned} expired cache entries")

                    await asyncio.sleep(300)  # Clean up every 5 minutes

                except Exception as e:
                    logger.error(f"Cache cleanup task failed: {e}")
                    await asyncio.sleep(60)  # Retry after 1 minute on error

        # Start the cleanup task
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_task())
        except RuntimeError:
            # No event loop running, cleanup task will be started later
            pass


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager()

    return _cache_manager


def cached(
    cache_name: str = "default",
    ttl_seconds: Optional[int] = None,
    key_func: Optional[Callable[..., str]] = None
):
    """
    Decorator for caching function results.

    Args:
        cache_name: Name of cache to use
        ttl_seconds: Time to live for cached result
        key_func: Function to generate cache key

    Example:
        @cached("api_cache", ttl_seconds=300)
        def get_market_data(symbol):
            # API call
            return data
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()

            # Get cache
            cache = get_cache_manager().get_cache(cache_name)

            # Try to get cached result
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl_seconds)
            return result

        return wrapper
    return decorator