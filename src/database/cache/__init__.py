"""
Cache package for the MetaTrader Python Framework.

This package provides enterprise-grade caching capabilities using Redis
with automatic failover, connection pooling, and intelligent cache management.
"""

from __future__ import annotations

# Import core cache components
from .redis_client import (
    RedisClient,
    get_redis_client,
    initialize_redis,
    close_redis,
)

from .cache_manager import (
    CacheManager,
    get_cache_manager,
    initialize_cache,
    cached,
    cached_async,
    cache_invalidate,
)

# Export all cache functionality
__all__ = [
    # Redis client
    "RedisClient",
    "get_redis_client",
    "initialize_redis",
    "close_redis",

    # Cache manager
    "CacheManager",
    "get_cache_manager",
    "initialize_cache",

    # Decorators
    "cached",
    "cached_async",
    "cache_invalidate",
]