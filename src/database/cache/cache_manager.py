"""
Cache manager for the MetaTrader Python Framework.

This module provides high-level caching functionality with decorators,
cache patterns, and automatic cache management for trading data.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from src.core.logging import get_logger
from .redis_client import RedisClient, get_redis_client

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

logger = get_logger(__name__)


class CacheManager:
    """
    High-level cache manager with intelligent caching strategies.

    Provides caching patterns optimized for trading data including
    market data, symbol information, and user sessions.
    """

    def __init__(self, redis_client: Optional[RedisClient] = None):
        """
        Initialize cache manager.

        Args:
            redis_client: Redis client instance (defaults to global client)
        """
        self._redis_client = redis_client or get_redis_client()
        self._default_ttl = 300  # 5 minutes
        self._key_prefix = "mt_framework"

        # Cache configuration for different data types
        self._cache_configs = {
            'symbol': {'ttl': 3600, 'prefix': 'sym'},  # 1 hour
            'market_data': {'ttl': 60, 'prefix': 'md'},  # 1 minute
            'tick_data': {'ttl': 30, 'prefix': 'tick'},  # 30 seconds
            'user_session': {'ttl': 1800, 'prefix': 'sess'},  # 30 minutes
            'account': {'ttl': 300, 'prefix': 'acc'},  # 5 minutes
            'order': {'ttl': 60, 'prefix': 'ord'},  # 1 minute
            'position': {'ttl': 60, 'prefix': 'pos'},  # 1 minute
            'strategy': {'ttl': 600, 'prefix': 'strat'},  # 10 minutes
            'configuration': {'ttl': 3600, 'prefix': 'cfg'},  # 1 hour
            'health_check': {'ttl': 30, 'prefix': 'health'},  # 30 seconds
        }

    def _build_key(self, cache_type: str, identifier: str) -> str:
        """
        Build cache key with proper prefixing.

        Args:
            cache_type: Type of cached data
            identifier: Unique identifier for the data

        Returns:
            Formatted cache key
        """
        config = self._cache_configs.get(cache_type, {'prefix': cache_type})
        return f"{self._key_prefix}:{config['prefix']}:{identifier}"

    def _get_ttl(self, cache_type: str, custom_ttl: Optional[int] = None) -> int:
        """
        Get TTL for cache type.

        Args:
            cache_type: Type of cached data
            custom_ttl: Custom TTL override

        Returns:
            TTL in seconds
        """
        if custom_ttl is not None:
            return custom_ttl

        config = self._cache_configs.get(cache_type, {})
        return config.get('ttl', self._default_ttl)

    def _serialize_args(self, *args, **kwargs) -> str:
        """
        Serialize function arguments to create cache key.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Serialized arguments as string
        """
        # Create a stable string representation of arguments
        args_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(args_str.encode()).hexdigest()

    def get(
        self,
        cache_type: str,
        identifier: str,
        default: Any = None
    ) -> Any:
        """
        Get value from cache.

        Args:
            cache_type: Type of cached data
            identifier: Unique identifier
            default: Default value if not found

        Returns:
            Cached value or default
        """
        key = self._build_key(cache_type, identifier)
        return self._redis_client.get(key, default)

    async def get_async(
        self,
        cache_type: str,
        identifier: str,
        default: Any = None
    ) -> Any:
        """
        Asynchronously get value from cache.

        Args:
            cache_type: Type of cached data
            identifier: Unique identifier
            default: Default value if not found

        Returns:
            Cached value or default
        """
        key = self._build_key(cache_type, identifier)
        return await self._redis_client.get_async(key, default)

    def set(
        self,
        cache_type: str,
        identifier: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            cache_type: Type of cached data
            identifier: Unique identifier
            value: Value to cache
            ttl: Custom TTL (overrides default)

        Returns:
            True if successful
        """
        key = self._build_key(cache_type, identifier)
        cache_ttl = self._get_ttl(cache_type, ttl)
        return self._redis_client.set(key, value, cache_ttl)

    async def set_async(
        self,
        cache_type: str,
        identifier: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Asynchronously set value in cache.

        Args:
            cache_type: Type of cached data
            identifier: Unique identifier
            value: Value to cache
            ttl: Custom TTL (overrides default)

        Returns:
            True if successful
        """
        key = self._build_key(cache_type, identifier)
        cache_ttl = self._get_ttl(cache_type, ttl)
        return await self._redis_client.set_async(key, value, cache_ttl)

    def delete(self, cache_type: str, *identifiers: str) -> int:
        """
        Delete cached values.

        Args:
            cache_type: Type of cached data
            *identifiers: Identifiers to delete

        Returns:
            Number of keys deleted
        """
        keys = [self._build_key(cache_type, identifier) for identifier in identifiers]
        return self._redis_client.delete(*keys)

    async def delete_async(self, cache_type: str, *identifiers: str) -> int:
        """
        Asynchronously delete cached values.

        Args:
            cache_type: Type of cached data
            *identifiers: Identifiers to delete

        Returns:
            Number of keys deleted
        """
        keys = [self._build_key(cache_type, identifier) for identifier in identifiers]
        return await self._redis_client.delete_async(*keys)

    def invalidate_pattern(self, cache_type: str, pattern: str = "*") -> int:
        """
        Invalidate cache entries matching pattern.

        Args:
            cache_type: Type of cached data
            pattern: Pattern to match (Redis glob pattern)

        Returns:
            Number of keys deleted
        """
        search_pattern = self._build_key(cache_type, pattern)
        keys = self._redis_client.keys(search_pattern)
        if keys:
            return self._redis_client.delete(*keys)
        return 0

    def exists(self, cache_type: str, identifier: str) -> bool:
        """
        Check if cache entry exists.

        Args:
            cache_type: Type of cached data
            identifier: Unique identifier

        Returns:
            True if exists
        """
        key = self._build_key(cache_type, identifier)
        return self._redis_client.exists(key) > 0

    def get_ttl(self, cache_type: str, identifier: str) -> int:
        """
        Get time to live for cache entry.

        Args:
            cache_type: Type of cached data
            identifier: Unique identifier

        Returns:
            TTL in seconds (-1 if no expiration, -2 if doesn't exist)
        """
        key = self._build_key(cache_type, identifier)
        return self._redis_client.ttl(key)

    def extend_ttl(self, cache_type: str, identifier: str, ttl: Optional[int] = None) -> bool:
        """
        Extend TTL for cache entry.

        Args:
            cache_type: Type of cached data
            identifier: Unique identifier
            ttl: New TTL (defaults to cache type default)

        Returns:
            True if successful
        """
        key = self._build_key(cache_type, identifier)
        cache_ttl = self._get_ttl(cache_type, ttl)
        return self._redis_client.expire(key, cache_ttl)

    # Trading-specific cache methods
    def cache_symbol(self, symbol_id: str, symbol_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache symbol data."""
        return self.set('symbol', symbol_id, symbol_data, ttl)

    def get_cached_symbol(self, symbol_id: str) -> Optional[Dict[str, Any]]:
        """Get cached symbol data."""
        return self.get('symbol', symbol_id)

    def cache_market_data(self, symbol_id: str, timeframe: str, data: Dict[str, Any]) -> bool:
        """Cache market data."""
        key = f"{symbol_id}:{timeframe}"
        return self.set('market_data', key, data)

    def get_cached_market_data(self, symbol_id: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get cached market data."""
        key = f"{symbol_id}:{timeframe}"
        return self.get('market_data', key)

    def cache_user_session(self, user_id: str, session_data: Dict[str, Any]) -> bool:
        """Cache user session data."""
        return self.set('user_session', user_id, session_data)

    def get_cached_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user session data."""
        return self.get('user_session', user_id)

    def invalidate_user_session(self, user_id: str) -> bool:
        """Invalidate user session cache."""
        return self.delete('user_session', user_id) > 0

    def cache_account_balance(self, account_id: str, balance_data: Dict[str, Any]) -> bool:
        """Cache account balance data."""
        return self.set('account', f"{account_id}:balance", balance_data)

    def get_cached_account_balance(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get cached account balance data."""
        return self.get('account', f"{account_id}:balance")

    def invalidate_account_cache(self, account_id: str) -> int:
        """Invalidate all account-related cache."""
        return self.invalidate_pattern('account', f"{account_id}:*")

    def cache_health_check(self, component: str, health_data: Dict[str, Any]) -> bool:
        """Cache health check results."""
        return self.set('health_check', component, health_data)

    def get_cached_health_check(self, component: str) -> Optional[Dict[str, Any]]:
        """Get cached health check results."""
        return self.get('health_check', component)

    # Bulk operations
    def get_multiple(self, cache_type: str, identifiers: List[str]) -> Dict[str, Any]:
        """
        Get multiple cached values.

        Args:
            cache_type: Type of cached data
            identifiers: List of identifiers

        Returns:
            Dictionary mapping identifiers to cached values
        """
        result = {}
        for identifier in identifiers:
            value = self.get(cache_type, identifier)
            if value is not None:
                result[identifier] = value
        return result

    def set_multiple(
        self,
        cache_type: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Set multiple cached values.

        Args:
            cache_type: Type of cached data
            data: Dictionary mapping identifiers to values
            ttl: Custom TTL

        Returns:
            Dictionary mapping identifiers to success status
        """
        result = {}
        for identifier, value in data.items():
            result[identifier] = self.set(cache_type, identifier, value, ttl)
        return result

    # Cache statistics and monitoring
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._redis_client.get_stats()

        # Add cache-specific statistics
        cache_stats = {}
        for cache_type in self._cache_configs:
            pattern = self._build_key(cache_type, "*")
            keys = self._redis_client.keys(pattern)
            cache_stats[cache_type] = {
                'key_count': len(keys),
                'pattern': pattern,
                'config': self._cache_configs[cache_type]
            }

        stats['cache_types'] = cache_stats
        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        return self._redis_client.health_check()

    def clear_all_cache(self, confirm: bool = False) -> bool:
        """
        Clear all framework cache (dangerous operation).

        Args:
            confirm: Must be True to proceed

        Returns:
            True if successful
        """
        if not confirm:
            logger.warning("clear_all_cache called without confirmation")
            return False

        pattern = f"{self._key_prefix}:*"
        keys = self._redis_client.keys(pattern)
        if keys:
            deleted = self._redis_client.delete(*keys)
            logger.warning(f"Cleared {deleted} cache entries")
            return deleted > 0
        return True


# Cache decorators
def cached(
    cache_type: str,
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
    condition: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Decorator for caching function results.

    Args:
        cache_type: Type of cached data
        ttl: Custom TTL
        key_func: Function to generate cache key from arguments
        condition: Function to determine if result should be cached

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        cache_manager = CacheManager()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager._serialize_args(*args, **kwargs)

            # Try to get from cache
            cached_result = cache_manager.get(cache_type, cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result if condition is met
            if condition is None or condition(result):
                success = cache_manager.set(cache_type, cache_key, result, ttl)
                if success:
                    logger.debug(f"Cached result for {func.__name__}: {cache_key}")
                else:
                    logger.warning(f"Failed to cache result for {func.__name__}: {cache_key}")

            return result

        return wrapper
    return decorator


def cached_async(
    cache_type: str,
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
    condition: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Decorator for caching async function results.

    Args:
        cache_type: Type of cached data
        ttl: Custom TTL
        key_func: Function to generate cache key from arguments
        condition: Function to determine if result should be cached

    Returns:
        Decorated async function
    """
    def decorator(func: F) -> F:
        cache_manager = CacheManager()

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager._serialize_args(*args, **kwargs)

            # Try to get from cache
            cached_result = await cache_manager.get_async(cache_type, cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result if condition is met
            if condition is None or condition(result):
                success = await cache_manager.set_async(cache_type, cache_key, result, ttl)
                if success:
                    logger.debug(f"Cached result for {func.__name__}: {cache_key}")
                else:
                    logger.warning(f"Failed to cache result for {func.__name__}: {cache_key}")

            return result

        return wrapper
    return decorator


def cache_invalidate(cache_type: str, key_func: Optional[Callable] = None):
    """
    Decorator for invalidating cache entries.

    Args:
        cache_type: Type of cached data
        key_func: Function to generate cache key from arguments

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        cache_manager = CacheManager()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute function first
            result = func(*args, **kwargs)

            # Invalidate cache
            if key_func:
                cache_key = key_func(*args, **kwargs)
                cache_manager.delete(cache_type, cache_key)
                logger.debug(f"Invalidated cache for {func.__name__}: {cache_key}")
            else:
                # Invalidate all entries of this type (dangerous)
                deleted = cache_manager.invalidate_pattern(cache_type)
                logger.debug(f"Invalidated {deleted} cache entries for {func.__name__}")

            return result

        return wrapper
    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get global cache manager instance.

    Returns:
        Cache manager instance
    """
    global _cache_manager

    if _cache_manager is None:
        _cache_manager = CacheManager()

    return _cache_manager


def initialize_cache(redis_client: Optional[RedisClient] = None) -> CacheManager:
    """
    Initialize global cache manager.

    Args:
        redis_client: Redis client instance

    Returns:
        Cache manager instance
    """
    global _cache_manager

    _cache_manager = CacheManager(redis_client)
    return _cache_manager