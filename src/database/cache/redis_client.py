"""
Redis client and cache management for the MetaTrader Python Framework.

This module provides enterprise-grade Redis caching capabilities with
connection pooling, health monitoring, and automatic failover support.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

# Optional Redis imports
try:
    import redis
    import redis.asyncio as aioredis
    from redis.connection import ConnectionPool
    from redis.exceptions import ConnectionError, RedisError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    # Create dummy classes when redis is not available
    redis = None
    aioredis = None
    ConnectionPool = None
    ConnectionError = Exception
    RedisError = Exception
    TimeoutError = Exception
    REDIS_AVAILABLE = False

from src.core.config import get_settings
from src.core.exceptions import CacheError, ConnectionError as CoreConnectionError
from src.core.logging import get_logger

T = TypeVar('T')

logger = get_logger(__name__)
settings = get_settings()


class RedisClient:
    """
    Enterprise-grade Redis client with connection pooling and health monitoring.

    Provides both synchronous and asynchronous Redis operations with
    automatic failover, connection pooling, and comprehensive error handling.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        retry_attempts: int = 3,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        health_check_interval: int = 30,
        **kwargs
    ):
        """
        Initialize Redis client.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password
            max_connections: Maximum connection pool size
            retry_attempts: Number of retry attempts for failed operations
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connection timeout in seconds
            health_check_interval: Health check interval in seconds
            **kwargs: Additional Redis connection parameters
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis is not available, cache operations will be disabled")
            self._redis_available = False
            return

        self._redis_available = True
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.retry_attempts = retry_attempts
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.health_check_interval = health_check_interval

        # Connection pools
        self._pool: Optional[ConnectionPool] = None
        self._async_pool: Optional[aioredis.ConnectionPool] = None

        # Clients
        self._client: Optional[redis.Redis] = None
        self._async_client: Optional[aioredis.Redis] = None

        # Status tracking
        self._is_connected = False
        self._last_health_check = 0
        self._connection_errors = 0
        self._operation_stats = {
            'operations_total': 0,
            'operations_successful': 0,
            'operations_failed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

        # Additional connection parameters
        self._connection_kwargs = kwargs

        logger.info(f"Redis client initialized for {host}:{port}/{db}")

    def connect(self) -> None:
        """Initialize Redis connections and connection pools."""
        try:
            # Create connection pool
            self._pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=True,
                health_check_interval=self.health_check_interval,
                **self._connection_kwargs
            )

            # Create sync client
            self._client = redis.Redis(connection_pool=self._pool)

            # Create async connection pool
            self._async_pool = aioredis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=True,
                health_check_interval=self.health_check_interval,
                **self._connection_kwargs
            )

            # Create async client
            self._async_client = aioredis.Redis(connection_pool=self._async_pool)

            # Test connection
            self._test_connection()
            self._is_connected = True
            logger.info("Redis client connected successfully")

        except Exception as e:
            self._connection_errors += 1
            logger.error(f"Failed to connect to Redis: {e}")
            raise CacheError(f"Redis connection failed: {e}") from e

    def disconnect(self) -> None:
        """Close Redis connections and clean up resources."""
        try:
            if self._client:
                self._client.close()
                logger.debug("Sync Redis client closed")

            if self._async_client:
                # Note: async client cleanup should be done in async context
                # This is a synchronous method, so we'll just set to None
                self._async_client = None
                logger.debug("Async Redis client reference cleared")

            if self._pool:
                self._pool.disconnect()
                logger.debug("Connection pool disconnected")

            self._is_connected = False
            logger.info("Redis client disconnected")

        except Exception as e:
            logger.error(f"Error during Redis disconnect: {e}")

    async def disconnect_async(self) -> None:
        """Asynchronously close Redis connections."""
        try:
            if self._async_client:
                await self._async_client.close()
                logger.debug("Async Redis client closed")

            if self._async_pool:
                await self._async_pool.disconnect()
                logger.debug("Async connection pool disconnected")

        except Exception as e:
            logger.error(f"Error during async Redis disconnect: {e}")

    def _test_connection(self) -> None:
        """Test Redis connection."""
        if not self._client:
            raise CacheError("Redis client not initialized")

        try:
            response = self._client.ping()
            if not response:
                raise CacheError("Redis ping failed")
        except Exception as e:
            raise CacheError(f"Redis connection test failed: {e}") from e

    async def _test_connection_async(self) -> None:
        """Test async Redis connection."""
        if not self._async_client:
            raise CacheError("Async Redis client not initialized")

        try:
            response = await self._async_client.ping()
            if not response:
                raise CacheError("Async Redis ping failed")
        except Exception as e:
            raise CacheError(f"Async Redis connection test failed: {e}") from e

    def _execute_with_retry(self, operation: callable, *args, **kwargs) -> Any:
        """Execute Redis operation with retry logic."""
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                self._operation_stats['operations_total'] += 1
                result = operation(*args, **kwargs)
                self._operation_stats['operations_successful'] += 1
                return result

            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                self._connection_errors += 1
                logger.warning(f"Redis operation failed (attempt {attempt + 1}): {e}")

                if attempt < self.retry_attempts - 1:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                    try:
                        # Try to reconnect
                        self._test_connection()
                    except Exception:
                        pass

            except RedisError as e:
                last_exception = e
                logger.error(f"Redis error during operation: {e}")
                break

        self._operation_stats['operations_failed'] += 1
        raise CacheError(f"Redis operation failed after {self.retry_attempts} attempts") from last_exception

    async def _execute_with_retry_async(self, operation: callable, *args, **kwargs) -> Any:
        """Execute async Redis operation with retry logic."""
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                self._operation_stats['operations_total'] += 1
                result = await operation(*args, **kwargs)
                self._operation_stats['operations_successful'] += 1
                return result

            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                self._connection_errors += 1
                logger.warning(f"Async Redis operation failed (attempt {attempt + 1}): {e}")

                if attempt < self.retry_attempts - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                    try:
                        # Try to reconnect
                        await self._test_connection_async()
                    except Exception:
                        pass

            except RedisError as e:
                last_exception = e
                logger.error(f"Async Redis error during operation: {e}")
                break

        self._operation_stats['operations_failed'] += 1
        raise CacheError(f"Async Redis operation failed after {self.retry_attempts} attempts") from last_exception

    def _serialize_value(self, value: Any, serialization: str = "json") -> bytes:
        """Serialize value for Redis storage."""
        try:
            if serialization == "json":
                return json.dumps(value, default=str).encode('utf-8')
            elif serialization == "pickle":
                return pickle.dumps(value)
            else:
                return str(value).encode('utf-8')
        except Exception as e:
            raise CacheError(f"Failed to serialize value: {e}") from e

    def _deserialize_value(self, value: bytes, serialization: str = "json") -> Any:
        """Deserialize value from Redis storage."""
        try:
            if serialization == "json":
                return json.loads(value.decode('utf-8'))
            elif serialization == "pickle":
                return pickle.loads(value)
            else:
                return value.decode('utf-8')
        except Exception as e:
            raise CacheError(f"Failed to deserialize value: {e}") from e

    def get(
        self,
        key: str,
        default: Any = None,
        serialization: str = "json"
    ) -> Any:
        """
        Get value from Redis cache.

        Args:
            key: Cache key
            default: Default value if key not found
            serialization: Serialization method (json, pickle, string)

        Returns:
            Cached value or default
        """
        if not self._is_connected:
            self.connect()

        try:
            def _get():
                return self._client.get(key)

            value = self._execute_with_retry(_get)

            if value is None:
                self._operation_stats['cache_misses'] += 1
                return default

            self._operation_stats['cache_hits'] += 1
            return self._deserialize_value(value, serialization)

        except Exception as e:
            logger.error(f"Error getting cache key '{key}': {e}")
            return default

    async def get_async(
        self,
        key: str,
        default: Any = None,
        serialization: str = "json"
    ) -> Any:
        """
        Asynchronously get value from Redis cache.

        Args:
            key: Cache key
            default: Default value if key not found
            serialization: Serialization method (json, pickle, string)

        Returns:
            Cached value or default
        """
        if not self._async_client:
            raise CacheError("Async Redis client not initialized")

        try:
            async def _get():
                return await self._async_client.get(key)

            value = await self._execute_with_retry_async(_get)

            if value is None:
                self._operation_stats['cache_misses'] += 1
                return default

            self._operation_stats['cache_hits'] += 1
            return self._deserialize_value(value, serialization)

        except Exception as e:
            logger.error(f"Error getting cache key '{key}' (async): {e}")
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialization: str = "json"
    ) -> bool:
        """
        Set value in Redis cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialization: Serialization method (json, pickle, string)

        Returns:
            True if successful, False otherwise
        """
        if not self._is_connected:
            self.connect()

        try:
            def _set():
                serialized_value = self._serialize_value(value, serialization)
                return self._client.set(key, serialized_value, ex=ttl)

            return self._execute_with_retry(_set)

        except Exception as e:
            logger.error(f"Error setting cache key '{key}': {e}")
            return False

    async def set_async(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialization: str = "json"
    ) -> bool:
        """
        Asynchronously set value in Redis cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialization: Serialization method (json, pickle, string)

        Returns:
            True if successful, False otherwise
        """
        if not self._async_client:
            raise CacheError("Async Redis client not initialized")

        try:
            async def _set():
                serialized_value = self._serialize_value(value, serialization)
                return await self._async_client.set(key, serialized_value, ex=ttl)

            return await self._execute_with_retry_async(_set)

        except Exception as e:
            logger.error(f"Error setting cache key '{key}' (async): {e}")
            return False

    def delete(self, *keys: str) -> int:
        """
        Delete keys from Redis cache.

        Args:
            *keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        if not self._is_connected:
            self.connect()

        try:
            def _delete():
                return self._client.delete(*keys)

            return self._execute_with_retry(_delete)

        except Exception as e:
            logger.error(f"Error deleting cache keys {keys}: {e}")
            return 0

    async def delete_async(self, *keys: str) -> int:
        """
        Asynchronously delete keys from Redis cache.

        Args:
            *keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        if not self._async_client:
            raise CacheError("Async Redis client not initialized")

        try:
            async def _delete():
                return await self._async_client.delete(*keys)

            return await self._execute_with_retry_async(_delete)

        except Exception as e:
            logger.error(f"Error deleting cache keys {keys} (async): {e}")
            return 0

    def exists(self, *keys: str) -> int:
        """
        Check if keys exist in Redis cache.

        Args:
            *keys: Keys to check

        Returns:
            Number of keys that exist
        """
        if not self._is_connected:
            self.connect()

        try:
            def _exists():
                return self._client.exists(*keys)

            return self._execute_with_retry(_exists)

        except Exception as e:
            logger.error(f"Error checking cache keys existence {keys}: {e}")
            return 0

    def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for a key.

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self._is_connected:
            self.connect()

        try:
            def _expire():
                return self._client.expire(key, ttl)

            return self._execute_with_retry(_expire)

        except Exception as e:
            logger.error(f"Error setting expiration for cache key '{key}': {e}")
            return False

    def ttl(self, key: str) -> int:
        """
        Get time to live for a key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds (-1 if no expiration, -2 if key doesn't exist)
        """
        if not self._is_connected:
            self.connect()

        try:
            def _ttl():
                return self._client.ttl(key)

            return self._execute_with_retry(_ttl)

        except Exception as e:
            logger.error(f"Error getting TTL for cache key '{key}': {e}")
            return -2

    def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching pattern.

        Args:
            pattern: Key pattern (Redis glob pattern)

        Returns:
            List of matching keys
        """
        if not self._is_connected:
            self.connect()

        try:
            def _keys():
                return [key.decode('utf-8') for key in self._client.keys(pattern)]

            return self._execute_with_retry(_keys)

        except Exception as e:
            logger.error(f"Error getting keys with pattern '{pattern}': {e}")
            return []

    def flush_db(self) -> bool:
        """
        Flush current database.

        Returns:
            True if successful, False otherwise
        """
        if not self._is_connected:
            self.connect()

        try:
            def _flush():
                return self._client.flushdb()

            self._execute_with_retry(_flush)
            logger.warning("Redis database flushed")
            return True

        except Exception as e:
            logger.error(f"Error flushing Redis database: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Perform Redis health check.

        Returns:
            Health check results
        """
        current_time = time.time()

        # Skip frequent health checks
        if current_time - self._last_health_check < self.health_check_interval:
            return {'status': 'skipped', 'reason': 'too_frequent'}

        health_status = {
            'timestamp': current_time,
            'status': 'healthy',
            'details': {},
            'errors': []
        }

        try:
            # Test basic connectivity
            start_time = time.time()
            if self._is_connected:
                self._test_connection()
                response_time = time.time() - start_time

                health_status['details'].update({
                    'ping': 'ok',
                    'response_time_ms': round(response_time * 1000, 2),
                    'is_connected': self._is_connected,
                })

                # Get Redis info
                try:
                    info = self._client.info()
                    health_status['details']['redis_info'] = {
                        'version': info.get('redis_version'),
                        'connected_clients': info.get('connected_clients'),
                        'used_memory_human': info.get('used_memory_human'),
                        'keyspace_hits': info.get('keyspace_hits'),
                        'keyspace_misses': info.get('keyspace_misses'),
                    }
                except Exception as e:
                    health_status['details']['redis_info_error'] = str(e)

                # Add operation statistics
                health_status['details']['operation_stats'] = self._operation_stats.copy()

            else:
                health_status['status'] = 'unhealthy'
                health_status['errors'].append('Not connected to Redis')

        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['errors'].append(str(e))
            logger.error(f"Redis health check failed: {e}")

        self._last_health_check = current_time
        return health_status

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis client statistics."""
        return {
            'connection_info': {
                'host': self.host,
                'port': self.port,
                'db': self.db,
                'is_connected': self._is_connected,
                'connection_errors': self._connection_errors,
                'max_connections': self.max_connections,
            },
            'operation_stats': self._operation_stats.copy(),
            'pool_info': self._get_pool_info(),
        }

    def _get_pool_info(self) -> Dict[str, Any]:
        """Get connection pool information."""
        pool_info = {}

        try:
            if self._pool:
                pool_info['sync_pool'] = {
                    'max_connections': self._pool.max_connections,
                    'connection_kwargs': self._pool.connection_kwargs,
                }
        except Exception as e:
            pool_info['sync_pool_error'] = str(e)

        return pool_info

    @property
    def is_connected(self) -> bool:
        """Check if Redis client is connected."""
        return self._is_connected

    @property
    def client(self) -> redis.Redis:
        """Get sync Redis client."""
        if not self._client:
            raise CacheError("Redis client not initialized")
        return self._client

    @property
    def async_client(self) -> aioredis.Redis:
        """Get async Redis client."""
        if not self._async_client:
            raise CacheError("Async Redis client not initialized")
        return self._async_client

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    async def __aenter__(self):
        """Async context manager entry."""
        self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_async()


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


def get_redis_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[int] = None,
    **kwargs
) -> RedisClient:
    """
    Get or create global Redis client instance.

    Args:
        host: Redis host (defaults to localhost)
        port: Redis port (defaults to 6379)
        db: Redis database (defaults to 0)
        **kwargs: Additional Redis parameters

    Returns:
        Redis client instance
    """
    global _redis_client

    if _redis_client is None:
        _redis_client = RedisClient(
            host=host or "localhost",
            port=port or 6379,
            db=db or 0,
            **kwargs
        )

    return _redis_client


def initialize_redis(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    **kwargs
) -> RedisClient:
    """
    Initialize global Redis client.

    Args:
        host: Redis host
        port: Redis port
        db: Redis database
        **kwargs: Additional Redis parameters

    Returns:
        Initialized Redis client
    """
    global _redis_client

    if _redis_client is not None:
        _redis_client.disconnect()

    _redis_client = RedisClient(host=host, port=port, db=db, **kwargs)
    _redis_client.connect()

    return _redis_client


def close_redis() -> None:
    """Close global Redis client."""
    global _redis_client

    if _redis_client is not None:
        _redis_client.disconnect()
        _redis_client = None