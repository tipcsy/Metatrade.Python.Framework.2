"""
Database Connection Manager with Connection Pooling.

This module provides enterprise-grade database connection management with
connection pooling, health monitoring, automatic failover, and performance
optimization for the MetaTrader Python Framework.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Callable, ContextManager
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import threading
from contextlib import asynccontextmanager
import weakref

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncConnection, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from sqlalchemy import event

from src.core.config.settings import DatabaseSettings
from src.core.exceptions import DatabaseError, ConnectionError
from src.core.logging import get_logger

logger = get_logger(__name__)


class ConnectionStatus(Enum):
    """Database connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class PoolStatus(Enum):
    """Connection pool status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ConnectionMetrics:
    """Database connection metrics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    connection_attempts: int = 0
    successful_connections: int = 0
    query_count: int = 0
    query_errors: int = 0
    avg_query_time: float = 0.0
    last_activity: float = field(default_factory=time.time)


@dataclass
class PoolConfig:
    """Database pool configuration."""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    echo: bool = False
    echo_pool: bool = False


class DatabaseConnectionManager:
    """
    Enterprise database connection manager with advanced pooling.

    Features:
    - Multiple database connection support
    - Connection pooling with health monitoring
    - Automatic reconnection and failover
    - Performance metrics and monitoring
    - Connection leak detection
    - Query performance tracking
    """

    def __init__(self, database_settings: DatabaseSettings) -> None:
        """Initialize database connection manager.

        Args:
            database_settings: Database configuration settings
        """
        self.settings = database_settings
        self.is_initialized = False
        self.is_shutting_down = False

        # Database engines and session makers
        self._engines: Dict[str, AsyncEngine] = {}
        self._session_makers: Dict[str, async_sessionmaker] = {}
        self._pool_configs: Dict[str, PoolConfig] = {}

        # Connection monitoring
        self._connection_metrics: Dict[str, ConnectionMetrics] = defaultdict(ConnectionMetrics)
        self._pool_status: Dict[str, PoolStatus] = {}

        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Connection tracking
        self._active_connections: Dict[str, weakref.WeakSet] = defaultdict(lambda: weakref.WeakSet())
        self._session_registry: Dict[str, List[AsyncSession]] = defaultdict(list)

        # Event callbacks
        self._connection_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._error_callbacks: List[Callable[[Exception], None]] = []

        # Locks for thread safety
        self._manager_lock = asyncio.Lock()
        self._metrics_lock = asyncio.Lock()

        logger.info(
            "Database connection manager initialized",
            extra={
                "databases_configured": len(database_settings.databases),
                "default_pool_size": database_settings.pool_size,
                "max_overflow": database_settings.max_overflow,
            }
        )

    async def initialize(self) -> None:
        """Initialize the connection manager and create connection pools."""
        async with self._manager_lock:
            if self.is_initialized:
                logger.warning("Database connection manager already initialized")
                return

            try:
                logger.info("Initializing database connection manager")

                # Validate configuration
                await self._validate_configuration()

                # Initialize connection pools for each database
                for db_name, db_config in self.settings.databases.items():
                    if db_config.get("enabled", True):
                        await self._initialize_database_pool(db_name, db_config)

                # Start monitoring tasks
                await self._start_monitoring()

                self.is_initialized = True
                logger.info(
                    "Database connection manager initialized successfully",
                    extra={
                        "active_databases": len(self._engines),
                        "total_pools": len(self._engines),
                    }
                )

            except Exception as e:
                logger.error(
                    "Failed to initialize database connection manager",
                    extra={"error": str(e)},
                    exc_info=True
                )
                await self._cleanup_partial_initialization()
                raise DatabaseError(f"Connection manager initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """Shutdown the connection manager and cleanup resources."""
        async with self._manager_lock:
            if not self.is_initialized or self.is_shutting_down:
                return

            self.is_shutting_down = True
            logger.info("Shutting down database connection manager")

            try:
                # Stop monitoring tasks
                await self._stop_monitoring()

                # Close all sessions
                await self._close_all_sessions()

                # Shutdown all engines
                await self._shutdown_all_engines()

                # Clear state
                self._engines.clear()
                self._session_makers.clear()
                self._pool_configs.clear()
                self._connection_metrics.clear()
                self._pool_status.clear()

                self.is_initialized = False
                logger.info("Database connection manager shutdown completed")

            except Exception as e:
                logger.error(
                    "Error during database connection manager shutdown",
                    extra={"error": str(e)},
                    exc_info=True
                )

    async def get_engine(self, database_name: Optional[str] = None) -> AsyncEngine:
        """Get database engine.

        Args:
            database_name: Database name, uses default if None

        Returns:
            Database engine

        Raises:
            DatabaseError: If engine not found or not initialized
        """
        if not self.is_initialized:
            raise DatabaseError("Connection manager not initialized")

        if database_name is None:
            database_name = self.settings.default_database

        if database_name is None:
            # Use first available database
            if self._engines:
                database_name = next(iter(self._engines.keys()))
            else:
                raise DatabaseError("No databases configured")

        engine = self._engines.get(database_name)
        if engine is None:
            raise DatabaseError(f"Database '{database_name}' not found or not initialized")

        return engine

    @asynccontextmanager
    async def get_connection(self, database_name: Optional[str] = None) -> AsyncConnection:
        """Get database connection context manager.

        Args:
            database_name: Database name

        Returns:
            Database connection context manager
        """
        engine = await self.get_engine(database_name)

        try:
            async with engine.connect() as connection:
                # Track connection
                self._track_connection_acquired(database_name or "default")
                yield connection

        except Exception as e:
            self._track_connection_error(database_name or "default")
            logger.error(
                "Database connection error",
                extra={
                    "database": database_name,
                    "error": str(e),
                },
                exc_info=True
            )
            raise DatabaseError(f"Connection error: {e}") from e
        finally:
            self._track_connection_released(database_name or "default")

    @asynccontextmanager
    async def get_session(self, database_name: Optional[str] = None) -> AsyncSession:
        """Get database session context manager.

        Args:
            database_name: Database name

        Returns:
            Database session context manager
        """
        if not self.is_initialized:
            raise DatabaseError("Connection manager not initialized")

        if database_name is None:
            database_name = self.settings.default_database or next(iter(self._session_makers.keys()), None)

        if database_name is None:
            raise DatabaseError("No database specified and no default configured")

        session_maker = self._session_makers.get(database_name)
        if session_maker is None:
            raise DatabaseError(f"No session maker for database '{database_name}'")

        try:
            async with session_maker() as session:
                # Register session for tracking
                self._session_registry[database_name].append(session)
                self._track_connection_acquired(database_name)

                yield session

        except Exception as e:
            logger.error(
                "Database session error",
                extra={
                    "database": database_name,
                    "error": str(e),
                },
                exc_info=True
            )
            raise DatabaseError(f"Session error: {e}") from e
        finally:
            # Cleanup session registration
            try:
                self._session_registry[database_name].remove(session)
            except ValueError:
                pass  # Session not in registry
            self._track_connection_released(database_name)

    async def execute_query(
        self,
        query: Union[str, sa.text],
        parameters: Optional[Dict[str, Any]] = None,
        database_name: Optional[str] = None
    ) -> Any:
        """Execute a query and return results.

        Args:
            query: SQL query to execute
            parameters: Query parameters
            database_name: Database name

        Returns:
            Query results
        """
        start_time = time.time()

        try:
            async with self.get_connection(database_name) as conn:
                if isinstance(query, str):
                    query = sa.text(query)

                if parameters:
                    result = await conn.execute(query, parameters)
                else:
                    result = await conn.execute(query)

                # Track query metrics
                query_time = time.time() - start_time
                self._track_query_completed(database_name or "default", query_time)

                return result

        except Exception as e:
            self._track_query_error(database_name or "default")
            logger.error(
                "Query execution error",
                extra={
                    "database": database_name,
                    "query": str(query)[:200],
                    "error": str(e),
                },
                exc_info=True
            )
            raise DatabaseError(f"Query execution failed: {e}") from e

    async def health_check(self, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Perform database health check.

        Args:
            database_name: Specific database or None for all

        Returns:
            Health check results
        """
        results = {}

        if database_name:
            databases_to_check = [database_name]
        else:
            databases_to_check = list(self._engines.keys())

        for db_name in databases_to_check:
            results[db_name] = await self._check_database_health(db_name)

        return results

    async def get_metrics(self, database_name: Optional[str] = None) -> Union[ConnectionMetrics, Dict[str, ConnectionMetrics]]:
        """Get connection metrics.

        Args:
            database_name: Specific database or None for all

        Returns:
            Connection metrics
        """
        async with self._metrics_lock:
            if database_name:
                return self._connection_metrics.get(database_name, ConnectionMetrics())
            else:
                return dict(self._connection_metrics)

    async def get_pool_status(self, database_name: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Get connection pool status.

        Args:
            database_name: Specific database or None for all

        Returns:
            Pool status information
        """
        if database_name:
            return await self._get_single_pool_status(database_name)
        else:
            results = {}
            for db_name in self._engines.keys():
                results[db_name] = await self._get_single_pool_status(db_name)
            return results

    def add_connection_callback(self, database_name: str, callback: Callable[[str, ConnectionStatus], None]) -> None:
        """Add connection status callback.

        Args:
            database_name: Database name
            callback: Callback function
        """
        self._connection_callbacks[database_name].append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add error callback.

        Args:
            callback: Error callback function
        """
        self._error_callbacks.append(callback)

    async def _validate_configuration(self) -> None:
        """Validate database configuration."""
        if not self.settings.databases:
            raise DatabaseError("No databases configured")

        for db_name, db_config in self.settings.databases.items():
            if not db_config.get("url"):
                raise DatabaseError(f"No URL configured for database '{db_name}'")

        logger.debug("Database configuration validated successfully")

    async def _initialize_database_pool(self, db_name: str, db_config: Dict[str, Any]) -> None:
        """Initialize connection pool for a database."""
        try:
            logger.info(f"Initializing database pool: {db_name}")

            # Create pool configuration
            pool_config = PoolConfig(
                pool_size=db_config.get("pool_size", self.settings.pool_size),
                max_overflow=db_config.get("max_overflow", self.settings.max_overflow),
                pool_timeout=db_config.get("pool_timeout", self.settings.pool_timeout),
                pool_recycle=db_config.get("pool_recycle", self.settings.pool_recycle),
                pool_pre_ping=db_config.get("pool_pre_ping", True),
                echo=db_config.get("echo", False),
                echo_pool=db_config.get("echo_pool", False)
            )

            self._pool_configs[db_name] = pool_config

            # Create engine with connection pooling
            engine = create_async_engine(
                db_config["url"],
                poolclass=QueuePool,
                pool_size=pool_config.pool_size,
                max_overflow=pool_config.max_overflow,
                pool_timeout=pool_config.pool_timeout,
                pool_recycle=pool_config.pool_recycle,
                pool_pre_ping=pool_config.pool_pre_ping,
                echo=pool_config.echo,
                echo_pool=pool_config.echo_pool,
                future=True
            )

            # Set up event listeners
            self._setup_engine_events(engine, db_name)

            # Test connection
            await self._test_engine_connection(engine, db_name)

            # Create session maker
            session_maker = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Store components
            self._engines[db_name] = engine
            self._session_makers[db_name] = session_maker
            self._pool_status[db_name] = PoolStatus.RUNNING

            logger.info(
                f"Database pool initialized successfully: {db_name}",
                extra={
                    "pool_size": pool_config.pool_size,
                    "max_overflow": pool_config.max_overflow,
                }
            )

        except Exception as e:
            self._pool_status[db_name] = PoolStatus.ERROR
            logger.error(
                f"Failed to initialize database pool: {db_name}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise DatabaseError(f"Pool initialization failed for '{db_name}': {e}") from e

    def _setup_engine_events(self, engine: AsyncEngine, db_name: str) -> None:
        """Set up engine event listeners."""
        @event.listens_for(engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Handle connection events."""
            logger.debug(f"Database connection established: {db_name}")
            self._track_connection_acquired(db_name)

        @event.listens_for(engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout events."""
            self._track_connection_acquired(db_name)

        @event.listens_for(engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Handle connection checkin events."""
            self._track_connection_released(db_name)

        @event.listens_for(engine.sync_engine, "invalidate")
        def on_invalidate(dbapi_connection, connection_record, exception):
            """Handle connection invalidation events."""
            logger.warning(
                f"Database connection invalidated: {db_name}",
                extra={"error": str(exception) if exception else "Unknown"}
            )
            self._track_connection_error(db_name)

    async def _test_engine_connection(self, engine: AsyncEngine, db_name: str) -> None:
        """Test engine connection."""
        try:
            async with engine.connect() as conn:
                await conn.execute(sa.text("SELECT 1"))
            logger.debug(f"Database connection test passed: {db_name}")
        except Exception as e:
            logger.error(
                f"Database connection test failed: {db_name}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise

    async def _start_monitoring(self) -> None:
        """Start monitoring tasks."""
        if self.settings.monitoring.enabled:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Database monitoring started")

    async def _stop_monitoring(self) -> None:
        """Stop monitoring tasks."""
        tasks = [self._health_check_task, self._metrics_task, self._cleanup_task]

        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _health_check_loop(self) -> None:
        """Health check monitoring loop."""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(self.settings.monitoring.health_check_interval)

                if self.is_shutting_down:
                    break

                await self.health_check()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in database health check loop",
                    extra={"error": str(e)},
                    exc_info=True
                )

    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop."""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(self.settings.monitoring.metrics_collection_interval)

                if self.is_shutting_down:
                    break

                await self._collect_pool_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in metrics collection loop",
                    extra={"error": str(e)},
                    exc_info=True
                )

    async def _cleanup_loop(self) -> None:
        """Connection cleanup loop."""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(300)  # 5 minutes

                if self.is_shutting_down:
                    break

                await self._cleanup_stale_connections()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in cleanup loop",
                    extra={"error": str(e)},
                    exc_info=True
                )

    async def _check_database_health(self, db_name: str) -> Dict[str, Any]:
        """Check health of a specific database."""
        result = {
            "database": db_name,
            "status": "unknown",
            "connection_test": False,
            "pool_status": "unknown",
            "metrics": {},
            "error": None
        }

        try:
            # Test basic connection
            engine = self._engines.get(db_name)
            if engine:
                async with engine.connect() as conn:
                    await conn.execute(sa.text("SELECT 1"))
                result["connection_test"] = True
                result["status"] = "healthy"

            # Get pool status
            pool_info = await self._get_single_pool_status(db_name)
            result["pool_status"] = pool_info

            # Get metrics
            metrics = await self.get_metrics(db_name)
            result["metrics"] = metrics.__dict__ if hasattr(metrics, '__dict__') else metrics

        except Exception as e:
            result["status"] = "unhealthy"
            result["error"] = str(e)
            logger.warning(
                f"Database health check failed: {db_name}",
                extra={"error": str(e)}
            )

        return result

    async def _get_single_pool_status(self, db_name: str) -> Dict[str, Any]:
        """Get status for a single connection pool."""
        engine = self._engines.get(db_name)
        if not engine:
            return {"error": "Engine not found"}

        try:
            pool = engine.pool
            return {
                "pool_class": pool.__class__.__name__,
                "pool_size": getattr(pool, '_pool_size', 'unknown'),
                "checked_in": getattr(pool, 'checkedin', lambda: 'unknown')(),
                "checked_out": getattr(pool, 'checkedout', lambda: 'unknown')(),
                "overflow": getattr(pool, 'overflow', lambda: 'unknown')(),
                "invalid": getattr(pool, 'invalid', lambda: 'unknown')(),
                "status": self._pool_status.get(db_name, "unknown")
            }
        except Exception as e:
            return {"error": str(e)}

    async def _collect_pool_metrics(self) -> None:
        """Collect metrics from all connection pools."""
        for db_name, engine in self._engines.items():
            try:
                pool = engine.pool
                async with self._metrics_lock:
                    metrics = self._connection_metrics[db_name]

                    # Update pool metrics if available
                    if hasattr(pool, 'checkedin'):
                        metrics.idle_connections = pool.checkedin()
                    if hasattr(pool, 'checkedout'):
                        metrics.active_connections = pool.checkedout()

                    metrics.total_connections = metrics.active_connections + metrics.idle_connections
                    metrics.last_activity = time.time()

            except Exception as e:
                logger.warning(
                    f"Failed to collect metrics for database: {db_name}",
                    extra={"error": str(e)}
                )

    async def _cleanup_stale_connections(self) -> None:
        """Clean up stale connections and sessions."""
        try:
            # Clean up session registry
            for db_name in list(self._session_registry.keys()):
                sessions = self._session_registry[db_name]
                active_sessions = []

                for session in sessions:
                    try:
                        # Check if session is still valid
                        if not session.is_active:
                            await session.close()
                        else:
                            active_sessions.append(session)
                    except Exception:
                        # Session is invalid, remove it
                        pass

                self._session_registry[db_name] = active_sessions

            logger.debug("Connection cleanup completed")

        except Exception as e:
            logger.error(
                "Error during connection cleanup",
                extra={"error": str(e)},
                exc_info=True
            )

    def _track_connection_acquired(self, db_name: str) -> None:
        """Track connection acquisition."""
        try:
            metrics = self._connection_metrics[db_name]
            metrics.connection_attempts += 1
            metrics.successful_connections += 1
            metrics.last_activity = time.time()
        except Exception as e:
            logger.debug(f"Error tracking connection acquisition: {e}")

    def _track_connection_released(self, db_name: str) -> None:
        """Track connection release."""
        try:
            metrics = self._connection_metrics[db_name]
            metrics.last_activity = time.time()
        except Exception as e:
            logger.debug(f"Error tracking connection release: {e}")

    def _track_connection_error(self, db_name: str) -> None:
        """Track connection error."""
        try:
            metrics = self._connection_metrics[db_name]
            metrics.failed_connections += 1
            metrics.last_activity = time.time()

            # Notify error callbacks
            for callback in self._error_callbacks:
                try:
                    callback(DatabaseError(f"Connection error for database '{db_name}'"))
                except Exception as e:
                    logger.warning(f"Error in error callback: {e}")

        except Exception as e:
            logger.debug(f"Error tracking connection error: {e}")

    def _track_query_completed(self, db_name: str, query_time: float) -> None:
        """Track completed query."""
        try:
            metrics = self._connection_metrics[db_name]
            metrics.query_count += 1

            # Update average query time
            if metrics.avg_query_time == 0:
                metrics.avg_query_time = query_time
            else:
                metrics.avg_query_time = (metrics.avg_query_time * 0.9) + (query_time * 0.1)

            metrics.last_activity = time.time()
        except Exception as e:
            logger.debug(f"Error tracking query completion: {e}")

    def _track_query_error(self, db_name: str) -> None:
        """Track query error."""
        try:
            metrics = self._connection_metrics[db_name]
            metrics.query_errors += 1
            metrics.last_activity = time.time()
        except Exception as e:
            logger.debug(f"Error tracking query error: {e}")

    async def _close_all_sessions(self) -> None:
        """Close all active sessions."""
        for db_name, sessions in self._session_registry.items():
            for session in sessions:
                try:
                    await session.close()
                except Exception as e:
                    logger.warning(
                        f"Error closing session for database {db_name}",
                        extra={"error": str(e)}
                    )

        self._session_registry.clear()

    async def _shutdown_all_engines(self) -> None:
        """Shutdown all database engines."""
        for db_name, engine in self._engines.items():
            try:
                await engine.dispose()
                logger.debug(f"Engine disposed: {db_name}")
            except Exception as e:
                logger.error(
                    f"Error disposing engine: {db_name}",
                    extra={"error": str(e)},
                    exc_info=True
                )

    async def _cleanup_partial_initialization(self) -> None:
        """Cleanup partial initialization on failure."""
        try:
            await self._close_all_sessions()
            await self._shutdown_all_engines()
            self._engines.clear()
            self._session_makers.clear()
            self._pool_configs.clear()
        except Exception as e:
            logger.error(
                "Error during cleanup",
                extra={"error": str(e)},
                exc_info=True
            )

    async def __aenter__(self) -> DatabaseConnectionManager:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()