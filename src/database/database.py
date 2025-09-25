"""
Database connection management and session factory for the MetaTrader Python Framework.

This module provides enterprise-grade database connection management with:
- Connection pooling
- Session management
- Health monitoring
- Async support
- Transaction management
- Error handling and recovery
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Dict, Generator, Optional, Type, Union

from sqlalchemy import (
    create_engine,
    event,
    pool,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import get_settings
from src.core.exceptions import DatabaseError, ConnectionError
from src.core.logging import get_logger
from src.database.models.base import Base
# Import all models to register them with Base.metadata
from src.database.models import *  # noqa: F401,F403

logger = get_logger(__name__)
settings = get_settings()


class DatabaseManager:
    """
    Enterprise-grade database connection manager.

    Provides connection pooling, session management, health monitoring,
    and both sync and async database operations.
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the database manager.

        Args:
            database_url: Database connection URL. If None, uses settings.
        """
        self._database_url = database_url or settings.get_database_url()
        self._engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._async_session_factory: Optional[async_sessionmaker] = None
        self._is_initialized = False
        self._health_check_interval = 30  # seconds
        self._last_health_check = 0
        self._connection_pool_stats = {
            'connections_created': 0,
            'connections_closed': 0,
            'pool_hits': 0,
            'pool_misses': 0,
        }

        logger.info(f"Database manager initialized with URL: {self._mask_password(self._database_url)}")

    def _mask_password(self, url: str) -> str:
        """Mask password in database URL for logging."""
        if '://' not in url:
            return url

        scheme, rest = url.split('://', 1)
        if '@' not in rest:
            return url

        credentials, host_part = rest.split('@', 1)
        if ':' in credentials:
            username, _ = credentials.split(':', 1)
            return f"{scheme}://{username}:***@{host_part}"

        return url

    def _get_engine_config(self) -> Dict[str, Any]:
        """Get engine configuration based on database type and settings."""
        config = {
            'echo': settings.database.echo,
            'pool_pre_ping': True,  # Validate connections before use
            'pool_recycle': 3600,   # Recycle connections after 1 hour
        }

        # SQLite specific configuration
        if self._database_url.startswith('sqlite'):
            config.update({
                'poolclass': pool.StaticPool,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 30,
                }
            })
        # PostgreSQL specific configuration
        elif 'postgresql' in self._database_url:
            config.update({
                'poolclass': pool.QueuePool,
                'pool_size': settings.database.pool_size,
                'max_overflow': settings.database.max_overflow,
                'pool_timeout': 30,
                'connect_args': {
                    'application_name': 'MetaTrader_Framework',
                    'connect_timeout': 10,
                }
            })

        return config

    def initialize(self) -> None:
        """Initialize database engines and session factories."""
        if self._is_initialized:
            logger.warning("Database manager already initialized")
            return

        try:
            # Create synchronous engine
            engine_config = self._get_engine_config()
            self._engine = create_engine(self._database_url, **engine_config)

            # Create async engine (convert sync URL to async)
            async_url = self._database_url
            if async_url.startswith('sqlite'):
                async_url = async_url.replace('sqlite://', 'sqlite+aiosqlite://')
            elif async_url.startswith('postgresql'):
                async_url = async_url.replace('postgresql://', 'postgresql+asyncpg://')

            async_config = engine_config.copy()
            async_config.pop('poolclass', None)  # Remove poolclass for async engine

            self._async_engine = create_async_engine(async_url, **async_config)

            # Create session factories
            self._session_factory = sessionmaker(
                bind=self._engine,
                class_=Session,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )

            self._async_session_factory = async_sessionmaker(
                bind=self._async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )

            # Set up event listeners for connection monitoring
            self._setup_event_listeners()

            # Test connection
            self._test_connection()

            self._is_initialized = True
            logger.info("Database manager successfully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise DatabaseError(f"Database initialization failed: {e}") from e

    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners for monitoring."""
        if not self._engine:
            return

        @event.listens_for(self._engine, 'connect')
        def on_connect(dbapi_connection, connection_record):
            """Handle new database connections."""
            self._connection_pool_stats['connections_created'] += 1
            logger.debug("New database connection created")

        @event.listens_for(self._engine, 'close')
        def on_close(dbapi_connection, connection_record):
            """Handle database connection closures."""
            self._connection_pool_stats['connections_closed'] += 1
            logger.debug("Database connection closed")

        @event.listens_for(self._engine, 'checkout')
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection pool checkouts."""
            self._connection_pool_stats['pool_hits'] += 1

        @event.listens_for(self._engine, 'invalidate')
        def on_invalidate(dbapi_connection, connection_record, exception):
            """Handle connection invalidations."""
            logger.warning(f"Database connection invalidated: {exception}")

    def _test_connection(self) -> None:
        """Test database connection."""
        try:
            # Test connection directly using engine
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise ConnectionError(f"Database connection test failed: {e}") from e

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic transaction management.

        Yields:
            Session: SQLAlchemy session with transaction management

        Raises:
            DatabaseError: If session creation or transaction fails
        """
        if not self._is_initialized:
            raise DatabaseError("Database manager not initialized")

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database session error: {e}") from e
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session with automatic transaction management.

        Yields:
            AsyncSession: SQLAlchemy async session with transaction management

        Raises:
            DatabaseError: If session creation or transaction fails
        """
        if not self._is_initialized:
            raise DatabaseError("Database manager not initialized")

        async_session = self._async_session_factory()
        try:
            yield async_session
            await async_session.commit()
        except Exception as e:
            await async_session.rollback()
            logger.error(f"Async database session error: {e}")
            raise DatabaseError(f"Async database session error: {e}") from e
        finally:
            await async_session.close()

    def create_all_tables(self) -> None:
        """Create all database tables."""
        if not self._engine:
            raise DatabaseError("Database engine not initialized")

        try:
            Base.metadata.create_all(bind=self._engine)
            logger.info("All database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise DatabaseError(f"Failed to create database tables: {e}") from e

    async def create_all_tables_async(self) -> None:
        """Create all database tables asynchronously."""
        if not self._async_engine:
            raise DatabaseError("Async database engine not initialized")

        try:
            async with self._async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("All database tables created successfully (async)")
        except Exception as e:
            logger.error(f"Failed to create database tables (async): {e}")
            raise DatabaseError(f"Failed to create database tables (async): {e}") from e

    def drop_all_tables(self) -> None:
        """Drop all database tables."""
        if not self._engine:
            raise DatabaseError("Database engine not initialized")

        try:
            Base.metadata.drop_all(bind=self._engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise DatabaseError(f"Failed to drop database tables: {e}") from e

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive database health check.

        Returns:
            Dict with health check results
        """
        current_time = time.time()

        # Skip frequent health checks
        if current_time - self._last_health_check < self._health_check_interval:
            return {'status': 'skipped', 'reason': 'too_frequent'}

        health_status = {
            'timestamp': current_time,
            'status': 'healthy',
            'details': {},
            'errors': []
        }

        try:
            # Test basic connectivity
            with self.get_session() as session:
                start_time = time.time()
                result = session.execute(text("SELECT 1 as test"))
                result.fetchone()
                response_time = time.time() - start_time

                health_status['details'].update({
                    'connection': 'ok',
                    'response_time_ms': round(response_time * 1000, 2),
                })

            # Check connection pool status
            if self._engine and hasattr(self._engine.pool, 'size'):
                pool_status = {
                    'pool_size': self._engine.pool.size(),
                    'checked_in': self._engine.pool.checkedin(),
                    'checked_out': self._engine.pool.checkedout(),
                    'overflow': self._engine.pool.overflow(),
                    'invalid': self._engine.pool.invalid(),
                }
                health_status['details']['pool'] = pool_status

            # Add connection statistics
            health_status['details']['stats'] = self._connection_pool_stats.copy()

        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['errors'].append(str(e))
            logger.error(f"Database health check failed: {e}")

        self._last_health_check = current_time
        return health_status

    async def health_check_async(self) -> Dict[str, Any]:
        """
        Perform comprehensive async database health check.

        Returns:
            Dict with health check results
        """
        health_status = {
            'timestamp': time.time(),
            'status': 'healthy',
            'details': {},
            'errors': []
        }

        try:
            async with self.get_async_session() as session:
                start_time = time.time()
                result = await session.execute(text("SELECT 1 as test"))
                await result.fetchone()
                response_time = time.time() - start_time

                health_status['details'].update({
                    'async_connection': 'ok',
                    'response_time_ms': round(response_time * 1000, 2),
                })

        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['errors'].append(str(e))
            logger.error(f"Async database health check failed: {e}")

        return health_status

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection pool statistics."""
        stats = {
            'pool_stats': self._connection_pool_stats.copy(),
            'engine_info': {},
            'pool_info': {}
        }

        if self._engine:
            stats['engine_info'] = {
                'url': self._mask_password(str(self._engine.url)),
                'driver': self._engine.driver,
                'echo': self._engine.echo,
            }

            if hasattr(self._engine.pool, 'size'):
                stats['pool_info'] = {
                    'size': self._engine.pool.size(),
                    'checked_in': self._engine.pool.checkedin(),
                    'checked_out': self._engine.pool.checkedout(),
                    'overflow': self._engine.pool.overflow(),
                    'invalid': self._engine.pool.invalid(),
                }

        return stats

    def close(self) -> None:
        """Close database connections and clean up resources."""
        try:
            if self._engine:
                self._engine.dispose()
                logger.info("Synchronous database engine disposed")

            if self._async_engine:
                # For async engine, we need to run dispose in an event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._async_engine.dispose())
                    else:
                        loop.run_until_complete(self._async_engine.dispose())
                except RuntimeError:
                    # No event loop running, create one
                    asyncio.run(self._async_engine.dispose())
                logger.info("Asynchronous database engine disposed")

            self._is_initialized = False
            logger.info("Database manager closed successfully")

        except Exception as e:
            logger.error(f"Error closing database manager: {e}")

    def shutdown(self) -> None:
        """Shutdown database manager (alias for close)."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @property
    def is_initialized(self) -> bool:
        """Check if database manager is initialized."""
        return self._is_initialized

    @property
    def engine(self) -> Optional[Engine]:
        """Get the synchronous database engine."""
        return self._engine

    @property
    def async_engine(self) -> Optional[AsyncEngine]:
        """Get the asynchronous database engine."""
        return self._async_engine


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance.

    Returns:
        DatabaseManager: The global database manager

    Raises:
        DatabaseError: If database manager is not initialized
    """
    global _db_manager

    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.initialize()

    return _db_manager


def initialize_database(database_url: Optional[str] = None) -> DatabaseManager:
    """
    Initialize the global database manager.

    Args:
        database_url: Optional database URL override

    Returns:
        DatabaseManager: The initialized database manager
    """
    global _db_manager

    if _db_manager is not None:
        _db_manager.close()

    _db_manager = DatabaseManager(database_url)
    _db_manager.initialize()

    return _db_manager


def close_database() -> None:
    """Close the global database manager."""
    global _db_manager

    if _db_manager is not None:
        _db_manager.close()
        _db_manager = None


# Convenience functions for session management
def get_session() -> Generator[Session, None, None]:
    """Get a database session from the global manager."""
    return get_database_manager().get_session()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session from the global manager."""
    async with get_database_manager().get_async_session() as session:
        yield session