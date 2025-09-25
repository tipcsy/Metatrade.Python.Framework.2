"""
MT5 Connection Manager.

This module provides enterprise-grade connection management for MetaTrader 5,
including multi-account support, connection pooling, and advanced monitoring.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from src.core.config.settings import Mt5Settings, Mt5AccountSettings
from src.core.exceptions import (
    Mt5ConnectionError,
    Mt5SessionError,
    ConfigurationError,
)
from src.core.logging import get_logger
from .session import Mt5Session
from .pool import Mt5ConnectionPool
from .circuit_breaker import Mt5CircuitBreaker

logger = get_logger(__name__)


class Mt5ConnectionManager:
    """
    Enterprise MT5 connection manager with multi-account support,
    connection pooling, circuit breaker, and advanced monitoring.

    Features:
    - Multi-account connection management
    - Connection pooling for performance
    - Circuit breaker for fault tolerance
    - Health monitoring and metrics
    - Automatic failover and recovery
    - Load balancing across connections
    """

    def __init__(self, settings: Mt5Settings) -> None:
        """Initialize MT5 connection manager.

        Args:
            settings: MT5 configuration settings
        """
        self.settings = settings
        self.is_initialized = False
        self.is_shutting_down = False

        # Connection management
        self._sessions: Dict[str, Mt5Session] = {}
        self._pools: Dict[str, Mt5ConnectionPool] = {}
        self._circuit_breakers: Dict[str, Mt5CircuitBreaker] = {}
        self._active_accounts: Set[str] = set()

        # Thread pool for blocking operations
        self._thread_pool = ThreadPoolExecutor(
            max_workers=settings.performance.tick_processing_threads,
            thread_name_prefix="MT5-Manager"
        )

        # Monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._stats: Dict[str, Any] = defaultdict(int)
        self._last_health_check = 0.0

        # Locks for thread safety
        self._manager_lock = asyncio.Lock()
        self._session_locks: Dict[str, asyncio.Lock] = {}

        logger.info(
            "MT5 connection manager initialized",
            extra={
                "accounts_configured": len(settings.accounts),
                "pool_size": settings.performance.pool_size,
                "circuit_breaker_enabled": settings.circuit_breaker_enabled,
            }
        )

    async def initialize(self) -> None:
        """Initialize the connection manager and establish connections."""
        async with self._manager_lock:
            if self.is_initialized:
                logger.warning("MT5 connection manager already initialized")
                return

            try:
                logger.info("Initializing MT5 connection manager")

                # Validate configuration
                await self._validate_configuration()

                # Initialize circuit breakers
                await self._initialize_circuit_breakers()

                # Initialize connection pools
                await self._initialize_connection_pools()

                # Connect to accounts
                if self.settings.auto_connect:
                    await self._connect_all_accounts()

                # Start monitoring tasks
                await self._start_monitoring()

                self.is_initialized = True
                logger.info(
                    "MT5 connection manager initialized successfully",
                    extra={
                        "active_accounts": len(self._active_accounts),
                        "total_sessions": len(self._sessions),
                    }
                )

            except Exception as e:
                logger.error(
                    "Failed to initialize MT5 connection manager",
                    extra={"error": str(e)},
                    exc_info=True
                )
                await self._cleanup_partial_initialization()
                raise Mt5ConnectionError(f"Connection manager initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """Shutdown the connection manager and cleanup resources."""
        async with self._manager_lock:
            if not self.is_initialized or self.is_shutting_down:
                return

            self.is_shutting_down = True
            logger.info("Shutting down MT5 connection manager")

            try:
                # Stop monitoring tasks
                await self._stop_monitoring()

                # Disconnect all accounts
                await self._disconnect_all_accounts()

                # Shutdown connection pools
                await self._shutdown_connection_pools()

                # Shutdown thread pool
                self._thread_pool.shutdown(wait=True)

                # Reset state
                self._sessions.clear()
                self._pools.clear()
                self._circuit_breakers.clear()
                self._active_accounts.clear()

                self.is_initialized = False
                logger.info("MT5 connection manager shutdown completed")

            except Exception as e:
                logger.error(
                    "Error during MT5 connection manager shutdown",
                    extra={"error": str(e)},
                    exc_info=True
                )

    async def get_session(self, account_name: Optional[str] = None) -> Mt5Session:
        """Get an active MT5 session.

        Args:
            account_name: Account name, uses default if None

        Returns:
            Active MT5 session

        Raises:
            Mt5SessionError: If no session available
            Mt5ConnectionError: If connection failed
        """
        if not self.is_initialized:
            raise Mt5SessionError("Connection manager not initialized")

        # Determine account name
        if account_name is None:
            account_name = self.settings.default_account
            if account_name is None and self.settings.accounts:
                account_name = self.settings.accounts[0].name

        if account_name is None:
            raise Mt5SessionError("No account specified and no default account configured")

        # Check if account is active
        if account_name not in self._active_accounts:
            raise Mt5SessionError(f"Account '{account_name}' is not active")

        # Get session from pool
        pool = self._pools.get(account_name)
        if pool is None:
            raise Mt5SessionError(f"No connection pool for account '{account_name}'")

        try:
            session = await pool.get_connection()
            logger.debug(
                "Retrieved MT5 session",
                extra={
                    "account": account_name,
                    "session_id": session.session_id,
                }
            )
            return session

        except Exception as e:
            logger.error(
                "Failed to get MT5 session",
                extra={
                    "account": account_name,
                    "error": str(e),
                },
                exc_info=True
            )
            raise Mt5SessionError(f"Failed to get session for account '{account_name}': {e}") from e

    def get_session_sync(self, account_name: Optional[str] = None) -> Optional['Mt5Session']:
        """Get an active MT5 session synchronously (wrapper for GUI components).

        Args:
            account_name: Account name, uses default if None

        Returns:
            Active MT5 session or None if not available
        """
        try:
            # Check if we're in an async context
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, can't run sync
                    logger.warning("Cannot get session synchronously while event loop is running")
                    return None
                else:
                    # No running loop, create one
                    return loop.run_until_complete(self.get_session(account_name))
            except RuntimeError:
                # No event loop, create new one
                return asyncio.run(self.get_session(account_name))

        except Exception as e:
            logger.debug(f"Failed to get MT5 session synchronously: {e}")
            return None

    async def return_session(self, session: Mt5Session, account_name: Optional[str] = None) -> None:
        """Return a session to the pool.

        Args:
            session: Session to return
            account_name: Account name, auto-detected if None
        """
        if account_name is None:
            account_name = session.account_config.name

        pool = self._pools.get(account_name)
        if pool is not None:
            await pool.return_connection(session)
            logger.debug(
                "Returned MT5 session to pool",
                extra={
                    "account": account_name,
                    "session_id": session.session_id,
                }
            )

    async def connect_account(self, account_name: str) -> bool:
        """Connect to a specific account.

        Args:
            account_name: Account name to connect

        Returns:
            True if connection successful

        Raises:
            Mt5ConnectionError: If connection failed
        """
        if account_name in self._active_accounts:
            logger.debug(f"Account '{account_name}' already connected")
            return True

        # Find account configuration
        account_config = None
        for config in self.settings.accounts:
            if config.name == account_name:
                account_config = config
                break

        if account_config is None:
            raise Mt5ConnectionError(f"Account '{account_name}' not found in configuration")

        if not account_config.enabled:
            raise Mt5ConnectionError(f"Account '{account_name}' is disabled")

        try:
            logger.info(f"Connecting to MT5 account: {account_name}")

            # Initialize connection pool
            pool = Mt5ConnectionPool(
                account_config=account_config,
                pool_size=self.settings.performance.pool_size,
                max_overflow=self.settings.performance.max_overflow
            )
            await pool.initialize()

            self._pools[account_name] = pool
            self._active_accounts.add(account_name)

            logger.info(
                "Successfully connected to MT5 account",
                extra={
                    "account": account_name,
                    "login": account_config.login,
                    "server": account_config.server,
                }
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to connect to MT5 account",
                extra={
                    "account": account_name,
                    "error": str(e),
                },
                exc_info=True
            )

            # Cleanup on failure
            if account_name in self._pools:
                await self._pools[account_name].shutdown()
                del self._pools[account_name]
            self._active_accounts.discard(account_name)

            raise Mt5ConnectionError(f"Failed to connect to account '{account_name}': {e}") from e

    async def disconnect_account(self, account_name: str) -> None:
        """Disconnect from a specific account.

        Args:
            account_name: Account name to disconnect
        """
        if account_name not in self._active_accounts:
            logger.debug(f"Account '{account_name}' already disconnected")
            return

        try:
            logger.info(f"Disconnecting from MT5 account: {account_name}")

            # Shutdown connection pool
            pool = self._pools.get(account_name)
            if pool is not None:
                await pool.shutdown()
                del self._pools[account_name]

            # Remove from active accounts
            self._active_accounts.discard(account_name)

            logger.info(f"Successfully disconnected from MT5 account: {account_name}")

        except Exception as e:
            logger.error(
                "Error disconnecting from MT5 account",
                extra={
                    "account": account_name,
                    "error": str(e),
                },
                exc_info=True
            )

    async def get_account_status(self, account_name: str) -> Dict[str, Any]:
        """Get status information for an account.

        Args:
            account_name: Account name

        Returns:
            Dictionary containing account status
        """
        is_active = account_name in self._active_accounts
        pool = self._pools.get(account_name)
        circuit_breaker = self._circuit_breakers.get(account_name)

        status = {
            "account_name": account_name,
            "is_active": is_active,
            "is_healthy": False,
            "pool_stats": None,
            "circuit_breaker_state": None,
        }

        if pool is not None:
            status["pool_stats"] = pool.get_stats()
            status["is_healthy"] = pool.is_healthy()

        if circuit_breaker is not None:
            status["circuit_breaker_state"] = circuit_breaker.get_state()

        return status

    async def get_manager_status(self) -> Dict[str, Any]:
        """Get overall manager status.

        Returns:
            Dictionary containing manager status
        """
        return {
            "is_initialized": self.is_initialized,
            "is_shutting_down": self.is_shutting_down,
            "total_accounts": len(self.settings.accounts),
            "active_accounts": len(self._active_accounts),
            "active_account_names": list(self._active_accounts),
            "total_pools": len(self._pools),
            "health_check_enabled": self._health_check_task is not None,
            "metrics_collection_enabled": self._metrics_task is not None,
            "last_health_check": self._last_health_check,
            "stats": dict(self._stats),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check.

        Returns:
            Dictionary containing health check results
        """
        start_time = time.time()
        results = {
            "timestamp": start_time,
            "overall_healthy": True,
            "accounts": {},
            "summary": {
                "total_accounts": len(self._active_accounts),
                "healthy_accounts": 0,
                "unhealthy_accounts": 0,
                "total_connections": 0,
                "healthy_connections": 0,
            }
        }

        # Check each active account
        for account_name in self._active_accounts:
            try:
                account_status = await self.get_account_status(account_name)
                results["accounts"][account_name] = account_status

                if account_status["is_healthy"]:
                    results["summary"]["healthy_accounts"] += 1
                else:
                    results["summary"]["unhealthy_accounts"] += 1
                    results["overall_healthy"] = False

                # Add pool statistics
                if account_status["pool_stats"]:
                    stats = account_status["pool_stats"]
                    results["summary"]["total_connections"] += stats["total_connections"]
                    results["summary"]["healthy_connections"] += stats["healthy_connections"]

            except Exception as e:
                logger.error(
                    "Error checking account health",
                    extra={
                        "account": account_name,
                        "error": str(e),
                    },
                    exc_info=True
                )
                results["accounts"][account_name] = {
                    "error": str(e),
                    "is_healthy": False,
                }
                results["summary"]["unhealthy_accounts"] += 1
                results["overall_healthy"] = False

        # Update last health check time
        self._last_health_check = time.time()
        results["duration"] = self._last_health_check - start_time

        logger.debug(
            "Health check completed",
            extra={
                "duration": results["duration"],
                "overall_healthy": results["overall_healthy"],
                "healthy_accounts": results["summary"]["healthy_accounts"],
                "total_accounts": results["summary"]["total_accounts"],
            }
        )

        return results

    async def _validate_configuration(self) -> None:
        """Validate MT5 configuration."""
        if not self.settings.accounts:
            raise ConfigurationError("No MT5 accounts configured")

        # Validate account configurations
        account_names = set()
        for account in self.settings.accounts:
            if account.name in account_names:
                raise ConfigurationError(f"Duplicate account name: {account.name}")
            account_names.add(account.name)

        # Validate default account
        if self.settings.default_account:
            if self.settings.default_account not in account_names:
                raise ConfigurationError(
                    f"Default account '{self.settings.default_account}' not found in configuration"
                )

        logger.debug("MT5 configuration validated successfully")

    async def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for accounts."""
        if not self.settings.circuit_breaker_enabled:
            logger.debug("Circuit breaker disabled, skipping initialization")
            return

        for account in self.settings.accounts:
            if not account.enabled:
                continue

            circuit_breaker = Mt5CircuitBreaker(
                failure_threshold=self.settings.circuit_breaker_failure_threshold,
                timeout=self.settings.circuit_breaker_timeout,
                account_name=account.name
            )
            self._circuit_breakers[account.name] = circuit_breaker

            logger.debug(
                "Circuit breaker initialized",
                extra={"account": account.name}
            )

    async def _initialize_connection_pools(self) -> None:
        """Initialize connection pools for accounts."""
        for account in self.settings.accounts:
            if not account.enabled:
                logger.debug(f"Skipping disabled account: {account.name}")
                continue

            self._session_locks[account.name] = asyncio.Lock()
            logger.debug(f"Connection pool prepared for account: {account.name}")

    async def _connect_all_accounts(self) -> None:
        """Connect to all enabled accounts."""
        connect_tasks = []

        for account in self.settings.accounts:
            if account.enabled:
                task = asyncio.create_task(self.connect_account(account.name))
                connect_tasks.append(task)

        if connect_tasks:
            results = await asyncio.gather(*connect_tasks, return_exceptions=True)

            # Log results
            for i, result in enumerate(results):
                account = self.settings.accounts[i]
                if isinstance(result, Exception):
                    logger.warning(
                        "Failed to connect to account during initialization",
                        extra={
                            "account": account.name,
                            "error": str(result),
                        }
                    )

    async def _disconnect_all_accounts(self) -> None:
        """Disconnect from all accounts."""
        disconnect_tasks = []

        for account_name in list(self._active_accounts):
            task = asyncio.create_task(self.disconnect_account(account_name))
            disconnect_tasks.append(task)

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

    async def _shutdown_connection_pools(self) -> None:
        """Shutdown all connection pools."""
        for pool in self._pools.values():
            try:
                await pool.shutdown()
            except Exception as e:
                logger.error(
                    "Error shutting down connection pool",
                    extra={"error": str(e)},
                    exc_info=True
                )

    async def _start_monitoring(self) -> None:
        """Start monitoring tasks."""
        if self.settings.monitoring.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.debug("Health check monitoring started")

        if self.settings.monitoring.performance_metrics_enabled:
            self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
            logger.debug("Metrics collection started")

    async def _stop_monitoring(self) -> None:
        """Stop monitoring tasks."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._metrics_task and not self._metrics_task.done():
            self._metrics_task.cancel()
            try:
                await self._metrics_task
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
                    "Error in health check loop",
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

                # Collect metrics from all pools
                for account_name, pool in self._pools.items():
                    try:
                        stats = pool.get_stats()
                        logger.debug(
                            "Pool metrics collected",
                            extra={
                                "account": account_name,
                                "stats": stats,
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to collect pool metrics",
                            extra={
                                "account": account_name,
                                "error": str(e),
                            }
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in metrics collection loop",
                    extra={"error": str(e)},
                    exc_info=True
                )

    async def _cleanup_partial_initialization(self) -> None:
        """Cleanup partial initialization on failure."""
        try:
            await self._disconnect_all_accounts()
            await self._shutdown_connection_pools()
            self._sessions.clear()
            self._pools.clear()
            self._circuit_breakers.clear()
            self._active_accounts.clear()
        except Exception as e:
            logger.error(
                "Error during cleanup",
                extra={"error": str(e)},
                exc_info=True
            )

    async def __aenter__(self) -> Mt5ConnectionManager:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()


# Global instance management
_mt5_session_manager_instance: Optional[Mt5ConnectionManager] = None
_mt5_session_manager_lock = threading.Lock()


def get_mt5_session_manager() -> Mt5ConnectionManager:
    """
    Get or create the global MT5 session manager instance.

    Returns:
        Mt5ConnectionManager: Global session manager instance
    """
    global _mt5_session_manager_instance

    if _mt5_session_manager_instance is None:
        with _mt5_session_manager_lock:
            if _mt5_session_manager_instance is None:
                from src.core.config import get_settings
                settings = get_settings()
                _mt5_session_manager_instance = Mt5ConnectionManager(settings.mt5)

    return _mt5_session_manager_instance