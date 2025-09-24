"""
MT5 Connection Pool Implementation.

This module provides a connection pool for managing multiple MT5 connections
with load balancing, health monitoring, and automatic failover.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Set

from src.core.config.settings import Mt5AccountSettings, Mt5PerformanceSettings
from src.core.exceptions import (
    Mt5ConnectionError,
    Mt5PerformanceError,
    Mt5SessionError,
)
from src.core.logging import get_logger

from .circuit_breaker import Mt5CircuitBreaker
from .session import Mt5Session

logger = get_logger(__name__)


class Mt5ConnectionPool:
    """
    Enterprise-grade connection pool for MT5 sessions.

    Provides connection pooling, load balancing, health monitoring,
    and automatic failover for MT5 connections.
    """

    def __init__(
        self,
        account_configs: List[Mt5AccountSettings],
        performance_config: Mt5PerformanceSettings,
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize connection pool.

        Args:
            account_configs: List of account configurations
            performance_config: Performance configuration
            circuit_breaker_config: Circuit breaker configuration
        """
        self.account_configs = account_configs
        self.performance_config = performance_config
        self.circuit_breaker_config = circuit_breaker_config or {}

        # Connection pool
        self._pool: Dict[str, Mt5Session] = {}
        self._available_connections: Set[str] = set()
        self._busy_connections: Set[str] = set()
        self._failed_connections: Set[str] = set()

        # Circuit breakers for each account
        self._circuit_breakers: Dict[str, Mt5CircuitBreaker] = {}

        # Pool statistics
        self._pool_created_time = time.time()
        self._total_connections_created = 0
        self._total_connections_failed = 0
        self._total_requests = 0
        self._total_request_time = 0.0

        # Health monitoring
        self._health_check_interval = 30.0  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Synchronization
        self._pool_lock = asyncio.Lock()
        self._request_semaphore = asyncio.Semaphore(
            self.performance_config.pool_size + self.performance_config.max_overflow
        )

        logger.info(
            "MT5 connection pool initialized",
            extra={
                "accounts": len(self.account_configs),
                "pool_size": self.performance_config.pool_size,
                "max_overflow": self.performance_config.max_overflow,
            }
        )

    async def start(self) -> None:
        """Start the connection pool."""
        async with self._pool_lock:
            if self._is_running:
                logger.warning("Connection pool is already running")
                return

            logger.info("Starting MT5 connection pool")

            try:
                # Initialize circuit breakers
                await self._initialize_circuit_breakers()

                # Create initial connections
                await self._create_initial_connections()

                # Start health monitoring
                self._health_check_task = asyncio.create_task(self._health_check_loop())

                self._is_running = True

                logger.info(
                    "MT5 connection pool started successfully",
                    extra={
                        "total_connections": len(self._pool),
                        "available_connections": len(self._available_connections),
                    }
                )

            except Exception as e:
                logger.error(
                    "Failed to start MT5 connection pool",
                    extra={"error": str(e)},
                    exc_info=True
                )
                await self._cleanup()
                raise

    async def stop(self) -> None:
        """Stop the connection pool."""
        async with self._pool_lock:
            if not self._is_running:
                return

            logger.info("Stopping MT5 connection pool")

            self._is_running = False

            # Stop health monitoring
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Cleanup connections
            await self._cleanup()

            logger.info("MT5 connection pool stopped")

    async def get_connection(self, account_name: Optional[str] = None) -> Mt5Session:
        """Get a connection from the pool.

        Args:
            account_name: Specific account name to use

        Returns:
            MT5 session

        Raises:
            Mt5ConnectionError: If no connections available
            Mt5PerformanceError: If pool is overloaded
        """
        if not self._is_running:
            raise Mt5ConnectionError("Connection pool is not running")

        async with self._request_semaphore:
            start_time = time.time()
            self._total_requests += 1

            try:
                async with self._pool_lock:
                    # Find available connection
                    session = await self._find_available_connection(account_name)

                    if session is None:
                        # Try to create overflow connection
                        session = await self._create_overflow_connection(account_name)

                    if session is None:
                        raise Mt5ConnectionError(
                            "No available connections in pool",
                            pool_size=self.performance_config.pool_size,
                            busy_connections=len(self._busy_connections),
                            failed_connections=len(self._failed_connections)
                        )

                    # Mark connection as busy
                    self._available_connections.discard(session.session_id)
                    self._busy_connections.add(session.session_id)

                    # Record request time
                    request_time = time.time() - start_time
                    self._total_request_time += request_time

                    # Check performance thresholds
                    if request_time * 1000 > self.performance_config.max_response_latency:
                        logger.warning(
                            "MT5 connection request exceeded latency threshold",
                            extra={
                                "request_time_ms": request_time * 1000,
                                "threshold_ms": self.performance_config.max_response_latency,
                                "session_id": session.session_id,
                            }
                        )

                    logger.debug(
                        "Connection acquired from pool",
                        extra={
                            "session_id": session.session_id,
                            "account": session.account_config.name,
                            "request_time_ms": request_time * 1000,
                        }
                    )

                    return session

            except Exception as e:
                logger.error(
                    "Failed to get connection from pool",
                    extra={"error": str(e)},
                    exc_info=True
                )
                raise

    async def return_connection(
        self,
        session: Mt5Session,
        error: Optional[Exception] = None
    ) -> None:
        """Return a connection to the pool.

        Args:
            session: Session to return
            error: Optional error that occurred
        """
        async with self._pool_lock:
            session_id = session.session_id

            # Remove from busy connections
            self._busy_connections.discard(session_id)

            if error:
                logger.warning(
                    "Connection returned with error",
                    extra={
                        "session_id": session_id,
                        "error": str(error),
                        "error_type": type(error).__name__,
                    }
                )

                # Mark as failed and trigger circuit breaker
                self._failed_connections.add(session_id)
                self._available_connections.discard(session_id)

                # Trigger circuit breaker
                circuit_breaker = self._circuit_breakers.get(session.account_config.name)
                if circuit_breaker:
                    try:
                        await circuit_breaker._record_failure(error)
                    except Exception as cb_error:
                        logger.error(
                            "Circuit breaker error recording failure",
                            extra={"error": str(cb_error)},
                            exc_info=True
                        )

                # Attempt to recreate connection
                await self._recreate_failed_connection(session)

            else:
                # Check if connection is still healthy
                if session.is_healthy():
                    self._available_connections.add(session_id)
                    self._failed_connections.discard(session_id)

                    logger.debug(
                        "Healthy connection returned to pool",
                        extra={"session_id": session_id}
                    )
                else:
                    logger.warning(
                        "Unhealthy connection discarded",
                        extra={"session_id": session_id}
                    )

                    self._failed_connections.add(session_id)
                    await self._recreate_failed_connection(session)

    async def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for each account."""
        for account_config in self.account_configs:
            circuit_breaker = Mt5CircuitBreaker(
                name=f"mt5_{account_config.name}",
                failure_threshold=self.circuit_breaker_config.get("failure_threshold", 5),
                timeout_seconds=self.circuit_breaker_config.get("timeout_seconds", 30.0),
                expected_exception=Mt5SessionError,
            )

            self._circuit_breakers[account_config.name] = circuit_breaker

            logger.debug(
                "Circuit breaker initialized",
                extra={
                    "account": account_config.name,
                    "circuit_breaker": str(circuit_breaker),
                }
            )

    async def _create_initial_connections(self) -> None:
        """Create initial connections for the pool."""
        tasks = []

        for account_config in self.account_configs:
            # Create connections up to pool size
            connections_per_account = max(1, self.performance_config.pool_size // len(self.account_configs))

            for _ in range(connections_per_account):
                task = asyncio.create_task(self._create_connection(account_config))
                tasks.append(task)

        # Wait for all connections to be created
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_connections = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error(
                    "Failed to create initial connection",
                    extra={"error": str(result)},
                    exc_info=True
                )
                self._total_connections_failed += 1
            else:
                successful_connections += 1

        if successful_connections == 0:
            raise Mt5ConnectionError("Failed to create any initial connections")

        logger.info(
            "Initial connections created",
            extra={
                "successful": successful_connections,
                "failed": len(results) - successful_connections,
                "total": len(results),
            }
        )

    async def _create_connection(self, account_config: Mt5AccountSettings) -> Optional[Mt5Session]:
        """Create a new connection.

        Args:
            account_config: Account configuration

        Returns:
            Created session or None if failed
        """
        try:
            session = Mt5Session(account_config)

            await session.connect()
            await session.authenticate()

            self._pool[session.session_id] = session
            self._available_connections.add(session.session_id)
            self._total_connections_created += 1

            logger.info(
                "New MT5 connection created",
                extra={
                    "session_id": session.session_id,
                    "account": account_config.name,
                }
            )

            return session

        except Exception as e:
            logger.error(
                "Failed to create MT5 connection",
                extra={
                    "account": account_config.name,
                    "error": str(e),
                },
                exc_info=True
            )
            self._total_connections_failed += 1
            return None

    async def _find_available_connection(
        self,
        account_name: Optional[str] = None
    ) -> Optional[Mt5Session]:
        """Find an available connection.

        Args:
            account_name: Specific account name to use

        Returns:
            Available session or None
        """
        # Filter available connections by account if specified
        if account_name:
            available_sessions = [
                self._pool[session_id]
                for session_id in self._available_connections
                if self._pool[session_id].account_config.name == account_name
            ]
        else:
            available_sessions = [
                self._pool[session_id]
                for session_id in self._available_connections
            ]

        # Filter by health and circuit breaker status
        healthy_sessions = []
        for session in available_sessions:
            if session.is_healthy():
                circuit_breaker = self._circuit_breakers.get(session.account_config.name)
                if circuit_breaker and not circuit_breaker.is_open:
                    healthy_sessions.append(session)

        # Return the session with the lowest operation count (load balancing)
        if healthy_sessions:
            return min(healthy_sessions, key=lambda s: s.total_operations)

        return None

    async def _create_overflow_connection(
        self,
        account_name: Optional[str] = None
    ) -> Optional[Mt5Session]:
        """Create an overflow connection if within limits.

        Args:
            account_name: Specific account name to use

        Returns:
            Created session or None
        """
        total_connections = len(self._pool)
        max_connections = self.performance_config.pool_size + self.performance_config.max_overflow

        if total_connections >= max_connections:
            return None

        # Find account config
        if account_name:
            account_config = next(
                (config for config in self.account_configs if config.name == account_name),
                None
            )
        else:
            # Use round-robin selection
            account_config = self.account_configs[total_connections % len(self.account_configs)]

        if account_config is None:
            return None

        return await self._create_connection(account_config)

    async def _recreate_failed_connection(self, failed_session: Mt5Session) -> None:
        """Recreate a failed connection.

        Args:
            failed_session: Failed session to recreate
        """
        try:
            # Disconnect failed session
            await failed_session.disconnect()

            # Remove from pool
            session_id = failed_session.session_id
            self._pool.pop(session_id, None)
            self._available_connections.discard(session_id)
            self._busy_connections.discard(session_id)
            self._failed_connections.discard(session_id)

            # Create new connection
            new_session = await self._create_connection(failed_session.account_config)

            if new_session:
                logger.info(
                    "Failed connection recreated",
                    extra={
                        "old_session_id": session_id,
                        "new_session_id": new_session.session_id,
                        "account": failed_session.account_config.name,
                    }
                )
            else:
                logger.error(
                    "Failed to recreate connection",
                    extra={
                        "session_id": session_id,
                        "account": failed_session.account_config.name,
                    }
                )

        except Exception as e:
            logger.error(
                "Error recreating failed connection",
                extra={
                    "session_id": failed_session.session_id,
                    "error": str(e),
                },
                exc_info=True
            )

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._is_running:
            try:
                await asyncio.sleep(self._health_check_interval)

                if not self._is_running:
                    break

                await self._perform_health_checks()

            except asyncio.CancelledError:
                logger.debug("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(
                    "Error in health check loop",
                    extra={"error": str(e)},
                    exc_info=True
                )

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all connections."""
        async with self._pool_lock:
            unhealthy_sessions = []

            for session_id, session in self._pool.items():
                if not session.is_healthy():
                    unhealthy_sessions.append(session)

            # Handle unhealthy sessions
            for session in unhealthy_sessions:
                logger.warning(
                    "Unhealthy session detected",
                    extra={
                        "session_id": session.session_id,
                        "account": session.account_config.name,
                    }
                )

                # Move to failed connections
                self._available_connections.discard(session.session_id)
                self._failed_connections.add(session.session_id)

                # Recreate connection
                await self._recreate_failed_connection(session)

    async def _cleanup(self) -> None:
        """Cleanup all connections."""
        disconnect_tasks = []

        for session in self._pool.values():
            disconnect_tasks.append(session.disconnect())

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        # Clear all data structures
        self._pool.clear()
        self._available_connections.clear()
        self._busy_connections.clear()
        self._failed_connections.clear()

    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get pool statistics.

        Returns:
            Dictionary containing pool statistics
        """
        current_time = time.time()
        uptime = current_time - self._pool_created_time

        avg_request_time = (
            self._total_request_time / self._total_requests
            if self._total_requests > 0 else 0.0
        )

        return {
            "pool_size": self.performance_config.pool_size,
            "max_overflow": self.performance_config.max_overflow,
            "total_connections": len(self._pool),
            "available_connections": len(self._available_connections),
            "busy_connections": len(self._busy_connections),
            "failed_connections": len(self._failed_connections),
            "total_connections_created": self._total_connections_created,
            "total_connections_failed": self._total_connections_failed,
            "total_requests": self._total_requests,
            "avg_request_time_ms": avg_request_time * 1000,
            "requests_per_second": self._total_requests / uptime if uptime > 0 else 0.0,
            "uptime_seconds": uptime,
            "is_running": self._is_running,
            "circuit_breaker_stats": {
                name: cb.get_statistics()
                for name, cb in self._circuit_breakers.items()
            },
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get pool health status.

        Returns:
            Dictionary containing health status
        """
        stats = self.get_pool_statistics()

        # Determine overall health
        if not self._is_running:
            health = "stopped"
        elif len(self._available_connections) == 0:
            health = "critical"
        elif len(self._failed_connections) > len(self._available_connections):
            health = "degraded"
        else:
            health = "healthy"

        return {
            "health": health,
            "is_running": self._is_running,
            "connection_availability": (
                len(self._available_connections) / len(self._pool)
                if len(self._pool) > 0 else 0.0
            ),
            "failure_rate": (
                len(self._failed_connections) / len(self._pool)
                if len(self._pool) > 0 else 0.0
            ),
            "circuit_breaker_health": {
                name: cb.get_health_status()
                for name, cb in self._circuit_breakers.items()
            },
            "recommendations": self._get_health_recommendations(health),
        }

    def _get_health_recommendations(self, health: str) -> List[str]:
        """Get health recommendations.

        Args:
            health: Current health status

        Returns:
            List of recommendations
        """
        recommendations = []

        if health == "critical":
            recommendations.append("No available connections - check MT5 terminals")
            recommendations.append("Verify network connectivity")
        elif health == "degraded":
            recommendations.append("High failure rate detected")
            recommendations.append("Check MT5 terminal stability")
        elif health == "stopped":
            recommendations.append("Connection pool is not running")

        return recommendations

    async def __aenter__(self) -> Mt5ConnectionPool:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()