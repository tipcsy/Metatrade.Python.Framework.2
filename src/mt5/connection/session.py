"""
MT5 Session Management.

This module provides secure session management for MetaTrader 5 connections
with automatic session lifecycle handling and connection persistence.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from src.core.config.settings import Mt5AccountSettings
from src.core.exceptions import (
    Mt5AuthenticationError,
    Mt5ConnectionError,
    Mt5SessionError,
    Mt5TimeoutError,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


class Mt5Session:
    """
    Manages MT5 session lifecycle with automatic connection management.

    Provides secure authentication, session persistence, and automatic
    reconnection with proper error handling and monitoring.
    """

    def __init__(
        self,
        account_config: Mt5AccountSettings,
        session_timeout: int = 3600000,  # 1 hour in milliseconds
        heartbeat_interval: int = 30000,  # 30 seconds
    ) -> None:
        """Initialize MT5 session.

        Args:
            account_config: Account configuration
            session_timeout: Session timeout in milliseconds
            heartbeat_interval: Heartbeat interval in milliseconds
        """
        self.account_config = account_config
        self.session_timeout = session_timeout
        self.heartbeat_interval = heartbeat_interval

        # Session state
        self.session_id = str(uuid.uuid4())
        self.is_connected = False
        self.is_authenticated = False
        self.connection_time: Optional[float] = None
        self.last_heartbeat: Optional[float] = None
        self.authentication_time: Optional[float] = None

        # Connection stats
        self.connection_attempts = 0
        self.authentication_attempts = 0
        self.last_error: Optional[Exception] = None
        self.total_operations = 0
        self.failed_operations = 0

        # Heartbeat task
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._session_lock = asyncio.Lock()

        logger.info(
            "MT5 session initialized",
            extra={
                "session_id": self.session_id,
                "account": self.account_config.name,
                "login": self.account_config.login,
                "server": self.account_config.server,
            }
        )

    async def connect(self) -> bool:
        """Connect to MT5 terminal.

        Returns:
            True if connection successful, False otherwise

        Raises:
            Mt5ConnectionError: If connection fails
            Mt5TimeoutError: If connection times out
        """
        async with self._session_lock:
            if self.is_connected:
                logger.debug(
                    "Already connected to MT5",
                    extra={"session_id": self.session_id}
                )
                return True

            try:
                logger.info(
                    "Connecting to MT5",
                    extra={
                        "session_id": self.session_id,
                        "account": self.account_config.name,
                        "attempt": self.connection_attempts + 1,
                    }
                )

                self.connection_attempts += 1
                start_time = time.time()

                # Import MetaTrader5 here to avoid import errors in tests
                try:
                    import MetaTrader5 as mt5
                except ImportError:
                    raise Mt5ConnectionError(
                        "MetaTrader5 package not installed. Install with: pip install MetaTrader5",
                        account=self.account_config.name
                    )

                # Initialize MT5 terminal
                if not mt5.initialize(
                    path=str(self.account_config.path) if hasattr(self.account_config, 'path') and self.account_config.path else None,
                    login=self.account_config.login,
                    password=self.account_config.password,
                    server=self.account_config.server,
                    timeout=self.account_config.timeout
                ):
                    error = mt5.last_error()
                    raise Mt5ConnectionError(
                        f"Failed to initialize MT5 terminal: {error}",
                        error_code=error[0] if error else None,
                        mt5_error=error[1] if error else None,
                        account=self.account_config.name,
                        server=self.account_config.server,
                        login=self.account_config.login
                    )

                # Set connection state
                self.is_connected = True
                self.connection_time = time.time()
                connection_duration = self.connection_time - start_time

                logger.info(
                    "Successfully connected to MT5",
                    extra={
                        "session_id": self.session_id,
                        "account": self.account_config.name,
                        "connection_time": connection_duration,
                        "attempts": self.connection_attempts,
                    }
                )

                # Start heartbeat task
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                return True

            except Exception as e:
                self.last_error = e
                logger.error(
                    "Failed to connect to MT5",
                    extra={
                        "session_id": self.session_id,
                        "account": self.account_config.name,
                        "error": str(e),
                        "attempts": self.connection_attempts,
                    },
                    exc_info=True
                )

                if isinstance(e, (Mt5ConnectionError, Mt5TimeoutError)):
                    raise
                else:
                    raise Mt5ConnectionError(
                        f"Unexpected error during MT5 connection: {e}",
                        account=self.account_config.name
                    ) from e

    async def authenticate(self) -> bool:
        """Authenticate with MT5 account.

        Returns:
            True if authentication successful, False otherwise

        Raises:
            Mt5AuthenticationError: If authentication fails
            Mt5ConnectionError: If not connected
        """
        async with self._session_lock:
            if not self.is_connected:
                raise Mt5ConnectionError(
                    "Must be connected before authentication",
                    account=self.account_config.name
                )

            if self.is_authenticated:
                logger.debug(
                    "Already authenticated with MT5",
                    extra={"session_id": self.session_id}
                )
                return True

            try:
                logger.info(
                    "Authenticating with MT5",
                    extra={
                        "session_id": self.session_id,
                        "account": self.account_config.name,
                        "attempt": self.authentication_attempts + 1,
                    }
                )

                self.authentication_attempts += 1

                # Import MetaTrader5 here
                import MetaTrader5 as mt5

                # Login to account
                if not mt5.login(
                    login=self.account_config.login,
                    password=self.account_config.password,
                    server=self.account_config.server,
                    timeout=self.account_config.timeout
                ):
                    error = mt5.last_error()
                    raise Mt5AuthenticationError(
                        f"Failed to authenticate with MT5: {error}",
                        error_code=error[0] if error else None,
                        mt5_error=error[1] if error else None,
                        login=self.account_config.login,
                        server=self.account_config.server
                    )

                # Verify account info
                account_info = mt5.account_info()
                if account_info is None:
                    raise Mt5AuthenticationError(
                        "Failed to retrieve account information after login",
                        login=self.account_config.login
                    )

                # Set authentication state
                self.is_authenticated = True
                self.authentication_time = time.time()

                logger.info(
                    "Successfully authenticated with MT5",
                    extra={
                        "session_id": self.session_id,
                        "account": self.account_config.name,
                        "account_number": account_info.login,
                        "account_name": account_info.name,
                        "server": account_info.server,
                        "balance": account_info.balance,
                        "attempts": self.authentication_attempts,
                    }
                )

                return True

            except Exception as e:
                self.last_error = e
                logger.error(
                    "Failed to authenticate with MT5",
                    extra={
                        "session_id": self.session_id,
                        "account": self.account_config.name,
                        "error": str(e),
                        "attempts": self.authentication_attempts,
                    },
                    exc_info=True
                )

                if isinstance(e, Mt5AuthenticationError):
                    raise
                else:
                    raise Mt5AuthenticationError(
                        f"Unexpected error during MT5 authentication: {e}",
                        login=self.account_config.login
                    ) from e

    async def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        async with self._session_lock:
            if not self.is_connected:
                return

            try:
                logger.info(
                    "Disconnecting from MT5",
                    extra={
                        "session_id": self.session_id,
                        "account": self.account_config.name,
                    }
                )

                # Stop heartbeat task
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except asyncio.CancelledError:
                        pass

                # Shutdown MT5 terminal
                try:
                    import MetaTrader5 as mt5
                    mt5.shutdown()
                except ImportError:
                    pass

                # Reset connection state
                self.is_connected = False
                self.is_authenticated = False
                self.connection_time = None
                self.authentication_time = None
                self.last_heartbeat = None

                logger.info(
                    "Successfully disconnected from MT5",
                    extra={
                        "session_id": self.session_id,
                        "account": self.account_config.name,
                        "total_operations": self.total_operations,
                        "failed_operations": self.failed_operations,
                    }
                )

            except Exception as e:
                logger.error(
                    "Error during MT5 disconnection",
                    extra={
                        "session_id": self.session_id,
                        "account": self.account_config.name,
                        "error": str(e),
                    },
                    exc_info=True
                )

    async def _heartbeat_loop(self) -> None:
        """Internal heartbeat loop to maintain connection."""
        while self.is_connected:
            try:
                await asyncio.sleep(self.heartbeat_interval / 1000.0)

                if not self.is_connected:
                    break

                # Check MT5 connection
                try:
                    import MetaTrader5 as mt5

                    # Simple operation to test connection
                    terminal_info = mt5.terminal_info()
                    if terminal_info is None:
                        raise Mt5SessionError(
                            "Lost connection to MT5 terminal",
                            session_id=self.session_id
                        )

                    self.last_heartbeat = time.time()

                    logger.debug(
                        "MT5 heartbeat successful",
                        extra={
                            "session_id": self.session_id,
                            "account": self.account_config.name,
                        }
                    )

                except Exception as e:
                    logger.warning(
                        "MT5 heartbeat failed",
                        extra={
                            "session_id": self.session_id,
                            "account": self.account_config.name,
                            "error": str(e),
                        }
                    )

                    # Attempt reconnection
                    await self._handle_connection_loss()
                    break

            except asyncio.CancelledError:
                logger.debug(
                    "MT5 heartbeat task cancelled",
                    extra={"session_id": self.session_id}
                )
                break
            except Exception as e:
                logger.error(
                    "Unexpected error in MT5 heartbeat loop",
                    extra={
                        "session_id": self.session_id,
                        "error": str(e),
                    },
                    exc_info=True
                )

    async def _handle_connection_loss(self) -> None:
        """Handle connection loss and attempt reconnection."""
        logger.warning(
            "Handling MT5 connection loss",
            extra={
                "session_id": self.session_id,
                "account": self.account_config.name,
            }
        )

        # Reset connection state
        self.is_connected = False
        self.is_authenticated = False

        # Attempt reconnection with backoff
        for attempt in range(self.account_config.retry_attempts):
            try:
                await asyncio.sleep(self.account_config.retry_delay / 1000.0)

                if await self.connect() and await self.authenticate():
                    logger.info(
                        "Successfully reconnected to MT5",
                        extra={
                            "session_id": self.session_id,
                            "account": self.account_config.name,
                            "reconnect_attempt": attempt + 1,
                        }
                    )
                    return

            except Exception as e:
                logger.warning(
                    "MT5 reconnection attempt failed",
                    extra={
                        "session_id": self.session_id,
                        "account": self.account_config.name,
                        "attempt": attempt + 1,
                        "error": str(e),
                    }
                )

        logger.error(
            "Failed to reconnect to MT5 after all attempts",
            extra={
                "session_id": self.session_id,
                "account": self.account_config.name,
                "total_attempts": self.account_config.retry_attempts,
            }
        )

    def is_healthy(self) -> bool:
        """Check if session is healthy.

        Returns:
            True if session is healthy, False otherwise
        """
        if not self.is_connected or not self.is_authenticated:
            return False

        # Check session timeout
        if (
            self.connection_time and
            time.time() - self.connection_time > self.session_timeout / 1000.0
        ):
            return False

        # Check heartbeat
        if (
            self.last_heartbeat and
            time.time() - self.last_heartbeat > self.heartbeat_interval * 2 / 1000.0
        ):
            return False

        return True

    def get_session_info(self) -> Dict[str, Any]:
        """Get session information.

        Returns:
            Dictionary containing session information
        """
        return {
            "session_id": self.session_id,
            "account_name": self.account_config.name,
            "account_login": self.account_config.login,
            "server": self.account_config.server,
            "is_connected": self.is_connected,
            "is_authenticated": self.is_authenticated,
            "connection_time": self.connection_time,
            "authentication_time": self.authentication_time,
            "last_heartbeat": self.last_heartbeat,
            "connection_attempts": self.connection_attempts,
            "authentication_attempts": self.authentication_attempts,
            "total_operations": self.total_operations,
            "failed_operations": self.failed_operations,
            "success_rate": (
                (self.total_operations - self.failed_operations) / self.total_operations
                if self.total_operations > 0 else 0.0
            ),
            "is_healthy": self.is_healthy(),
            "last_error": str(self.last_error) if self.last_error else None,
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics.

        Returns:
            Dictionary containing performance metrics
        """
        current_time = time.time()

        uptime = (
            current_time - self.connection_time
            if self.connection_time else 0.0
        )

        return {
            "uptime_seconds": uptime,
            "total_operations": float(self.total_operations),
            "failed_operations": float(self.failed_operations),
            "success_rate": (
                (self.total_operations - self.failed_operations) / self.total_operations
                if self.total_operations > 0 else 0.0
            ),
            "operations_per_second": (
                self.total_operations / uptime
                if uptime > 0 else 0.0
            ),
            "connection_attempts": float(self.connection_attempts),
            "authentication_attempts": float(self.authentication_attempts),
        }

    def increment_operation_count(self, success: bool = True) -> None:
        """Increment operation counters.

        Args:
            success: Whether the operation was successful
        """
        self.total_operations += 1
        if not success:
            self.failed_operations += 1

    async def __aenter__(self) -> Mt5Session:
        """Async context manager entry."""
        await self.connect()
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    def __del__(self) -> None:
        """Cleanup on destruction."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()