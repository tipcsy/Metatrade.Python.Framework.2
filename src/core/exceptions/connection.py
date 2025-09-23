"""
Connection-specific exception classes for MetaTrader Python Framework.

This module defines exceptions related to MetaTrader 5 connections,
authentication, network issues, and broker communication.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import BaseFrameworkError, RetryableError, TimeoutError


class ConnectionError(BaseFrameworkError):
    """Base exception for all connection-related errors."""

    error_code = "CONNECTION_ERROR"
    error_category = "connection"
    severity = "error"


class Mt5ConnectionError(ConnectionError):
    """Exception raised for MetaTrader 5 connection errors."""

    error_code = "MT5_CONNECTION_ERROR"
    error_category = "mt5_connection"

    def __init__(
        self,
        message: str,
        *,
        server: Optional[str] = None,
        login: Optional[str] = None,
        terminal_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MT5 connection error.

        Args:
            message: Error message
            server: MT5 server name
            login: MT5 login (without password for security)
            terminal_path: Path to MT5 terminal
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if server:
            self.add_context("server", server)
        if login:
            # Only store login number, never password
            self.add_context("login", str(login))
        if terminal_path:
            self.add_context("terminal_path", terminal_path)


class Mt5InitializationError(Mt5ConnectionError):
    """Exception raised when MT5 initialization fails."""

    error_code = "MT5_INIT_ERROR"

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MT5 initialization error.

        Args:
            message: Error message
            error_code: MT5-specific error code
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if error_code is not None:
            self.add_context("mt5_error_code", error_code)


class Mt5AuthenticationError(Mt5ConnectionError):
    """Exception raised for MT5 authentication failures."""

    error_code = "MT5_AUTH_ERROR"
    severity = "critical"

    def __init__(
        self,
        message: str,
        *,
        auth_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MT5 authentication error.

        Args:
            message: Error message
            auth_type: Type of authentication (login, certificate, etc.)
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if auth_type:
            self.add_context("auth_type", auth_type)


class Mt5TerminalNotFoundError(Mt5ConnectionError):
    """Exception raised when MT5 terminal is not found."""

    error_code = "MT5_TERMINAL_NOT_FOUND_ERROR"

    def __init__(
        self,
        message: str,
        *,
        searched_paths: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MT5 terminal not found error.

        Args:
            message: Error message
            searched_paths: Paths that were searched for MT5 terminal
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if searched_paths:
            self.add_context("searched_paths", searched_paths)


class Mt5TerminalError(Mt5ConnectionError):
    """Exception raised for MT5 terminal-related errors."""

    error_code = "MT5_TERMINAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        terminal_version: Optional[str] = None,
        build: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MT5 terminal error.

        Args:
            message: Error message
            terminal_version: MT5 terminal version
            build: MT5 build number
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if terminal_version:
            self.add_context("terminal_version", terminal_version)
        if build:
            self.add_context("build", build)


class Mt5AccountError(Mt5ConnectionError):
    """Exception raised for MT5 account-related errors."""

    error_code = "MT5_ACCOUNT_ERROR"

    def __init__(
        self,
        message: str,
        *,
        account_number: Optional[str] = None,
        account_type: Optional[str] = None,
        broker: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MT5 account error.

        Args:
            message: Error message
            account_number: Account number
            account_type: Account type (demo, real, etc.)
            broker: Broker name
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if account_number:
            self.add_context("account_number", str(account_number))
        if account_type:
            self.add_context("account_type", account_type)
        if broker:
            self.add_context("broker", broker)


class NetworkConnectionError(ConnectionError, RetryableError):
    """Exception raised for network connection issues."""

    error_code = "NETWORK_CONNECTION_ERROR"
    error_category = "network"

    def __init__(
        self,
        message: str,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        protocol: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize network connection error.

        Args:
            message: Error message
            host: Target host
            port: Target port
            protocol: Network protocol (TCP, HTTP, etc.)
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if host:
            self.add_context("host", host)
        if port:
            self.add_context("port", port)
        if protocol:
            self.add_context("protocol", protocol)


class ConnectionTimeoutError(ConnectionError, TimeoutError):
    """Exception raised when connection attempts timeout."""

    error_code = "CONNECTION_TIMEOUT_ERROR"

    def __init__(
        self,
        message: str,
        *,
        connection_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize connection timeout error.

        Args:
            message: Error message
            connection_type: Type of connection (MT5, database, API, etc.)
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if connection_type:
            self.add_context("connection_type", connection_type)


class ServerUnavailableError(ConnectionError, RetryableError):
    """Exception raised when server is unavailable."""

    error_code = "SERVER_UNAVAILABLE_ERROR"

    def __init__(
        self,
        message: str,
        *,
        server_status: Optional[str] = None,
        maintenance_window: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize server unavailable error.

        Args:
            message: Error message
            server_status: Current server status
            maintenance_window: Maintenance window information
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if server_status:
            self.add_context("server_status", server_status)
        if maintenance_window:
            self.add_context("maintenance_window", maintenance_window)


class BrokerConnectionError(ConnectionError):
    """Exception raised for broker-specific connection errors."""

    error_code = "BROKER_CONNECTION_ERROR"
    error_category = "broker"

    def __init__(
        self,
        message: str,
        *,
        broker_name: Optional[str] = None,
        broker_status: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize broker connection error.

        Args:
            message: Error message
            broker_name: Name of the broker
            broker_status: Current broker status
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if broker_name:
            self.add_context("broker_name", broker_name)
        if broker_status:
            self.add_context("broker_status", broker_status)


class TradingSessionError(ConnectionError):
    """Exception raised for trading session-related errors."""

    error_code = "TRADING_SESSION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        session_status: Optional[str] = None,
        market_hours: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize trading session error.

        Args:
            message: Error message
            session_status: Current session status
            market_hours: Market trading hours information
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if session_status:
            self.add_context("session_status", session_status)
        if market_hours:
            self.add_context("market_hours", market_hours)


class DatabaseConnectionError(ConnectionError, RetryableError):
    """Exception raised for database connection errors."""

    error_code = "DATABASE_CONNECTION_ERROR"
    error_category = "database"

    def __init__(
        self,
        message: str,
        *,
        database_url: Optional[str] = None,
        database_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize database connection error.

        Args:
            message: Error message
            database_url: Database URL (sanitized)
            database_type: Type of database (SQLite, PostgreSQL, etc.)
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if database_url:
            # Sanitize URL to remove credentials
            sanitized_url = self._sanitize_url(database_url)
            self.add_context("database_url", sanitized_url)
        if database_type:
            self.add_context("database_type", database_type)

    def _sanitize_url(self, url: str) -> str:
        """
        Sanitize database URL to remove credentials.

        Args:
            url: Original database URL

        Returns:
            Sanitized URL without credentials
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            # Remove username and password
            sanitized = f"{parsed.scheme}://***:***@{parsed.hostname}"
            if parsed.port:
                sanitized += f":{parsed.port}"
            if parsed.path:
                sanitized += parsed.path
            return sanitized
        except Exception:
            return "***sanitized***"


class ApiConnectionError(ConnectionError, RetryableError):
    """Exception raised for API connection errors."""

    error_code = "API_CONNECTION_ERROR"
    error_category = "api"

    def __init__(
        self,
        message: str,
        *,
        api_endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize API connection error.

        Args:
            message: Error message
            api_endpoint: API endpoint that failed
            status_code: HTTP status code
            response_body: Response body (truncated)
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if api_endpoint:
            self.add_context("api_endpoint", api_endpoint)
        if status_code:
            self.add_context("status_code", status_code)
        if response_body:
            # Truncate response body to avoid huge logs
            truncated_body = response_body[:500] + "..." if len(response_body) > 500 else response_body
            self.add_context("response_body", truncated_body)


class ConnectionPoolError(ConnectionError):
    """Exception raised for connection pool-related errors."""

    error_code = "CONNECTION_POOL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        pool_size: Optional[int] = None,
        active_connections: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize connection pool error.

        Args:
            message: Error message
            pool_size: Maximum pool size
            active_connections: Current active connections
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if pool_size:
            self.add_context("pool_size", pool_size)
        if active_connections:
            self.add_context("active_connections", active_connections)


class SslConnectionError(ConnectionError):
    """Exception raised for SSL/TLS connection errors."""

    error_code = "SSL_CONNECTION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        certificate_issue: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize SSL connection error.

        Args:
            message: Error message
            certificate_issue: Description of certificate issue
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if certificate_issue:
            self.add_context("certificate_issue", certificate_issue)