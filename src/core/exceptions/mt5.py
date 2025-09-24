"""
MetaTrader 5 specific exception classes.

This module provides comprehensive exception handling for all MT5-related operations
including connection management, trading operations, data retrieval, and real-time processing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from .base import BaseFrameworkError, RetryableError, SecurityError, TimeoutError


class Mt5Error(BaseFrameworkError):
    """Base exception for all MT5-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        mt5_error: Optional[str] = None,
        account: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize MT5 error.

        Args:
            message: Error message
            error_code: MT5 error code
            mt5_error: MT5 error description
            account: Account name/login
            **kwargs: Additional error context
        """
        super().__init__(message, **kwargs)
        self.error_code = error_code
        self.mt5_error = mt5_error
        self.account = account


# Connection-related exceptions
class Mt5ConnectionError(Mt5Error, RetryableError):
    """MT5 connection error."""

    def __init__(
        self,
        message: str = "Failed to connect to MT5 terminal",
        server: Optional[str] = None,
        login: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.server = server
        self.login = login


class Mt5InitializationError(Mt5Error):
    """MT5 initialization error."""

    def __init__(
        self,
        message: str = "Failed to initialize MT5 terminal",
        path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.path = path


class Mt5AuthenticationError(Mt5Error, SecurityError):
    """MT5 authentication error."""

    def __init__(
        self,
        message: str = "MT5 authentication failed",
        login: Optional[int] = None,
        server: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.login = login
        self.server = server


class Mt5TerminalNotFoundError(Mt5Error):
    """MT5 terminal not found error."""

    def __init__(
        self,
        message: str = "MT5 terminal not found",
        path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.path = path


class Mt5TerminalError(Mt5Error):
    """MT5 terminal error."""

    def __init__(
        self,
        message: str = "MT5 terminal error",
        terminal_error: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.terminal_error = terminal_error


class Mt5AccountError(Mt5Error):
    """MT5 account error."""

    def __init__(
        self,
        message: str = "MT5 account error",
        account_number: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.account_number = account_number


class Mt5SessionError(Mt5Error, RetryableError):
    """MT5 session error."""

    def __init__(
        self,
        message: str = "MT5 session error",
        session_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.session_id = session_id


class Mt5TimeoutError(Mt5Error, TimeoutError):
    """MT5 operation timeout error."""

    def __init__(
        self,
        message: str = "MT5 operation timeout",
        operation: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.operation = operation
        self.timeout_ms = timeout_ms


# Trading-related exceptions
class Mt5TradingError(Mt5Error):
    """Base exception for MT5 trading errors."""

    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        order_id: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.symbol = symbol
        self.order_id = order_id


class Mt5OrderError(Mt5TradingError):
    """MT5 order error."""

    def __init__(
        self,
        message: str = "MT5 order error",
        order_type: Optional[str] = None,
        volume: Optional[float] = None,
        price: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.order_type = order_type
        self.volume = volume
        self.price = price


class Mt5OrderExecutionError(Mt5OrderError):
    """MT5 order execution error."""

    def __init__(
        self,
        message: str = "Failed to execute MT5 order",
        execution_time: Optional[float] = None,
        slippage: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.execution_time = execution_time
        self.slippage = slippage


class Mt5OrderValidationError(Mt5OrderError):
    """MT5 order validation error."""

    def __init__(
        self,
        message: str = "MT5 order validation failed",
        validation_errors: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or {}


class Mt5InsufficientFundsError(Mt5TradingError):
    """MT5 insufficient funds error."""

    def __init__(
        self,
        message: str = "Insufficient funds for MT5 trade",
        required_margin: Optional[float] = None,
        available_margin: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.required_margin = required_margin
        self.available_margin = available_margin


class Mt5PositionError(Mt5TradingError):
    """MT5 position error."""

    def __init__(
        self,
        message: str = "MT5 position error",
        position_id: Optional[int] = None,
        ticket: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.position_id = position_id
        self.ticket = ticket


class Mt5PositionNotFoundError(Mt5PositionError):
    """MT5 position not found error."""

    def __init__(
        self,
        message: str = "MT5 position not found",
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)


class Mt5PositionModificationError(Mt5PositionError):
    """MT5 position modification error."""

    def __init__(
        self,
        message: str = "Failed to modify MT5 position",
        modification_type: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.modification_type = modification_type


# Market data exceptions
class Mt5MarketDataError(Mt5Error):
    """Base exception for MT5 market data errors."""

    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.symbol = symbol
        self.timeframe = timeframe


class Mt5DataNotAvailableError(Mt5MarketDataError):
    """MT5 data not available error."""

    def __init__(
        self,
        message: str = "MT5 market data not available",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.start_date = start_date
        self.end_date = end_date


class Mt5DataTimeoutError(Mt5MarketDataError, Mt5TimeoutError):
    """MT5 data timeout error."""

    def __init__(
        self,
        message: str = "MT5 data request timeout",
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)


class Mt5DataValidationError(Mt5MarketDataError):
    """MT5 data validation error."""

    def __init__(
        self,
        message: str = "MT5 data validation failed",
        data_issues: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.data_issues = data_issues or {}


class Mt5TickDataError(Mt5MarketDataError):
    """MT5 tick data error."""

    def __init__(
        self,
        message: str = "MT5 tick data error",
        tick_count: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.tick_count = tick_count


class Mt5QuoteDataError(Mt5MarketDataError):
    """MT5 quote data error."""

    def __init__(
        self,
        message: str = "MT5 quote data error",
        quote_time: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.quote_time = quote_time


class Mt5HistoricalDataError(Mt5MarketDataError):
    """MT5 historical data error."""

    def __init__(
        self,
        message: str = "MT5 historical data error",
        bars_requested: Optional[int] = None,
        bars_received: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.bars_requested = bars_requested
        self.bars_received = bars_received


# Symbol and instrument exceptions
class Mt5SymbolError(Mt5Error):
    """MT5 symbol error."""

    def __init__(
        self,
        message: str = "MT5 symbol error",
        symbol: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.symbol = symbol


class Mt5SymbolNotFoundError(Mt5SymbolError):
    """MT5 symbol not found error."""

    def __init__(
        self,
        message: str = "MT5 symbol not found",
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)


class Mt5SymbolInfoError(Mt5SymbolError):
    """MT5 symbol information error."""

    def __init__(
        self,
        message: str = "Failed to get MT5 symbol information",
        info_type: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.info_type = info_type


class Mt5MarketClosedError(Mt5SymbolError):
    """MT5 market closed error."""

    def __init__(
        self,
        message: str = "MT5 market is closed",
        trading_hours: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.trading_hours = trading_hours


# Risk management exceptions
class Mt5RiskManagementError(Mt5Error):
    """MT5 risk management error."""

    def __init__(
        self,
        message: str = "MT5 risk management error",
        risk_check: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.risk_check = risk_check


class Mt5ExcessiveRiskError(Mt5RiskManagementError):
    """MT5 excessive risk error."""

    def __init__(
        self,
        message: str = "MT5 trade exceeds risk limits",
        risk_percentage: Optional[float] = None,
        max_risk: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.risk_percentage = risk_percentage
        self.max_risk = max_risk


class Mt5MaxPositionsError(Mt5RiskManagementError):
    """MT5 maximum positions error."""

    def __init__(
        self,
        message: str = "MT5 maximum positions exceeded",
        current_positions: Optional[int] = None,
        max_positions: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.current_positions = current_positions
        self.max_positions = max_positions


class Mt5TradingHaltError(Mt5RiskManagementError):
    """MT5 trading halt error."""

    def __init__(
        self,
        message: str = "MT5 trading halted",
        halt_reason: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.halt_reason = halt_reason


# Performance and resource exceptions
class Mt5PerformanceError(Mt5Error):
    """MT5 performance error."""

    def __init__(
        self,
        message: str = "MT5 performance error",
        metric: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.metric = metric
        self.current_value = current_value
        self.threshold = threshold


class Mt5LatencyError(Mt5PerformanceError):
    """MT5 latency error."""

    def __init__(
        self,
        message: str = "MT5 latency threshold exceeded",
        latency_ms: Optional[float] = None,
        threshold_ms: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.latency_ms = latency_ms
        self.threshold_ms = threshold_ms


class Mt5ThroughputError(Mt5PerformanceError):
    """MT5 throughput error."""

    def __init__(
        self,
        message: str = "MT5 throughput threshold exceeded",
        current_tps: Optional[float] = None,
        max_tps: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.current_tps = current_tps
        self.max_tps = max_tps


class Mt5MemoryError(Mt5PerformanceError):
    """MT5 memory error."""

    def __init__(
        self,
        message: str = "MT5 memory limit exceeded",
        memory_usage_mb: Optional[float] = None,
        memory_limit_mb: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.memory_usage_mb = memory_usage_mb
        self.memory_limit_mb = memory_limit_mb


# Circuit breaker exceptions
class Mt5CircuitBreakerError(Mt5Error):
    """MT5 circuit breaker error."""

    def __init__(
        self,
        message: str = "MT5 circuit breaker triggered",
        failure_count: Optional[int] = None,
        threshold: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.failure_count = failure_count
        self.threshold = threshold


class Mt5CircuitBreakerOpenError(Mt5CircuitBreakerError):
    """MT5 circuit breaker open error."""

    def __init__(
        self,
        message: str = "MT5 circuit breaker is open",
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)


# Multi-account exceptions
class Mt5MultiAccountError(Mt5Error):
    """MT5 multi-account error."""

    def __init__(
        self,
        message: str = "MT5 multi-account error",
        account_count: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.account_count = account_count


class Mt5AccountSwitchError(Mt5MultiAccountError):
    """MT5 account switch error."""

    def __init__(
        self,
        message: str = "Failed to switch MT5 account",
        from_account: Optional[str] = None,
        to_account: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.from_account = from_account
        self.to_account = to_account


# Event and streaming exceptions
class Mt5EventError(Mt5Error):
    """MT5 event error."""

    def __init__(
        self,
        message: str = "MT5 event error",
        event_type: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.event_type = event_type
        self.event_data = event_data


class Mt5StreamingError(Mt5Error):
    """MT5 streaming error."""

    def __init__(
        self,
        message: str = "MT5 streaming error",
        stream_type: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.stream_type = stream_type


class Mt5EventBufferOverflowError(Mt5EventError):
    """MT5 event buffer overflow error."""

    def __init__(
        self,
        message: str = "MT5 event buffer overflow",
        buffer_size: Optional[int] = None,
        events_lost: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.buffer_size = buffer_size
        self.events_lost = events_lost


class Mt5EventProcessingError(Mt5EventError):
    """MT5 event processing error."""

    def __init__(
        self,
        message: str = "MT5 event processing error",
        processor_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.processor_id = processor_id


# API-specific exceptions
class Mt5ApiError(Mt5Error):
    """MT5 API error."""

    def __init__(
        self,
        message: str = "MT5 API error",
        api_function: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.api_function = api_function


class Mt5ApiNotInitializedError(Mt5ApiError):
    """MT5 API not initialized error."""

    def __init__(
        self,
        message: str = "MT5 API not initialized",
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)


class Mt5ApiVersionError(Mt5ApiError):
    """MT5 API version error."""

    def __init__(
        self,
        message: str = "MT5 API version mismatch",
        required_version: Optional[str] = None,
        current_version: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.required_version = required_version
        self.current_version = current_version


def create_mt5_error_from_code(
    error_code: int,
    mt5_error: str,
    operation: str = "unknown",
    **kwargs: Any
) -> Mt5Error:
    """Create appropriate MT5 error from error code.

    Args:
        error_code: MT5 error code
        mt5_error: MT5 error description
        operation: Operation that failed
        **kwargs: Additional error context

    Returns:
        Appropriate MT5 error instance
    """
    error_mapping = {
        # Connection errors
        10004: Mt5ConnectionError,
        10005: Mt5TerminalNotFoundError,
        10006: Mt5InitializationError,
        10007: Mt5AuthenticationError,
        10008: Mt5AccountError,

        # Trading errors
        10009: Mt5OrderError,
        10010: Mt5OrderExecutionError,
        10011: Mt5InsufficientFundsError,
        10012: Mt5PositionError,

        # Market data errors
        10013: Mt5DataNotAvailableError,
        10014: Mt5SymbolNotFoundError,
        10015: Mt5MarketClosedError,

        # API errors
        10016: Mt5ApiNotInitializedError,
        10017: Mt5ApiVersionError,

        # Timeout errors
        10018: Mt5TimeoutError,

        # Performance errors
        10019: Mt5LatencyError,
        10020: Mt5ThroughputError,
        10021: Mt5MemoryError,
    }

    error_class = error_mapping.get(error_code, Mt5Error)
    message = f"MT5 {operation} failed: {mt5_error} (Error: {error_code})"

    return error_class(
        message=message,
        error_code=error_code,
        mt5_error=mt5_error,
        **kwargs
    )


__all__ = [
    # Base MT5 exceptions
    "Mt5Error",

    # Connection exceptions
    "Mt5ConnectionError",
    "Mt5InitializationError",
    "Mt5AuthenticationError",
    "Mt5TerminalNotFoundError",
    "Mt5TerminalError",
    "Mt5AccountError",
    "Mt5SessionError",
    "Mt5TimeoutError",

    # Trading exceptions
    "Mt5TradingError",
    "Mt5OrderError",
    "Mt5OrderExecutionError",
    "Mt5OrderValidationError",
    "Mt5InsufficientFundsError",
    "Mt5PositionError",
    "Mt5PositionNotFoundError",
    "Mt5PositionModificationError",

    # Market data exceptions
    "Mt5MarketDataError",
    "Mt5DataNotAvailableError",
    "Mt5DataTimeoutError",
    "Mt5DataValidationError",
    "Mt5TickDataError",
    "Mt5QuoteDataError",
    "Mt5HistoricalDataError",

    # Symbol exceptions
    "Mt5SymbolError",
    "Mt5SymbolNotFoundError",
    "Mt5SymbolInfoError",
    "Mt5MarketClosedError",

    # Risk management exceptions
    "Mt5RiskManagementError",
    "Mt5ExcessiveRiskError",
    "Mt5MaxPositionsError",
    "Mt5TradingHaltError",

    # Performance exceptions
    "Mt5PerformanceError",
    "Mt5LatencyError",
    "Mt5ThroughputError",
    "Mt5MemoryError",

    # Circuit breaker exceptions
    "Mt5CircuitBreakerError",
    "Mt5CircuitBreakerOpenError",

    # Multi-account exceptions
    "Mt5MultiAccountError",
    "Mt5AccountSwitchError",

    # Event exceptions
    "Mt5EventError",
    "Mt5StreamingError",
    "Mt5EventBufferOverflowError",
    "Mt5EventProcessingError",

    # API exceptions
    "Mt5ApiError",
    "Mt5ApiNotInitializedError",
    "Mt5ApiVersionError",

    # Utility functions
    "create_mt5_error_from_code",
]