"""
Core exception handling module for MetaTrader Python Framework.

This module provides a comprehensive exception hierarchy for handling
various types of errors that can occur in the trading framework.
"""

# Base exceptions
from .base import (
    BaseFrameworkError,
    ConfigurationError,
    DependencyError,
    FrameworkWarning,
    InitializationError,
    RateLimitError,
    RetryableError,
    SecurityError,
    TimeoutError,
    ValidationError,
    create_error_context,
    handle_exception,
)

# Connection exceptions
from .connection import (
    ApiConnectionError,
    BrokerConnectionError,
    ConnectionError,
    ConnectionPoolError,
    ConnectionTimeoutError,
    DatabaseConnectionError,
    Mt5AccountError,
    Mt5AuthenticationError,
    Mt5ConnectionError,
    Mt5InitializationError,
    Mt5TerminalError,
    Mt5TerminalNotFoundError,
    NetworkConnectionError,
    ServerUnavailableError,
    SslConnectionError,
    TradingSessionError,
)

# Trading exceptions
from .trading import (
    DataNotAvailableError,
    DataTimeoutError,
    DataValidationError,
    ExcessiveRiskError,
    IndicatorError,
    InsufficientFundsError,
    MarketDataError,
    MaxPositionsError,
    NetworkTradingError,
    OrderError,
    OrderExecutionError,
    OrderValidationError,
    PatternError,
    PositionError,
    PositionModificationError,
    PositionNotFoundError,
    RiskManagementError,
    StrategyError,
    StrategyExecutionError,
    StrategyInitializationError,
    TradingError,
    TradingHaltError,
)

__all__ = [
    # Base exceptions
    "BaseFrameworkError",
    "ConfigurationError",
    "ValidationError",
    "InitializationError",
    "DependencyError",
    "SecurityError",
    "RateLimitError",
    "TimeoutError",
    "RetryableError",
    "FrameworkWarning",
    "handle_exception",
    "create_error_context",
    # Connection exceptions
    "ConnectionError",
    "Mt5ConnectionError",
    "Mt5InitializationError",
    "Mt5AuthenticationError",
    "Mt5TerminalNotFoundError",
    "Mt5TerminalError",
    "Mt5AccountError",
    "NetworkConnectionError",
    "ConnectionTimeoutError",
    "ServerUnavailableError",
    "BrokerConnectionError",
    "TradingSessionError",
    "DatabaseConnectionError",
    "ApiConnectionError",
    "ConnectionPoolError",
    "SslConnectionError",
    # Trading exceptions
    "TradingError",
    "OrderError",
    "OrderExecutionError",
    "OrderValidationError",
    "InsufficientFundsError",
    "PositionError",
    "PositionNotFoundError",
    "PositionModificationError",
    "MarketDataError",
    "DataNotAvailableError",
    "DataTimeoutError",
    "DataValidationError",
    "StrategyError",
    "StrategyInitializationError",
    "StrategyExecutionError",
    "IndicatorError",
    "PatternError",
    "RiskManagementError",
    "ExcessiveRiskError",
    "MaxPositionsError",
    "TradingHaltError",
    "NetworkTradingError",
]