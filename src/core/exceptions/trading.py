"""
Trading-specific exception classes for MetaTrader Python Framework.

This module defines exceptions related to trading operations, order management,
market data, strategy execution, and risk management.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import BaseFrameworkError, RetryableError, TimeoutError


class TradingError(BaseFrameworkError):
    """Base exception for all trading-related errors."""

    error_code = "TRADING_ERROR"
    error_category = "trading"
    severity = "error"


class OrderError(TradingError):
    """Exception raised for order-related errors."""

    error_code = "ORDER_ERROR"
    error_category = "order"

    def __init__(
        self,
        message: str,
        *,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize order error.

        Args:
            message: Error message
            order_id: Order identifier
            symbol: Trading symbol
            order_type: Type of order (BUY, SELL, etc.)
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if order_id:
            self.add_context("order_id", order_id)
        if symbol:
            self.add_context("symbol", symbol)
        if order_type:
            self.add_context("order_type", order_type)


class OrderExecutionError(OrderError):
    """Exception raised when order execution fails."""

    error_code = "ORDER_EXECUTION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        execution_price: Optional[float] = None,
        requested_price: Optional[float] = None,
        volume: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize order execution error.

        Args:
            message: Error message
            execution_price: Actual execution price
            requested_price: Requested execution price
            volume: Order volume
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if execution_price:
            self.add_context("execution_price", execution_price)
        if requested_price:
            self.add_context("requested_price", requested_price)
        if volume:
            self.add_context("volume", volume)


class OrderValidationError(OrderError):
    """Exception raised when order validation fails."""

    error_code = "ORDER_VALIDATION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        validation_rule: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize order validation error.

        Args:
            message: Error message
            validation_rule: Validation rule that failed
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if validation_rule:
            self.add_context("validation_rule", validation_rule)


class InsufficientFundsError(OrderError):
    """Exception raised when there are insufficient funds for an order."""

    error_code = "INSUFFICIENT_FUNDS_ERROR"

    def __init__(
        self,
        message: str,
        *,
        required_margin: Optional[float] = None,
        available_margin: Optional[float] = None,
        account_balance: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize insufficient funds error.

        Args:
            message: Error message
            required_margin: Required margin for the order
            available_margin: Available margin
            account_balance: Current account balance
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if required_margin:
            self.add_context("required_margin", required_margin)
        if available_margin:
            self.add_context("available_margin", available_margin)
        if account_balance:
            self.add_context("account_balance", account_balance)


class PositionError(TradingError):
    """Exception raised for position-related errors."""

    error_code = "POSITION_ERROR"
    error_category = "position"

    def __init__(
        self,
        message: str,
        *,
        position_id: Optional[str] = None,
        symbol: Optional[str] = None,
        volume: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize position error.

        Args:
            message: Error message
            position_id: Position identifier
            symbol: Trading symbol
            volume: Position volume
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if position_id:
            self.add_context("position_id", position_id)
        if symbol:
            self.add_context("symbol", symbol)
        if volume:
            self.add_context("volume", volume)


class PositionNotFoundError(PositionError):
    """Exception raised when a position is not found."""

    error_code = "POSITION_NOT_FOUND_ERROR"


class PositionModificationError(PositionError):
    """Exception raised when position modification fails."""

    error_code = "POSITION_MODIFICATION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        modification_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize position modification error.

        Args:
            message: Error message
            modification_type: Type of modification (SL, TP, volume, etc.)
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if modification_type:
            self.add_context("modification_type", modification_type)


class MarketDataError(TradingError):
    """Exception raised for market data-related errors."""

    error_code = "MARKET_DATA_ERROR"
    error_category = "market_data"

    def __init__(
        self,
        message: str,
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        data_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize market data error.

        Args:
            message: Error message
            symbol: Trading symbol
            timeframe: Data timeframe
            data_type: Type of data (tick, ohlc, etc.)
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if symbol:
            self.add_context("symbol", symbol)
        if timeframe:
            self.add_context("timeframe", timeframe)
        if data_type:
            self.add_context("data_type", data_type)


class DataNotAvailableError(MarketDataError):
    """Exception raised when requested market data is not available."""

    error_code = "DATA_NOT_AVAILABLE_ERROR"


class DataTimeoutError(MarketDataError, TimeoutError):
    """Exception raised when market data request times out."""

    error_code = "DATA_TIMEOUT_ERROR"


class DataValidationError(MarketDataError):
    """Exception raised when market data validation fails."""

    error_code = "DATA_VALIDATION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        validation_issue: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize data validation error.

        Args:
            message: Error message
            validation_issue: Description of validation issue
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if validation_issue:
            self.add_context("validation_issue", validation_issue)


class StrategyError(TradingError):
    """Exception raised for strategy-related errors."""

    error_code = "STRATEGY_ERROR"
    error_category = "strategy"

    def __init__(
        self,
        message: str,
        *,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize strategy error.

        Args:
            message: Error message
            strategy_name: Name of the strategy
            symbol: Trading symbol
            timeframe: Strategy timeframe
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if strategy_name:
            self.add_context("strategy_name", strategy_name)
        if symbol:
            self.add_context("symbol", symbol)
        if timeframe:
            self.add_context("timeframe", timeframe)


class StrategyInitializationError(StrategyError):
    """Exception raised when strategy initialization fails."""

    error_code = "STRATEGY_INIT_ERROR"


class StrategyExecutionError(StrategyError):
    """Exception raised when strategy execution fails."""

    error_code = "STRATEGY_EXECUTION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        execution_phase: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize strategy execution error.

        Args:
            message: Error message
            execution_phase: Phase where execution failed (signal, order, etc.)
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if execution_phase:
            self.add_context("execution_phase", execution_phase)


class IndicatorError(TradingError):
    """Exception raised for indicator calculation errors."""

    error_code = "INDICATOR_ERROR"
    error_category = "indicator"

    def __init__(
        self,
        message: str,
        *,
        indicator_name: Optional[str] = None,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize indicator error.

        Args:
            message: Error message
            indicator_name: Name of the indicator
            parameter: Parameter that caused the error
            value: Invalid parameter value
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if indicator_name:
            self.add_context("indicator_name", indicator_name)
        if parameter:
            self.add_context("parameter", parameter)
        if value is not None:
            self.add_context("value", str(value))


class PatternError(TradingError):
    """Exception raised for pattern recognition errors."""

    error_code = "PATTERN_ERROR"
    error_category = "pattern"

    def __init__(
        self,
        message: str,
        *,
        pattern_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize pattern error.

        Args:
            message: Error message
            pattern_name: Name of the pattern
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if pattern_name:
            self.add_context("pattern_name", pattern_name)


class RiskManagementError(TradingError):
    """Exception raised for risk management violations."""

    error_code = "RISK_MANAGEMENT_ERROR"
    error_category = "risk_management"
    severity = "critical"

    def __init__(
        self,
        message: str,
        *,
        risk_rule: Optional[str] = None,
        current_risk: Optional[float] = None,
        max_risk: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize risk management error.

        Args:
            message: Error message
            risk_rule: Risk rule that was violated
            current_risk: Current risk level
            max_risk: Maximum allowed risk
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if risk_rule:
            self.add_context("risk_rule", risk_rule)
        if current_risk:
            self.add_context("current_risk", current_risk)
        if max_risk:
            self.add_context("max_risk", max_risk)


class ExcessiveRiskError(RiskManagementError):
    """Exception raised when risk limits are exceeded."""

    error_code = "EXCESSIVE_RISK_ERROR"


class MaxPositionsError(RiskManagementError):
    """Exception raised when maximum position limit is reached."""

    error_code = "MAX_POSITIONS_ERROR"

    def __init__(
        self,
        message: str,
        *,
        current_positions: Optional[int] = None,
        max_positions: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize max positions error.

        Args:
            message: Error message
            current_positions: Current number of positions
            max_positions: Maximum allowed positions
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if current_positions:
            self.add_context("current_positions", current_positions)
        if max_positions:
            self.add_context("max_positions", max_positions)


class TradingHaltError(TradingError):
    """Exception raised when trading is halted."""

    error_code = "TRADING_HALT_ERROR"
    severity = "critical"

    def __init__(
        self,
        message: str,
        *,
        halt_reason: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize trading halt error.

        Args:
            message: Error message
            halt_reason: Reason for trading halt
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if halt_reason:
            self.add_context("halt_reason", halt_reason)


class NetworkTradingError(TradingError, RetryableError):
    """Exception raised for network-related trading errors."""

    error_code = "NETWORK_TRADING_ERROR"
    error_category = "network"

    def __init__(
        self,
        message: str,
        *,
        endpoint: Optional[str] = None,
        response_code: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize network trading error.

        Args:
            message: Error message
            endpoint: API endpoint that failed
            response_code: HTTP response code
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if endpoint:
            self.add_context("endpoint", endpoint)
        if response_code:
            self.add_context("response_code", response_code)