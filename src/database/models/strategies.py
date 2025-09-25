"""
Strategy and performance tracking models for the MetaTrader Python Framework.

This module defines database models for trading strategies, their configurations,
performance tracking, and execution history.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    DECIMAL,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models.base import BaseModel
from src.database.models.mixins import ConfigurationMixin, PerformanceMixin


class Strategy(BaseModel, ConfigurationMixin, PerformanceMixin):
    """
    Model for trading strategies.

    Represents trading strategies with their configurations,
    parameters, and performance tracking capabilities.
    """

    __tablename__ = "strategies"

    # Strategy classification
    strategy_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Strategy type (TREND_FOLLOWING, MEAN_REVERSION, ARBITRAGE, etc.)"
    )

    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Strategy category (SCALPING, DAY_TRADING, SWING, POSITION)"
    )

    # Strategy version and development
    version: Mapped[str] = mapped_column(
        String(20),
        default="1.0.0",
        nullable=False,
        doc="Strategy version"
    )

    author: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Strategy author/developer"
    )

    # Strategy parameters (stored as JSON)
    parameters: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Strategy parameters as JSON string"
    )

    # Risk management settings
    max_risk_per_trade: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=5, scale=2),
        default=Decimal('2.0'),
        nullable=False,
        doc="Maximum risk per trade (percentage)"
    )

    max_daily_risk: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=5, scale=2),
        default=Decimal('10.0'),
        nullable=False,
        doc="Maximum daily risk (percentage)"
    )

    max_concurrent_trades: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
        doc="Maximum number of concurrent trades"
    )

    # Strategy status
    is_live: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether strategy is running live"
    )

    is_backtested: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether strategy has been backtested"
    )

    # Performance tracking
    start_balance: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=True,
        doc="Starting balance for performance calculation"
    )

    current_balance: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=True,
        doc="Current balance"
    )

    peak_balance: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=True,
        doc="Peak balance reached"
    )

    # Strategy execution info
    last_execution: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last strategy execution time"
    )

    execution_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Total number of strategy executions"
    )

    # Relationships
    trades = relationship(
        "Trade",
        back_populates="strategy",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    strategy_sessions = relationship(
        "StrategySession",
        back_populates="strategy",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    backtest_results = relationship(
        "BacktestResult",
        back_populates="strategy",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_strategy_name_version'),
        Index('ix_strategy_type_active', 'strategy_type', 'is_active'),
        Index('ix_strategy_live_active', 'is_live', 'is_active'),
    )

    @property
    def total_return(self) -> Optional[Decimal]:
        """Calculate total return percentage."""
        if self.start_balance is None or self.current_balance is None or self.start_balance == 0:
            return None
        return ((self.current_balance - self.start_balance) / self.start_balance) * Decimal('100')

    @property
    def current_drawdown(self) -> Optional[Decimal]:
        """Calculate current drawdown percentage."""
        if self.peak_balance is None or self.current_balance is None or self.peak_balance == 0:
            return None
        return ((self.peak_balance - self.current_balance) / self.peak_balance) * Decimal('100')

    def update_balance(self, new_balance: Decimal) -> None:
        """
        Update strategy balance and performance metrics.

        Args:
            new_balance: New balance amount
        """
        self.current_balance = new_balance

        # Update peak balance
        if self.peak_balance is None or new_balance > self.peak_balance:
            self.peak_balance = new_balance

        # Update max drawdown
        current_dd = self.current_drawdown
        if current_dd is not None:
            if self.max_drawdown is None or current_dd > self.max_drawdown:
                self.max_drawdown = current_dd

    def record_execution(self) -> None:
        """Record a strategy execution."""
        self.last_execution = datetime.now(timezone.utc)
        self.execution_count += 1

    def __repr__(self) -> str:
        return f"<Strategy(name='{self.name}', type='{self.strategy_type}', version='{self.version}')>"


class StrategySession(BaseModel):
    """
    Model for strategy execution sessions.

    Tracks individual strategy execution sessions with
    performance metrics and session-specific information.
    """

    __tablename__ = "strategy_sessions"

    # Strategy relationship
    strategy_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("strategies.id"),
        nullable=False,
        index=True,
        doc="Reference to strategy"
    )

    # Session information
    session_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Session name/identifier"
    )

    # Session timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Session start time"
    )

    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Session end time"
    )

    # Session status
    status: Mapped[str] = mapped_column(
        String(20),
        default="RUNNING",
        nullable=False,
        index=True,
        doc="Session status (RUNNING, COMPLETED, STOPPED, ERROR)"
    )

    # Performance tracking (specific to this session)
    starting_balance: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=False,
        doc="Balance at session start"
    )

    ending_balance: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=True,
        doc="Balance at session end"
    )

    session_trades: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of trades in this session"
    )

    session_profit: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        default=Decimal('0'),
        nullable=False,
        doc="Total profit/loss for this session"
    )

    # Additional session metrics
    max_concurrent_positions: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Maximum concurrent positions during session"
    )

    # Error tracking
    error_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of errors during session"
    )

    last_error: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Last error message"
    )

    # Relationships
    strategy = relationship(
        "Strategy",
        back_populates="strategy_sessions",
        lazy="select"
    )

    __table_args__ = (
        Index('ix_strategy_session_strategy_time', 'strategy_id', 'started_at'),
        Index('ix_strategy_session_status', 'status'),
    )

    @property
    def duration_seconds(self) -> Optional[int]:
        """Get session duration in seconds."""
        if self.ended_at:
            return int((self.ended_at - self.started_at).total_seconds())
        else:
            return int((datetime.now(timezone.utc) - self.started_at).total_seconds())

    @property
    def session_return(self) -> Optional[Decimal]:
        """Calculate session return percentage."""
        if self.ending_balance is None or self.starting_balance == 0:
            return None
        return ((self.ending_balance - self.starting_balance) / self.starting_balance) * Decimal('100')

    def end_session(self, ending_balance: Decimal, status: str = "COMPLETED") -> None:
        """
        End the strategy session.

        Args:
            ending_balance: Final balance at session end
            status: Final session status
        """
        self.ended_at = datetime.now(timezone.utc)
        self.ending_balance = ending_balance
        self.session_profit = ending_balance - self.starting_balance
        self.status = status

    def record_error(self, error_message: str) -> None:
        """
        Record an error during session execution.

        Args:
            error_message: Error message to record
        """
        self.error_count += 1
        self.last_error = error_message[:500]  # Truncate if too long

    def __repr__(self) -> str:
        return f"<StrategySession(strategy_id='{self.strategy_id}', status='{self.status}')>"


class BacktestResult(BaseModel, PerformanceMixin):
    """
    Model for backtest results.

    Stores comprehensive backtest results and performance metrics
    for strategy validation and optimization.
    """

    __tablename__ = "backtest_results"

    # Strategy relationship
    strategy_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("strategies.id"),
        nullable=False,
        index=True,
        doc="Reference to strategy"
    )

    # Backtest identification
    backtest_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="Backtest name/identifier"
    )

    # Backtest period
    start_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Backtest start date"
    )

    end_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Backtest end date"
    )

    # Backtest parameters
    initial_balance: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=False,
        doc="Starting balance for backtest"
    )

    final_balance: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=False,
        doc="Final balance at backtest end"
    )

    # Performance metrics
    total_return: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=False,
        doc="Total return percentage"
    )

    annualized_return: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=True,
        doc="Annualized return percentage"
    )

    sharpe_ratio: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=True,
        doc="Sharpe ratio"
    )

    sortino_ratio: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=True,
        doc="Sortino ratio"
    )

    calmar_ratio: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=True,
        doc="Calmar ratio"
    )

    # Risk metrics
    volatility: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=True,
        doc="Volatility (standard deviation of returns)"
    )

    var_95: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=True,
        doc="Value at Risk (95% confidence)"
    )

    # Drawdown metrics
    max_drawdown_duration: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Maximum drawdown duration in days"
    )

    recovery_factor: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=True,
        doc="Recovery factor (net profit / max drawdown)"
    )

    # Trade statistics
    consecutive_wins: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Maximum consecutive winning trades"
    )

    consecutive_losses: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Maximum consecutive losing trades"
    )

    # Market exposure
    time_in_market: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=5, scale=2),
        nullable=True,
        doc="Percentage of time with open positions"
    )

    # Backtest quality metrics
    data_quality_score: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=5, scale=2),
        nullable=True,
        doc="Data quality score (0-100)"
    )

    # Additional metadata
    backtest_parameters: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Backtest parameters as JSON string"
    )

    comments: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Additional comments about the backtest"
    )

    # Relationships
    strategy = relationship(
        "Strategy",
        back_populates="backtest_results",
        lazy="select"
    )

    __table_args__ = (
        UniqueConstraint('strategy_id', 'backtest_name', name='uq_backtest_strategy_name'),
        Index('ix_backtest_strategy_date', 'strategy_id', 'start_date', 'end_date'),
        Index('ix_backtest_performance', 'total_return', 'sharpe_ratio'),
    )

    @property
    def backtest_duration_days(self) -> int:
        """Get backtest duration in days."""
        return (self.end_date - self.start_date).days

    @property
    def profit_factor_calculated(self) -> Optional[Decimal]:
        """Calculate profit factor from gross profit and loss."""
        if self.gross_loss == 0:
            return None
        return self.gross_profit / abs(self.gross_loss)

    def calculate_risk_adjusted_return(self) -> Optional[Decimal]:
        """Calculate risk-adjusted return (return / max drawdown)."""
        if self.max_drawdown is None or self.max_drawdown == 0:
            return None
        return self.total_return / self.max_drawdown

    def __repr__(self) -> str:
        return f"<BacktestResult(name='{self.backtest_name}', return={self.total_return}%)>"


class StrategyParameter(BaseModel):
    """
    Model for strategy parameters.

    Stores individual strategy parameters with their values,
    types, and optimization ranges for strategy tuning.
    """

    __tablename__ = "strategy_parameters"

    # Strategy relationship
    strategy_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("strategies.id"),
        nullable=False,
        index=True,
        doc="Reference to strategy"
    )

    # Parameter information
    parameter_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="Parameter name"
    )

    parameter_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="Parameter type (INT, FLOAT, STRING, BOOL)"
    )

    current_value: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Current parameter value (as string)"
    )

    default_value: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Default parameter value (as string)"
    )

    # Optimization ranges
    min_value: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Minimum value for optimization"
    )

    max_value: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Maximum value for optimization"
    )

    step_size: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Step size for optimization"
    )

    # Parameter metadata
    description: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Parameter description"
    )

    is_optimizable: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether parameter can be optimized"
    )

    # Relationships
    strategy = relationship(
        "Strategy",
        lazy="select"
    )

    __table_args__ = (
        UniqueConstraint('strategy_id', 'parameter_name', name='uq_strategy_parameter'),
        Index('ix_strategy_parameter_name', 'parameter_name'),
    )

    def get_typed_value(self):
        """Get parameter value in its proper type."""
        if self.parameter_type == "INT":
            return int(self.current_value)
        elif self.parameter_type == "FLOAT":
            return float(self.current_value)
        elif self.parameter_type == "BOOL":
            return self.current_value.lower() in ('true', '1', 'yes', 'on')
        else:
            return self.current_value

    def set_typed_value(self, value) -> None:
        """Set parameter value from its proper type."""
        self.current_value = str(value)

    def __repr__(self) -> str:
        return f"<StrategyParameter(name='{self.parameter_name}', value='{self.current_value}')>"