"""
Advanced mixin classes for the MetaTrader Python Framework database models.

This module provides specialized mixin classes for common trading and financial
data patterns including time series data, financial metrics, and performance tracking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional, Type

from sqlalchemy import (
    DECIMAL,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column

from src.core.logging import get_logger

logger = get_logger(__name__)


class TimeSeriesMixin:
    """
    Mixin for time series data models.

    Provides timestamp indexing and time-based querying capabilities
    optimized for financial time series data.
    """

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Timestamp for the time series data point (UTC)"
    )

    @classmethod
    def __table_args__(cls):
        """Add composite indexes for time series queries."""
        return (
            Index(f'ix_{cls.__tablename__}_timestamp_desc', 'timestamp', postgresql_using='btree'),
            Index(f'ix_{cls.__tablename__}_symbol_timestamp', 'symbol_id', 'timestamp'),
        )


class FinancialMixin:
    """
    Mixin for financial data models.

    Provides common financial fields with proper decimal precision
    for accurate monetary calculations.
    """

    # Price fields with 8 decimal places for cryptocurrency support
    price: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Price value with high precision"
    )

    # Volume fields
    volume: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Volume value with high precision"
    )

    # Value fields for monetary amounts
    value: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=True,
        doc="Monetary value (price * volume)"
    )

    @hybrid_property
    def formatted_price(self) -> Optional[str]:
        """Get formatted price string."""
        if self.price is None:
            return None
        return f"{self.price:.8f}".rstrip('0').rstrip('.')

    @hybrid_property
    def formatted_volume(self) -> Optional[str]:
        """Get formatted volume string."""
        if self.volume is None:
            return None
        return f"{self.volume:.8f}".rstrip('0').rstrip('.')


class OHLCVMixin(FinancialMixin):
    """
    Mixin for OHLCV (Open, High, Low, Close, Volume) data.

    Extends FinancialMixin with specific OHLCV fields and validation.
    """

    open_price: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Opening price for the time period"
    )

    high_price: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Highest price for the time period"
    )

    low_price: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Lowest price for the time period"
    )

    close_price: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Closing price for the time period"
    )

    @hybrid_property
    def price_range(self) -> Decimal:
        """Calculate the price range (high - low)."""
        return self.high_price - self.low_price

    @hybrid_property
    def price_change(self) -> Decimal:
        """Calculate the price change (close - open)."""
        return self.close_price - self.open_price

    @hybrid_property
    def price_change_percent(self) -> Decimal:
        """Calculate the percentage price change."""
        if self.open_price == 0:
            return Decimal('0')
        return (self.price_change / self.open_price) * Decimal('100')

    @hybrid_property
    def typical_price(self) -> Decimal:
        """Calculate the typical price (H+L+C)/3."""
        return (self.high_price + self.low_price + self.close_price) / Decimal('3')

    @hybrid_property
    def weighted_price(self) -> Decimal:
        """Calculate the weighted price (O+H+L+C)/4."""
        return (self.open_price + self.high_price + self.low_price + self.close_price) / Decimal('4')

    def validate_ohlcv_data(self) -> bool:
        """
        Validate OHLCV data integrity.

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # High should be >= Low
            if self.high_price < self.low_price:
                logger.error(f"Invalid OHLCV: High ({self.high_price}) < Low ({self.low_price})")
                return False

            # High should be >= Open and Close
            if self.high_price < self.open_price or self.high_price < self.close_price:
                logger.error(f"Invalid OHLCV: High price is not the highest")
                return False

            # Low should be <= Open and Close
            if self.low_price > self.open_price or self.low_price > self.close_price:
                logger.error(f"Invalid OHLCV: Low price is not the lowest")
                return False

            # Volume should be non-negative
            if self.volume is not None and self.volume < 0:
                logger.error(f"Invalid OHLCV: Negative volume ({self.volume})")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating OHLCV data: {e}")
            return False


class TradingMixin:
    """
    Mixin for trading-related models.

    Provides common trading fields and calculations for orders, positions, etc.
    """

    # Trading direction
    direction: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        doc="Trading direction: BUY, SELL, LONG, SHORT"
    )

    # Quantity/Size
    quantity: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Trade quantity/size"
    )

    # Entry price
    entry_price: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Entry price for the trade"
    )

    # Exit price
    exit_price: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Exit price for the trade"
    )

    # Commission/Fees
    commission: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        default=Decimal('0'),
        nullable=True,
        doc="Commission/fees paid for the trade"
    )

    # Swap/Interest
    swap: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        default=Decimal('0'),
        nullable=True,
        doc="Swap/interest charges"
    )

    @hybrid_property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.direction.upper() in ('BUY', 'LONG')

    @hybrid_property
    def is_short(self) -> bool:
        """Check if this is a short position."""
        return self.direction.upper() in ('SELL', 'SHORT')

    @hybrid_property
    def notional_value(self) -> Optional[Decimal]:
        """Calculate notional value (quantity * entry_price)."""
        if self.entry_price is None:
            return None
        return self.quantity * self.entry_price

    @hybrid_property
    def gross_pnl(self) -> Optional[Decimal]:
        """Calculate gross P&L without fees."""
        if self.entry_price is None or self.exit_price is None:
            return None

        if self.is_long:
            return self.quantity * (self.exit_price - self.entry_price)
        else:
            return self.quantity * (self.entry_price - self.exit_price)

    @hybrid_property
    def net_pnl(self) -> Optional[Decimal]:
        """Calculate net P&L including fees and swap."""
        gross = self.gross_pnl
        if gross is None:
            return None

        total_fees = (self.commission or Decimal('0')) + (self.swap or Decimal('0'))
        return gross - total_fees

    @hybrid_property
    def pnl_percent(self) -> Optional[Decimal]:
        """Calculate P&L percentage."""
        notional = self.notional_value
        net_pnl = self.net_pnl

        if notional is None or net_pnl is None or notional == 0:
            return None

        return (net_pnl / notional) * Decimal('100')


class PerformanceMixin:
    """
    Mixin for performance tracking models.

    Provides fields and calculations for tracking trading performance metrics.
    """

    # Performance metrics
    total_trades: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Total number of trades"
    )

    winning_trades: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of winning trades"
    )

    losing_trades: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of losing trades"
    )

    gross_profit: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        default=Decimal('0'),
        nullable=False,
        doc="Total gross profit"
    )

    gross_loss: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        default=Decimal('0'),
        nullable=False,
        doc="Total gross loss"
    )

    max_drawdown: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=True,
        doc="Maximum drawdown percentage"
    )

    max_profit: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=True,
        doc="Maximum single trade profit"
    )

    max_loss: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=True,
        doc="Maximum single trade loss"
    )

    @hybrid_property
    def net_profit(self) -> Decimal:
        """Calculate net profit (gross profit - gross loss)."""
        return self.gross_profit + self.gross_loss  # gross_loss is stored as negative

    @hybrid_property
    def win_rate(self) -> Optional[Decimal]:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return None
        return (Decimal(self.winning_trades) / Decimal(self.total_trades)) * Decimal('100')

    @hybrid_property
    def profit_factor(self) -> Optional[Decimal]:
        """Calculate profit factor (gross profit / abs(gross loss))."""
        if self.gross_loss == 0:
            return None
        return self.gross_profit / abs(self.gross_loss)

    @hybrid_property
    def average_win(self) -> Optional[Decimal]:
        """Calculate average winning trade."""
        if self.winning_trades == 0:
            return None
        return self.gross_profit / Decimal(self.winning_trades)

    @hybrid_property
    def average_loss(self) -> Optional[Decimal]:
        """Calculate average losing trade."""
        if self.losing_trades == 0:
            return None
        return self.gross_loss / Decimal(self.losing_trades)

    def update_performance_metrics(
        self,
        trade_pnl: Decimal,
        current_equity: Optional[Decimal] = None,
        peak_equity: Optional[Decimal] = None
    ) -> None:
        """
        Update performance metrics with a new trade.

        Args:
            trade_pnl: P&L of the completed trade
            current_equity: Current account equity
            peak_equity: Peak account equity (for drawdown calculation)
        """
        self.total_trades += 1

        if trade_pnl > 0:
            self.winning_trades += 1
            self.gross_profit += trade_pnl
            if self.max_profit is None or trade_pnl > self.max_profit:
                self.max_profit = trade_pnl
        else:
            self.losing_trades += 1
            self.gross_loss += trade_pnl  # trade_pnl is negative for losses
            if self.max_loss is None or trade_pnl < self.max_loss:
                self.max_loss = trade_pnl

        # Update drawdown if equity values are provided
        if current_equity is not None and peak_equity is not None and peak_equity > 0:
            drawdown = ((peak_equity - current_equity) / peak_equity) * Decimal('100')
            if self.max_drawdown is None or drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

        logger.debug(
            f"Updated performance metrics: "
            f"Total trades: {self.total_trades}, "
            f"Win rate: {self.win_rate}%, "
            f"Net profit: {self.net_profit}"
        )


class ConfigurationMixin:
    """
    Mixin for configuration-related models.

    Provides fields for storing and validating configuration parameters.
    """

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="Configuration name/identifier"
    )

    description: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Configuration description"
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        doc="Whether the configuration is active"
    )

    config_data: Mapped[Optional[str]] = mapped_column(
        String,  # JSON string for flexible configuration storage
        nullable=True,
        doc="Configuration parameters as JSON string"
    )

    @hybrid_property
    def is_enabled(self) -> bool:
        """Check if configuration is enabled (active and not deleted)."""
        return self.is_active and not self.is_deleted


# Event listeners for data validation
@event.listens_for(OHLCVMixin, 'before_insert', propagate=True)
@event.listens_for(OHLCVMixin, 'before_update', propagate=True)
def validate_ohlcv_before_save(mapper, connection, target):
    """Validate OHLCV data before saving to database."""
    if hasattr(target, 'validate_ohlcv_data'):
        if not target.validate_ohlcv_data():
            raise ValueError("Invalid OHLCV data detected")


@event.listens_for(TradingMixin, 'before_insert', propagate=True)
@event.listens_for(TradingMixin, 'before_update', propagate=True)
def validate_trading_before_save(mapper, connection, target):
    """Validate trading data before saving to database."""
    # Validate direction
    valid_directions = {'BUY', 'SELL', 'LONG', 'SHORT'}
    if target.direction.upper() not in valid_directions:
        raise ValueError(f"Invalid trading direction: {target.direction}")

    # Validate quantity
    if target.quantity <= 0:
        raise ValueError("Quantity must be positive")

    # Validate prices are positive if provided
    for price_field in ['entry_price', 'exit_price']:
        price = getattr(target, price_field, None)
        if price is not None and price <= 0:
            raise ValueError(f"{price_field} must be positive")