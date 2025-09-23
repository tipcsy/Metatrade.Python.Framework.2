"""
Market data models for the MetaTrader Python Framework.

This module defines database models for market data including OHLCV bars,
tick data, and related market information with optimizations for time series storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    DECIMAL,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models.base import BaseModel
from src.database.models.mixins import TimeSeriesMixin, OHLCVMixin, FinancialMixin


class MarketData(BaseModel, TimeSeriesMixin, OHLCVMixin):
    """
    Model for OHLCV market data bars.

    Stores historical and real-time OHLCV (Open, High, Low, Close, Volume) data
    for different timeframes with optimizations for time series queries.
    """

    __tablename__ = "market_data"

    # Symbol relationship
    symbol_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        doc="Reference to symbol"
    )

    # Timeframe information
    timeframe: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        index=True,
        doc="Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)"
    )

    # Time information
    time_close: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Bar close time (for some data sources)"
    )

    # Additional market data
    tick_volume: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of ticks in this bar"
    )

    real_volume: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Real volume (if available)"
    )

    # Spread information
    spread: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        doc="Spread in points"
    )

    # Data quality flags
    is_complete: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether this bar is complete (not being updated)"
    )

    data_source: Mapped[str] = mapped_column(
        String(50),
        default="MT5",
        nullable=False,
        doc="Data source identifier"
    )

    # Relationships
    symbol = relationship(
        "Symbol",
        back_populates="market_data",
        lazy="select"
    )

    __table_args__ = (
        UniqueConstraint('symbol_id', 'timeframe', 'timestamp', name='uq_market_data'),
        Index('ix_market_data_symbol_timeframe', 'symbol_id', 'timeframe'),
        Index('ix_market_data_timestamp_desc', 'timestamp', postgresql_using='btree'),
        Index('ix_market_data_symbol_time_tf', 'symbol_id', 'timestamp', 'timeframe'),
        Index('ix_market_data_complete', 'is_complete'),
        # Partitioning hint for PostgreSQL (can be implemented later)
        # {'postgresql_partition_by': 'RANGE (timestamp)'}
    )

    @property
    def timeframe_minutes(self) -> int:
        """Get timeframe in minutes."""
        timeframe_map = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440,
            'W1': 10080, 'MN1': 43200
        }
        return timeframe_map.get(self.timeframe, 1)

    @property
    def bar_duration_seconds(self) -> int:
        """Get bar duration in seconds."""
        return self.timeframe_minutes * 60

    def is_gap_up(self) -> bool:
        """Check if this bar gapped up from previous close."""
        # This would require querying the previous bar
        # Implementation depends on specific use case
        return False

    def is_gap_down(self) -> bool:
        """Check if this bar gapped down from previous close."""
        # This would require querying the previous bar
        # Implementation depends on specific use case
        return False

    def get_body_size(self) -> Decimal:
        """Get the size of the bar body (close - open)."""
        return abs(self.close_price - self.open_price)

    def get_upper_shadow(self) -> Decimal:
        """Get the upper shadow/wick length."""
        return self.high_price - max(self.open_price, self.close_price)

    def get_lower_shadow(self) -> Decimal:
        """Get the lower shadow/wick length."""
        return min(self.open_price, self.close_price) - self.low_price

    def is_bullish(self) -> bool:
        """Check if this is a bullish bar (close > open)."""
        return self.close_price > self.open_price

    def is_bearish(self) -> bool:
        """Check if this is a bearish bar (close < open)."""
        return self.close_price < self.open_price

    def is_doji(self, threshold: Decimal = Decimal('0.001')) -> bool:
        """
        Check if this is a doji bar (open ≈ close).

        Args:
            threshold: Threshold for considering open and close equal

        Returns:
            True if bar is a doji
        """
        return abs(self.close_price - self.open_price) <= threshold

    def __repr__(self) -> str:
        return f"<MarketData(symbol_id='{self.symbol_id}', timeframe='{self.timeframe}', timestamp='{self.timestamp}')>"


class TickData(BaseModel, TimeSeriesMixin, FinancialMixin):
    """
    Model for tick-level market data.

    Stores individual price ticks with bid/ask information for
    high-frequency analysis and backtesting.
    """

    __tablename__ = "tick_data"

    # Symbol relationship
    symbol_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        doc="Reference to symbol"
    )

    # Tick information
    bid: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Bid price"
    )

    ask: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Ask price"
    )

    last: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Last trade price"
    )

    volume_bid: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Volume at bid"
    )

    volume_ask: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Volume at ask"
    )

    # Tick flags
    tick_flags: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        doc="Tick flags (MT5 tick flags)"
    )

    # Data source
    data_source: Mapped[str] = mapped_column(
        String(50),
        default="MT5",
        nullable=False,
        doc="Data source identifier"
    )

    # Relationships
    symbol = relationship(
        "Symbol",
        back_populates="tick_data",
        lazy="select"
    )

    __table_args__ = (
        Index('ix_tick_data_symbol_timestamp', 'symbol_id', 'timestamp'),
        Index('ix_tick_data_timestamp', 'timestamp'),
        # Consider partitioning for large tick data volumes
        # {'postgresql_partition_by': 'RANGE (timestamp)'}
    )

    @property
    def spread(self) -> Decimal:
        """Calculate spread (ask - bid)."""
        return self.ask - self.bid

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price ((bid + ask) / 2)."""
        return (self.bid + self.ask) / Decimal('2')

    def is_valid_tick(self) -> bool:
        """Validate tick data integrity."""
        # Basic validation
        if self.bid <= 0 or self.ask <= 0:
            return False

        # Ask should be >= bid
        if self.ask < self.bid:
            return False

        # Volume should be non-negative if provided
        if self.volume_bid is not None and self.volume_bid < 0:
            return False
        if self.volume_ask is not None and self.volume_ask < 0:
            return False

        return True

    def __repr__(self) -> str:
        return f"<TickData(symbol_id='{self.symbol_id}', timestamp='{self.timestamp}', bid={self.bid}, ask={self.ask})>"


class MarketDepth(BaseModel, TimeSeriesMixin):
    """
    Model for market depth (Level II) data.

    Stores order book information showing bid/ask levels
    and their corresponding volumes.
    """

    __tablename__ = "market_depth"

    # Symbol relationship
    symbol_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        doc="Reference to symbol"
    )

    # Depth level
    level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Depth level (0 = best bid/ask, 1 = second best, etc.)"
    )

    # Side
    side: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        doc="Order side (BID or ASK)"
    )

    # Price and volume
    price: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Price level"
    )

    volume: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Volume at this price level"
    )

    # Data source
    data_source: Mapped[str] = mapped_column(
        String(50),
        default="MT5",
        nullable=False,
        doc="Data source identifier"
    )

    # Relationships
    symbol = relationship(
        "Symbol",
        lazy="select"
    )

    __table_args__ = (
        UniqueConstraint('symbol_id', 'timestamp', 'level', 'side', name='uq_market_depth'),
        Index('ix_market_depth_symbol_timestamp', 'symbol_id', 'timestamp'),
        Index('ix_market_depth_level_side', 'level', 'side'),
    )

    def __repr__(self) -> str:
        return f"<MarketDepth(symbol_id='{self.symbol_id}', level={self.level}, side='{self.side}')>"


class MarketSession(BaseModel):
    """
    Model for market trading sessions.

    Records information about market sessions including
    session statistics and trading activity.
    """

    __tablename__ = "market_sessions"

    # Session identification
    session_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Session date"
    )

    session_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        doc="Session type (ASIAN, EUROPEAN, AMERICAN, OVERNIGHT)"
    )

    # Session times
    session_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="Session start time"
    )

    session_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="Session end time"
    )

    # Session statistics
    total_volume: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Total volume traded in session"
    )

    total_trades: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        doc="Total number of trades in session"
    )

    # Session status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether session is currently active"
    )

    __table_args__ = (
        UniqueConstraint('session_date', 'session_type', name='uq_market_session'),
        Index('ix_market_session_date_type', 'session_date', 'session_type'),
        Index('ix_market_session_active', 'is_active'),
    )

    def __repr__(self) -> str:
        return f"<MarketSession(date='{self.session_date.date()}', type='{self.session_type}')>"


class DataQuality(BaseModel):
    """
    Model for tracking data quality metrics.

    Monitors data completeness, accuracy, and other quality metrics
    for market data feeds and sources.
    """

    __tablename__ = "data_quality"

    # Symbol and timeframe
    symbol_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
        index=True,
        doc="Reference to symbol (null for global metrics)"
    )

    timeframe: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        doc="Timeframe (null for tick data or global metrics)"
    )

    # Time period
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Start of measurement period"
    )

    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="End of measurement period"
    )

    # Quality metrics
    total_expected_bars: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        doc="Expected number of bars in period"
    )

    total_received_bars: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        doc="Actual number of bars received"
    )

    missing_bars: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of missing bars"
    )

    duplicate_bars: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of duplicate bars detected"
    )

    invalid_bars: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of invalid bars (failed validation)"
    )

    # Completeness percentage
    completeness_ratio: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=5, scale=2),
        nullable=True,
        doc="Data completeness ratio (0-100%)"
    )

    # Data source
    data_source: Mapped[str] = mapped_column(
        String(50),
        default="MT5",
        nullable=False,
        doc="Data source identifier"
    )

    # Quality status
    quality_status: Mapped[str] = mapped_column(
        String(20),
        default="GOOD",
        nullable=False,
        doc="Overall quality status (EXCELLENT, GOOD, FAIR, POOR)"
    )

    # Relationships
    symbol = relationship(
        "Symbol",
        lazy="select"
    )

    __table_args__ = (
        Index('ix_data_quality_symbol_period', 'symbol_id', 'period_start', 'period_end'),
        Index('ix_data_quality_status', 'quality_status'),
    )

    def calculate_quality_score(self) -> Optional[Decimal]:
        """Calculate overall quality score (0-100)."""
        if self.completeness_ratio is None:
            return None

        score = self.completeness_ratio

        # Adjust for data issues
        if self.total_received_bars and self.total_received_bars > 0:
            duplicate_penalty = (self.duplicate_bars or 0) / self.total_received_bars * 10
            invalid_penalty = (self.invalid_bars or 0) / self.total_received_bars * 20
            score = max(Decimal('0'), score - duplicate_penalty - invalid_penalty)

        return min(Decimal('100'), score)

    def __repr__(self) -> str:
        return f"<DataQuality(symbol_id='{self.symbol_id}', period='{self.period_start.date()}', status='{self.quality_status}')>"