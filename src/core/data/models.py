"""
Data models for real-time market data processing.

This module defines the data structures used throughout the framework
for representing market data, events, and processing states.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class MarketEventType(str, Enum):
    """Types of market events."""

    TICK_RECEIVED = "tick_received"
    OHLC_UPDATED = "ohlc_updated"
    SYMBOL_ADDED = "symbol_added"
    SYMBOL_REMOVED = "symbol_removed"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    DATA_GAP_DETECTED = "data_gap_detected"
    TREND_CHANGE = "trend_change"


class DataQuality(str, Enum):
    """Data quality levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SUSPECT = "suspect"
    INVALID = "invalid"


class TrendDirection(str, Enum):
    """Trend directions."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class TickData(BaseModel):
    """
    Real-time tick data model.

    Represents a single price tick with comprehensive metadata
    for performance tracking and quality assessment.
    """

    symbol: str = Field(description="Trading symbol")
    timestamp: datetime = Field(description="Tick timestamp (server time)")
    bid: Decimal = Field(description="Bid price")
    ask: Decimal = Field(description="Ask price")
    volume: int = Field(default=0, description="Tick volume")

    # Metadata
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When tick was received by framework"
    )
    sequence_id: Optional[int] = Field(
        default=None,
        description="Sequence ID from MT5"
    )
    source_account: Optional[str] = Field(
        default=None,
        description="Source account name"
    )

    # Quality metrics
    latency_ms: Optional[float] = Field(
        default=None,
        description="Processing latency in milliseconds"
    )
    quality: DataQuality = Field(
        default=DataQuality.HIGH,
        description="Data quality assessment"
    )

    @field_validator("bid", "ask", mode='before')
    def validate_prices(cls, v: Union[float, Decimal, str]) -> Decimal:
        """Convert and validate prices to Decimal."""
        if isinstance(v, (float, str)):
            return Decimal(str(v))
        return v

    @field_validator("timestamp", "received_at")
    def validate_timestamps(cls, v: datetime) -> datetime:
        """Ensure timestamps are timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    def calculate_latency(self) -> float:
        """Calculate processing latency in milliseconds."""
        delta = self.received_at - self.timestamp
        return delta.total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid": float(self.bid),
            "ask": float(self.ask),
            "volume": self.volume,
            "received_at": self.received_at.isoformat(),
            "sequence_id": self.sequence_id,
            "source_account": self.source_account,
            "latency_ms": self.latency_ms,
            "quality": self.quality,
            "mid_price": float(self.mid_price),
            "spread": float(self.spread)
        }

    # Pydantic v2 configuration moved to model_config

class OHLCData(BaseModel):
    """
    OHLC (Open, High, Low, Close) candlestick data model.

    Represents aggregated price data over a specific timeframe
    with volume and trend analysis information.
    """

    symbol: str = Field(description="Trading symbol")
    timeframe: str = Field(description="Timeframe (M1, M5, H1, etc.)")
    timestamp: datetime = Field(description="Bar timestamp (open time)")
    open: Decimal = Field(description="Opening price")
    high: Decimal = Field(description="Highest price")
    low: Decimal = Field(description="Lowest price")
    close: Decimal = Field(description="Closing price")
    volume: int = Field(default=0, description="Volume")
    tick_count: int = Field(default=0, description="Number of ticks")

    # Additional metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When OHLC was created"
    )
    is_complete: bool = Field(
        default=False,
        description="Whether this bar is complete"
    )

    # Quality and validation
    quality: DataQuality = Field(
        default=DataQuality.HIGH,
        description="Data quality assessment"
    )
    data_gaps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detected data gaps"
    )

    @field_validator("open", "high", "low", "close", mode='before')
    def validate_prices(cls, v: Union[float, Decimal, str]) -> Decimal:
        """Convert and validate prices to Decimal."""
        if isinstance(v, (float, str)):
            return Decimal(str(v))
        return v

    @field_validator("timestamp", "created_at")
    def validate_timestamps(cls, v: datetime) -> datetime:
        """Ensure timestamps are timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @property
    def typical_price(self) -> Decimal:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3

    @property
    def weighted_price(self) -> Decimal:
        """Calculate weighted price (OHLC/4)."""
        return (self.open + self.high + self.low + self.close) / 4

    @property
    def range_price(self) -> Decimal:
        """Calculate price range (High - Low)."""
        return self.high - self.low

    @property
    def body_size(self) -> Decimal:
        """Calculate candlestick body size."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """Check if candlestick is bullish."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if candlestick is bearish."""
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        """Check if candlestick is a doji (small body)."""
        body_size = float(self.body_size)
        range_size = float(self.range_price)
        return range_size > 0 and (body_size / range_size) < 0.1

    def update_from_tick(self, tick: TickData) -> None:
        """Update OHLC data from a new tick."""
        price = tick.mid_price

        if self.tick_count == 0:
            # First tick for this bar
            self.open = price
            self.high = price
            self.low = price
            self.close = price
        else:
            # Update high and low
            if price > self.high:
                self.high = price
            if price < self.low:
                self.low = price

            # Always update close
            self.close = price

        self.volume += tick.volume
        self.tick_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": self.volume,
            "tick_count": self.tick_count,
            "created_at": self.created_at.isoformat(),
            "is_complete": self.is_complete,
            "quality": self.quality,
            "data_gaps": self.data_gaps,
            "typical_price": float(self.typical_price),
            "weighted_price": float(self.weighted_price),
            "range_price": float(self.range_price),
            "body_size": float(self.body_size),
            "is_bullish": self.is_bullish,
            "is_bearish": self.is_bearish,
            "is_doji": self.is_doji
        }

    # Pydantic v2 configuration moved to model_config

class MarketEvent(BaseModel):
    """
    Market event model for real-time notifications.

    Represents various market events that occur during trading
    sessions and data processing.
    """

    event_id: str = Field(description="Unique event ID")
    event_type: MarketEventType = Field(description="Type of market event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    symbol: Optional[str] = Field(default=None, description="Related symbol")
    account: Optional[str] = Field(default=None, description="Related account")

    # Event data
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data"
    )

    # Metadata
    source: str = Field(
        default="framework",
        description="Event source"
    )
    severity: str = Field(
        default="info",
        description="Event severity (info, warning, error, critical)"
    )

    @field_validator("timestamp")
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "account": self.account,
            "data": self.data,
            "source": self.source,
            "severity": self.severity
        }

    # Pydantic v2 configuration moved to model_config

class ProcessingState(BaseModel):
    """
    Processing state information for tracking data flow.

    Used to monitor and control data processing pipelines
    and maintain state across restarts.
    """

    component_id: str = Field(description="Component identifier")
    state: str = Field(description="Current state")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="State timestamp"
    )

    # Metrics
    items_processed: int = Field(default=0, description="Items processed")
    errors_count: int = Field(default=0, description="Number of errors")
    last_error: Optional[str] = Field(default=None, description="Last error message")

    # Performance metrics
    throughput_per_second: float = Field(default=0.0, description="Items per second")
    average_latency_ms: float = Field(default=0.0, description="Average latency")
    memory_usage_mb: float = Field(default=0.0, description="Memory usage")

    # Configuration
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Component configuration"
    )

    @field_validator("timestamp")
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "component_id": self.component_id,
            "state": self.state,
            "timestamp": self.timestamp.isoformat(),
            "items_processed": self.items_processed,
            "errors_count": self.errors_count,
            "last_error": self.last_error,
            "throughput_per_second": self.throughput_per_second,
            "average_latency_ms": self.average_latency_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "config": self.config
        }

    # Pydantic v2 configuration moved to model_config

class TrendAnalysis(BaseModel):
    """
    Trend analysis result model.

    Contains trend analysis results from MACD and other
    technical indicators.
    """

    symbol: str = Field(description="Trading symbol")
    timeframe: str = Field(description="Analysis timeframe")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Trend information
    trend: TrendDirection = Field(description="Current trend direction")
    strength: float = Field(
        description="Trend strength (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    confidence: float = Field(
        description="Analysis confidence (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

    # MACD values
    macd_line: Optional[float] = Field(default=None, description="MACD line value")
    signal_line: Optional[float] = Field(default=None, description="Signal line value")
    histogram: Optional[float] = Field(default=None, description="MACD histogram value")

    # Additional indicators
    indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional indicator values"
    )

    # Signals
    buy_signal: bool = Field(default=False, description="Buy signal detected")
    sell_signal: bool = Field(default=False, description="Sell signal detected")

    @field_validator("timestamp")
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "trend": self.trend,
            "strength": self.strength,
            "confidence": self.confidence,
            "macd_line": self.macd_line,
            "signal_line": self.signal_line,
            "histogram": self.histogram,
            "indicators": self.indicators,
            "buy_signal": self.buy_signal,
            "sell_signal": self.sell_signal
        }

