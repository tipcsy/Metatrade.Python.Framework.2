"""
Symbol management data models.

This module defines data structures for symbol information,
market sessions, and symbol groups.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SymbolStatus(str, Enum):
    """Symbol status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELISTED = "delisted"
    UNKNOWN = "unknown"


class SymbolType(str, Enum):
    """Symbol type enumeration."""

    FOREX = "forex"
    STOCK = "stock"
    INDEX = "index"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    BOND = "bond"
    OPTION = "option"
    FUTURE = "future"
    CFD = "cfd"
    UNKNOWN = "unknown"


class MarketSession(BaseModel):
    """Market session information."""

    name: str = Field(description="Session name")
    start_time: time = Field(description="Session start time")
    end_time: time = Field(description="Session end time")
    timezone: str = Field(description="Session timezone")
    days_of_week: List[int] = Field(
        description="Active days (0=Monday, 6=Sunday)"
    )

    def is_active(self, current_time: datetime = None) -> bool:
        """Check if session is currently active."""
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Check day of week
        weekday = current_time.weekday()
        if weekday not in self.days_of_week:
            return False

        # Check time range
        current_time_only = current_time.time()

        if self.start_time <= self.end_time:
            # Same day session
            return self.start_time <= current_time_only <= self.end_time
        else:
            # Overnight session
            return current_time_only >= self.start_time or current_time_only <= self.end_time

    # Pydantic v2 configuration moved to model_config

class SymbolInfo(BaseModel):
    """Comprehensive symbol information."""

    # Basic identification
    symbol: str = Field(description="Symbol name")
    description: str = Field(default="", description="Symbol description")
    base_currency: Optional[str] = Field(default=None, description="Base currency")
    quote_currency: Optional[str] = Field(default=None, description="Quote currency")

    # Classification
    symbol_type: SymbolType = Field(
        default=SymbolType.UNKNOWN,
        description="Symbol type"
    )
    category: Optional[str] = Field(default=None, description="Symbol category")
    sector: Optional[str] = Field(default=None, description="Economic sector")
    exchange: Optional[str] = Field(default=None, description="Exchange name")

    # Trading parameters
    tick_size: Optional[float] = Field(default=None, description="Minimum tick size")
    tick_value: Optional[float] = Field(default=None, description="Tick value")
    contract_size: Optional[float] = Field(default=None, description="Contract size")
    margin_required: Optional[float] = Field(default=None, description="Margin required")

    # Price information
    bid: Optional[float] = Field(default=None, description="Current bid price")
    ask: Optional[float] = Field(default=None, description="Current ask price")
    last_price: Optional[float] = Field(default=None, description="Last trade price")
    high_24h: Optional[float] = Field(default=None, description="24h high")
    low_24h: Optional[float] = Field(default=None, description="24h low")

    # Volume information
    volume_24h: Optional[int] = Field(default=None, description="24h volume")
    volume_today: Optional[int] = Field(default=None, description="Today's volume")

    # Status and availability
    status: SymbolStatus = Field(
        default=SymbolStatus.UNKNOWN,
        description="Symbol status"
    )
    is_tradable: bool = Field(default=True, description="Is symbol tradable")
    is_visible: bool = Field(default=True, description="Is symbol visible in UI")

    # Market sessions
    trading_sessions: List[MarketSession] = Field(
        default_factory=list,
        description="Trading sessions"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When symbol was added"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time"
    )
    last_quote_time: Optional[datetime] = Field(
        default=None,
        description="Last quote update time"
    )

    # MT5 specific
    mt5_symbol: Optional[str] = Field(
        default=None,
        description="MT5 symbol name (if different)"
    )
    mt5_digits: Optional[int] = Field(default=None, description="MT5 digits")
    mt5_point: Optional[float] = Field(default=None, description="MT5 point value")

    # Custom properties
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom symbol properties"
    )

    @field_validator("created_at", "updated_at", "last_quote_time", mode="before")
    @classmethod
    def validate_timestamps(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure timestamps are timezone-aware."""
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    def is_market_open(self, current_time: datetime = None) -> bool:
        """Check if market is currently open for this symbol."""
        if not self.trading_sessions:
            return True  # Assume 24/7 if no sessions defined

        return any(
            session.is_active(current_time)
            for session in self.trading_sessions
        )

    def update_quote(
        self,
        bid: float = None,
        ask: float = None,
        last_price: float = None,
        volume: int = None
    ) -> None:
        """Update quote information."""
        if bid is not None:
            self.bid = bid
        if ask is not None:
            self.ask = ask
        if last_price is not None:
            self.last_price = last_price
        if volume is not None:
            self.volume_today = volume

        self.last_quote_time = datetime.now(timezone.utc)
        self.updated_at = self.last_quote_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "description": self.description,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "symbol_type": self.symbol_type.value,
            "category": self.category,
            "sector": self.sector,
            "exchange": self.exchange,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "contract_size": self.contract_size,
            "margin_required": self.margin_required,
            "bid": self.bid,
            "ask": self.ask,
            "last_price": self.last_price,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "volume_24h": self.volume_24h,
            "volume_today": self.volume_today,
            "status": self.status.value,
            "is_tradable": self.is_tradable,
            "is_visible": self.is_visible,
            "spread": self.spread,
            "mid_price": self.mid_price,
            "is_market_open": self.is_market_open(),
            "trading_sessions": [
                {
                    "name": session.name,
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat(),
                    "timezone": session.timezone,
                    "days_of_week": session.days_of_week
                }
                for session in self.trading_sessions
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_quote_time": self.last_quote_time.isoformat() if self.last_quote_time else None,
            "mt5_symbol": self.mt5_symbol,
            "mt5_digits": self.mt5_digits,
            "mt5_point": self.mt5_point,
            "properties": self.properties
        }

    # Pydantic v2 configuration moved to model_config

class SymbolGroup(BaseModel):
    """Symbol group for organizing symbols."""

    group_id: str = Field(description="Group identifier")
    name: str = Field(description="Group name")
    description: str = Field(default="", description="Group description")
    symbols: List[str] = Field(
        default_factory=list,
        description="List of symbols in group"
    )

    # Group properties
    is_active: bool = Field(default=True, description="Is group active")
    priority: int = Field(default=0, description="Group priority")
    color: Optional[str] = Field(default=None, description="Display color")
    tags: List[str] = Field(default_factory=list, description="Group tags")

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time"
    )

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def validate_timestamps(cls, v: datetime) -> datetime:
        """Ensure timestamps are timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def add_symbol(self, symbol: str) -> bool:
        """Add symbol to group."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from group."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def contains_symbol(self, symbol: str) -> bool:
        """Check if group contains symbol."""
        return symbol in self.symbols

    def symbol_count(self) -> int:
        """Get number of symbols in group."""
        return len(self.symbols)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "group_id": self.group_id,
            "name": self.name,
            "description": self.description,
            "symbols": self.symbols,
            "is_active": self.is_active,
            "priority": self.priority,
            "color": self.color,
            "tags": self.tags,
            "symbol_count": self.symbol_count(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    # Pydantic v2 configuration moved to model_config

class SymbolStats(BaseModel):
    """Symbol statistics and metrics."""

    symbol: str = Field(description="Symbol name")

    # Price statistics
    price_change_24h: Optional[float] = Field(default=None, description="24h price change")
    price_change_percent_24h: Optional[float] = Field(default=None, description="24h % change")
    volatility_24h: Optional[float] = Field(default=None, description="24h volatility")

    # Volume statistics
    volume_change_24h: Optional[float] = Field(default=None, description="24h volume change")
    avg_volume_30d: Optional[float] = Field(default=None, description="30d average volume")

    # Trading statistics
    trades_count_24h: Optional[int] = Field(default=None, description="24h trade count")
    avg_spread_24h: Optional[float] = Field(default=None, description="24h average spread")

    # Data quality metrics
    data_completeness: float = Field(default=100.0, description="Data completeness %")
    last_update_lag_seconds: Optional[float] = Field(default=None, description="Update lag")

    # Calculated at
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When stats were calculated"
    )

    @field_validator("calculated_at", mode="before")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "price_change_24h": self.price_change_24h,
            "price_change_percent_24h": self.price_change_percent_24h,
            "volatility_24h": self.volatility_24h,
            "volume_change_24h": self.volume_change_24h,
            "avg_volume_30d": self.avg_volume_30d,
            "trades_count_24h": self.trades_count_24h,
            "avg_spread_24h": self.avg_spread_24h,
            "data_completeness": self.data_completeness,
            "last_update_lag_seconds": self.last_update_lag_seconds,
            "calculated_at": self.calculated_at.isoformat()
        }

