"""
Symbol and instrument models for the MetaTrader Python Framework.

This module defines database models for trading symbols, instruments, and their
related metadata including specifications, contract details, and market information.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    DECIMAL,
    Boolean,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models.base import BaseModel
from src.database.models.mixins import ConfigurationMixin


class SymbolGroup(BaseModel, ConfigurationMixin):
    """
    Model for symbol groups (e.g., Major Forex, Cryptocurrencies, Indices).

    Groups symbols by market type, asset class, or trading characteristics
    for organization and risk management purposes.
    """

    __tablename__ = "symbol_groups"

    # Group classification
    group_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Type of symbol group (FOREX, CRYPTO, STOCK, COMMODITY, INDEX)"
    )

    # Display order for UI
    display_order: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Display order in UI"
    )

    # Market session information
    market_open_hour: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Market opening hour (UTC)"
    )

    market_close_hour: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Market closing hour (UTC)"
    )

    # Relationships
    symbols = relationship(
        "Symbol",
        back_populates="symbol_group",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    __table_args__ = (
        UniqueConstraint('name', name='uq_symbol_group_name'),
        Index('ix_symbol_group_type_active', 'group_type', 'is_active'),
    )

    def __repr__(self) -> str:
        return f"<SymbolGroup(name='{self.name}', type='{self.group_type}')>"


class Symbol(BaseModel):
    """
    Model for trading symbols/instruments.

    Represents a tradeable instrument with all its specifications,
    contract details, and trading parameters.
    """

    __tablename__ = "symbols"

    # Basic symbol information
    symbol: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
        doc="Symbol identifier (e.g., EURUSD, BTCUSD)"
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="Full name of the symbol"
    )

    # Symbol group relationship
    symbol_group_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
        index=True,
        doc="Reference to symbol group"
    )

    # Market information
    market: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Market identifier (FOREX, CRYPTO, STOCK, etc.)"
    )

    exchange: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        doc="Exchange name"
    )

    base_currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        doc="Base currency (e.g., EUR in EURUSD)"
    )

    quote_currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        doc="Quote currency (e.g., USD in EURUSD)"
    )

    # Trading specifications
    digits: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
        doc="Number of decimal places for price quotes"
    )

    point: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=10, scale=8),
        nullable=False,
        doc="Point value (smallest price change)"
    )

    tick_size: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=10, scale=8),
        nullable=False,
        doc="Minimum price change step"
    )

    tick_value: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=10, scale=2),
        nullable=False,
        doc="Value of one tick in account currency"
    )

    # Contract specifications
    contract_size: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        default=Decimal('100000'),
        nullable=False,
        doc="Standard contract size"
    )

    min_lot: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=10, scale=8),
        default=Decimal('0.01'),
        nullable=False,
        doc="Minimum lot size"
    )

    max_lot: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=10, scale=8),
        default=Decimal('100'),
        nullable=False,
        doc="Maximum lot size"
    )

    lot_step: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=10, scale=8),
        default=Decimal('0.01'),
        nullable=False,
        doc="Lot size step"
    )

    # Margin requirements
    margin_initial: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=True,
        doc="Initial margin requirement percentage"
    )

    margin_maintenance: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=4),
        nullable=True,
        doc="Maintenance margin requirement percentage"
    )

    # Trading costs
    spread_typical: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=5),
        nullable=True,
        doc="Typical spread in pips/points"
    )

    commission: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=2),
        default=Decimal('0'),
        nullable=True,
        doc="Commission per lot"
    )

    swap_long: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=2),
        default=Decimal('0'),
        nullable=True,
        doc="Swap rate for long positions"
    )

    swap_short: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=2),
        default=Decimal('0'),
        nullable=True,
        doc="Swap rate for short positions"
    )

    # Trading status
    trade_mode: Mapped[str] = mapped_column(
        String(20),
        default="FULL",
        nullable=False,
        doc="Trading mode (FULL, LONG_ONLY, SHORT_ONLY, CLOSE_ONLY, DISABLED)"
    )

    is_tradeable: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        doc="Whether the symbol is currently tradeable"
    )

    # Session information
    session_deals: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of deals in current session"
    )

    session_buy_orders: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of buy orders in current session"
    )

    session_sell_orders: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of sell orders in current session"
    )

    # Last quote information
    last_bid: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Last bid price"
    )

    last_ask: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Last ask price"
    )

    last_volume: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Last volume"
    )

    # Relationships
    symbol_group = relationship(
        "SymbolGroup",
        back_populates="symbols",
        lazy="select"
    )

    # Market data relationships (will be defined in market_data.py)
    market_data = relationship(
        "MarketData",
        back_populates="symbol",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    tick_data = relationship(
        "TickData",
        back_populates="symbol",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    # Trading relationships (will be defined in trading.py)
    orders = relationship(
        "Order",
        back_populates="symbol",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    positions = relationship(
        "Position",
        back_populates="symbol",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    __table_args__ = (
        UniqueConstraint('symbol', name='uq_symbol_symbol'),
        Index('ix_symbol_market_tradeable', 'market', 'is_tradeable'),
        Index('ix_symbol_base_quote', 'base_currency', 'quote_currency'),
        Index('ix_symbol_group_tradeable', 'symbol_group_id', 'is_tradeable'),
    )

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate current spread."""
        if self.last_bid is None or self.last_ask is None:
            return None
        return self.last_ask - self.last_bid

    @property
    def spread_pips(self) -> Optional[Decimal]:
        """Calculate current spread in pips."""
        spread = self.spread
        if spread is None:
            return None
        return spread / self.point

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price (bid + ask) / 2."""
        if self.last_bid is None or self.last_ask is None:
            return None
        return (self.last_bid + self.last_ask) / Decimal('2')

    def update_quote(
        self,
        bid: Decimal,
        ask: Decimal,
        volume: Optional[Decimal] = None
    ) -> None:
        """
        Update the symbol's last quote information.

        Args:
            bid: Current bid price
            ask: Current ask price
            volume: Current volume
        """
        self.last_bid = bid
        self.last_ask = ask
        if volume is not None:
            self.last_volume = volume

    def calculate_pip_value(self, lot_size: Decimal = None) -> Decimal:
        """
        Calculate pip value for the given lot size.

        Args:
            lot_size: Lot size (defaults to 1.0)

        Returns:
            Pip value in account currency
        """
        if lot_size is None:
            lot_size = Decimal('1.0')

        return self.tick_value * lot_size * (self.point / self.tick_size)

    def calculate_margin_required(self, lot_size: Decimal) -> Optional[Decimal]:
        """
        Calculate margin required for the given lot size.

        Args:
            lot_size: Lot size

        Returns:
            Margin required in account currency
        """
        if self.margin_initial is None or self.last_bid is None:
            return None

        notional_value = lot_size * self.contract_size * self.last_bid
        return notional_value * (self.margin_initial / Decimal('100'))

    def is_market_open(self) -> bool:
        """
        Check if the market is currently open for this symbol.

        Returns:
            True if market is open, False otherwise
        """
        # This is a simplified implementation
        # In a real application, you would check against market hours
        return self.is_tradeable and self.trade_mode == "FULL"

    def validate_lot_size(self, lot_size: Decimal) -> bool:
        """
        Validate if the given lot size is within allowed limits.

        Args:
            lot_size: Lot size to validate

        Returns:
            True if lot size is valid, False otherwise
        """
        if lot_size < self.min_lot or lot_size > self.max_lot:
            return False

        # Check if lot size is a multiple of lot step
        remainder = (lot_size - self.min_lot) % self.lot_step
        return remainder == 0

    def __repr__(self) -> str:
        return f"<Symbol(symbol='{self.symbol}', name='{self.name}')>"


class SymbolSession(BaseModel):
    """
    Model for symbol trading sessions.

    Defines trading sessions for symbols including session times,
    trading modes, and session-specific parameters.
    """

    __tablename__ = "symbol_sessions"

    # Symbol reference
    symbol_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        doc="Reference to symbol"
    )

    # Session information
    session_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Session name (e.g., LONDON, NEW_YORK, TOKYO)"
    )

    # Session times (in minutes from Sunday 00:00 UTC)
    session_start: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Session start time in minutes from Sunday 00:00 UTC"
    )

    session_end: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Session end time in minutes from Sunday 00:00 UTC"
    )

    # Session parameters
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether the session is active"
    )

    trade_mode: Mapped[str] = mapped_column(
        String(20),
        default="FULL",
        nullable=False,
        doc="Trading mode during this session"
    )

    # Relationships
    symbol = relationship(
        "Symbol",
        lazy="select"
    )

    __table_args__ = (
        UniqueConstraint('symbol_id', 'session_name', name='uq_symbol_session'),
        Index('ix_symbol_session_times', 'session_start', 'session_end'),
    )

    def __repr__(self) -> str:
        return f"<SymbolSession(symbol_id='{self.symbol_id}', session='{self.session_name}')>"