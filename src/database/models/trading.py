"""
Trading models for the MetaTrader Python Framework.

This module defines database models for trading operations including
orders, positions, trades, and related trading information.
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
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models.base import BaseModel
from src.database.models.mixins import TradingMixin, FinancialMixin


class Order(BaseModel, TradingMixin):
    """
    Model for trading orders.

    Represents trading orders with all their specifications,
    status tracking, and execution details.
    """

    __tablename__ = "orders"

    # Account and symbol relationships
    account_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("accounts.id"),
        nullable=False,
        index=True,
        doc="Reference to trading account"
    )

    symbol_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("symbols.id"),
        nullable=False,
        index=True,
        doc="Reference to symbol"
    )

    # Position relationship
    position_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("positions.id"),
        nullable=True,
        index=True,
        doc="Reference to position (if order is part of a position)"
    )

    # Order identification
    order_ticket: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        unique=True,
        index=True,
        doc="Broker order ticket number"
    )

    client_order_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        doc="Client-side order identifier"
    )

    # Order type and specifications
    order_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        doc="Order type (MARKET, LIMIT, STOP, STOP_LIMIT)"
    )

    time_in_force: Mapped[str] = mapped_column(
        String(10),
        default="GTC",
        nullable=False,
        doc="Time in force (GTC, IOC, FOK, DAY)"
    )

    # Price levels
    price: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Order price (for limit/stop orders)"
    )

    stop_price: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Stop price (for stop-limit orders)"
    )

    # Risk management
    stop_loss: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Stop loss price"
    )

    take_profit: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Take profit price"
    )

    # Order status and timing
    status: Mapped[str] = mapped_column(
        String(20),
        default="PENDING",
        nullable=False,
        index=True,
        doc="Order status (PENDING, FILLED, PARTIALLY_FILLED, CANCELLED, REJECTED, EXPIRED)"
    )

    filled_quantity: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        default=Decimal('0'),
        nullable=False,
        doc="Quantity filled so far"
    )

    remaining_quantity: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Remaining quantity to fill"
    )

    # Execution details
    avg_fill_price: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Average fill price"
    )

    executed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Execution timestamp"
    )

    # Order validity
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Order expiration time"
    )

    # Additional information
    comment: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Order comment"
    )

    magic_number: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        doc="Expert advisor magic number"
    )

    # Parent order (for OCO, bracket orders)
    parent_order_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
        index=True,
        doc="Parent order ID for related orders"
    )

    # Relationships
    account = relationship(
        "Account",
        back_populates="orders",
        lazy="select"
    )

    symbol = relationship(
        "Symbol",
        back_populates="orders",
        lazy="select"
    )

    position = relationship(
        "Position",
        back_populates="orders",
        lazy="select"
    )

    order_fills = relationship(
        "OrderFill",
        back_populates="order",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    __table_args__ = (
        Index('ix_order_account_status', 'account_id', 'status'),
        Index('ix_order_symbol_type', 'symbol_id', 'order_type'),
        Index('ix_order_status_created', 'status', 'created_at'),
        Index('ix_order_magic_number', 'magic_number'),
    )

    @property
    def is_pending(self) -> bool:
        """Check if order is pending execution."""
        return self.status == "PENDING"

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == "FILLED"

    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return self.status == "PARTIALLY_FILLED"

    @property
    def is_active(self) -> bool:
        """Check if order is active (pending or partially filled)."""
        return self.status in ("PENDING", "PARTIALLY_FILLED")

    @property
    def fill_ratio(self) -> Decimal:
        """Calculate fill ratio (0-1)."""
        if self.quantity == 0:
            return Decimal('0')
        return self.filled_quantity / self.quantity

    def update_fill(self, fill_quantity: Decimal, fill_price: Decimal) -> None:
        """
        Update order with new fill information.

        Args:
            fill_quantity: Quantity filled in this execution
            fill_price: Price of this fill
        """
        # Update filled quantity
        self.filled_quantity += fill_quantity
        self.remaining_quantity = self.quantity - self.filled_quantity

        # Update average fill price
        if self.avg_fill_price is None:
            self.avg_fill_price = fill_price
        else:
            total_value = (self.avg_fill_price * (self.filled_quantity - fill_quantity) +
                         fill_price * fill_quantity)
            self.avg_fill_price = total_value / self.filled_quantity

        # Update status
        if self.remaining_quantity <= 0:
            self.status = "FILLED"
            self.executed_at = datetime.now(timezone.utc)
        elif self.filled_quantity > 0:
            self.status = "PARTIALLY_FILLED"

    def cancel(self, reason: str = "USER_CANCELLED") -> bool:
        """
        Cancel the order.

        Args:
            reason: Cancellation reason

        Returns:
            True if order was cancelled, False if not possible
        """
        if not self.is_active:
            return False

        self.status = "CANCELLED"
        self.comment = f"{self.comment or ''} | Cancelled: {reason}".strip(" |")
        return True

    def __repr__(self) -> str:
        return f"<Order(ticket='{self.order_ticket}', type='{self.order_type}', status='{self.status}')>"


class OrderFill(BaseModel):
    """
    Model for order fill executions.

    Records individual fills/executions for orders, allowing
    for partial fills and detailed execution tracking.
    """

    __tablename__ = "order_fills"

    # Order relationship
    order_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("orders.id"),
        nullable=False,
        index=True,
        doc="Reference to order"
    )

    # Fill identification
    fill_id: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        unique=True,
        index=True,
        doc="Broker fill/execution ID"
    )

    # Fill details
    fill_quantity: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Quantity filled in this execution"
    )

    fill_price: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=False,
        doc="Price of this fill"
    )

    fill_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Fill execution time"
    )

    # Costs
    commission_charged: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=True,
        doc="Commission charged for this fill"
    )

    # Relationships
    order = relationship(
        "Order",
        back_populates="order_fills",
        lazy="select"
    )

    __table_args__ = (
        Index('ix_order_fill_order_time', 'order_id', 'fill_time'),
    )

    def __repr__(self) -> str:
        return f"<OrderFill(order_id='{self.order_id}', quantity={self.fill_quantity}, price={self.fill_price})>"


class Position(BaseModel, TradingMixin):
    """
    Model for trading positions.

    Represents open or closed trading positions with P&L tracking,
    risk management, and position lifecycle management.
    """

    __tablename__ = "positions"

    # Account and symbol relationships
    account_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("accounts.id"),
        nullable=False,
        index=True,
        doc="Reference to trading account"
    )

    symbol_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("symbols.id"),
        nullable=False,
        index=True,
        doc="Reference to symbol"
    )

    # Position identification
    position_ticket: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        unique=True,
        index=True,
        doc="Broker position ticket number"
    )

    # Position status
    status: Mapped[str] = mapped_column(
        String(20),
        default="OPEN",
        nullable=False,
        index=True,
        doc="Position status (OPEN, CLOSED, PARTIAL)"
    )

    # Position timing
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Position opening time"
    )

    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Position closing time"
    )

    # Current price information
    current_price: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Current market price for P&L calculation"
    )

    # Risk management
    stop_loss: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Stop loss price"
    )

    take_profit: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=8),
        nullable=True,
        doc="Take profit price"
    )

    # P&L tracking
    unrealized_pnl: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=True,
        doc="Current unrealized P&L"
    )

    realized_pnl: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=True,
        doc="Realized P&L (for closed positions)"
    )

    # Additional information
    comment: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Position comment"
    )

    magic_number: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        doc="Expert advisor magic number"
    )

    # Relationships
    account = relationship(
        "Account",
        back_populates="positions",
        lazy="select"
    )

    symbol = relationship(
        "Symbol",
        back_populates="positions",
        lazy="select"
    )

    orders = relationship(
        "Order",
        back_populates="position",
        lazy="dynamic"
    )

    trades = relationship(
        "Trade",
        back_populates="position",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    __table_args__ = (
        Index('ix_position_account_status', 'account_id', 'status'),
        Index('ix_position_symbol_status', 'symbol_id', 'status'),
        Index('ix_position_opened_closed', 'opened_at', 'closed_at'),
        Index('ix_position_magic_number', 'magic_number'),
    )

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.status in ("OPEN", "PARTIAL")

    @property
    def is_closed(self) -> bool:
        """Check if position is closed."""
        return self.status == "CLOSED"

    @property
    def duration_seconds(self) -> Optional[int]:
        """Get position duration in seconds."""
        if self.closed_at:
            return int((self.closed_at - self.opened_at).total_seconds())
        else:
            return int((datetime.now(timezone.utc) - self.opened_at).total_seconds())

    def update_current_price(self, price: Decimal) -> None:
        """
        Update current price and recalculate unrealized P&L.

        Args:
            price: Current market price
        """
        self.current_price = price

        if self.entry_price is not None:
            if self.is_long:
                pnl = self.quantity * (price - self.entry_price)
            else:
                pnl = self.quantity * (self.entry_price - price)

            # Subtract commission and swap
            total_costs = (self.commission or Decimal('0')) + (self.swap or Decimal('0'))
            self.unrealized_pnl = pnl - total_costs

    def close_position(self, close_price: Decimal, close_time: Optional[datetime] = None) -> None:
        """
        Close the position.

        Args:
            close_price: Closing price
            close_time: Closing time (defaults to now)
        """
        self.exit_price = close_price
        self.closed_at = close_time or datetime.now(timezone.utc)
        self.status = "CLOSED"

        # Calculate final realized P&L
        if self.entry_price is not None:
            if self.is_long:
                gross_pnl = self.quantity * (close_price - self.entry_price)
            else:
                gross_pnl = self.quantity * (self.entry_price - close_price)

            total_costs = (self.commission or Decimal('0')) + (self.swap or Decimal('0'))
            self.realized_pnl = gross_pnl - total_costs
            self.unrealized_pnl = Decimal('0')

    def __repr__(self) -> str:
        return f"<Position(ticket='{self.position_ticket}', status='{self.status}', direction='{self.direction}')>"


class Trade(BaseModel, TradingMixin):
    """
    Model for completed trades.

    Records completed trade transactions with full execution details,
    P&L calculations, and performance metrics.
    """

    __tablename__ = "trades"

    # Position relationship
    position_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("positions.id"),
        nullable=True,
        index=True,
        doc="Reference to position (if applicable)"
    )

    # Account and symbol relationships
    account_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("accounts.id"),
        nullable=False,
        index=True,
        doc="Reference to trading account"
    )

    symbol_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("symbols.id"),
        nullable=False,
        index=True,
        doc="Reference to symbol"
    )

    # Trade identification
    trade_ticket: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        unique=True,
        index=True,
        doc="Broker trade ticket number"
    )

    # Trade timing
    open_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Trade opening time"
    )

    close_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Trade closing time"
    )

    # Trade result
    gross_profit: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=False,
        doc="Gross profit/loss before costs"
    )

    net_profit: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=False,
        doc="Net profit/loss after all costs"
    )

    # Additional information
    comment: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Trade comment"
    )

    magic_number: Mapped[Optional[Integer]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        doc="Expert advisor magic number"
    )

    # Strategy information
    strategy_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("strategies.id"),
        nullable=True,
        index=True,
        doc="Reference to strategy that generated this trade"
    )

    # Relationships
    position = relationship(
        "Position",
        back_populates="trades",
        lazy="select"
    )

    account = relationship(
        "Account",
        lazy="select"
    )

    symbol = relationship(
        "Symbol",
        lazy="select"
    )

    strategy = relationship(
        "Strategy",
        back_populates="trades",
        lazy="select"
    )

    __table_args__ = (
        Index('ix_trade_account_time', 'account_id', 'open_time'),
        Index('ix_trade_symbol_time', 'symbol_id', 'open_time'),
        Index('ix_trade_strategy_time', 'strategy_id', 'open_time'),
        Index('ix_trade_profit', 'net_profit'),
    )

    @property
    def duration_seconds(self) -> int:
        """Get trade duration in seconds."""
        return int((self.close_time - self.open_time).total_seconds())

    @property
    def duration_minutes(self) -> int:
        """Get trade duration in minutes."""
        return self.duration_seconds // 60

    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable."""
        return self.net_profit > 0

    @property
    def return_on_margin(self) -> Optional[Decimal]:
        """Calculate return on margin (if margin information available)."""
        # This would require margin calculation from the symbol and account
        # Implementation depends on having margin requirements available
        return None

    def __repr__(self) -> str:
        return f"<Trade(ticket='{self.trade_ticket}', profit={self.net_profit})>"


# Event listeners for automatic calculations
@event.listens_for(Position, 'before_update')
def update_position_pnl(mapper, connection, target):
    """Update position P&L when position is modified."""
    if target.current_price is not None and target.entry_price is not None:
        target.update_current_price(target.current_price)


@event.listens_for(Order, 'before_update')
def update_order_remaining_quantity(mapper, connection, target):
    """Update remaining quantity when order is modified."""
    if target.quantity is not None and target.filled_quantity is not None:
        target.remaining_quantity = target.quantity - target.filled_quantity