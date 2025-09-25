"""
Account and user management models for the MetaTrader Python Framework.

This module defines database models for trading accounts, users, and their
related information including account specifications, balances, and access control.
"""

from __future__ import annotations

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
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.models.base import BaseModel
from src.database.models.mixins import PerformanceMixin


class User(BaseModel):
    """
    Model for system users.

    Represents users who can access the trading system with
    authentication and authorization information.
    """

    __tablename__ = "users"

    # User identification
    username: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
        doc="Unique username for login"
    )

    email: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        doc="User email address"
    )

    # User information
    first_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="User's first name"
    )

    last_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="User's last name"
    )

    # Authentication
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Hashed password"
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        doc="Whether the user account is active"
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether the user email is verified"
    )

    # Access control
    role: Mapped[str] = mapped_column(
        String(20),
        default="USER",
        nullable=False,
        index=True,
        doc="User role (ADMIN, TRADER, USER, READONLY)"
    )

    # Login tracking
    last_login: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last login timestamp"
    )

    login_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Total number of logins"
    )

    # User preferences
    timezone: Mapped[str] = mapped_column(
        String(50),
        default="UTC",
        nullable=False,
        doc="User's preferred timezone"
    )

    language: Mapped[str] = mapped_column(
        String(10),
        default="en",
        nullable=False,
        doc="User's preferred language"
    )

    # Relationships
    accounts = relationship(
        "Account",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    __table_args__ = (
        Index('ix_user_role_active', 'role', 'is_active'),
    )

    @property
    def full_name(self) -> str:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        else:
            return self.username

    def can_trade(self) -> bool:
        """Check if user has trading permissions."""
        return self.is_active and self.role in ("ADMIN", "TRADER")

    def can_view_all_accounts(self) -> bool:
        """Check if user can view all accounts."""
        return self.role == "ADMIN"

    def __repr__(self) -> str:
        return f"<User(username='{self.username}', email='{self.email}')>"


class Account(BaseModel, PerformanceMixin):
    """
    Model for trading accounts.

    Represents a trading account with balance information,
    trading permissions, and performance tracking.
    """

    __tablename__ = "accounts"

    # User relationship
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
        doc="Reference to user who owns this account"
    )

    # Account identification
    account_number: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
        doc="Unique account number"
    )

    account_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="Account display name"
    )

    # Account type and status
    account_type: Mapped[str] = mapped_column(
        String(20),
        default="DEMO",
        nullable=False,
        index=True,
        doc="Account type (DEMO, LIVE, PRACTICE)"
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        doc="Whether the account is active"
    )

    # Broker information
    broker_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Broker name"
    )

    server_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Trading server name"
    )

    # Account specifications
    currency: Mapped[str] = mapped_column(
        String(10),
        default="USD",
        nullable=False,
        index=True,
        doc="Account base currency"
    )

    leverage: Mapped[int] = mapped_column(
        Integer,
        default=100,
        nullable=False,
        doc="Account leverage ratio"
    )

    # Balance information
    balance: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        default=Decimal('0'),
        nullable=False,
        doc="Account balance"
    )

    equity: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        default=Decimal('0'),
        nullable=False,
        doc="Account equity (balance + floating P&L)"
    )

    margin: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        default=Decimal('0'),
        nullable=False,
        doc="Used margin"
    )

    free_margin: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        default=Decimal('0'),
        nullable=False,
        doc="Free margin available for trading"
    )

    margin_level: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=2),
        nullable=True,
        doc="Margin level percentage"
    )

    # Trading limits
    max_leverage: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Maximum allowed leverage"
    )

    max_lot_size: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=2),
        nullable=True,
        doc="Maximum lot size per trade"
    )

    max_daily_loss: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=True,
        doc="Maximum daily loss limit"
    )

    # Risk management
    stop_out_level: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=2),
        default=Decimal('20'),
        nullable=True,
        doc="Stop out level percentage"
    )

    margin_call_level: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(precision=10, scale=2),
        default=Decimal('50'),
        nullable=True,
        doc="Margin call level percentage"
    )

    # Trading permissions
    trading_allowed: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether trading is allowed on this account"
    )

    expert_advisors_allowed: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether expert advisors are allowed"
    )

    # Account statistics
    total_deposits: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        default=Decimal('0'),
        nullable=False,
        doc="Total deposits made to account"
    )

    total_withdrawals: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        default=Decimal('0'),
        nullable=False,
        doc="Total withdrawals from account"
    )

    # Relationships
    user = relationship(
        "User",
        back_populates="accounts",
        lazy="select"
    )

    orders = relationship(
        "Order",
        back_populates="account",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    positions = relationship(
        "Position",
        back_populates="account",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    transactions = relationship(
        "Transaction",
        back_populates="account",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )

    __table_args__ = (
        UniqueConstraint('account_number', name='uq_account_number'),
        Index('ix_account_user_type', 'user_id', 'account_type'),
        Index('ix_account_active_type', 'is_active', 'account_type'),
    )

    @property
    def margin_utilization(self) -> Optional[Decimal]:
        """Calculate margin utilization percentage."""
        if self.equity == 0:
            return None
        return (self.margin / self.equity) * Decimal('100')

    @property
    def available_margin(self) -> Decimal:
        """Calculate available margin for new positions."""
        return max(Decimal('0'), self.equity - self.margin)

    @property
    def is_margin_call(self) -> bool:
        """Check if account is in margin call."""
        if self.margin_call_level is None or self.margin_level is None:
            return False
        return self.margin_level <= self.margin_call_level

    @property
    def is_stop_out(self) -> bool:
        """Check if account is in stop out."""
        if self.stop_out_level is None or self.margin_level is None:
            return False
        return self.margin_level <= self.stop_out_level

    def update_balance(
        self,
        balance: Decimal,
        equity: Decimal,
        margin: Decimal,
        free_margin: Decimal
    ) -> None:
        """
        Update account balance information.

        Args:
            balance: New account balance
            equity: New account equity
            margin: New used margin
            free_margin: New free margin
        """
        self.balance = balance
        self.equity = equity
        self.margin = margin
        self.free_margin = free_margin

        # Calculate margin level
        if self.margin > 0:
            self.margin_level = (self.equity / self.margin) * Decimal('100')
        else:
            self.margin_level = None

    def can_open_position(self, required_margin: Decimal) -> bool:
        """
        Check if account can open a new position with required margin.

        Args:
            required_margin: Margin required for the new position

        Returns:
            True if position can be opened, False otherwise
        """
        if not self.trading_allowed or not self.is_active:
            return False

        # Check if there's enough free margin
        if self.free_margin < required_margin:
            return False

        # Check if opening position would trigger margin call
        if self.margin_call_level is not None:
            new_margin = self.margin + required_margin
            new_margin_level = (self.equity / new_margin) * Decimal('100')
            if new_margin_level <= self.margin_call_level:
                return False

        return True

    def add_deposit(self, amount: Decimal) -> None:
        """
        Add a deposit to the account.

        Args:
            amount: Deposit amount
        """
        self.balance += amount
        self.equity += amount
        self.free_margin += amount
        self.total_deposits += amount

    def add_withdrawal(self, amount: Decimal) -> bool:
        """
        Add a withdrawal from the account.

        Args:
            amount: Withdrawal amount

        Returns:
            True if withdrawal is successful, False if insufficient funds
        """
        if self.free_margin < amount:
            return False

        self.balance -= amount
        self.equity -= amount
        self.free_margin -= amount
        self.total_withdrawals += amount
        return True

    def __repr__(self) -> str:
        return f"<Account(number='{self.account_number}', type='{self.account_type}')>"


class Transaction(BaseModel):
    """
    Model for account transactions.

    Records all financial transactions on trading accounts including
    deposits, withdrawals, trading profits/losses, and fees.
    """

    __tablename__ = "transactions"

    # Account relationship
    account_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("accounts.id"),
        nullable=False,
        index=True,
        doc="Reference to account"
    )

    # Transaction identification
    transaction_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        unique=True,
        index=True,
        doc="External transaction ID"
    )

    # Transaction details
    transaction_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Transaction type (DEPOSIT, WITHDRAWAL, TRADE_PROFIT, TRADE_LOSS, COMMISSION, SWAP)"
    )

    amount: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=False,
        doc="Transaction amount (positive for credit, negative for debit)"
    )

    balance_before: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=False,
        doc="Account balance before transaction"
    )

    balance_after: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=20, scale=2),
        nullable=False,
        doc="Account balance after transaction"
    )

    # Additional information
    comment: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Transaction comment/description"
    )

    reference_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        doc="Reference to related order/position ID"
    )

    # Relationships
    account = relationship(
        "Account",
        back_populates="transactions",
        lazy="select"
    )

    __table_args__ = (
        Index('ix_transaction_account_type', 'account_id', 'transaction_type'),
        Index('ix_transaction_created', 'created_at'),
    )

    def __repr__(self) -> str:
        return f"<Transaction(type='{self.transaction_type}', amount={self.amount})>"


class AccountSettings(BaseModel):
    """
    Model for account-specific settings and preferences.

    Stores user preferences and configuration specific to each trading account.
    """

    __tablename__ = "account_settings"

    # Account relationship
    account_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("accounts.id"),
        nullable=False,
        unique=True,
        index=True,
        doc="Reference to account"
    )

    # Trading preferences
    default_lot_size: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=10, scale=2),
        default=Decimal('0.1'),
        nullable=False,
        doc="Default lot size for trades"
    )

    risk_per_trade: Mapped[Decimal] = mapped_column(
        DECIMAL(precision=5, scale=2),
        default=Decimal('2.0'),
        nullable=False,
        doc="Default risk percentage per trade"
    )

    # Risk management settings
    auto_stop_loss: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Automatically set stop loss on trades"
    )

    auto_take_profit: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Automatically set take profit on trades"
    )

    default_stop_loss_pips: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Default stop loss in pips"
    )

    default_take_profit_pips: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Default take profit in pips"
    )

    # Notification settings
    email_notifications: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Enable email notifications"
    )

    sms_notifications: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Enable SMS notifications"
    )

    push_notifications: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Enable push notifications"
    )

    # Display preferences
    chart_theme: Mapped[str] = mapped_column(
        String(20),
        default="dark",
        nullable=False,
        doc="Preferred chart theme"
    )

    default_timeframe: Mapped[str] = mapped_column(
        String(10),
        default="H1",
        nullable=False,
        doc="Default chart timeframe"
    )

    # Relationships
    account = relationship(
        "Account",
        lazy="select"
    )

    __table_args__ = (
        UniqueConstraint('account_id', name='uq_account_settings'),
    )

    def __repr__(self) -> str:
        return f"<AccountSettings(account_id='{self.account_id}')>"