"""
Account and user management Pydantic schemas for the MetaTrader Python Framework.

This module provides validation schemas for account and user-related API operations
including user registration, account management, and transaction tracking.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import Field, field_validator, EmailStr

from .base import (
    BaseEntitySchema,
    CreateRequestSchema,
    UpdateRequestSchema,
    ResponseSchema,
    CurrencyCode,
    PositiveDecimal,
    NonNegativeDecimal,
    PercentageDecimal,
)


# User Schemas
class UserCreateSchema(CreateRequestSchema):
    """Schema for creating a new user."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9_]+$",
        description="Unique username (alphanumeric and underscore only)",
        json_schema_extra={"example": "john_doe"}
    )

    email: EmailStr = Field(
        ...,
        description="User email address",
        json_schema_extra={"example": "john.doe@example.com"}
    )

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User password (will be hashed)",
        json_schema_extra={"example": "SecurePassword123!"}
    )

    first_name: Optional[str] = Field(
        None,
        max_length=100,
        description="User's first name",
        json_schema_extra={"example": "John"}
    )

    last_name: Optional[str] = Field(
        None,
        max_length=100,
        description="User's last name",
        json_schema_extra={"example": "Doe"}
    )

    role: str = Field(
        "USER",
        description="User role",
        json_schema_extra={"example": "USER"}
    )

    timezone: str = Field(
        "UTC",
        max_length=50,
        description="User's preferred timezone",
        json_schema_extra={"example": "America/New_York"}
    )

    language: str = Field(
        "en",
        min_length=2,
        max_length=10,
        description="User's preferred language",
        json_schema_extra={"example": "en"}
    )

    @field_validator('role')
    def validate_role(cls, v):
        """Validate user role."""
        valid_roles = ['ADMIN', 'TRADER', 'USER', 'READONLY']
        if v.upper() not in valid_roles:
            raise ValueError(f'Role must be one of: {", ".join(valid_roles)}')
        return v.upper()

    @field_validator('password')
    def validate_password_strength(cls, v):
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserUpdateSchema(UpdateRequestSchema):
    """Schema for updating user information."""

    email: Optional[EmailStr] = Field(
        None,
        description="User email address",
        json_schema_extra={"example": "john.doe@example.com"}
    )

    first_name: Optional[str] = Field(
        None,
        max_length=100,
        description="User's first name",
        json_schema_extra={"example": "John"}
    )

    last_name: Optional[str] = Field(
        None,
        max_length=100,
        description="User's last name",
        json_schema_extra={"example": "Doe"}
    )

    is_active: Optional[bool] = Field(
        None,
        description="Whether the user account is active",
        json_schema_extra={"example": True}
    )

    role: Optional[str] = Field(
        None,
        description="User role",
        json_schema_extra={"example": "TRADER"}
    )

    timezone: Optional[str] = Field(
        None,
        max_length=50,
        description="User's preferred timezone",
        json_schema_extra={"example": "America/New_York"}
    )

    language: Optional[str] = Field(
        None,
        min_length=2,
        max_length=10,
        description="User's preferred language",
        json_schema_extra={"example": "en"}
    )

    @field_validator('role')
    def validate_role(cls, v):
        """Validate user role."""
        if v is None:
            return v
        valid_roles = ['ADMIN', 'TRADER', 'USER', 'READONLY']
        if v.upper() not in valid_roles:
            raise ValueError(f'Role must be one of: {", ".join(valid_roles)}')
        return v.upper()


class UserPasswordChangeSchema(CreateRequestSchema):
    """Schema for changing user password."""

    current_password: str = Field(
        ...,
        min_length=1,
        description="Current password",
        json_schema_extra={"example": "OldPassword123!"}
    )

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password",
        json_schema_extra={"example": "NewSecurePassword456!"}
    )

    confirm_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Confirm new password",
        json_schema_extra={"example": "NewSecurePassword456!"}
    )

    @field_validator('new_password')
    def validate_password_strength(cls, v):
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

    @field_validator('confirm_password')
    def validate_passwords_match(cls, v, values):
        """Validate that new password and confirm password match."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class UserResponseSchema(ResponseSchema):
    """Schema for user responses."""

    username: str = Field(
        ...,
        description="Username",
        json_schema_extra={"example": "john_doe"}
    )

    email: str = Field(
        ...,
        description="Email address",
        json_schema_extra={"example": "john.doe@example.com"}
    )

    first_name: Optional[str] = Field(
        None,
        description="First name",
        json_schema_extra={"example": "John"}
    )

    last_name: Optional[str] = Field(
        None,
        description="Last name",
        json_schema_extra={"example": "Doe"}
    )

    full_name: Optional[str] = Field(
        None,
        description="Full name",
        json_schema_extra={"example": "John Doe"}
    )

    is_active: bool = Field(
        ...,
        description="Whether the user account is active",
        json_schema_extra={"example": True}
    )

    is_verified: bool = Field(
        ...,
        description="Whether the user email is verified",
        json_schema_extra={"example": True}
    )

    role: str = Field(
        ...,
        description="User role",
        json_schema_extra={"example": "USER"}
    )

    last_login: Optional[datetime] = Field(
        None,
        description="Last login timestamp",
        json_schema_extra={"example": "2024-01-01T12:00:00Z"}
    )

    login_count: int = Field(
        ...,
        description="Total number of logins",
        json_schema_extra={"example": 42}
    )

    timezone: str = Field(
        ...,
        description="Preferred timezone",
        json_schema_extra={"example": "America/New_York"}
    )

    language: str = Field(
        ...,
        description="Preferred language",
        json_schema_extra={"example": "en"}
    )


# Account Schemas
class AccountCreateSchema(CreateRequestSchema):
    """Schema for creating a new account."""

    user_id: str = Field(
        ...,
        min_length=36,
        max_length=36,
        description="Reference to user who owns this account",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    account_number: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique account number",
        json_schema_extra={"example": "ACC123456789"}
    )

    account_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Account display name",
        json_schema_extra={"example": "Main Trading Account"}
    )

    account_type: str = Field(
        "DEMO",
        description="Account type",
        json_schema_extra={"example": "DEMO"}
    )

    currency: str = CurrencyCode

    leverage: int = Field(
        100,
        ge=1,
        le=1000,
        description="Account leverage ratio",
        json_schema_extra={"example": 100}
    )

    broker_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Broker name",
        json_schema_extra={"example": "MetaTrader Broker"}
    )

    server_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Trading server name",
        json_schema_extra={"example": "Demo-Server-01"}
    )

    balance: Decimal = Field(
        Decimal('10000'),
        ge=0,
        decimal_places=2,
        description="Initial account balance",
        json_schema_extra={"example": "10000.00"}
    )

    max_leverage: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Maximum allowed leverage",
        json_schema_extra={"example": 500}
    )

    max_lot_size: Optional[Decimal] = Field(
        None,
        gt=0,
        decimal_places=2,
        description="Maximum lot size per trade",
        json_schema_extra={"example": "10.00"}
    )

    max_daily_loss: Optional[Decimal] = Field(
        None,
        gt=0,
        decimal_places=2,
        description="Maximum daily loss limit",
        json_schema_extra={"example": "1000.00"}
    )

    @field_validator('account_type')
    def validate_account_type(cls, v):
        """Validate account type."""
        valid_types = ['DEMO', 'LIVE', 'PRACTICE']
        if v.upper() not in valid_types:
            raise ValueError(f'Account type must be one of: {", ".join(valid_types)}')
        return v.upper()


class AccountUpdateSchema(UpdateRequestSchema):
    """Schema for updating account information."""

    account_name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Account display name",
        json_schema_extra={"example": "Main Trading Account"}
    )

    is_active: Optional[bool] = Field(
        None,
        description="Whether the account is active",
        json_schema_extra={"example": True}
    )

    leverage: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Account leverage ratio",
        json_schema_extra={"example": 200}
    )

    broker_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Broker name",
        json_schema_extra={"example": "MetaTrader Broker"}
    )

    server_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Trading server name",
        json_schema_extra={"example": "Live-Server-01"}
    )

    max_leverage: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Maximum allowed leverage",
        json_schema_extra={"example": 500}
    )

    max_lot_size: Optional[Decimal] = Field(
        None,
        gt=0,
        decimal_places=2,
        description="Maximum lot size per trade",
        json_schema_extra={"example": "10.00"}
    )

    max_daily_loss: Optional[Decimal] = Field(
        None,
        gt=0,
        decimal_places=2,
        description="Maximum daily loss limit",
        json_schema_extra={"example": "1000.00"}
    )

    trading_allowed: Optional[bool] = Field(
        None,
        description="Whether trading is allowed on this account",
        json_schema_extra={"example": True}
    )


class AccountBalanceUpdateSchema(CreateRequestSchema):
    """Schema for updating account balance."""

    balance: Decimal = Field(
        ...,
        ge=0,
        decimal_places=2,
        description="New account balance",
        json_schema_extra={"example": "15000.00"}
    )

    equity: Decimal = Field(
        ...,
        ge=0,
        decimal_places=2,
        description="New account equity",
        json_schema_extra={"example": "15250.75"}
    )

    margin: Decimal = Field(
        ...,
        ge=0,
        decimal_places=2,
        description="Used margin",
        json_schema_extra={"example": "500.00"}
    )

    free_margin: Decimal = Field(
        ...,
        ge=0,
        decimal_places=2,
        description="Free margin",
        json_schema_extra={"example": "14750.75"}
    )


class AccountResponseSchema(ResponseSchema):
    """Schema for account responses."""

    user_id: str = Field(
        ...,
        description="Reference to user who owns this account",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    account_number: str = Field(
        ...,
        description="Unique account number",
        json_schema_extra={"example": "ACC123456789"}
    )

    account_name: str = Field(
        ...,
        description="Account display name",
        json_schema_extra={"example": "Main Trading Account"}
    )

    account_type: str = Field(
        ...,
        description="Account type",
        json_schema_extra={"example": "DEMO"}
    )

    is_active: bool = Field(
        ...,
        description="Whether the account is active",
        json_schema_extra={"example": True}
    )

    broker_name: Optional[str] = Field(
        None,
        description="Broker name",
        json_schema_extra={"example": "MetaTrader Broker"}
    )

    server_name: Optional[str] = Field(
        None,
        description="Trading server name",
        json_schema_extra={"example": "Demo-Server-01"}
    )

    currency: str = Field(
        ...,
        description="Account base currency",
        json_schema_extra={"example": "USD"}
    )

    leverage: int = Field(
        ...,
        description="Account leverage ratio",
        json_schema_extra={"example": 100}
    )

    balance: Decimal = Field(
        ...,
        description="Account balance",
        json_schema_extra={"example": "10000.00"}
    )

    equity: Decimal = Field(
        ...,
        description="Account equity",
        json_schema_extra={"example": "10250.75"}
    )

    margin: Decimal = Field(
        ...,
        description="Used margin",
        json_schema_extra={"example": "500.00"}
    )

    free_margin: Decimal = Field(
        ...,
        description="Free margin",
        json_schema_extra={"example": "9750.75"}
    )

    margin_level: Optional[Decimal] = Field(
        None,
        description="Margin level percentage",
        json_schema_extra={"example": "2050.15"}
    )

    trading_allowed: bool = Field(
        ...,
        description="Whether trading is allowed",
        json_schema_extra={"example": True}
    )

    # Performance metrics
    total_trades: int = Field(
        ...,
        description="Total number of trades",
        json_schema_extra={"example": 50}
    )

    winning_trades: int = Field(
        ...,
        description="Number of winning trades",
        json_schema_extra={"example": 30}
    )

    losing_trades: int = Field(
        ...,
        description="Number of losing trades",
        json_schema_extra={"example": 20}
    )

    gross_profit: Decimal = Field(
        ...,
        description="Total gross profit",
        json_schema_extra={"example": "1500.00"}
    )

    gross_loss: Decimal = Field(
        ...,
        description="Total gross loss",
        json_schema_extra={"example": "-800.00"}
    )

    max_drawdown: Optional[Decimal] = Field(
        None,
        description="Maximum drawdown percentage",
        json_schema_extra={"example": "15.50"}
    )

    # Calculated fields
    net_profit: Optional[Decimal] = Field(
        None,
        description="Net profit (gross profit + gross loss)",
        json_schema_extra={"example": "700.00"}
    )

    win_rate: Optional[Decimal] = Field(
        None,
        description="Win rate percentage",
        json_schema_extra={"example": "60.00"}
    )

    profit_factor: Optional[Decimal] = Field(
        None,
        description="Profit factor (gross profit / abs(gross loss))",
        json_schema_extra={"example": "1.875"}
    )


# Transaction Schemas
class TransactionCreateSchema(CreateRequestSchema):
    """Schema for creating a transaction."""

    account_id: str = Field(
        ...,
        min_length=36,
        max_length=36,
        description="Reference to account",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    transaction_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Transaction type",
        json_schema_extra={"example": "DEPOSIT"}
    )

    amount: Decimal = Field(
        ...,
        decimal_places=2,
        description="Transaction amount (positive for credit, negative for debit)",
        json_schema_extra={"example": "1000.00"}
    )

    balance_before: Decimal = Field(
        ...,
        decimal_places=2,
        description="Account balance before transaction",
        json_schema_extra={"example": "9000.00"}
    )

    balance_after: Decimal = Field(
        ...,
        decimal_places=2,
        description="Account balance after transaction",
        json_schema_extra={"example": "10000.00"}
    )

    comment: Optional[str] = Field(
        None,
        max_length=255,
        description="Transaction comment",
        json_schema_extra={"example": "Initial deposit"}
    )

    reference_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Reference to related order/position ID",
        json_schema_extra={"example": "ORDER123456"}
    )

    transaction_id: Optional[str] = Field(
        None,
        max_length=100,
        description="External transaction ID",
        json_schema_extra={"example": "TXN987654321"}
    )

    @field_validator('transaction_type')
    def validate_transaction_type(cls, v):
        """Validate transaction type."""
        valid_types = [
            'DEPOSIT', 'WITHDRAWAL', 'TRADE_PROFIT', 'TRADE_LOSS',
            'COMMISSION', 'SWAP', 'DIVIDEND', 'TAX', 'BONUS', 'CREDIT_ADJUSTMENT'
        ]
        if v.upper() not in valid_types:
            raise ValueError(f'Transaction type must be one of: {", ".join(valid_types)}')
        return v.upper()


class TransactionResponseSchema(ResponseSchema):
    """Schema for transaction responses."""

    account_id: str = Field(
        ...,
        description="Reference to account",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    transaction_id: Optional[str] = Field(
        None,
        description="External transaction ID",
        json_schema_extra={"example": "TXN987654321"}
    )

    transaction_type: str = Field(
        ...,
        description="Transaction type",
        json_schema_extra={"example": "DEPOSIT"}
    )

    amount: Decimal = Field(
        ...,
        description="Transaction amount",
        json_schema_extra={"example": "1000.00"}
    )

    balance_before: Decimal = Field(
        ...,
        description="Account balance before transaction",
        json_schema_extra={"example": "9000.00"}
    )

    balance_after: Decimal = Field(
        ...,
        description="Account balance after transaction",
        json_schema_extra={"example": "10000.00"}
    )

    comment: Optional[str] = Field(
        None,
        description="Transaction comment",
        json_schema_extra={"example": "Initial deposit"}
    )

    reference_id: Optional[str] = Field(
        None,
        description="Reference to related order/position ID",
        json_schema_extra={"example": "ORDER123456"}
    )


# Account Settings Schemas
class AccountSettingsCreateSchema(CreateRequestSchema):
    """Schema for creating account settings."""

    account_id: str = Field(
        ...,
        min_length=36,
        max_length=36,
        description="Reference to account",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    default_lot_size: Decimal = Field(
        Decimal('0.1'),
        gt=0,
        decimal_places=2,
        description="Default lot size for trades",
        json_schema_extra={"example": "0.10"}
    )

    risk_per_trade: Decimal = Field(
        Decimal('2.0'),
        gt=0,
        le=100,
        decimal_places=2,
        description="Default risk percentage per trade",
        json_schema_extra={"example": "2.00"}
    )

    auto_stop_loss: bool = Field(
        False,
        description="Automatically set stop loss on trades",
        json_schema_extra={"example": False}
    )

    auto_take_profit: bool = Field(
        False,
        description="Automatically set take profit on trades",
        json_schema_extra={"example": False}
    )

    default_stop_loss_pips: Optional[int] = Field(
        None,
        gt=0,
        description="Default stop loss in pips",
        json_schema_extra={"example": 50}
    )

    default_take_profit_pips: Optional[int] = Field(
        None,
        gt=0,
        description="Default take profit in pips",
        json_schema_extra={"example": 100}
    )

    email_notifications: bool = Field(
        True,
        description="Enable email notifications",
        json_schema_extra={"example": True}
    )

    chart_theme: str = Field(
        "dark",
        description="Preferred chart theme",
        json_schema_extra={"example": "dark"}
    )

    default_timeframe: str = Field(
        "H1",
        description="Default chart timeframe",
        json_schema_extra={"example": "H1"}
    )

    @field_validator('chart_theme')
    def validate_chart_theme(cls, v):
        """Validate chart theme."""
        valid_themes = ['dark', 'light', 'blue', 'classic']
        if v.lower() not in valid_themes:
            raise ValueError(f'Chart theme must be one of: {", ".join(valid_themes)}')
        return v.lower()

    @field_validator('default_timeframe')
    def validate_timeframe(cls, v):
        """Validate timeframe."""
        valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
        if v.upper() not in valid_timeframes:
            raise ValueError(f'Timeframe must be one of: {", ".join(valid_timeframes)}')
        return v.upper()


class AccountSettingsUpdateSchema(UpdateRequestSchema):
    """Schema for updating account settings."""

    default_lot_size: Optional[Decimal] = Field(
        None,
        gt=0,
        decimal_places=2,
        description="Default lot size for trades",
        json_schema_extra={"example": "0.10"}
    )

    risk_per_trade: Optional[Decimal] = Field(
        None,
        gt=0,
        le=100,
        decimal_places=2,
        description="Default risk percentage per trade",
        json_schema_extra={"example": "2.00"}
    )

    auto_stop_loss: Optional[bool] = Field(
        None,
        description="Automatically set stop loss on trades",
        json_schema_extra={"example": False}
    )

    auto_take_profit: Optional[bool] = Field(
        None,
        description="Automatically set take profit on trades",
        json_schema_extra={"example": False}
    )

    default_stop_loss_pips: Optional[int] = Field(
        None,
        gt=0,
        description="Default stop loss in pips",
        json_schema_extra={"example": 50}
    )

    default_take_profit_pips: Optional[int] = Field(
        None,
        gt=0,
        description="Default take profit in pips",
        json_schema_extra={"example": 100}
    )

    email_notifications: Optional[bool] = Field(
        None,
        description="Enable email notifications",
        json_schema_extra={"example": True}
    )

    chart_theme: Optional[str] = Field(
        None,
        description="Preferred chart theme",
        json_schema_extra={"example": "dark"}
    )

    default_timeframe: Optional[str] = Field(
        None,
        description="Default chart timeframe",
        json_schema_extra={"example": "H1"}
    )


class AccountSettingsResponseSchema(ResponseSchema):
    """Schema for account settings responses."""

    account_id: str = Field(
        ...,
        description="Reference to account",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    default_lot_size: Decimal = Field(
        ...,
        description="Default lot size for trades",
        json_schema_extra={"example": "0.10"}
    )

    risk_per_trade: Decimal = Field(
        ...,
        description="Default risk percentage per trade",
        json_schema_extra={"example": "2.00"}
    )

    auto_stop_loss: bool = Field(
        ...,
        description="Automatically set stop loss on trades",
        json_schema_extra={"example": False}
    )

    auto_take_profit: bool = Field(
        ...,
        description="Automatically set take profit on trades",
        json_schema_extra={"example": False}
    )

    default_stop_loss_pips: Optional[int] = Field(
        None,
        description="Default stop loss in pips",
        json_schema_extra={"example": 50}
    )

    default_take_profit_pips: Optional[int] = Field(
        None,
        description="Default take profit in pips",
        json_schema_extra={"example": 100}
    )

    email_notifications: bool = Field(
        ...,
        description="Enable email notifications",
        json_schema_extra={"example": True}
    )

    chart_theme: str = Field(
        ...,
        description="Preferred chart theme",
        json_schema_extra={"example": "dark"}
    )

    default_timeframe: str = Field(
        ...,
        description="Default chart timeframe",
        json_schema_extra={"example": "H1"}
    )


# List and filter schemas
class AccountFilterSchema(CreateRequestSchema):
    """Schema for filtering accounts."""

    user_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Filter by user",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    account_type: Optional[str] = Field(
        None,
        description="Filter by account type",
        json_schema_extra={"example": "DEMO"}
    )

    currency: Optional[str] = Field(
        None,
        min_length=3,
        max_length=3,
        description="Filter by currency",
        json_schema_extra={"example": "USD"}
    )

    is_active: Optional[bool] = Field(
        None,
        description="Filter by active status",
        json_schema_extra={"example": True}
    )


class UserListResponseSchema(CreateRequestSchema):
    """Schema for user list responses."""

    items: List[UserResponseSchema] = Field(
        default_factory=list,
        description="List of users"
    )

    total: int = Field(
        ...,
        description="Total number of users",
        json_schema_extra={"example": 100}
    )


class AccountListResponseSchema(CreateRequestSchema):
    """Schema for account list responses."""

    items: List[AccountResponseSchema] = Field(
        default_factory=list,
        description="List of accounts"
    )

    total: int = Field(
        ...,
        description="Total number of accounts",
        json_schema_extra={"example": 50}
    )


class TransactionListResponseSchema(CreateRequestSchema):
    """Schema for transaction list responses."""

    items: List[TransactionResponseSchema] = Field(
        default_factory=list,
        description="List of transactions"
    )

    total: int = Field(
        ...,
        description="Total number of transactions",
        json_schema_extra={"example": 1000}
    )