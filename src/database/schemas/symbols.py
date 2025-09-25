"""
Symbol and instrument Pydantic schemas for the MetaTrader Python Framework.

This module provides validation schemas for symbol-related API operations
including symbol creation, updates, and responses.
"""

from __future__ import annotations

from decimal import Decimal
from typing import List, Optional

from pydantic import Field, validator

from .base import (
    BaseEntitySchema,
    CreateRequestSchema,
    UpdateRequestSchema,
    ResponseSchema,
    ConfigurationSchema,
    CurrencyCode,
    SymbolName,
    PositiveDecimal,
    NonNegativeDecimal,
)


# Symbol Group Schemas
class SymbolGroupCreateSchema(CreateRequestSchema, ConfigurationSchema):
    """Schema for creating a new symbol group."""

    group_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Type of symbol group",
        json_schema_extra={"example": "FOREX"}
    )

    display_order: int = Field(
        0,
        ge=0,
        description="Display order in UI",
        json_schema_extra={"example": 1}
    )

    market_open_hour: Optional[int] = Field(
        None,
        ge=0,
        le=23,
        description="Market opening hour (UTC)",
        json_schema_extra={"example": 9}
    )

    market_close_hour: Optional[int] = Field(
        None,
        ge=0,
        le=23,
        description="Market closing hour (UTC)",
        json_schema_extra={"example": 17}
    )

    @field_validator('group_type')
    def validate_group_type(cls, v):
        """Validate group type."""
        valid_types = ['FOREX', 'CRYPTO', 'STOCK', 'COMMODITY', 'INDEX', 'BOND', 'ETF']
        if v.upper() not in valid_types:
            raise ValueError(f'Group type must be one of: {", ".join(valid_types)}')
        return v.upper()


class SymbolGroupUpdateSchema(UpdateRequestSchema):
    """Schema for updating a symbol group."""

    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Group name",
        json_schema_extra={"example": "Major Forex Pairs"}
    )

    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Group description",
        json_schema_extra={"example": "Major currency pairs with high liquidity"}
    )

    group_type: Optional[str] = Field(
        None,
        min_length=1,
        max_length=50,
        description="Type of symbol group",
        json_schema_extra={"example": "FOREX"}
    )

    display_order: Optional[int] = Field(
        None,
        ge=0,
        description="Display order in UI",
        json_schema_extra={"example": 1}
    )

    market_open_hour: Optional[int] = Field(
        None,
        ge=0,
        le=23,
        description="Market opening hour (UTC)",
        json_schema_extra={"example": 9}
    )

    market_close_hour: Optional[int] = Field(
        None,
        ge=0,
        le=23,
        description="Market closing hour (UTC)",
        json_schema_extra={"example": 17}
    )

    is_active: Optional[bool] = Field(
        None,
        description="Whether the group is active",
        json_schema_extra={"example": True}
    )


class SymbolGroupResponseSchema(ResponseSchema, ConfigurationSchema):
    """Schema for symbol group responses."""

    group_type: str = Field(
        ...,
        description="Type of symbol group",
        json_schema_extra={"example": "FOREX"}
    )

    display_order: int = Field(
        ...,
        description="Display order in UI",
        json_schema_extra={"example": 1}
    )

    market_open_hour: Optional[int] = Field(
        None,
        description="Market opening hour (UTC)",
        json_schema_extra={"example": 9}
    )

    market_close_hour: Optional[int] = Field(
        None,
        description="Market closing hour (UTC)",
        json_schema_extra={"example": 17}
    )


# Symbol Schemas
class SymbolCreateSchema(CreateRequestSchema):
    """Schema for creating a new symbol."""

    symbol: str = SymbolName

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Full name of the symbol",
        json_schema_extra={"example": "Euro vs US Dollar"}
    )

    symbol_group_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Reference to symbol group",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    market: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Market identifier",
        json_schema_extra={"example": "FOREX"}
    )

    exchange: Optional[str] = Field(
        None,
        max_length=50,
        description="Exchange name",
        json_schema_extra={"example": "NYSE"}
    )

    base_currency: str = CurrencyCode
    quote_currency: str = CurrencyCode

    digits: int = Field(
        5,
        ge=0,
        le=8,
        description="Number of decimal places for price quotes",
        json_schema_extra={"example": 5}
    )

    point: Decimal = Field(
        ...,
        gt=0,
        decimal_places=8,
        description="Point value (smallest price change)",
        json_schema_extra={"example": "0.00001"}
    )

    tick_size: Decimal = Field(
        ...,
        gt=0,
        decimal_places=8,
        description="Minimum price change step",
        json_schema_extra={"example": "0.00001"}
    )

    tick_value: Decimal = Field(
        ...,
        gt=0,
        decimal_places=2,
        description="Value of one tick in account currency",
        json_schema_extra={"example": "1.00"}
    )

    contract_size: Decimal = Field(
        Decimal('100000'),
        gt=0,
        decimal_places=8,
        description="Standard contract size",
        json_schema_extra={"example": "100000.00000000"}
    )

    min_lot: Decimal = Field(
        Decimal('0.01'),
        gt=0,
        decimal_places=8,
        description="Minimum lot size",
        json_schema_extra={"example": "0.01000000"}
    )

    max_lot: Decimal = Field(
        Decimal('100'),
        gt=0,
        decimal_places=8,
        description="Maximum lot size",
        json_schema_extra={"example": "100.00000000"}
    )

    lot_step: Decimal = Field(
        Decimal('0.01'),
        gt=0,
        decimal_places=8,
        description="Lot size step",
        json_schema_extra={"example": "0.01000000"}
    )

    margin_initial: Optional[Decimal] = Field(
        None,
        ge=0,
        le=100,
        decimal_places=4,
        description="Initial margin requirement percentage",
        json_schema_extra={"example": "1.0000"}
    )

    margin_maintenance: Optional[Decimal] = Field(
        None,
        ge=0,
        le=100,
        decimal_places=4,
        description="Maintenance margin requirement percentage",
        json_schema_extra={"example": "0.5000"}
    )

    spread_typical: Optional[Decimal] = Field(
        None,
        ge=0,
        decimal_places=5,
        description="Typical spread in pips/points",
        json_schema_extra={"example": "1.50000"}
    )

    commission: Optional[Decimal] = Field(
        Decimal('0'),
        ge=0,
        decimal_places=2,
        description="Commission per lot",
        json_schema_extra={"example": "7.00"}
    )

    swap_long: Optional[Decimal] = Field(
        Decimal('0'),
        decimal_places=2,
        description="Swap rate for long positions",
        json_schema_extra={"example": "-2.50"}
    )

    swap_short: Optional[Decimal] = Field(
        Decimal('0'),
        decimal_places=2,
        description="Swap rate for short positions",
        json_schema_extra={"example": "1.20"}
    )

    trade_mode: str = Field(
        "FULL",
        description="Trading mode",
        json_schema_extra={"example": "FULL"}
    )

    is_tradeable: bool = Field(
        True,
        description="Whether the symbol is currently tradeable",
        json_schema_extra={"example": True}
    )

    @field_validator('trade_mode')
    def validate_trade_mode(cls, v):
        """Validate trade mode."""
        valid_modes = ['FULL', 'LONG_ONLY', 'SHORT_ONLY', 'CLOSE_ONLY', 'DISABLED']
        if v.upper() not in valid_modes:
            raise ValueError(f'Trade mode must be one of: {", ".join(valid_modes)}')
        return v.upper()

    @field_validator('max_lot')
    def validate_max_lot_greater_than_min(cls, v, values):
        """Validate that max_lot is greater than min_lot."""
        min_lot = values.get('min_lot')
        if min_lot and v <= min_lot:
            raise ValueError('max_lot must be greater than min_lot')
        return v


class SymbolUpdateSchema(UpdateRequestSchema):
    """Schema for updating a symbol."""

    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Full name of the symbol",
        json_schema_extra={"example": "Euro vs US Dollar"}
    )

    symbol_group_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Reference to symbol group",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    exchange: Optional[str] = Field(
        None,
        max_length=50,
        description="Exchange name",
        json_schema_extra={"example": "NYSE"}
    )

    digits: Optional[int] = Field(
        None,
        ge=0,
        le=8,
        description="Number of decimal places for price quotes",
        json_schema_extra={"example": 5}
    )

    spread_typical: Optional[Decimal] = Field(
        None,
        ge=0,
        decimal_places=5,
        description="Typical spread in pips/points",
        json_schema_extra={"example": "1.50000"}
    )

    commission: Optional[Decimal] = Field(
        None,
        ge=0,
        decimal_places=2,
        description="Commission per lot",
        json_schema_extra={"example": "7.00"}
    )

    swap_long: Optional[Decimal] = Field(
        None,
        decimal_places=2,
        description="Swap rate for long positions",
        json_schema_extra={"example": "-2.50"}
    )

    swap_short: Optional[Decimal] = Field(
        None,
        decimal_places=2,
        description="Swap rate for short positions",
        json_schema_extra={"example": "1.20"}
    )

    trade_mode: Optional[str] = Field(
        None,
        description="Trading mode",
        json_schema_extra={"example": "FULL"}
    )

    is_tradeable: Optional[bool] = Field(
        None,
        description="Whether the symbol is currently tradeable",
        json_schema_extra={"example": True}
    )

    @field_validator('trade_mode')
    def validate_trade_mode(cls, v):
        """Validate trade mode."""
        if v is None:
            return v
        valid_modes = ['FULL', 'LONG_ONLY', 'SHORT_ONLY', 'CLOSE_ONLY', 'DISABLED']
        if v.upper() not in valid_modes:
            raise ValueError(f'Trade mode must be one of: {", ".join(valid_modes)}')
        return v.upper()


class SymbolQuoteUpdateSchema(CreateRequestSchema):
    """Schema for updating symbol quotes."""

    bid: Decimal = Field(
        ...,
        gt=0,
        decimal_places=8,
        description="Current bid price",
        json_schema_extra={"example": "1.23456"}
    )

    ask: Decimal = Field(
        ...,
        gt=0,
        decimal_places=8,
        description="Current ask price",
        json_schema_extra={"example": "1.23458"}
    )

    volume: Optional[Decimal] = Field(
        None,
        ge=0,
        decimal_places=8,
        description="Current volume",
        json_schema_extra={"example": "1000.50000000"}
    )

    @field_validator('ask')
    def validate_ask_greater_than_bid(cls, v, values):
        """Validate that ask price is greater than or equal to bid price."""
        bid = values.get('bid')
        if bid and v < bid:
            raise ValueError('ask price must be greater than or equal to bid price')
        return v


class SymbolResponseSchema(ResponseSchema):
    """Schema for symbol responses."""

    symbol: str = Field(
        ...,
        description="Symbol identifier",
        json_schema_extra={"example": "EURUSD"}
    )

    name: str = Field(
        ...,
        description="Full name of the symbol",
        json_schema_extra={"example": "Euro vs US Dollar"}
    )

    symbol_group_id: Optional[str] = Field(
        None,
        description="Reference to symbol group",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    market: str = Field(
        ...,
        description="Market identifier",
        json_schema_extra={"example": "FOREX"}
    )

    exchange: Optional[str] = Field(
        None,
        description="Exchange name",
        json_schema_extra={"example": "NYSE"}
    )

    base_currency: str = Field(
        ...,
        description="Base currency",
        json_schema_extra={"example": "EUR"}
    )

    quote_currency: str = Field(
        ...,
        description="Quote currency",
        json_schema_extra={"example": "USD"}
    )

    digits: int = Field(
        ...,
        description="Number of decimal places for price quotes",
        json_schema_extra={"example": 5}
    )

    point: Decimal = Field(
        ...,
        description="Point value (smallest price change)",
        json_schema_extra={"example": "0.00001"}
    )

    tick_size: Decimal = Field(
        ...,
        description="Minimum price change step",
        json_schema_extra={"example": "0.00001"}
    )

    tick_value: Decimal = Field(
        ...,
        description="Value of one tick in account currency",
        json_schema_extra={"example": "1.00"}
    )

    contract_size: Decimal = Field(
        ...,
        description="Standard contract size",
        json_schema_extra={"example": "100000.00000000"}
    )

    min_lot: Decimal = Field(
        ...,
        description="Minimum lot size",
        json_schema_extra={"example": "0.01000000"}
    )

    max_lot: Decimal = Field(
        ...,
        description="Maximum lot size",
        json_schema_extra={"example": "100.00000000"}
    )

    lot_step: Decimal = Field(
        ...,
        description="Lot size step",
        json_schema_extra={"example": "0.01000000"}
    )

    margin_initial: Optional[Decimal] = Field(
        None,
        description="Initial margin requirement percentage",
        json_schema_extra={"example": "1.0000"}
    )

    margin_maintenance: Optional[Decimal] = Field(
        None,
        description="Maintenance margin requirement percentage",
        json_schema_extra={"example": "0.5000"}
    )

    spread_typical: Optional[Decimal] = Field(
        None,
        description="Typical spread in pips/points",
        json_schema_extra={"example": "1.50000"}
    )

    commission: Optional[Decimal] = Field(
        None,
        description="Commission per lot",
        json_schema_extra={"example": "7.00"}
    )

    swap_long: Optional[Decimal] = Field(
        None,
        description="Swap rate for long positions",
        json_schema_extra={"example": "-2.50"}
    )

    swap_short: Optional[Decimal] = Field(
        None,
        description="Swap rate for short positions",
        json_schema_extra={"example": "1.20"}
    )

    trade_mode: str = Field(
        ...,
        description="Trading mode",
        json_schema_extra={"example": "FULL"}
    )

    is_tradeable: bool = Field(
        ...,
        description="Whether the symbol is currently tradeable",
        json_schema_extra={"example": True}
    )

    # Current quote information
    last_bid: Optional[Decimal] = Field(
        None,
        description="Last bid price",
        json_schema_extra={"example": "1.23456"}
    )

    last_ask: Optional[Decimal] = Field(
        None,
        description="Last ask price",
        json_schema_extra={"example": "1.23458"}
    )

    last_volume: Optional[Decimal] = Field(
        None,
        description="Last volume",
        json_schema_extra={"example": "1000.50000000"}
    )

    # Calculated fields
    spread: Optional[Decimal] = Field(
        None,
        description="Current spread (ask - bid)",
        json_schema_extra={"example": "0.00002"}
    )

    spread_pips: Optional[Decimal] = Field(
        None,
        description="Current spread in pips",
        json_schema_extra={"example": "2.0"}
    )

    mid_price: Optional[Decimal] = Field(
        None,
        description="Mid price ((bid + ask) / 2)",
        json_schema_extra={"example": "1.23457"}
    )


# Symbol Session Schemas
class SymbolSessionCreateSchema(CreateRequestSchema):
    """Schema for creating a symbol session."""

    symbol_id: str = Field(
        ...,
        min_length=36,
        max_length=36,
        description="Reference to symbol",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    session_name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Session name",
        json_schema_extra={"example": "LONDON"}
    )

    session_start: int = Field(
        ...,
        ge=0,
        le=10080,  # Maximum minutes in a week
        description="Session start time in minutes from Sunday 00:00 UTC",
        json_schema_extra={"example": 480}  # Monday 08:00
    )

    session_end: int = Field(
        ...,
        ge=0,
        le=10080,
        description="Session end time in minutes from Sunday 00:00 UTC",
        json_schema_extra={"example": 1020}  # Monday 17:00
    )

    is_active: bool = Field(
        True,
        description="Whether the session is active",
        json_schema_extra={"example": True}
    )

    trade_mode: str = Field(
        "FULL",
        description="Trading mode during this session",
        json_schema_extra={"example": "FULL"}
    )

    @field_validator('session_name')
    def validate_session_name(cls, v):
        """Validate session name."""
        valid_sessions = ['SYDNEY', 'TOKYO', 'LONDON', 'NEW_YORK', 'FRANKFURT', 'HONG_KONG']
        if v.upper() not in valid_sessions:
            raise ValueError(f'Session name must be one of: {", ".join(valid_sessions)}')
        return v.upper()

    @field_validator('trade_mode')
    def validate_trade_mode(cls, v):
        """Validate trade mode."""
        valid_modes = ['FULL', 'LONG_ONLY', 'SHORT_ONLY', 'CLOSE_ONLY', 'DISABLED']
        if v.upper() not in valid_modes:
            raise ValueError(f'Trade mode must be one of: {", ".join(valid_modes)}')
        return v.upper()


class SymbolSessionResponseSchema(ResponseSchema):
    """Schema for symbol session responses."""

    symbol_id: str = Field(
        ...,
        description="Reference to symbol",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    session_name: str = Field(
        ...,
        description="Session name",
        json_schema_extra={"example": "LONDON"}
    )

    session_start: int = Field(
        ...,
        description="Session start time in minutes from Sunday 00:00 UTC",
        json_schema_extra={"example": 480}
    )

    session_end: int = Field(
        ...,
        description="Session end time in minutes from Sunday 00:00 UTC",
        json_schema_extra={"example": 1020}
    )

    is_active: bool = Field(
        ...,
        description="Whether the session is active",
        json_schema_extra={"example": True}
    )

    trade_mode: str = Field(
        ...,
        description="Trading mode during this session",
        json_schema_extra={"example": "FULL"}
    )


# List and filter schemas
class SymbolFilterSchema(CreateRequestSchema):
    """Schema for filtering symbols."""

    symbol: Optional[str] = Field(
        None,
        description="Filter by symbol name (partial match)",
        json_schema_extra={"example": "EUR"}
    )

    market: Optional[str] = Field(
        None,
        description="Filter by market",
        json_schema_extra={"example": "FOREX"}
    )

    base_currency: Optional[str] = Field(
        None,
        min_length=3,
        max_length=3,
        description="Filter by base currency",
        json_schema_extra={"example": "EUR"}
    )

    quote_currency: Optional[str] = Field(
        None,
        min_length=3,
        max_length=3,
        description="Filter by quote currency",
        json_schema_extra={"example": "USD"}
    )

    symbol_group_id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Filter by symbol group",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )

    is_tradeable: Optional[bool] = Field(
        None,
        description="Filter by tradeable status",
        json_schema_extra={"example": True}
    )


class SymbolListResponseSchema(CreateRequestSchema):
    """Schema for symbol list responses."""

    items: List[SymbolResponseSchema] = Field(
        default_factory=list,
        description="List of symbols"
    )

    total: int = Field(
        ...,
        description="Total number of symbols",
        json_schema_extra={"example": 100}
    )


class SymbolGroupListResponseSchema(CreateRequestSchema):
    """Schema for symbol group list responses."""

    items: List[SymbolGroupResponseSchema] = Field(
        default_factory=list,
        description="List of symbol groups"
    )

    total: int = Field(
        ...,
        description="Total number of symbol groups",
        json_schema_extra={"example": 10}
    )