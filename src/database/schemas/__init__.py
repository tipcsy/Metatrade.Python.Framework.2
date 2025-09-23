"""
Database schemas package for the MetaTrader Python Framework.

This package contains all Pydantic schema definitions for API validation
and data serialization organized by domain.
"""

from __future__ import annotations

# Import base schemas
from .base import (
    BaseSchema,
    BaseEntitySchema,
    CreateRequestSchema,
    UpdateRequestSchema,
    ResponseSchema,
    ListResponseSchema,
    ErrorSchema,
    ValidationErrorSchema,
    FilterSchema,
    SortingSchema,
    PaginationSchema,
    DateRangeSchema,
    HealthCheckSchema,
    ConfigurationSchema,
    # Mixins
    TimestampMixin,
    SoftDeleteMixin,
    AuditMixin,
    VersionMixin,
    # Common field types
    PositiveDecimal,
    NonNegativeDecimal,
    PercentageDecimal,
    CurrencyCode,
    SymbolName,
    PriceField,
    VolumeField,
    LotSizeField,
)

# Import symbol schemas
from .symbols import (
    # Symbol Group schemas
    SymbolGroupCreateSchema,
    SymbolGroupUpdateSchema,
    SymbolGroupResponseSchema,
    SymbolGroupListResponseSchema,
    # Symbol schemas
    SymbolCreateSchema,
    SymbolUpdateSchema,
    SymbolQuoteUpdateSchema,
    SymbolResponseSchema,
    SymbolListResponseSchema,
    # Symbol Session schemas
    SymbolSessionCreateSchema,
    SymbolSessionResponseSchema,
    # Filter schemas
    SymbolFilterSchema,
)

# Import account schemas
from .accounts import (
    # User schemas
    UserCreateSchema,
    UserUpdateSchema,
    UserPasswordChangeSchema,
    UserResponseSchema,
    UserListResponseSchema,
    # Account schemas
    AccountCreateSchema,
    AccountUpdateSchema,
    AccountBalanceUpdateSchema,
    AccountResponseSchema,
    AccountListResponseSchema,
    # Transaction schemas
    TransactionCreateSchema,
    TransactionResponseSchema,
    TransactionListResponseSchema,
    # Account Settings schemas
    AccountSettingsCreateSchema,
    AccountSettingsUpdateSchema,
    AccountSettingsResponseSchema,
    # Filter schemas
    AccountFilterSchema,
)

# Export all schemas for external use
__all__ = [
    # Base schemas
    "BaseSchema",
    "BaseEntitySchema",
    "CreateRequestSchema",
    "UpdateRequestSchema",
    "ResponseSchema",
    "ListResponseSchema",
    "ErrorSchema",
    "ValidationErrorSchema",
    "FilterSchema",
    "SortingSchema",
    "PaginationSchema",
    "DateRangeSchema",
    "HealthCheckSchema",
    "ConfigurationSchema",

    # Mixins
    "TimestampMixin",
    "SoftDeleteMixin",
    "AuditMixin",
    "VersionMixin",

    # Common field types
    "PositiveDecimal",
    "NonNegativeDecimal",
    "PercentageDecimal",
    "CurrencyCode",
    "SymbolName",
    "PriceField",
    "VolumeField",
    "LotSizeField",

    # Symbol schemas
    "SymbolGroupCreateSchema",
    "SymbolGroupUpdateSchema",
    "SymbolGroupResponseSchema",
    "SymbolGroupListResponseSchema",
    "SymbolCreateSchema",
    "SymbolUpdateSchema",
    "SymbolQuoteUpdateSchema",
    "SymbolResponseSchema",
    "SymbolListResponseSchema",
    "SymbolSessionCreateSchema",
    "SymbolSessionResponseSchema",
    "SymbolFilterSchema",

    # Account schemas
    "UserCreateSchema",
    "UserUpdateSchema",
    "UserPasswordChangeSchema",
    "UserResponseSchema",
    "UserListResponseSchema",
    "AccountCreateSchema",
    "AccountUpdateSchema",
    "AccountBalanceUpdateSchema",
    "AccountResponseSchema",
    "AccountListResponseSchema",
    "TransactionCreateSchema",
    "TransactionResponseSchema",
    "TransactionListResponseSchema",
    "AccountSettingsCreateSchema",
    "AccountSettingsUpdateSchema",
    "AccountSettingsResponseSchema",
    "AccountFilterSchema",
]