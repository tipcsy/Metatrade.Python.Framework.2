"""
Base Pydantic schemas for the MetaTrader Python Framework.

This module provides base schema classes and common validation patterns
for API requests, responses, and data validation throughout the system.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.config import ConfigDict


class BaseSchema(BaseModel):
    """
    Base schema class for all Pydantic models.

    Provides common configuration and validation patterns
    for consistent data handling across the application.
    """

    model_config = ConfigDict(
        # Allow extra fields but don't include them in serialization
        extra='forbid',
        # Use enum values instead of enum objects
        use_enum_values=True,
        # Validate assignment of new values
        validate_assignment=True,
        # Allow population by field name and alias
        populate_by_name=True,
        # Serialize by alias
        ser_by_alias=True,
        # Validate all fields
        validate_all=True,
        # JSON schema extra
        json_schema_extra={
            "examples": []
        }
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: Optional[datetime] = Field(
        None,
        description="Record creation timestamp (UTC)",
        json_schema_extra={"example": "2024-01-01T00:00:00Z"}
    )

    updated_at: Optional[datetime] = Field(
        None,
        description="Record last update timestamp (UTC)",
        json_schema_extra={"example": "2024-01-01T00:00:00Z"}
    )


class SoftDeleteMixin(BaseModel):
    """Mixin for soft delete fields."""

    deleted_at: Optional[datetime] = Field(
        None,
        description="Record deletion timestamp (UTC)",
        json_schema_extra={"example": "2024-01-01T00:00:00Z"}
    )

    is_deleted: Optional[bool] = Field(
        None,
        description="Whether the record is soft deleted",
        json_schema_extra={"example": False}
    )


class AuditMixin(BaseModel):
    """Mixin for audit trail fields."""

    created_by: Optional[str] = Field(
        None,
        max_length=100,
        description="User who created the record",
        json_schema_extra={"example": "user123"}
    )

    updated_by: Optional[str] = Field(
        None,
        max_length=100,
        description="User who last updated the record",
        json_schema_extra={"example": "user456"}
    )


class VersionMixin(BaseModel):
    """Mixin for optimistic locking version field."""

    version: Optional[int] = Field(
        None,
        ge=1,
        description="Version number for optimistic locking",
        json_schema_extra={"example": 1}
    )


class BaseEntitySchema(BaseSchema, TimestampMixin, SoftDeleteMixin, AuditMixin, VersionMixin):
    """
    Base schema for database entities.

    Includes all common fields that appear in database models:
    - ID field
    - Timestamp fields
    - Soft delete fields
    - Audit fields
    - Version field
    """

    id: Optional[str] = Field(
        None,
        min_length=36,
        max_length=36,
        description="Unique identifier (UUID)",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )


class CreateRequestSchema(BaseSchema):
    """
    Base schema for create requests.

    Excludes read-only fields that are set by the system.
    """

    pass


class UpdateRequestSchema(BaseSchema):
    """
    Base schema for update requests.

    Allows partial updates and includes version for optimistic locking.
    """

    version: Optional[int] = Field(
        None,
        ge=1,
        description="Current version for optimistic locking",
        json_schema_extra={"example": 1}
    )


class ResponseSchema(BaseEntitySchema):
    """
    Base schema for API responses.

    Includes all entity fields for complete data representation.
    """

    pass


class ListResponseSchema(BaseSchema):
    """
    Base schema for paginated list responses.

    Provides consistent structure for list endpoints with pagination.
    """

    items: List[Any] = Field(
        default_factory=list,
        description="List of items"
    )

    total: int = Field(
        ge=0,
        description="Total number of items",
        json_schema_extra={"example": 100}
    )

    page: int = Field(
        ge=1,
        description="Current page number",
        json_schema_extra={"example": 1}
    )

    size: int = Field(
        ge=1,
        le=1000,
        description="Page size",
        json_schema_extra={"example": 50}
    )

    pages: int = Field(
        ge=0,
        description="Total number of pages",
        json_schema_extra={"example": 2}
    )

    has_next: bool = Field(
        description="Whether there is a next page",
        json_schema_extra={"example": True}
    )

    has_prev: bool = Field(
        description="Whether there is a previous page",
        json_schema_extra={"example": False}
    )


class ErrorSchema(BaseSchema):
    """Schema for error responses."""

    error_code: str = Field(
        description="Error code identifier",
        json_schema_extra={"example": "VALIDATION_ERROR"}
    )

    message: str = Field(
        description="Human-readable error message",
        json_schema_extra={"example": "Validation failed for field 'amount'"}
    )

    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details",
        json_schema_extra={"example": {"field": "amount", "constraint": "must be positive"}}
    )

    field: Optional[str] = Field(
        None,
        description="Field that caused the error",
        json_schema_extra={"example": "amount"}
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp",
        json_schema_extra={"example": "2024-01-01T00:00:00Z"}
    )


class ValidationErrorSchema(BaseSchema):
    """Schema for validation error responses."""

    errors: List[Dict[str, Any]] = Field(
        description="List of validation errors",
        json_schema_extra={
            "example": [
                {
                    "field": "amount",
                    "message": "must be greater than 0",
                    "value": -10,
                    "constraint": "gt"
                }
            ]
        }
    )


# Common field validators
def validate_positive_decimal(value: Decimal) -> Decimal:
    """Validate that a decimal value is positive."""
    if value <= 0:
        raise ValueError("Value must be positive")
    return value


def validate_non_negative_decimal(value: Decimal) -> Decimal:
    """Validate that a decimal value is non-negative."""
    if value < 0:
        raise ValueError("Value must be non-negative")
    return value


def validate_percentage(value: Decimal) -> Decimal:
    """Validate that a decimal value is a valid percentage (0-100)."""
    if not (0 <= value <= 100):
        raise ValueError("Percentage must be between 0 and 100")
    return value


def validate_currency_code(value: str) -> str:
    """Validate currency code format."""
    if not value or len(value) != 3 or not value.isupper():
        raise ValueError("Currency code must be 3 uppercase letters")
    return value


def validate_symbol_name(value: str) -> str:
    """Validate trading symbol name format."""
    if not value or len(value) < 2 or len(value) > 50:
        raise ValueError("Symbol name must be between 2 and 50 characters")
    # Allow only alphanumeric characters, dots, and underscores
    if not all(c.isalnum() or c in '._' for c in value):
        raise ValueError("Symbol name can only contain letters, numbers, dots, and underscores")
    return value.upper()


# Common field types with validation
PositiveDecimal = Field(
    ...,
    gt=0,
    description="Positive decimal value",
    json_schema_extra={"example": "100.50"}
)

NonNegativeDecimal = Field(
    ...,
    ge=0,
    description="Non-negative decimal value",
    json_schema_extra={"example": "50.25"}
)

PercentageDecimal = Field(
    ...,
    ge=0,
    le=100,
    description="Percentage value (0-100)",
    json_schema_extra={"example": "25.50"}
)

CurrencyCode = Field(
    ...,
    min_length=3,
    max_length=3,
    pattern="^[A-Z]{3}$",
    description="Three-letter currency code",
    json_schema_extra={"example": "USD"}
)

SymbolName = Field(
    ...,
    min_length=2,
    max_length=50,
    pattern="^[A-Z0-9._]+$",
    description="Trading symbol name",
    json_schema_extra={"example": "EURUSD"}
)

# Custom field types for common trading concepts
PriceField = Field(
    ...,
    gt=0,
    decimal_places=8,
    description="Price value with up to 8 decimal places",
    json_schema_extra={"example": "1.23456789"}
)

VolumeField = Field(
    ...,
    ge=0,
    decimal_places=8,
    description="Volume value with up to 8 decimal places",
    json_schema_extra={"example": "100000.50000000"}
)

LotSizeField = Field(
    ...,
    gt=0,
    decimal_places=2,
    description="Lot size with up to 2 decimal places",
    json_schema_extra={"example": "0.10"}
)

# Validation schemas for common patterns
class FilterSchema(BaseSchema):
    """Base schema for filtering options."""

    search: Optional[str] = Field(
        None,
        max_length=255,
        description="Search term",
        json_schema_extra={"example": "EURUSD"}
    )

    active_only: Optional[bool] = Field(
        True,
        description="Filter to active records only",
        json_schema_extra={"example": True}
    )


class SortingSchema(BaseSchema):
    """Schema for sorting options."""

    sort_by: Optional[str] = Field(
        "created_at",
        description="Field to sort by",
        json_schema_extra={"example": "created_at"}
    )

    sort_order: Optional[str] = Field(
        "desc",
        pattern="^(asc|desc)$",
        description="Sort order (asc or desc)",
        json_schema_extra={"example": "desc"}
    )


class PaginationSchema(BaseSchema):
    """Schema for pagination parameters."""

    page: int = Field(
        1,
        ge=1,
        description="Page number (1-based)",
        json_schema_extra={"example": 1}
    )

    size: int = Field(
        50,
        ge=1,
        le=1000,
        description="Page size (max 1000)",
        json_schema_extra={"example": 50}
    )


class DateRangeSchema(BaseSchema):
    """Schema for date range filtering."""

    start_date: Optional[datetime] = Field(
        None,
        description="Start date (inclusive)",
        json_schema_extra={"example": "2024-01-01T00:00:00Z"}
    )

    end_date: Optional[datetime] = Field(
        None,
        description="End date (inclusive)",
        json_schema_extra={"example": "2024-12-31T23:59:59Z"}
    )

    @root_validator
    def validate_date_range(cls, values):
        """Validate that start_date is before end_date."""
        start_date = values.get('start_date')
        end_date = values.get('end_date')

        if start_date and end_date and start_date >= end_date:
            raise ValueError('start_date must be before end_date')

        return values


# Health check schemas
class HealthCheckSchema(BaseSchema):
    """Schema for health check responses."""

    status: str = Field(
        description="Overall health status",
        json_schema_extra={"example": "healthy"}
    )

    timestamp: datetime = Field(
        description="Health check timestamp",
        json_schema_extra={"example": "2024-01-01T00:00:00Z"}
    )

    checks: Dict[str, Any] = Field(
        description="Individual component health checks",
        json_schema_extra={
            "example": {
                "database": {"status": "healthy", "response_time_ms": 15},
                "cache": {"status": "healthy", "response_time_ms": 5}
            }
        }
    )

    uptime_seconds: Optional[int] = Field(
        None,
        description="System uptime in seconds",
        json_schema_extra={"example": 86400}
    )


# Configuration schemas
class ConfigurationSchema(BaseSchema):
    """Base schema for configuration objects."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Configuration name",
        json_schema_extra={"example": "default_config"}
    )

    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Configuration description",
        json_schema_extra={"example": "Default trading configuration"}
    )

    is_active: bool = Field(
        True,
        description="Whether the configuration is active",
        json_schema_extra={"example": True}
    )

    config_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuration parameters",
        json_schema_extra={"example": {"param1": "value1", "param2": 123}}
    )