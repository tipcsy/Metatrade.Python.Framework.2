"""
Base model classes for the MetaTrader Python Framework database layer.

This module provides the foundational database model classes with enterprise-grade
features including audit trails, soft delete, timezone awareness, and performance optimizations.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type, TypeVar, Union

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    String,
    event,
    inspect,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func

from src.core.exceptions import DatabaseError
from src.core.logging import get_logger

# Type variable for model classes
ModelType = TypeVar("ModelType", bound="BaseModel")

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """
    Base declarative class for all database models.

    This class provides the foundation for all SQLAlchemy models with
    modern SQLAlchemy 2.0+ syntax and type safety.
    """

    # Configure SQLAlchemy to use UUID as primary key type for PostgreSQL
    type_annotation_map = {
        str: String(255),  # Default string length
    }


class TimestampMixin:
    """
    Mixin class to add timestamp fields to models.

    Provides created_at and updated_at fields with automatic management.
    All timestamps are stored in UTC timezone.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
        doc="Timestamp when the record was created (UTC)"
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
        doc="Timestamp when the record was last updated (UTC)"
    )


class SoftDeleteMixin:
    """
    Mixin class to add soft delete functionality to models.

    Provides deleted_at field and related query methods for soft deletion.
    """

    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        default=None,
        nullable=True,
        index=True,
        doc="Timestamp when the record was soft deleted (UTC)"
    )

    @hybrid_property
    def is_deleted(self) -> bool:
        """Check if the record is soft deleted."""
        return self.deleted_at is not None

    def soft_delete(self) -> None:
        """Soft delete the record by setting deleted_at timestamp."""
        self.deleted_at = datetime.now(timezone.utc)
        logger.debug(f"Soft deleted {self.__class__.__name__} record")

    def restore(self) -> None:
        """Restore a soft deleted record by clearing deleted_at."""
        self.deleted_at = None
        logger.debug(f"Restored {self.__class__.__name__} record")


class AuditMixin:
    """
    Mixin class to add audit trail functionality to models.

    Tracks who created and last modified records for compliance and debugging.
    """

    created_by: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        doc="User ID or system identifier that created the record"
    )

    updated_by: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        doc="User ID or system identifier that last updated the record"
    )


class VersionMixin:
    """
    Mixin class to add optimistic locking support to models.

    Provides version field for preventing concurrent update conflicts.
    """

    version: Mapped[int] = mapped_column(
        default=1,
        nullable=False,
        doc="Version number for optimistic locking"
    )


class BaseModel(Base, TimestampMixin, SoftDeleteMixin, AuditMixin, VersionMixin):
    """
    Abstract base model class for all application models.

    This class combines all mixins and provides common functionality
    for all database models including:
    - UUID primary keys
    - Automatic timestamps
    - Soft delete functionality
    - Audit trail support
    - Optimistic locking
    - Utility methods for serialization and querying
    """

    __abstract__ = True

    # Use UUID as primary key for better distributed system support
    id: Mapped[str] = mapped_column(
        String(36),  # UUID string length
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        nullable=False,
        doc="Unique identifier for the record"
    )

    @declared_attr
    def __tablename__(cls) -> str:
        """
        Generate table name from class name.

        Converts CamelCase to snake_case for table names.
        Example: MarketData -> market_data
        """
        import re
        # Convert CamelCase to snake_case
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def to_dict(self, exclude_private: bool = True, exclude_none: bool = False) -> Dict[str, Any]:
        """
        Convert model instance to dictionary.

        Args:
            exclude_private: Whether to exclude private attributes (starting with _)
            exclude_none: Whether to exclude None values

        Returns:
            Dictionary representation of the model
        """
        result = {}

        # Get all mapped columns
        mapper = inspect(self.__class__)

        for column in mapper.columns:
            value = getattr(self, column.name)

            # Skip private attributes if requested
            if exclude_private and column.name.startswith('_'):
                continue

            # Skip None values if requested
            if exclude_none and value is None:
                continue

            # Convert datetime objects to ISO format strings
            if isinstance(value, datetime):
                value = value.isoformat()

            result[column.name] = value

        return result

    def update_from_dict(self, data: Dict[str, Any], exclude_keys: Optional[set] = None) -> None:
        """
        Update model instance from dictionary.

        Args:
            data: Dictionary with field names and values
            exclude_keys: Set of keys to exclude from update

        Raises:
            DatabaseError: If invalid field names are provided
        """
        if exclude_keys is None:
            exclude_keys = {'id', 'created_at', 'created_by'}

        mapper = inspect(self.__class__)
        column_names = {column.name for column in mapper.columns}

        for key, value in data.items():
            if key in exclude_keys:
                continue

            if key not in column_names:
                raise DatabaseError(f"Invalid field name: {key}")

            setattr(self, key, value)

        # Update audit fields
        self.updated_at = datetime.now(timezone.utc)
        logger.debug(f"Updated {self.__class__.__name__} record with {len(data)} fields")

    @classmethod
    def get_table_name(cls) -> str:
        """Get the table name for this model."""
        return cls.__tablename__

    @classmethod
    def get_primary_key_column(cls) -> str:
        """Get the primary key column name."""
        return 'id'

    def get_pk_value(self) -> Any:
        """Get the primary key value for this instance."""
        return getattr(self, self.get_primary_key_column())

    def __repr__(self) -> str:
        """String representation of the model."""
        pk_value = self.get_pk_value()
        return f"<{self.__class__.__name__}(id={pk_value})>"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


# Event listeners for automatic audit trail updates
@event.listens_for(BaseModel, 'before_insert', propagate=True)
def before_insert_listener(mapper, connection, target):
    """Set created_by and updated_by on insert."""
    # In a real application, you would get the current user from context
    # For now, we'll use a system identifier
    if hasattr(target, 'created_by') and target.created_by is None:
        target.created_by = 'system'
    if hasattr(target, 'updated_by') and target.updated_by is None:
        target.updated_by = 'system'


@event.listens_for(BaseModel, 'before_update', propagate=True)
def before_update_listener(mapper, connection, target):
    """Set updated_by and increment version on update."""
    if hasattr(target, 'updated_by'):
        target.updated_by = 'system'  # In real app, get from current user context
    if hasattr(target, 'version'):
        target.version += 1


# Utility functions for model operations
def get_model_columns(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Get all columns for a model class.

    Args:
        model_class: The model class to inspect

    Returns:
        Dictionary mapping column names to column objects
    """
    mapper = inspect(model_class)
    return {column.name: column for column in mapper.columns}


def get_model_relationships(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Get all relationships for a model class.

    Args:
        model_class: The model class to inspect

    Returns:
        Dictionary mapping relationship names to relationship objects
    """
    mapper = inspect(model_class)
    return {rel.key: rel for rel in mapper.relationships}


def model_to_dict(
    instance: BaseModel,
    include_relationships: bool = False,
    exclude_private: bool = True
) -> Dict[str, Any]:
    """
    Convert a model instance to dictionary with optional relationship loading.

    Args:
        instance: The model instance to convert
        include_relationships: Whether to include relationship data
        exclude_private: Whether to exclude private attributes

    Returns:
        Dictionary representation of the model
    """
    result = instance.to_dict(exclude_private=exclude_private)

    if include_relationships:
        relationships = get_model_relationships(instance.__class__)
        for rel_name in relationships:
            rel_value = getattr(instance, rel_name, None)
            if rel_value is not None:
                if hasattr(rel_value, '__iter__') and not isinstance(rel_value, str):
                    # Collection relationship
                    result[rel_name] = [
                        item.to_dict(exclude_private=exclude_private)
                        for item in rel_value
                    ]
                else:
                    # Single relationship
                    result[rel_name] = rel_value.to_dict(exclude_private=exclude_private)

    return result