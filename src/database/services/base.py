"""
Base service classes for the MetaTrader Python Framework.

This module provides base service classes with common CRUD operations,
caching, validation, and business logic patterns.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from sqlalchemy import and_, desc, func, or_
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from src.core.exceptions import (
    DatabaseError,
    ValidationError,
    NotFoundError,
    DuplicateError,
)
from src.core.logging import get_logger
from src.database.database import get_database_manager
from src.database.cache import get_cache_manager, cached, cached_async, cache_invalidate
from src.database.models.base import BaseModel as DBBaseModel

# Type variables
ModelType = TypeVar("ModelType", bound=DBBaseModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)
ResponseSchemaType = TypeVar("ResponseSchemaType", bound=BaseModel)

logger = get_logger(__name__)


class BaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType, ResponseSchemaType], ABC):
    """
    Abstract base service class for database operations.

    Provides common CRUD operations, caching, validation, and business logic
    patterns for all domain services.
    """

    def __init__(
        self,
        model: Type[ModelType],
        cache_enabled: bool = True,
        cache_ttl: Optional[int] = None
    ):
        """
        Initialize base service.

        Args:
            model: SQLAlchemy model class
            cache_enabled: Whether to enable caching
            cache_ttl: Cache TTL in seconds
        """
        self.model = model
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.cache_type = model.__tablename__

        self._db_manager = get_database_manager()
        self._cache_manager = get_cache_manager()

        logger.debug(f"Initialized {self.__class__.__name__} for {model.__name__}")

    @property
    @abstractmethod
    def create_schema(self) -> Type[CreateSchemaType]:
        """Pydantic schema for creation."""
        pass

    @property
    @abstractmethod
    def update_schema(self) -> Type[UpdateSchemaType]:
        """Pydantic schema for updates."""
        pass

    @property
    @abstractmethod
    def response_schema(self) -> Type[ResponseSchemaType]:
        """Pydantic schema for responses."""
        pass

    def _get_cache_key(self, identifier: Union[str, int]) -> str:
        """Generate cache key for model instance."""
        return f"{identifier}"

    def _model_to_dict(self, instance: ModelType) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        return instance.to_dict(exclude_private=True)

    def _validate_create_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data for creation.

        Args:
            data: Data to validate

        Returns:
            Validated data

        Raises:
            ValidationError: If validation fails
        """
        try:
            schema = self.create_schema(**data)
            return schema.model_dump(exclude_unset=True)
        except Exception as e:
            logger.error(f"Validation error in {self.__class__.__name__}.create: {e}")
            raise ValidationError(f"Validation failed: {e}") from e

    def _validate_update_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data for update.

        Args:
            data: Data to validate

        Returns:
            Validated data

        Raises:
            ValidationError: If validation fails
        """
        try:
            schema = self.update_schema(**data)
            return schema.model_dump(exclude_unset=True)
        except Exception as e:
            logger.error(f"Validation error in {self.__class__.__name__}.update: {e}")
            raise ValidationError(f"Validation failed: {e}") from e

    def _apply_business_rules(self, instance: ModelType, operation: str) -> None:
        """
        Apply business rules to model instance.

        Args:
            instance: Model instance
            operation: Operation type (create, update, delete)
        """
        # Override in subclasses to implement specific business rules
        pass

    def _build_filters(self, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Build SQLAlchemy filters from dictionary.

        Args:
            filters: Filter conditions

        Returns:
            List of SQLAlchemy filter conditions
        """
        if not filters:
            return []

        conditions = []
        for key, value in filters.items():
            if hasattr(self.model, key):
                column = getattr(self.model, key)
                if isinstance(value, dict):
                    # Handle complex filters
                    for op, op_value in value.items():
                        if op == 'eq':
                            conditions.append(column == op_value)
                        elif op == 'ne':
                            conditions.append(column != op_value)
                        elif op == 'gt':
                            conditions.append(column > op_value)
                        elif op == 'gte':
                            conditions.append(column >= op_value)
                        elif op == 'lt':
                            conditions.append(column < op_value)
                        elif op == 'lte':
                            conditions.append(column <= op_value)
                        elif op == 'in':
                            conditions.append(column.in_(op_value))
                        elif op == 'like':
                            conditions.append(column.like(f"%{op_value}%"))
                        elif op == 'ilike':
                            conditions.append(column.ilike(f"%{op_value}%"))
                        elif op == 'is_null':
                            if op_value:
                                conditions.append(column.is_(None))
                            else:
                                conditions.append(column.isnot(None))
                else:
                    # Simple equality filter
                    conditions.append(column == value)

        return conditions

    def get_by_id(self, id: Union[str, int], use_cache: bool = True) -> Optional[ModelType]:
        """
        Get model instance by ID.

        Args:
            id: Model ID
            use_cache: Whether to use cache

        Returns:
            Model instance or None
        """
        cache_key = self._get_cache_key(id)

        # Try cache first
        if use_cache and self.cache_enabled:
            cached_data = self._cache_manager.get(self.cache_type, cache_key)
            if cached_data:
                logger.debug(f"Cache hit for {self.model.__name__} ID: {id}")
                return self.model(**cached_data)

        # Query database
        with self._db_manager.get_session() as session:
            instance = session.query(self.model).filter(self.model.id == id).first()

            if instance and use_cache and self.cache_enabled:
                # Cache the result
                self._cache_manager.set(
                    self.cache_type,
                    cache_key,
                    self._model_to_dict(instance),
                    self.cache_ttl
                )

            return instance

    async def get_by_id_async(
        self,
        id: Union[str, int],
        use_cache: bool = True
    ) -> Optional[ModelType]:
        """
        Asynchronously get model instance by ID.

        Args:
            id: Model ID
            use_cache: Whether to use cache

        Returns:
            Model instance or None
        """
        cache_key = self._get_cache_key(id)

        # Try cache first
        if use_cache and self.cache_enabled:
            cached_data = await self._cache_manager.get_async(self.cache_type, cache_key)
            if cached_data:
                logger.debug(f"Cache hit for {self.model.__name__} ID: {id}")
                return self.model(**cached_data)

        # Query database
        async with self._db_manager.get_async_session() as session:
            from sqlalchemy import select
            result = await session.execute(select(self.model).filter(self.model.id == id))
            instance = result.scalar_one_or_none()

            if instance and use_cache and self.cache_enabled:
                # Cache the result
                await self._cache_manager.set_async(
                    self.cache_type,
                    cache_key,
                    self._model_to_dict(instance),
                    self.cache_ttl
                )

            return instance

    def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include_deleted: bool = False
    ) -> List[ModelType]:
        """
        Get all model instances with filtering and pagination.

        Args:
            filters: Filter conditions
            order_by: Field to order by
            order_desc: Whether to order descending
            limit: Maximum number of results
            offset: Number of results to skip
            include_deleted: Whether to include soft deleted records

        Returns:
            List of model instances
        """
        with self._db_manager.get_session() as session:
            query = session.query(self.model)

            # Apply filters
            filter_conditions = self._build_filters(filters)
            if filter_conditions:
                query = query.filter(and_(*filter_conditions))

            # Handle soft delete
            if not include_deleted and hasattr(self.model, 'deleted_at'):
                query = query.filter(self.model.deleted_at.is_(None))

            # Apply ordering
            if order_by and hasattr(self.model, order_by):
                order_column = getattr(self.model, order_by)
                if order_desc:
                    query = query.order_by(desc(order_column))
                else:
                    query = query.order_by(order_column)

            # Apply pagination
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            return query.all()

    def count(
        self,
        filters: Optional[Dict[str, Any]] = None,
        include_deleted: bool = False
    ) -> int:
        """
        Count model instances.

        Args:
            filters: Filter conditions
            include_deleted: Whether to include soft deleted records

        Returns:
            Number of instances
        """
        with self._db_manager.get_session() as session:
            query = session.query(func.count(self.model.id))

            # Apply filters
            filter_conditions = self._build_filters(filters)
            if filter_conditions:
                query = query.filter(and_(*filter_conditions))

            # Handle soft delete
            if not include_deleted and hasattr(self.model, 'deleted_at'):
                query = query.filter(self.model.deleted_at.is_(None))

            return query.scalar()

    def create(self, data: Dict[str, Any], commit: bool = True) -> ModelType:
        """
        Create new model instance.

        Args:
            data: Creation data
            commit: Whether to commit transaction

        Returns:
            Created model instance

        Raises:
            ValidationError: If validation fails
            DuplicateError: If unique constraint violation
            DatabaseError: If database operation fails
        """
        try:
            # Validate data
            validated_data = self._validate_create_data(data)

            with self._db_manager.get_session() as session:
                # Create instance
                instance = self.model(**validated_data)

                # Apply business rules
                self._apply_business_rules(instance, 'create')

                # Save to database
                session.add(instance)

                if commit:
                    session.commit()
                    session.refresh(instance)

                    # Invalidate cache
                    if self.cache_enabled:
                        cache_key = self._get_cache_key(instance.id)
                        self._cache_manager.set(
                            self.cache_type,
                            cache_key,
                            self._model_to_dict(instance),
                            self.cache_ttl
                        )

                logger.info(f"Created {self.model.__name__} with ID: {instance.id}")
                return instance

        except Exception as e:
            logger.error(f"Error creating {self.model.__name__}: {e}")
            if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                raise DuplicateError(f"Duplicate {self.model.__name__}") from e
            raise DatabaseError(f"Failed to create {self.model.__name__}") from e

    def update(
        self,
        id: Union[str, int],
        data: Dict[str, Any],
        commit: bool = True
    ) -> Optional[ModelType]:
        """
        Update model instance.

        Args:
            id: Model ID
            data: Update data
            commit: Whether to commit transaction

        Returns:
            Updated model instance or None

        Raises:
            ValidationError: If validation fails
            NotFoundError: If instance not found
            DatabaseError: If database operation fails
        """
        try:
            # Validate data
            validated_data = self._validate_update_data(data)

            with self._db_manager.get_session() as session:
                # Get instance
                instance = session.query(self.model).filter(self.model.id == id).first()
                if not instance:
                    raise NotFoundError(f"{self.model.__name__} with ID {id} not found")

                # Update instance
                instance.update_from_dict(validated_data)

                # Apply business rules
                self._apply_business_rules(instance, 'update')

                if commit:
                    session.commit()
                    session.refresh(instance)

                    # Update cache
                    if self.cache_enabled:
                        cache_key = self._get_cache_key(instance.id)
                        self._cache_manager.set(
                            self.cache_type,
                            cache_key,
                            self._model_to_dict(instance),
                            self.cache_ttl
                        )

                logger.info(f"Updated {self.model.__name__} with ID: {id}")
                return instance

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error updating {self.model.__name__} ID {id}: {e}")
            raise DatabaseError(f"Failed to update {self.model.__name__}") from e

    def delete(self, id: Union[str, int], soft_delete: bool = True, commit: bool = True) -> bool:
        """
        Delete model instance.

        Args:
            id: Model ID
            soft_delete: Whether to use soft delete
            commit: Whether to commit transaction

        Returns:
            True if deleted successfully

        Raises:
            NotFoundError: If instance not found
            DatabaseError: If database operation fails
        """
        try:
            with self._db_manager.get_session() as session:
                # Get instance
                instance = session.query(self.model).filter(self.model.id == id).first()
                if not instance:
                    raise NotFoundError(f"{self.model.__name__} with ID {id} not found")

                # Apply business rules
                self._apply_business_rules(instance, 'delete')

                # Delete instance
                if soft_delete and hasattr(instance, 'soft_delete'):
                    instance.soft_delete()
                else:
                    session.delete(instance)

                if commit:
                    session.commit()

                    # Remove from cache
                    if self.cache_enabled:
                        cache_key = self._get_cache_key(id)
                        self._cache_manager.delete(self.cache_type, cache_key)

                logger.info(f"Deleted {self.model.__name__} with ID: {id}")
                return True

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error deleting {self.model.__name__} ID {id}: {e}")
            raise DatabaseError(f"Failed to delete {self.model.__name__}") from e

    def exists(self, id: Union[str, int]) -> bool:
        """
        Check if model instance exists.

        Args:
            id: Model ID

        Returns:
            True if exists
        """
        with self._db_manager.get_session() as session:
            return session.query(
                session.query(self.model).filter(self.model.id == id).exists()
            ).scalar()

    def get_or_create(
        self,
        defaults: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> tuple[ModelType, bool]:
        """
        Get existing instance or create new one.

        Args:
            defaults: Default values for creation
            **kwargs: Filter conditions

        Returns:
            Tuple of (instance, created_flag)
        """
        with self._db_manager.get_session() as session:
            # Try to get existing instance
            filter_conditions = self._build_filters(kwargs)
            query = session.query(self.model)
            if filter_conditions:
                query = query.filter(and_(*filter_conditions))

            instance = query.first()

            if instance:
                return instance, False

            # Create new instance
            create_data = kwargs.copy()
            if defaults:
                create_data.update(defaults)

            return self.create(create_data), True

    def bulk_create(self, data_list: List[Dict[str, Any]], batch_size: int = 1000) -> List[ModelType]:
        """
        Create multiple instances in bulk.

        Args:
            data_list: List of creation data
            batch_size: Batch size for processing

        Returns:
            List of created instances
        """
        created_instances = []

        with self._db_manager.get_session() as session:
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                batch_instances = []

                for data in batch:
                    # Validate data
                    validated_data = self._validate_create_data(data)
                    instance = self.model(**validated_data)
                    self._apply_business_rules(instance, 'create')
                    batch_instances.append(instance)

                # Add batch to session
                session.add_all(batch_instances)
                session.commit()

                # Refresh instances to get IDs
                for instance in batch_instances:
                    session.refresh(instance)

                created_instances.extend(batch_instances)

                logger.info(f"Created batch of {len(batch_instances)} {self.model.__name__} instances")

        return created_instances

    def bulk_update(
        self,
        updates: List[Dict[str, Any]],
        id_field: str = 'id'
    ) -> int:
        """
        Update multiple instances in bulk.

        Args:
            updates: List of update data with ID field
            id_field: Name of ID field

        Returns:
            Number of updated instances
        """
        updated_count = 0

        with self._db_manager.get_session() as session:
            for update_data in updates:
                if id_field not in update_data:
                    continue

                instance_id = update_data.pop(id_field)
                validated_data = self._validate_update_data(update_data)

                # Update instance
                updated = session.query(self.model).filter(
                    self.model.id == instance_id
                ).update(validated_data)

                updated_count += updated

            session.commit()

        # Clear cache for updated type
        if self.cache_enabled:
            self._cache_manager.invalidate_pattern(self.cache_type)

        logger.info(f"Bulk updated {updated_count} {self.model.__name__} instances")
        return updated_count

    def to_response_model(self, instance: ModelType) -> ResponseSchemaType:
        """
        Convert model instance to response schema.

        Args:
            instance: Model instance

        Returns:
            Response schema instance
        """
        data = self._model_to_dict(instance)
        return self.response_schema(**data)

    def to_response_models(self, instances: List[ModelType]) -> List[ResponseSchemaType]:
        """
        Convert list of model instances to response schemas.

        Args:
            instances: List of model instances

        Returns:
            List of response schema instances
        """
        return [self.to_response_model(instance) for instance in instances]


class CachedService(BaseService[ModelType, CreateSchemaType, UpdateSchemaType, ResponseSchemaType]):
    """
    Service with enhanced caching capabilities.

    Provides additional caching methods and cache management
    for frequently accessed data.
    """

    def __init__(
        self,
        model: Type[ModelType],
        cache_ttl: int = 300,
        aggressive_caching: bool = False
    ):
        """
        Initialize cached service.

        Args:
            model: SQLAlchemy model class
            cache_ttl: Cache TTL in seconds
            aggressive_caching: Whether to cache all queries
        """
        super().__init__(model, cache_enabled=True, cache_ttl=cache_ttl)
        self.aggressive_caching = aggressive_caching

    @cached("query", ttl=300)
    def cached_query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ModelType]:
        """
        Cached query method.

        Args:
            filters: Filter conditions
            order_by: Field to order by
            limit: Maximum number of results

        Returns:
            List of model instances
        """
        return self.get_all(filters=filters, order_by=order_by, limit=limit)

    def invalidate_all_cache(self) -> int:
        """
        Invalidate all cache for this model type.

        Returns:
            Number of cache entries invalidated
        """
        return self._cache_manager.invalidate_pattern(self.cache_type)

    def warm_cache(self, ids: List[Union[str, int]]) -> int:
        """
        Warm cache with specific instances.

        Args:
            ids: List of IDs to cache

        Returns:
            Number of instances cached
        """
        cached_count = 0

        for instance_id in ids:
            instance = self.get_by_id(instance_id, use_cache=False)
            if instance:
                cache_key = self._get_cache_key(instance_id)
                success = self._cache_manager.set(
                    self.cache_type,
                    cache_key,
                    self._model_to_dict(instance),
                    self.cache_ttl
                )
                if success:
                    cached_count += 1

        logger.info(f"Warmed cache for {cached_count} {self.model.__name__} instances")
        return cached_count