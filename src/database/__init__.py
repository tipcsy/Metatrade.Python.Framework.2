"""
Database package for the MetaTrader Python Framework.

This package provides a complete database layer implementation including:
- SQLAlchemy models with enterprise-grade features
- Business logic services with caching support
- Redis caching layer with intelligent cache management
- Pydantic schemas for data validation
- Database health monitoring and performance tracking
- Alembic migrations for schema management

The database layer is designed for:
- Multi-million record support with partitioning
- ACID compliance with full transaction management
- Enterprise-grade connection pooling
- Audit trails and soft delete functionality
- High-performance caching with Redis
- Comprehensive monitoring and health checks

Key Components:
- models: SQLAlchemy database models
- services: Business logic layer with caching
- schemas: Pydantic validation schemas
- cache: Redis caching infrastructure
- utils: Database utilities and health checks
- migrations: Alembic migration management

Usage:
    from src.database import initialize_database, get_database_manager
    from src.database.services import SymbolService, AccountService
    from src.database.cache import initialize_redis

    # Initialize database
    db_manager = initialize_database("postgresql://user:pass@host/db")

    # Initialize caching
    initialize_redis(host="localhost", port=6379)

    # Use services
    symbol_service = SymbolService()
    symbols = symbol_service.get_all()
"""

from __future__ import annotations

# Core database components
from .database import (
    DatabaseManager,
    get_database_manager,
    initialize_database,
    close_database,
    get_session,
    get_async_session,
)

# Models
from .models import *

# Services
from .services import *

# Schemas
from .schemas import *

# Cache
from .cache import (
    RedisClient,
    CacheManager,
    get_redis_client,
    get_cache_manager,
    initialize_redis,
    close_redis,
    cached,
    cached_async,
    cache_invalidate,
)

# Utilities
from .utils.health_check import (
    DatabaseHealthChecker,
    run_health_check,
    run_async_health_check,
    get_health_report,
)

# Export all public components
__all__ = [
    # Core database
    "DatabaseManager",
    "get_database_manager",
    "initialize_database",
    "close_database",
    "get_session",
    "get_async_session",

    # Cache
    "RedisClient",
    "CacheManager",
    "get_redis_client",
    "get_cache_manager",
    "initialize_redis",
    "close_redis",
    "cached",
    "cached_async",
    "cache_invalidate",

    # Health monitoring
    "DatabaseHealthChecker",
    "run_health_check",
    "run_async_health_check",
    "get_health_report",

    # Note: Models, services, and schemas are exported via their respective packages
]

# Version information
__version__ = "2.0.0"
__description__ = "Enterprise-grade database layer for MetaTrader Python Framework"
__author__ = "MetaTrader Python Framework Team"