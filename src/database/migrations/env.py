"""
Alembic environment configuration for the MetaTrader Python Framework.

This module configures Alembic for database migrations with support for
both offline and online migration modes, custom migration tracking,
and integration with the application's configuration system.
"""

from __future__ import annotations

import asyncio
import logging
from logging.config import fileConfig
from pathlib import Path
from typing import Any, Dict

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

from src.core.config import get_settings
from src.core.logging import get_logger
from src.database.models.base import Base

# Import all model modules to ensure they are registered with SQLAlchemy
# This is critical for autogenerate to work properly
from src.database.models import (
    accounts,
    market_data,
    strategies,
    symbols,
    system,
    trading,
)

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Get application settings
settings = get_settings()
logger = get_logger(__name__)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

# Set the database URL from application settings
config.set_main_option("sqlalchemy.url", settings.get_database_url())


def get_url() -> str:
    """Get database URL from settings."""
    return settings.get_database_url()


def include_object(object, name, type_, reflected, compare_to):
    """
    Filter objects to include in migrations.

    This function allows fine-grained control over which database objects
    are included in the migration process.
    """
    # Skip tables that start with underscore (internal tables)
    if type_ == "table" and name.startswith("_"):
        return False

    # Skip certain system tables
    if type_ == "table" and name in ("alembic_version", "spatial_ref_sys"):
        return False

    # Include everything else
    return True


def include_name(name, type_, parent_names):
    """
    Filter names to include in migrations.

    This provides another level of filtering for migration objects.
    """
    if type_ == "schema":
        # Include only specific schemas if needed
        return name in (None, "public")
    else:
        return True


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        include_name=include_name,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,  # Enable batch mode for SQLite
    )

    with context.begin_transaction():
        context.run_migrations()

    logger.info("Offline migrations completed successfully")


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with a database connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        include_name=include_name,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True,  # Enable batch mode for SQLite
        transaction_per_migration=True,  # Each migration in its own transaction
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in async mode.

    This is used when the database URL is an async one.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    # Convert sync URL to async URL if needed
    url = get_url()
    if url.startswith('sqlite://'):
        url = url.replace('sqlite://', 'sqlite+aiosqlite://')
    elif url.startswith('postgresql://'):
        url = url.replace('postgresql://', 'postgresql+asyncpg://')

    connectable = create_async_engine(
        url,
        poolclass=pool.NullPool,
        echo=settings.database.echo,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()
    logger.info("Async migrations completed successfully")


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate
    a connection with the context.
    """
    from sqlalchemy import create_engine

    # Get database configuration
    engine_config = {
        'echo': settings.database.echo,
        'poolclass': pool.NullPool,  # Don't use connection pooling for migrations
    }

    # SQLite specific configuration
    url = get_url()
    if url.startswith('sqlite'):
        engine_config['connect_args'] = {'check_same_thread': False}

    connectable = create_engine(url, **engine_config)

    with connectable.connect() as connection:
        do_run_migrations(connection)

    logger.info("Online migrations completed successfully")


# Determine which migration mode to use
if context.is_offline_mode():
    run_migrations_offline()
else:
    # Check if we need to run async migrations
    url = get_url()
    if '+aiosqlite' in url or '+asyncpg' in url:
        asyncio.run(run_async_migrations())
    else:
        run_migrations_online()


def create_migration_context() -> Dict[str, Any]:
    """
    Create a custom migration context with additional metadata.

    This can be used to add custom information to migration files.
    """
    return {
        'framework_version': settings.app_version,
        'environment': settings.environment.value,
        'migration_timestamp': context.get_revision_argument(),
        'author': 'MetaTrader Python Framework',
    }


def validate_migration_environment() -> bool:
    """
    Validate that the migration environment is properly configured.

    Returns:
        True if environment is valid, False otherwise
    """
    try:
        # Check if database URL is configured
        url = get_url()
        if not url:
            logger.error("Database URL not configured")
            return False

        # Check if target metadata is available
        if target_metadata is None:
            logger.error("Target metadata not available")
            return False

        # Validate that models are imported
        if not target_metadata.tables:
            logger.warning("No tables found in metadata - models may not be imported")

        logger.info(f"Migration environment validated successfully")
        logger.info(f"Database URL: {url.split('@')[-1] if '@' in url else url}")
        logger.info(f"Tables found: {len(target_metadata.tables)}")

        return True

    except Exception as e:
        logger.error(f"Migration environment validation failed: {e}")
        return False


# Validate environment on import
if not validate_migration_environment():
    logger.warning("Migration environment validation failed - migrations may not work correctly")