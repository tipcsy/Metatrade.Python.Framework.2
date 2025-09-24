"""
Database Migration System.

This module provides a comprehensive database migration system with
version control, rollback capabilities, and automatic schema management
for the MetaTrader Python Framework.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import inspect
import importlib
from abc import ABC, abstractmethod
from enum import Enum

import sqlalchemy as sa
from sqlalchemy import select, insert, update, delete, text, MetaData, Table, Column
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from sqlalchemy.exc import SQLAlchemyError

from src.core.config.settings import DatabaseSettings
from src.core.exceptions import DatabaseError, MigrationError
from src.core.logging import get_logger
from src.database.connection_manager import DatabaseConnectionManager

logger = get_logger(__name__)


class MigrationDirection(Enum):
    """Migration direction."""
    UP = "up"
    DOWN = "down"


class MigrationStatus(Enum):
    """Migration status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationRecord:
    """Database migration record."""
    version: str
    name: str
    applied_at: Optional[datetime] = None
    rollback_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    execution_time: float = 0.0
    error_message: Optional[str] = None
    checksum: Optional[str] = None


class BaseMigration(ABC):
    """Base class for database migrations."""

    def __init__(self, version: str, name: str, description: str = "") -> None:
        """Initialize migration.

        Args:
            version: Migration version (e.g., "001", "002", etc.)
            name: Migration name
            description: Migration description
        """
        self.version = version
        self.name = name
        self.description = description

    @abstractmethod
    async def up(self, session: AsyncSession, engine: AsyncEngine) -> None:
        """Apply the migration.

        Args:
            session: Database session
            engine: Database engine
        """
        pass

    @abstractmethod
    async def down(self, session: AsyncSession, engine: AsyncEngine) -> None:
        """Rollback the migration.

        Args:
            session: Database session
            engine: Database engine
        """
        pass

    def get_checksum(self) -> str:
        """Get migration checksum for integrity verification."""
        import hashlib
        content = inspect.getsource(self.__class__)
        return hashlib.md5(content.encode()).hexdigest()


class MigrationManager:
    """
    Database migration manager with version control and rollback support.

    Features:
    - Automatic migration discovery and execution
    - Version control and dependency management
    - Rollback capabilities
    - Migration integrity verification
    - Parallel migration execution
    - Migration history tracking
    """

    def __init__(
        self,
        connection_manager: DatabaseConnectionManager,
        migrations_path: Optional[Union[str, Path]] = None,
        database_name: Optional[str] = None
    ) -> None:
        """Initialize migration manager.

        Args:
            connection_manager: Database connection manager
            migrations_path: Path to migrations directory
            database_name: Database name to migrate
        """
        self.connection_manager = connection_manager
        self.database_name = database_name

        # Migration discovery
        if migrations_path is None:
            migrations_path = Path(__file__).parent / "versions"
        self.migrations_path = Path(migrations_path)

        # Migration tracking
        self._migrations: Dict[str, BaseMigration] = {}
        self._migration_records: Dict[str, MigrationRecord] = {}
        self._migration_table_name = "migration_history"

        # State management
        self._is_initialized = False
        self._lock = asyncio.Lock()

        logger.info(
            "Migration manager initialized",
            extra={
                "migrations_path": str(self.migrations_path),
                "database": database_name,
            }
        )

    async def initialize(self) -> None:
        """Initialize the migration system."""
        async with self._lock:
            if self._is_initialized:
                return

            try:
                logger.info("Initializing migration system")

                # Create migration history table
                await self._create_migration_table()

                # Load existing migration records
                await self._load_migration_history()

                # Discover available migrations
                await self._discover_migrations()

                self._is_initialized = True
                logger.info(
                    "Migration system initialized successfully",
                    extra={
                        "available_migrations": len(self._migrations),
                        "applied_migrations": len([r for r in self._migration_records.values() if r.status == MigrationStatus.COMPLETED]),
                    }
                )

            except Exception as e:
                logger.error(
                    "Failed to initialize migration system",
                    extra={"error": str(e)},
                    exc_info=True
                )
                raise MigrationError(f"Migration system initialization failed: {e}") from e

    async def migrate(self, target_version: Optional[str] = None) -> List[str]:
        """Run database migrations.

        Args:
            target_version: Target migration version (None for latest)

        Returns:
            List of applied migration versions
        """
        if not self._is_initialized:
            await self.initialize()

        async with self._lock:
            try:
                logger.info(
                    "Starting database migration",
                    extra={"target_version": target_version}
                )

                # Determine migrations to apply
                pending_migrations = await self._get_pending_migrations(target_version)

                if not pending_migrations:
                    logger.info("No pending migrations to apply")
                    return []

                applied_versions = []

                # Apply migrations in order
                for migration in pending_migrations:
                    try:
                        await self._apply_migration(migration, MigrationDirection.UP)
                        applied_versions.append(migration.version)
                        logger.info(
                            f"Applied migration {migration.version}: {migration.name}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Migration {migration.version} failed",
                            extra={"error": str(e)},
                            exc_info=True
                        )
                        # Mark as failed
                        await self._update_migration_record(
                            migration.version,
                            status=MigrationStatus.FAILED,
                            error_message=str(e)
                        )
                        raise MigrationError(f"Migration {migration.version} failed: {e}") from e

                logger.info(
                    f"Migration completed successfully",
                    extra={
                        "applied_migrations": len(applied_versions),
                        "versions": applied_versions,
                    }
                )

                return applied_versions

            except Exception as e:
                logger.error(
                    "Migration process failed",
                    extra={"error": str(e)},
                    exc_info=True
                )
                raise MigrationError(f"Migration failed: {e}") from e

    async def rollback(self, target_version: Optional[str] = None, steps: int = 1) -> List[str]:
        """Rollback database migrations.

        Args:
            target_version: Target version to rollback to
            steps: Number of migrations to rollback (if target_version not specified)

        Returns:
            List of rolled back migration versions
        """
        if not self._is_initialized:
            await self.initialize()

        async with self._lock:
            try:
                logger.info(
                    "Starting migration rollback",
                    extra={"target_version": target_version, "steps": steps}
                )

                # Determine migrations to rollback
                rollback_migrations = await self._get_rollback_migrations(target_version, steps)

                if not rollback_migrations:
                    logger.info("No migrations to rollback")
                    return []

                rolled_back_versions = []

                # Rollback migrations in reverse order
                for migration in reversed(rollback_migrations):
                    try:
                        await self._apply_migration(migration, MigrationDirection.DOWN)
                        rolled_back_versions.append(migration.version)
                        logger.info(
                            f"Rolled back migration {migration.version}: {migration.name}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Rollback of migration {migration.version} failed",
                            extra={"error": str(e)},
                            exc_info=True
                        )
                        raise MigrationError(f"Rollback of migration {migration.version} failed: {e}") from e

                logger.info(
                    f"Rollback completed successfully",
                    extra={
                        "rolled_back_migrations": len(rolled_back_versions),
                        "versions": rolled_back_versions,
                    }
                )

                return rolled_back_versions

            except Exception as e:
                logger.error(
                    "Rollback process failed",
                    extra={"error": str(e)},
                    exc_info=True
                )
                raise MigrationError(f"Rollback failed: {e}") from e

    async def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status information.

        Returns:
            Migration status dictionary
        """
        if not self._is_initialized:
            await self.initialize()

        try:
            applied_migrations = [
                record for record in self._migration_records.values()
                if record.status == MigrationStatus.COMPLETED
            ]

            pending_migrations = await self._get_pending_migrations()

            current_version = None
            if applied_migrations:
                current_version = max(applied_migrations, key=lambda x: x.version).version

            return {
                "current_version": current_version,
                "total_migrations": len(self._migrations),
                "applied_migrations": len(applied_migrations),
                "pending_migrations": len(pending_migrations),
                "migration_details": {
                    version: {
                        "name": record.name,
                        "status": record.status.value,
                        "applied_at": record.applied_at.isoformat() if record.applied_at else None,
                        "execution_time": record.execution_time,
                        "error": record.error_message,
                    }
                    for version, record in self._migration_records.items()
                }
            }

        except Exception as e:
            logger.error(
                "Failed to get migration status",
                extra={"error": str(e)},
                exc_info=True
            )
            return {"error": str(e)}

    async def verify_migrations(self) -> Dict[str, bool]:
        """Verify migration integrity using checksums.

        Returns:
            Dictionary of migration versions and their verification status
        """
        if not self._is_initialized:
            await self.initialize()

        verification_results = {}

        for version, migration in self._migrations.items():
            try:
                current_checksum = migration.get_checksum()
                record = self._migration_records.get(version)

                if record and record.checksum:
                    verification_results[version] = (current_checksum == record.checksum)
                else:
                    # No checksum recorded, consider as valid
                    verification_results[version] = True

            except Exception as e:
                logger.warning(
                    f"Failed to verify migration {version}",
                    extra={"error": str(e)}
                )
                verification_results[version] = False

        return verification_results

    async def create_migration(self, name: str, description: str = "") -> str:
        """Create a new migration file template.

        Args:
            name: Migration name
            description: Migration description

        Returns:
            Migration version
        """
        # Generate version number
        existing_versions = list(self._migrations.keys()) if self._migrations else []
        if existing_versions:
            latest_version = max(existing_versions, key=lambda x: int(x))
            new_version = f"{int(latest_version) + 1:03d}"
        else:
            new_version = "001"

        # Create migration file
        migration_filename = f"migration_{new_version}_{name.lower().replace(' ', '_')}.py"
        migration_path = self.migrations_path / migration_filename

        # Ensure migrations directory exists
        self.migrations_path.mkdir(parents=True, exist_ok=True)

        # Migration template
        template = f'''"""
Migration {new_version}: {name}

{description}
"""

from src.database.migrations.migration_manager import BaseMigration
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from sqlalchemy import text


class Migration{new_version}(BaseMigration):
    """Migration {new_version}: {name}"""

    def __init__(self):
        super().__init__(
            version="{new_version}",
            name="{name}",
            description="{description}"
        )

    async def up(self, session: AsyncSession, engine: AsyncEngine) -> None:
        """Apply the migration."""
        # Add your migration logic here
        # Example:
        # await session.execute(text("""
        #     CREATE TABLE example_table (
        #         id INTEGER PRIMARY KEY,
        #         name VARCHAR(255) NOT NULL
        #     )
        # """))
        pass

    async def down(self, session: AsyncSession, engine: AsyncEngine) -> None:
        """Rollback the migration."""
        # Add your rollback logic here
        # Example:
        # await session.execute(text("DROP TABLE IF EXISTS example_table"))
        pass
'''

        # Write migration file
        with open(migration_path, 'w') as f:
            f.write(template)

        logger.info(
            f"Created migration {new_version}: {name}",
            extra={"file": str(migration_path)}
        )

        return new_version

    async def _create_migration_table(self) -> None:
        """Create the migration history table."""
        try:
            async with self.connection_manager.get_session(self.database_name) as session:
                # Check if table exists
                engine = session.bind
                inspector = sa.inspect(engine.sync_engine)

                if not inspector.has_table(self._migration_table_name):
                    # Create migration history table
                    create_table_sql = f"""
                    CREATE TABLE {self._migration_table_name} (
                        id INTEGER PRIMARY KEY,
                        version VARCHAR(50) NOT NULL UNIQUE,
                        name VARCHAR(255) NOT NULL,
                        applied_at TIMESTAMP,
                        rollback_at TIMESTAMP,
                        status VARCHAR(20) NOT NULL DEFAULT 'pending',
                        execution_time REAL DEFAULT 0.0,
                        error_message TEXT,
                        checksum VARCHAR(32)
                    )
                    """
                    await session.execute(text(create_table_sql))
                    await session.commit()

                    logger.debug("Created migration history table")

        except Exception as e:
            logger.error(
                "Failed to create migration table",
                extra={"error": str(e)},
                exc_info=True
            )
            raise MigrationError(f"Failed to create migration table: {e}") from e

    async def _load_migration_history(self) -> None:
        """Load migration history from database."""
        try:
            async with self.connection_manager.get_session(self.database_name) as session:
                # Load existing migration records
                result = await session.execute(
                    text(f"SELECT * FROM {self._migration_table_name} ORDER BY version")
                )

                records = result.fetchall()
                for record in records:
                    migration_record = MigrationRecord(
                        version=record.version,
                        name=record.name,
                        applied_at=record.applied_at,
                        rollback_at=record.rollback_at,
                        status=MigrationStatus(record.status),
                        execution_time=record.execution_time or 0.0,
                        error_message=record.error_message,
                        checksum=record.checksum
                    )
                    self._migration_records[record.version] = migration_record

                logger.debug(
                    f"Loaded {len(self._migration_records)} migration records from database"
                )

        except Exception as e:
            # Table might not exist yet, which is okay
            logger.debug(f"Could not load migration history: {e}")

    async def _discover_migrations(self) -> None:
        """Discover available migration files."""
        try:
            if not self.migrations_path.exists():
                self.migrations_path.mkdir(parents=True, exist_ok=True)
                logger.debug("Created migrations directory")
                return

            # Find migration files
            migration_files = list(self.migrations_path.glob("migration_*.py"))
            migration_files.sort()

            for migration_file in migration_files:
                try:
                    # Import migration module
                    module_name = migration_file.stem
                    spec = importlib.util.spec_from_file_location(module_name, migration_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find migration class
                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        if (inspect.isclass(item) and
                            issubclass(item, BaseMigration) and
                            item != BaseMigration):

                            migration = item()
                            self._migrations[migration.version] = migration
                            break

                except Exception as e:
                    logger.warning(
                        f"Failed to load migration from {migration_file}",
                        extra={"error": str(e)}
                    )

            logger.debug(
                f"Discovered {len(self._migrations)} migrations",
                extra={"migrations": list(self._migrations.keys())}
            )

        except Exception as e:
            logger.error(
                "Failed to discover migrations",
                extra={"error": str(e)},
                exc_info=True
            )
            raise MigrationError(f"Failed to discover migrations: {e}") from e

    async def _get_pending_migrations(self, target_version: Optional[str] = None) -> List[BaseMigration]:
        """Get list of pending migrations."""
        pending = []

        # Get applied migrations
        applied_versions = {
            version for version, record in self._migration_records.items()
            if record.status == MigrationStatus.COMPLETED
        }

        # Filter migrations
        for version, migration in sorted(self._migrations.items()):
            if version not in applied_versions:
                pending.append(migration)

                # Stop at target version if specified
                if target_version and version == target_version:
                    break

        return pending

    async def _get_rollback_migrations(self, target_version: Optional[str] = None, steps: int = 1) -> List[BaseMigration]:
        """Get list of migrations to rollback."""
        rollback = []

        # Get applied migrations in reverse order
        applied_migrations = [
            (version, migration) for version, migration in sorted(self._migrations.items(), reverse=True)
            if version in self._migration_records and
            self._migration_records[version].status == MigrationStatus.COMPLETED
        ]

        if target_version:
            # Rollback to specific version
            for version, migration in applied_migrations:
                if version != target_version:
                    rollback.append(migration)
                else:
                    break
        else:
            # Rollback specified number of steps
            rollback = [migration for _, migration in applied_migrations[:steps]]

        return rollback

    async def _apply_migration(self, migration: BaseMigration, direction: MigrationDirection) -> None:
        """Apply a single migration."""
        start_time = time.time()

        try:
            # Update status to running
            await self._update_migration_record(
                migration.version,
                status=MigrationStatus.RUNNING
            )

            async with self.connection_manager.get_session(self.database_name) as session:
                engine = session.bind

                # Apply migration
                if direction == MigrationDirection.UP:
                    await migration.up(session, engine)
                    await session.commit()

                    # Update record as completed
                    execution_time = time.time() - start_time
                    await self._update_migration_record(
                        migration.version,
                        status=MigrationStatus.COMPLETED,
                        applied_at=datetime.now(),
                        execution_time=execution_time,
                        checksum=migration.get_checksum()
                    )

                else:  # DOWN
                    await migration.down(session, engine)
                    await session.commit()

                    # Update record as rolled back
                    execution_time = time.time() - start_time
                    await self._update_migration_record(
                        migration.version,
                        status=MigrationStatus.ROLLED_BACK,
                        rollback_at=datetime.now(),
                        execution_time=execution_time
                    )

        except Exception as e:
            # Update record as failed
            await self._update_migration_record(
                migration.version,
                status=MigrationStatus.FAILED,
                error_message=str(e)
            )
            raise

    async def _update_migration_record(
        self,
        version: str,
        status: Optional[MigrationStatus] = None,
        applied_at: Optional[datetime] = None,
        rollback_at: Optional[datetime] = None,
        execution_time: Optional[float] = None,
        error_message: Optional[str] = None,
        checksum: Optional[str] = None
    ) -> None:
        """Update migration record in database."""
        try:
            async with self.connection_manager.get_session(self.database_name) as session:
                # Get migration info
                migration = self._migrations.get(version)
                if not migration:
                    raise MigrationError(f"Migration {version} not found")

                # Check if record exists
                record = self._migration_records.get(version)

                if record is None:
                    # Create new record
                    insert_sql = f"""
                    INSERT INTO {self._migration_table_name}
                    (version, name, status, applied_at, rollback_at, execution_time, error_message, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    await session.execute(text(insert_sql), {
                        'version': version,
                        'name': migration.name,
                        'status': status.value if status else MigrationStatus.PENDING.value,
                        'applied_at': applied_at,
                        'rollback_at': rollback_at,
                        'execution_time': execution_time,
                        'error_message': error_message,
                        'checksum': checksum,
                    })

                    # Create local record
                    record = MigrationRecord(
                        version=version,
                        name=migration.name,
                        status=status or MigrationStatus.PENDING
                    )
                    self._migration_records[version] = record

                else:
                    # Update existing record
                    update_sql = f"""
                    UPDATE {self._migration_table_name}
                    SET status = ?, applied_at = ?, rollback_at = ?, execution_time = ?, error_message = ?, checksum = ?
                    WHERE version = ?
                    """
                    await session.execute(text(update_sql), {
                        'status': status.value if status else record.status.value,
                        'applied_at': applied_at or record.applied_at,
                        'rollback_at': rollback_at or record.rollback_at,
                        'execution_time': execution_time or record.execution_time,
                        'error_message': error_message or record.error_message,
                        'checksum': checksum or record.checksum,
                        'version': version,
                    })

                # Update local record
                if status:
                    record.status = status
                if applied_at:
                    record.applied_at = applied_at
                if rollback_at:
                    record.rollback_at = rollback_at
                if execution_time is not None:
                    record.execution_time = execution_time
                if error_message:
                    record.error_message = error_message
                if checksum:
                    record.checksum = checksum

                await session.commit()

        except Exception as e:
            logger.error(
                f"Failed to update migration record for {version}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise MigrationError(f"Failed to update migration record: {e}") from e