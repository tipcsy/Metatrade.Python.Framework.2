"""
Comprehensive Backend Service Integration.

This module provides a unified backend service that integrates all Phase 4
components including MT5 connection management, data streaming, processing
pipelines, database services, monitoring, and configuration management.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.core.config.settings import Settings
from src.core.config.advanced_config_manager import AdvancedConfigManager
from src.core.logging.advanced_logger import AdvancedLogger
from src.core.performance.monitor import PerformanceMonitor
from src.core.pipeline.tick_processor import TickProcessor, ProcessingConfig
from src.core.pipeline.ohlc_aggregator import OHLCAggregator, AggregatorConfig, TimeFrame
from src.core.exceptions import BackendServiceError
from src.core.logging import get_logger

from src.mt5.connection.manager import Mt5ConnectionManager
from src.mt5.streaming import Mt5StreamingService

from src.database.connection_manager import DatabaseConnectionManager
from src.database.persistence.market_data_persistence import MarketDataPersistence, PersistenceConfig
from src.database.migrations.migration_manager import MigrationManager

logger = get_logger(__name__)


class ServiceStatus(Enum):
    """Backend service status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ServiceMetrics:
    """Backend service metrics."""
    start_time: float
    uptime_seconds: float = 0.0
    requests_processed: int = 0
    errors_encountered: int = 0
    data_points_processed: int = 0
    database_operations: int = 0
    mt5_connections_active: int = 0
    streaming_subscriptions: int = 0


class BackendService:
    """
    Comprehensive backend service integrating all Phase 4 components.

    Features:
    - MT5 connection management with pooling
    - Real-time data streaming and processing
    - High-performance tick processing pipeline
    - OHLC aggregation engine
    - Database connection management
    - Market data persistence layer
    - Database migration management
    - Performance monitoring and logging
    - Centralized configuration management
    - Health checks and status monitoring
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize backend service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._status = ServiceStatus.STOPPED
        self._start_time: Optional[float] = None
        self._metrics = ServiceMetrics(start_time=0.0)

        # Core services
        self._config_manager: Optional[AdvancedConfigManager] = None
        self._logger: Optional[AdvancedLogger] = None
        self._performance_monitor: Optional[PerformanceMonitor] = None

        # MT5 services
        self._mt5_connection_manager: Optional[Mt5ConnectionManager] = None
        self._mt5_streaming_service: Optional[Mt5StreamingService] = None

        # Data processing services
        self._tick_processor: Optional[TickProcessor] = None
        self._ohlc_aggregator: Optional[OHLCAggregator] = None

        # Database services
        self._db_connection_manager: Optional[DatabaseConnectionManager] = None
        self._data_persistence: Optional[MarketDataPersistence] = None
        self._migration_manager: Optional[MigrationManager] = None

        # Health checks
        self._health_checks: Dict[str, Callable[[], bool]] = {}

        # Event handlers
        self._startup_handlers: List[Callable[[], None]] = []
        self._shutdown_handlers: List[Callable[[], None]] = []

        logger.info(
            "Backend service initialized",
            extra={"version": "4.0", "environment": settings.environment}
        )

    async def start(self) -> None:
        """Start all backend services."""
        if self._status != ServiceStatus.STOPPED:
            logger.warning("Backend service not in stopped state")
            return

        try:
            self._status = ServiceStatus.STARTING
            self._start_time = time.time()
            self._metrics.start_time = self._start_time

            logger.info("Starting backend service...")

            # Initialize core services first
            await self._initialize_core_services()

            # Initialize MT5 services
            await self._initialize_mt5_services()

            # Initialize data processing services
            await self._initialize_processing_services()

            # Initialize database services
            await self._initialize_database_services()

            # Set up integrations between services
            await self._setup_service_integrations()

            # Set up health checks
            await self._setup_health_checks()

            # Run startup handlers
            await self._run_startup_handlers()

            self._status = ServiceStatus.RUNNING
            logger.info(
                "Backend service started successfully",
                extra={"startup_time": time.time() - self._start_time}
            )

        except Exception as e:
            self._status = ServiceStatus.ERROR
            logger.error(
                "Failed to start backend service",
                extra={"error": str(e)},
                exc_info=True
            )
            await self._cleanup_partial_startup()
            raise BackendServiceError(f"Backend service startup failed: {e}") from e

    async def stop(self) -> None:
        """Stop all backend services."""
        if self._status == ServiceStatus.STOPPED:
            return

        try:
            self._status = ServiceStatus.STOPPING
            logger.info("Stopping backend service...")

            # Run shutdown handlers
            await self._run_shutdown_handlers()

            # Stop services in reverse order
            await self._shutdown_database_services()
            await self._shutdown_processing_services()
            await self._shutdown_mt5_services()
            await self._shutdown_core_services()

            self._status = ServiceStatus.STOPPED
            logger.info("Backend service stopped successfully")

        except Exception as e:
            self._status = ServiceStatus.ERROR
            logger.error(
                "Error stopping backend service",
                extra={"error": str(e)},
                exc_info=True
            )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive service status.

        Returns:
            Service status dictionary
        """
        try:
            current_time = time.time()
            if self._start_time:
                self._metrics.uptime_seconds = current_time - self._start_time

            status = {
                "service_status": self._status.value,
                "uptime_seconds": self._metrics.uptime_seconds,
                "start_time": datetime.fromtimestamp(self._start_time) if self._start_time else None,
                "metrics": self._metrics.__dict__.copy(),
                "services": {}
            }

            # Core services status
            if self._config_manager:
                status["services"]["config_manager"] = self._config_manager.get_config_summary()

            if self._performance_monitor:
                status["services"]["performance_monitor"] = self._performance_monitor.get_performance_summary()

            # MT5 services status
            if self._mt5_connection_manager:
                status["services"]["mt5_connection_manager"] = asyncio.create_task(
                    self._mt5_connection_manager.get_manager_status()
                )

            if self._mt5_streaming_service:
                status["services"]["mt5_streaming"] = self._mt5_streaming_service.get_service_status()

            # Processing services status
            if self._tick_processor:
                status["services"]["tick_processor"] = self._tick_processor.get_status()

            if self._ohlc_aggregator:
                status["services"]["ohlc_aggregator"] = self._ohlc_aggregator.get_status()

            # Database services status
            if self._db_connection_manager:
                status["services"]["database"] = asyncio.create_task(
                    self._db_connection_manager.get_pool_status()
                )

            if self._data_persistence:
                status["services"]["data_persistence"] = asyncio.create_task(
                    self._data_persistence.get_storage_statistics()
                )

            return status

        except Exception as e:
            logger.error(
                "Error getting service status",
                extra={"error": str(e)},
                exc_info=True
            )
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check.

        Returns:
            Health check results
        """
        try:
            results = {
                "timestamp": datetime.now().isoformat(),
                "overall_healthy": True,
                "checks": {}
            }

            # Run all registered health checks
            for check_name, check_func in self._health_checks.items():
                try:
                    is_healthy = check_func()
                    results["checks"][check_name] = {
                        "healthy": is_healthy,
                        "message": "OK" if is_healthy else "Health check failed"
                    }

                    if not is_healthy:
                        results["overall_healthy"] = False

                except Exception as e:
                    results["checks"][check_name] = {
                        "healthy": False,
                        "message": f"Health check error: {e}"
                    }
                    results["overall_healthy"] = False

            # Service-specific health checks
            if self._mt5_connection_manager:
                mt5_health = await self._mt5_connection_manager.health_check()
                results["checks"]["mt5_connections"] = mt5_health

            if self._db_connection_manager:
                db_health = await self._db_connection_manager.health_check()
                results["checks"]["database"] = db_health

            return results

        except Exception as e:
            logger.error(
                "Error performing health check",
                extra={"error": str(e)},
                exc_info=True
            )
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_healthy": False,
                "error": str(e)
            }

    def add_startup_handler(self, handler: Callable[[], None]) -> None:
        """Add startup handler.

        Args:
            handler: Startup handler function
        """
        self._startup_handlers.append(handler)

    def add_shutdown_handler(self, handler: Callable[[], None]) -> None:
        """Add shutdown handler.

        Args:
            handler: Shutdown handler function
        """
        self._shutdown_handlers.append(handler)

    def add_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Add health check function.

        Args:
            name: Health check name
            check_func: Health check function
        """
        self._health_checks[name] = check_func

    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics.

        Returns:
            Service metrics dictionary
        """
        try:
            metrics = {
                "backend_service": self._metrics.__dict__.copy(),
                "services": {}
            }

            # Collect metrics from all services
            if self._performance_monitor:
                metrics["services"]["performance"] = self._performance_monitor.get_performance_summary()

            if self._tick_processor:
                tick_metrics = self._tick_processor.get_metrics()
                metrics["services"]["tick_processing"] = tick_metrics.__dict__ if hasattr(tick_metrics, '__dict__') else tick_metrics

            if self._ohlc_aggregator:
                ohlc_metrics = self._ohlc_aggregator.get_metrics()
                metrics["services"]["ohlc_aggregation"] = ohlc_metrics

            if self._mt5_streaming_service:
                streaming_metrics = self._mt5_streaming_service.get_stream_metrics()
                metrics["services"]["streaming"] = streaming_metrics

            if self._db_connection_manager:
                db_metrics = await self._db_connection_manager.get_metrics()
                metrics["services"]["database"] = db_metrics

            return metrics

        except Exception as e:
            logger.error(
                "Error getting service metrics",
                extra={"error": str(e)},
                exc_info=True
            )
            return {"error": str(e)}

    async def _initialize_core_services(self) -> None:
        """Initialize core services."""
        logger.debug("Initializing core services...")

        # Configuration manager
        self._config_manager = AdvancedConfigManager()
        # Initialize with default config paths
        await self._config_manager.initialize([])

        # Advanced logging
        self._logger = AdvancedLogger(self.settings.logging)
        await self._logger.initialize()

        # Performance monitoring
        self._performance_monitor = PerformanceMonitor(self.settings.performance)
        await self._performance_monitor.start()

        logger.debug("Core services initialized")

    async def _initialize_mt5_services(self) -> None:
        """Initialize MT5 services."""
        logger.debug("Initializing MT5 services...")

        # MT5 connection manager
        self._mt5_connection_manager = Mt5ConnectionManager(self.settings.mt5)
        await self._mt5_connection_manager.initialize()

        # MT5 streaming service
        self._mt5_streaming_service = Mt5StreamingService(
            connection_manager=self._mt5_connection_manager,
            market_data_settings=self.settings.market_data
        )
        await self._mt5_streaming_service.start_streaming()

        logger.debug("MT5 services initialized")

    async def _initialize_processing_services(self) -> None:
        """Initialize data processing services."""
        logger.debug("Initializing processing services...")

        # Tick processor
        processing_config = ProcessingConfig(
            enable_validation=True,
            enable_filtering=True,
            enable_transformation=True,
            batch_size=self.settings.performance.batch_size,
            worker_threads=self.settings.performance.worker_threads
        )
        self._tick_processor = TickProcessor(processing_config, self.settings.performance)
        await self._tick_processor.start()

        # OHLC aggregator
        aggregator_config = AggregatorConfig(
            timeframes=[],  # Will be configured based on requirements
            enable_volume_profile=True,
            enable_advanced_metrics=True
        )
        self._ohlc_aggregator = OHLCAggregator(aggregator_config)
        self._ohlc_aggregator.start()

        logger.debug("Processing services initialized")

    async def _initialize_database_services(self) -> None:
        """Initialize database services."""
        logger.debug("Initializing database services...")

        # Database connection manager
        self._db_connection_manager = DatabaseConnectionManager(self.settings.database)
        await self._db_connection_manager.initialize()

        # Data persistence layer
        persistence_config = PersistenceConfig(
            batch_size=self.settings.performance.batch_size,
            enable_compression=True,
            enable_partitioning=True
        )
        self._data_persistence = MarketDataPersistence(
            connection_manager=self._db_connection_manager,
            config=persistence_config
        )
        await self._data_persistence.start()

        # Migration manager
        self._migration_manager = MigrationManager(
            connection_manager=self._db_connection_manager
        )
        await self._migration_manager.initialize()

        logger.debug("Database services initialized")

    async def _setup_service_integrations(self) -> None:
        """Set up integrations between services."""
        logger.debug("Setting up service integrations...")

        # Connect streaming to tick processor
        if self._mt5_streaming_service and self._tick_processor:
            def process_ticks(ticks):
                if ticks:
                    self._tick_processor.process_batch(ticks)
                    self._metrics.data_points_processed += len(ticks)

            # Subscribe to tick data
            subscription_id = self._mt5_streaming_service.subscribe_ticks(
                symbols=["EURUSD", "GBPUSD", "USDJPY"],  # Example symbols
                callback=process_ticks
            )

        # Connect tick processor to OHLC aggregator
        if self._tick_processor and self._ohlc_aggregator:
            def aggregate_ticks(processed_ticks):
                if processed_ticks:
                    self._ohlc_aggregator.process_batch(processed_ticks)

            self._tick_processor.add_output_callback(aggregate_ticks)

        # Connect OHLC aggregator to persistence
        if self._ohlc_aggregator and self._data_persistence:
            def persist_bars(bar):
                if bar:
                    asyncio.create_task(self._data_persistence.store_ohlc_bar(bar))
                    self._metrics.database_operations += 1

            for timeframe in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.HOUR_1]:
                self._ohlc_aggregator.add_bar_callback(timeframe, persist_bars)

        logger.debug("Service integrations configured")

    async def _setup_health_checks(self) -> None:
        """Set up health checks for all services."""
        logger.debug("Setting up health checks...")

        # Core service health checks
        def config_manager_health():
            return self._config_manager is not None

        def logger_health():
            return self._logger is not None

        def performance_monitor_health():
            return self._performance_monitor is not None and self._performance_monitor._is_running

        # MT5 service health checks
        def mt5_connection_health():
            return (self._mt5_connection_manager is not None and
                   self._mt5_connection_manager.is_initialized)

        def mt5_streaming_health():
            return (self._mt5_streaming_service is not None and
                   self._mt5_streaming_service._stream_status.name == "RUNNING")

        # Processing service health checks
        def tick_processor_health():
            return (self._tick_processor is not None and
                   self._tick_processor._status.name == "RUNNING")

        def ohlc_aggregator_health():
            return self._ohlc_aggregator is not None

        # Database service health checks
        def database_health():
            return (self._db_connection_manager is not None and
                   self._db_connection_manager.is_initialized)

        def persistence_health():
            return self._data_persistence is not None

        # Register health checks
        health_checks = {
            "config_manager": config_manager_health,
            "logger": logger_health,
            "performance_monitor": performance_monitor_health,
            "mt5_connections": mt5_connection_health,
            "mt5_streaming": mt5_streaming_health,
            "tick_processor": tick_processor_health,
            "ohlc_aggregator": ohlc_aggregator_health,
            "database": database_health,
            "data_persistence": persistence_health,
        }

        for name, check_func in health_checks.items():
            self.add_health_check(name, check_func)

        # Add performance monitor health checks
        if self._performance_monitor:
            for name, check_func in health_checks.items():
                self._performance_monitor.add_health_check(
                    name, check_func, critical=(name in ["database", "mt5_connections"])
                )

        logger.debug("Health checks configured")

    async def _run_startup_handlers(self) -> None:
        """Run all startup handlers."""
        for handler in self._startup_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.error(
                    "Error in startup handler",
                    extra={"error": str(e)},
                    exc_info=True
                )

    async def _run_shutdown_handlers(self) -> None:
        """Run all shutdown handlers."""
        for handler in self._shutdown_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.error(
                    "Error in shutdown handler",
                    extra={"error": str(e)},
                    exc_info=True
                )

    async def _shutdown_database_services(self) -> None:
        """Shutdown database services."""
        try:
            if self._data_persistence:
                await self._data_persistence.stop()

            if self._db_connection_manager:
                await self._db_connection_manager.shutdown()

        except Exception as e:
            logger.error(
                "Error shutting down database services",
                extra={"error": str(e)},
                exc_info=True
            )

    async def _shutdown_processing_services(self) -> None:
        """Shutdown processing services."""
        try:
            if self._ohlc_aggregator:
                self._ohlc_aggregator.stop()

            if self._tick_processor:
                await self._tick_processor.stop()

        except Exception as e:
            logger.error(
                "Error shutting down processing services",
                extra={"error": str(e)},
                exc_info=True
            )

    async def _shutdown_mt5_services(self) -> None:
        """Shutdown MT5 services."""
        try:
            if self._mt5_streaming_service:
                await self._mt5_streaming_service.stop_streaming()

            if self._mt5_connection_manager:
                await self._mt5_connection_manager.shutdown()

        except Exception as e:
            logger.error(
                "Error shutting down MT5 services",
                extra={"error": str(e)},
                exc_info=True
            )

    async def _shutdown_core_services(self) -> None:
        """Shutdown core services."""
        try:
            if self._performance_monitor:
                await self._performance_monitor.stop()

            if self._logger:
                await self._logger.shutdown()

            if self._config_manager:
                await self._config_manager.shutdown()

        except Exception as e:
            logger.error(
                "Error shutting down core services",
                extra={"error": str(e)},
                exc_info=True
            )

    async def _cleanup_partial_startup(self) -> None:
        """Cleanup after partial startup failure."""
        try:
            await self.stop()
        except Exception as e:
            logger.error(
                "Error during cleanup after startup failure",
                extra={"error": str(e)},
                exc_info=True
            )

    async def __aenter__(self) -> BackendService:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()