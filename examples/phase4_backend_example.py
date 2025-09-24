#!/usr/bin/env python3
"""
Phase 4 Backend Components Example.

This example demonstrates how to use the enhanced Phase 4 backend components
including MT5 connection management, real-time data streaming, processing
pipelines, database services, and performance monitoring.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config.settings import Settings, Environment
from src.core.backend_service import BackendService
from src.core.logging import get_logger

# Configure basic logging for the example
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = get_logger(__name__)


class Phase4BackendExample:
    """Phase 4 backend components example."""

    def __init__(self):
        """Initialize the example."""
        self.backend_service: Optional[BackendService] = None
        self.running = True

    async def run(self):
        """Run the Phase 4 backend example."""
        try:
            logger.info("Starting Phase 4 Backend Components Example")

            # Create settings with example configuration
            settings = self._create_example_settings()

            # Initialize backend service
            self.backend_service = BackendService(settings)

            # Set up signal handlers
            self._setup_signal_handlers()

            # Add custom startup/shutdown handlers
            self._setup_custom_handlers()

            # Start the backend service
            await self.backend_service.start()

            # Demonstrate various features
            await self._demonstrate_features()

            # Keep running until stopped
            logger.info("Backend service running. Press Ctrl+C to stop.")
            while self.running:
                await asyncio.sleep(1)

                # Print periodic status
                if datetime.now().second % 30 == 0:
                    await self._print_status()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in example: {e}", exc_info=True)
        finally:
            await self._cleanup()

    def _create_example_settings(self) -> Settings:
        """Create example settings configuration."""
        return Settings(
            # Basic settings
            app_name="Phase4BackendExample",
            environment=Environment.DEVELOPMENT,
            debug=True,

            # MT5 settings - you would configure these with your actual MT5 credentials
            mt5=Mt5Settings(
                accounts=[
                    Mt5AccountSettings(
                        name="demo_account",
                        login=12345678,  # Your MT5 login
                        password="your_password",  # Your MT5 password
                        server="MetaQuotes-Demo",  # Your MT5 server
                        enabled=False  # Disabled for demo - enable with real credentials
                    )
                ],
                default_account="demo_account",
                auto_connect=False,  # Set to True when you have valid credentials
                timeout=30,
                performance=Mt5PerformanceSettings(
                    pool_size=5,
                    max_overflow=10,
                    tick_processing_threads=4
                )
            ),

            # Database settings
            database=DatabaseSettings(
                databases={
                    "default": {
                        "url": "sqlite:///./data/metatrader.db",
                        "enabled": True,
                        "pool_size": 10,
                        "max_overflow": 20
                    }
                },
                default_database="default",
                pool_size=10,
                max_overflow=20
            ),

            # Market data settings
            market_data=MarketDataSettings(
                symbols=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
                timeframes=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                tick_rate_limit=1000,
                enable_tick_storage=True,
                enable_ohlc_storage=True,
                buffer_size=10000
            ),

            # Logging settings
            logging=LoggingSettings(
                level="INFO",
                format="json",
                console=LoggingConsoleSettings(enabled=True, level="INFO"),
                file=LoggingFileSettings(
                    enabled=True,
                    path="./logs/phase4_backend.log",
                    level="DEBUG",
                    rotation="time",
                    max_size=10485760,  # 10MB
                    backup_count=5
                ),
                error_file=LoggingErrorFileSettings(
                    enabled=True,
                    path="./logs/phase4_backend_errors.log",
                    max_size=10485760,
                    backup_count=3
                ),
                enable_metrics=True,
                max_performance_logs=1000
            ),

            # Performance settings
            performance=PerformanceSettings(
                enable_monitoring=True,
                collection_interval=10.0,
                enable_memory_profiling=True,
                max_metrics_history=1000,
                max_alerts_history=100,
                retention_hours=24,
                batch_size=1000,
                worker_threads=4
            )
        )

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_custom_handlers(self):
        """Set up custom startup and shutdown handlers."""
        def startup_handler():
            logger.info("Custom startup handler executed")
            # Add any custom initialization logic here

        async def shutdown_handler():
            logger.info("Custom shutdown handler executed")
            # Add any custom cleanup logic here

        if self.backend_service:
            self.backend_service.add_startup_handler(startup_handler)
            self.backend_service.add_shutdown_handler(shutdown_handler)

    async def _demonstrate_features(self):
        """Demonstrate various Phase 4 backend features."""
        if not self.backend_service:
            return

        logger.info("Demonstrating Phase 4 backend features...")

        # 1. Health Check
        logger.info("--- Health Check ---")
        health_status = await self.backend_service.health_check()
        logger.info(f"Overall Health: {health_status.get('overall_healthy', 'Unknown')}")
        for check_name, result in health_status.get('checks', {}).items():
            logger.info(f"  {check_name}: {result.get('healthy', 'Unknown')}")

        # 2. Service Status
        logger.info("--- Service Status ---")
        status = self.backend_service.get_status()
        logger.info(f"Service Status: {status.get('service_status', 'Unknown')}")
        logger.info(f"Uptime: {status.get('uptime_seconds', 0):.1f} seconds")

        # 3. Metrics
        logger.info("--- Service Metrics ---")
        metrics = await self.backend_service.get_service_metrics()
        if "backend_service" in metrics:
            backend_metrics = metrics["backend_service"]
            logger.info(f"Requests Processed: {backend_metrics.get('requests_processed', 0)}")
            logger.info(f"Data Points Processed: {backend_metrics.get('data_points_processed', 0)}")
            logger.info(f"Database Operations: {backend_metrics.get('database_operations', 0)}")

        # 4. Configuration Management Example
        logger.info("--- Configuration Management ---")
        if hasattr(self.backend_service, '_config_manager') and self.backend_service._config_manager:
            config_manager = self.backend_service._config_manager

            # Set some example configuration values
            config_manager.set_config("example.setting1", "value1")
            config_manager.set_config("example.setting2", {"nested": "value2"})

            # Retrieve configuration values
            setting1 = config_manager.get_config("example.setting1")
            setting2 = config_manager.get_config("example.setting2.nested")

            logger.info(f"Configuration setting1: {setting1}")
            logger.info(f"Configuration nested setting: {setting2}")

        # 5. Performance Monitoring Example
        logger.info("--- Performance Monitoring ---")
        if hasattr(self.backend_service, '_performance_monitor') and self.backend_service._performance_monitor:
            monitor = self.backend_service._performance_monitor

            # Record some example metrics
            monitor.increment_counter("example.requests")
            monitor.record_gauge("example.active_users", 42)

            with monitor.profile("example_operation"):
                await asyncio.sleep(0.1)  # Simulate some work

            # Get performance summary
            summary = monitor.get_performance_summary()
            logger.info(f"Performance metrics collected: {summary.get('metrics_collected', 0)}")
            logger.info(f"System CPU: {summary.get('system_metrics', {}).get('cpu_percent', 0):.1f}%")
            logger.info(f"System Memory: {summary.get('system_metrics', {}).get('memory_percent', 0):.1f}%")

        logger.info("Feature demonstration completed")

    async def _print_status(self):
        """Print periodic status information."""
        if not self.backend_service:
            return

        try:
            status = self.backend_service.get_status()
            metrics = await self.backend_service.get_service_metrics()

            logger.info("--- Periodic Status ---")
            logger.info(f"Status: {status.get('service_status', 'Unknown')}")
            logger.info(f"Uptime: {status.get('uptime_seconds', 0):.1f}s")

            if "backend_service" in metrics:
                backend_metrics = metrics["backend_service"]
                logger.info(f"Data processed: {backend_metrics.get('data_points_processed', 0)}")
                logger.info(f"DB operations: {backend_metrics.get('database_operations', 0)}")

        except Exception as e:
            logger.error(f"Error printing status: {e}")

    async def _cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up...")

        if self.backend_service:
            try:
                await self.backend_service.stop()
                logger.info("Backend service stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping backend service: {e}")

        logger.info("Cleanup completed")


async def main():
    """Main function."""
    # Create and run the example
    example = Phase4BackendExample()
    await example.run()


if __name__ == "__main__":
    # Run the example
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Example interrupted by user")
    except Exception as e:
        print(f"Example failed: {e}")
        sys.exit(1)