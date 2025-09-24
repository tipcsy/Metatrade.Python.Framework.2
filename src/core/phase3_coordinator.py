"""
Phase 3 system coordinator for the MetaTrader Python Framework.

This module provides centralized coordination of all Phase 3 components including:
- Real-time data collection and processing
- Background task management
- Symbol management and monitoring
- MACD trend analysis
- Market data pipeline processing
- Historical data synchronization
- Performance monitoring and optimization
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from core.config.settings import Settings
from core.exceptions import ConfigurationError, InitializationError
from core.logging import get_logger

# Import Phase 3 components
from core.data import (
    get_data_collection_manager,
    get_buffer_manager,
    get_event_publisher,
    OHLCProcessor,
)
from core.tasks import get_task_manager
from core.symbols.manager import get_symbol_manager
from indicators.macd_analyzer import get_macd_analyzer
from core.pipeline.market_data_processor import get_market_data_pipeline
from core.sync.historical_sync_manager import get_historical_sync_manager
from core.performance.optimizer import get_performance_monitor, get_performance_optimizer

logger = get_logger(__name__)


class Phase3Status(str, Enum):
    """Phase 3 system status."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ComponentStatus:
    """Individual component status."""
    name: str
    status: str
    is_healthy: bool
    error_message: Optional[str] = None
    last_health_check: Optional[datetime] = None
    metrics: Dict[str, Any] = None


class Phase3Coordinator:
    """
    Central coordinator for all Phase 3 components.

    Features:
    - Centralized component lifecycle management
    - Health monitoring and alerting
    - Performance tracking and optimization
    - Configuration management
    - Error handling and recovery
    - Graceful shutdown
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.status = Phase3Status.STOPPED

        # Component instances
        self._components = {}
        self._component_health: Dict[str, ComponentStatus] = {}
        self._initialization_lock = threading.RLock()

        # System state
        self.start_time: Optional[datetime] = None
        self.active_symbols: Set[str] = set()
        self.active_timeframes: Set[str] = set()

        # Monitoring and coordination
        self.health_check_interval = 60.0  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        self.coordination_task: Optional[asyncio.Task] = None

        # Error tracking
        self.system_errors: List[Dict[str, Any]] = []
        self.max_error_history = 100

        logger.info("Phase 3 coordinator initialized")

    async def initialize_system(self) -> None:
        """Initialize all Phase 3 components."""
        if self.status != Phase3Status.STOPPED:
            logger.warning("System already initialized or initializing")
            return

        try:
            self.status = Phase3Status.INITIALIZING
            logger.info("Initializing Phase 3 system...")

            with self._initialization_lock:
                # Initialize components in dependency order
                await self._initialize_core_components()
                await self._initialize_data_components()
                await self._initialize_analysis_components()
                await self._initialize_optimization_components()

            self.status = Phase3Status.RUNNING
            self.start_time = datetime.now(timezone.utc)

            # Start system coordination
            await self._start_coordination_tasks()

            logger.info("Phase 3 system initialized successfully")

        except Exception as e:
            self.status = Phase3Status.ERROR
            logger.error(f"Failed to initialize Phase 3 system: {e}")
            await self._handle_initialization_error(e)
            raise InitializationError(f"Phase 3 initialization failed: {e}")

    async def _initialize_core_components(self) -> None:
        """Initialize core infrastructure components."""
        logger.info("Initializing core components...")

        # Task Manager
        task_manager = get_task_manager()
        if not task_manager.start():
            raise InitializationError("Failed to start task manager")
        self._components['task_manager'] = task_manager

        # Buffer Manager
        buffer_manager = get_buffer_manager()
        await buffer_manager.start()
        self._components['buffer_manager'] = buffer_manager

        # Event Publisher
        event_publisher = get_event_publisher()
        self._components['event_publisher'] = event_publisher

        logger.info("Core components initialized")

    async def _initialize_data_components(self) -> None:
        """Initialize data collection and processing components."""
        logger.info("Initializing data components...")

        # Data Collection Manager
        data_collection_manager = get_data_collection_manager()
        self._components['data_collection_manager'] = data_collection_manager

        # Market Data Pipeline
        market_data_pipeline = get_market_data_pipeline()
        await market_data_pipeline.start_pipeline()
        self._components['market_data_pipeline'] = market_data_pipeline

        # Historical Sync Manager
        historical_sync_manager = get_historical_sync_manager()
        await historical_sync_manager.start_sync_manager()
        self._components['historical_sync_manager'] = historical_sync_manager

        # Symbol Manager
        symbol_manager = get_symbol_manager()
        if not symbol_manager.start():
            raise InitializationError("Failed to start symbol manager")
        self._components['symbol_manager'] = symbol_manager

        logger.info("Data components initialized")

    async def _initialize_analysis_components(self) -> None:
        """Initialize analysis and indicator components."""
        logger.info("Initializing analysis components...")

        # MACD Analyzer
        macd_analyzer = get_macd_analyzer()
        # Note: MACD analyzer will be started when symbols are added
        self._components['macd_analyzer'] = macd_analyzer

        logger.info("Analysis components initialized")

    async def _initialize_optimization_components(self) -> None:
        """Initialize performance and optimization components."""
        logger.info("Initializing optimization components...")

        # Performance Monitor
        performance_monitor = get_performance_monitor()
        await performance_monitor.start_monitoring()
        self._components['performance_monitor'] = performance_monitor

        # Performance Optimizer
        performance_optimizer = get_performance_optimizer()
        self._components['performance_optimizer'] = performance_optimizer

        logger.info("Optimization components initialized")

    async def _start_coordination_tasks(self) -> None:
        """Start system coordination tasks."""
        self.health_check_task = asyncio.create_task(
            self._health_check_loop(),
            name="phase3-health-check"
        )

        self.coordination_task = asyncio.create_task(
            self._coordination_loop(),
            name="phase3-coordination"
        )

        logger.info("Coordination tasks started")

    async def start_trading_session(
        self,
        symbols: List[str],
        timeframes: List[str] = None
    ) -> None:
        """
        Start a trading session with specified symbols.

        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes (default: from settings)
        """
        if self.status != Phase3Status.RUNNING:
            raise ConfigurationError("System not running, cannot start trading session")

        if not timeframes:
            timeframes = [tf.value.replace('TIMEFRAME_', '') for tf in self.settings.mt5.default_timeframes]

        try:
            logger.info(f"Starting trading session with {len(symbols)} symbols and {len(timeframes)} timeframes")

            # Update active symbols and timeframes
            self.active_symbols.update(symbols)
            self.active_timeframes.update(timeframes)

            # Start data collection
            data_collection_manager = self._components['data_collection_manager']
            if not data_collection_manager.start_collection(tick_symbols=symbols):
                raise ConfigurationError("Failed to start data collection")

            # Start MACD analysis
            macd_analyzer = self._components['macd_analyzer']
            await macd_analyzer.start_analysis(symbols)

            # Initialize historical synchronization for new symbols
            historical_sync_manager = self._components['historical_sync_manager']
            for symbol in symbols:
                await historical_sync_manager.sync_symbol_history(
                    symbol=symbol,
                    timeframes=timeframes,
                    days_back=7,  # Start with 1 week of history
                    priority=7
                )

            logger.info(f"Trading session started successfully")

        except Exception as e:
            logger.error(f"Failed to start trading session: {e}")
            raise

    async def stop_trading_session(self) -> None:
        """Stop the current trading session."""
        try:
            logger.info("Stopping trading session...")

            # Stop data collection
            data_collection_manager = self._components.get('data_collection_manager')
            if data_collection_manager:
                data_collection_manager.stop_collection()

            # Stop MACD analysis
            macd_analyzer = self._components.get('macd_analyzer')
            if macd_analyzer:
                await macd_analyzer.stop_analysis()

            # Clear active symbols and timeframes
            self.active_symbols.clear()
            self.active_timeframes.clear()

            logger.info("Trading session stopped")

        except Exception as e:
            logger.error(f"Error stopping trading session: {e}")

    async def shutdown_system(self) -> None:
        """Graceful shutdown of the entire Phase 3 system."""
        if self.status in [Phase3Status.STOPPING, Phase3Status.STOPPED]:
            return

        try:
            self.status = Phase3Status.STOPPING
            logger.info("Shutting down Phase 3 system...")

            # Stop trading session first
            await self.stop_trading_session()

            # Stop coordination tasks
            if self.health_check_task:
                self.health_check_task.cancel()
            if self.coordination_task:
                self.coordination_task.cancel()

            # Shutdown components in reverse order
            await self._shutdown_components()

            self.status = Phase3Status.STOPPED
            logger.info("Phase 3 system shutdown complete")

        except Exception as e:
            self.status = Phase3Status.ERROR
            logger.error(f"Error during system shutdown: {e}")

    async def _shutdown_components(self) -> None:
        """Shutdown components in proper order."""
        shutdown_order = [
            'macd_analyzer',
            'historical_sync_manager',
            'market_data_pipeline',
            'data_collection_manager',
            'performance_monitor',
            'symbol_manager',
            'task_manager',
            'buffer_manager',
        ]

        for component_name in shutdown_order:
            component = self._components.get(component_name)
            if not component:
                continue

            try:
                logger.debug(f"Shutting down {component_name}")

                if hasattr(component, 'stop_analysis'):
                    await component.stop_analysis()
                elif hasattr(component, 'stop_sync_manager'):
                    await component.stop_sync_manager()
                elif hasattr(component, 'stop_pipeline'):
                    await component.stop_pipeline()
                elif hasattr(component, 'stop_collection'):
                    component.stop_collection()
                elif hasattr(component, 'stop_monitoring'):
                    await component.stop_monitoring()
                elif hasattr(component, 'stop'):
                    if asyncio.iscoroutinefunction(component.stop):
                        await component.stop()
                    else:
                        component.stop()

                logger.debug(f"Successfully shut down {component_name}")

            except Exception as e:
                logger.error(f"Error shutting down {component_name}: {e}")

    async def _health_check_loop(self) -> None:
        """Continuous health checking of all components."""
        while self.status == Phase3Status.RUNNING:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all components."""
        health_checks = []

        for component_name, component in self._components.items():
            if hasattr(component, 'health_check'):
                health_checks.append(self._check_component_health(component_name, component))

        # Run all health checks concurrently
        if health_checks:
            await asyncio.gather(*health_checks, return_exceptions=True)

    async def _check_component_health(self, component_name: str, component: Any) -> None:
        """Check health of a single component."""
        try:
            health_result = await component.health_check()

            status = ComponentStatus(
                name=component_name,
                status=health_result.get('status', 'unknown'),
                is_healthy=health_result.get('status') in ['healthy', 'running'],
                last_health_check=datetime.now(timezone.utc),
                metrics=health_result
            )

            self._component_health[component_name] = status

            # Log health issues
            if not status.is_healthy:
                logger.warning(f"Component {component_name} is unhealthy: {status.status}")

        except Exception as e:
            error_status = ComponentStatus(
                name=component_name,
                status='error',
                is_healthy=False,
                error_message=str(e),
                last_health_check=datetime.now(timezone.utc),
            )

            self._component_health[component_name] = error_status
            logger.error(f"Health check failed for {component_name}: {e}")

    async def _coordination_loop(self) -> None:
        """Main coordination loop for system management."""
        while self.status == Phase3Status.RUNNING:
            try:
                # Coordinate data flow between components
                await self._coordinate_data_flow()

                # Check for system-wide issues
                await self._check_system_health()

                # Perform maintenance tasks
                await self._perform_maintenance()

                await asyncio.sleep(30.0)  # Run every 30 seconds

            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(30.0)

    async def _coordinate_data_flow(self) -> None:
        """Coordinate data flow between components."""
        # This could include:
        # - Ensuring data buffers are not overflowing
        # - Coordinating between data collection and analysis
        # - Managing resource allocation
        pass

    async def _check_system_health(self) -> None:
        """Check overall system health."""
        unhealthy_components = [
            name for name, status in self._component_health.items()
            if not status.is_healthy
        ]

        if unhealthy_components:
            logger.warning(f"Unhealthy components detected: {unhealthy_components}")

            # If critical components are unhealthy, consider system restart
            critical_components = ['task_manager', 'buffer_manager', 'data_collection_manager']
            unhealthy_critical = [c for c in unhealthy_components if c in critical_components]

            if unhealthy_critical:
                logger.critical(f"Critical components unhealthy: {unhealthy_critical}")
                # Could trigger automatic recovery procedures here

    async def _perform_maintenance(self) -> None:
        """Perform periodic maintenance tasks."""
        current_time = datetime.now(timezone.utc)

        # Clean up old error history
        cutoff_time = current_time - timedelta(hours=24)
        self.system_errors = [
            error for error in self.system_errors
            if datetime.fromisoformat(error['timestamp'].replace('Z', '+00:00')) > cutoff_time
        ]

        # Trigger garbage collection if memory usage is high
        performance_monitor = self._components.get('performance_monitor')
        if performance_monitor:
            current_metrics = performance_monitor.get_current_metrics()
            if current_metrics and current_metrics.memory_percent > 80:
                import gc
                collected = gc.collect()
                logger.debug(f"Triggered garbage collection, collected {collected} objects")

    async def _handle_initialization_error(self, error: Exception) -> None:
        """Handle initialization errors."""
        error_info = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'phase': 'initialization',
        }

        self.system_errors.append(error_info)
        logger.error(f"Initialization error recorded: {error_info}")

    def add_symbols(self, symbols: List[str]) -> None:
        """Add symbols to active trading session."""
        if self.status != Phase3Status.RUNNING:
            logger.warning("System not running, cannot add symbols")
            return

        new_symbols = [s for s in symbols if s not in self.active_symbols]
        if not new_symbols:
            return

        try:
            # Add to data collection
            data_collection_manager = self._components.get('data_collection_manager')
            if data_collection_manager:
                for symbol in new_symbols:
                    data_collection_manager.add_tick_symbol(symbol)

            self.active_symbols.update(new_symbols)
            logger.info(f"Added {len(new_symbols)} symbols to trading session")

        except Exception as e:
            logger.error(f"Error adding symbols: {e}")

    def remove_symbols(self, symbols: List[str]) -> None:
        """Remove symbols from active trading session."""
        if self.status != Phase3Status.RUNNING:
            logger.warning("System not running, cannot remove symbols")
            return

        symbols_to_remove = [s for s in symbols if s in self.active_symbols]
        if not symbols_to_remove:
            return

        try:
            # Remove from data collection
            data_collection_manager = self._components.get('data_collection_manager')
            if data_collection_manager:
                for symbol in symbols_to_remove:
                    data_collection_manager.remove_tick_symbol(symbol)

            self.active_symbols.difference_update(symbols_to_remove)
            logger.info(f"Removed {len(symbols_to_remove)} symbols from trading session")

        except Exception as e:
            logger.error(f"Error removing symbols: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = None
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        component_statuses = {}
        for name, status in self._component_health.items():
            component_statuses[name] = {
                'status': status.status,
                'is_healthy': status.is_healthy,
                'last_check': status.last_health_check.isoformat() if status.last_health_check else None,
                'error': status.error_message,
            }

        return {
            'system_status': self.status.value,
            'uptime_seconds': uptime,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'active_symbols': list(self.active_symbols),
            'active_timeframes': list(self.active_timeframes),
            'component_count': len(self._components),
            'healthy_components': len([s for s in self._component_health.values() if s.is_healthy]),
            'component_statuses': component_statuses,
            'error_count': len(self.system_errors),
            'recent_errors': self.system_errors[-5:] if self.system_errors else [],
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get system performance summary."""
        performance_monitor = self._components.get('performance_monitor')
        if not performance_monitor:
            return {'status': 'monitor_not_available'}

        return performance_monitor.get_performance_summary()

    async def health_check(self) -> Dict[str, Any]:
        """Perform overall system health check."""
        system_health = {
            'status': 'healthy' if self.status == Phase3Status.RUNNING else 'unhealthy',
            'system_status': self.status.value,
            'components_healthy': True,
            'issues': [],
        }

        # Check component health
        unhealthy_components = [
            name for name, status in self._component_health.items()
            if not status.is_healthy
        ]

        if unhealthy_components:
            system_health['components_healthy'] = False
            system_health['status'] = 'degraded'
            system_health['issues'].append(f"Unhealthy components: {unhealthy_components}")

        # Check system resources
        performance_monitor = self._components.get('performance_monitor')
        if performance_monitor:
            current_metrics = performance_monitor.get_current_metrics()
            if current_metrics:
                if current_metrics.cpu_percent > 90:
                    system_health['status'] = 'degraded'
                    system_health['issues'].append('High CPU usage')

                if current_metrics.memory_percent > 95:
                    system_health['status'] = 'degraded'
                    system_health['issues'].append('High memory usage')

        return system_health


# Global Phase 3 coordinator instance
_phase3_coordinator: Optional[Phase3Coordinator] = None


def get_phase3_coordinator() -> Phase3Coordinator:
    """Get the global Phase 3 coordinator instance."""
    global _phase3_coordinator

    if _phase3_coordinator is None:
        from core.config.settings import Settings
        settings = Settings()
        _phase3_coordinator = Phase3Coordinator(settings)

    return _phase3_coordinator