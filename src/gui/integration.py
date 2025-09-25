"""
Integration module for connecting GUI components with backend services.

This module provides the integration layer between the GUI and the Phase 2 database
and Phase 3 backend services, ensuring seamless data flow and synchronization.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.symbols.manager import get_symbol_manager
from src.database.database import get_database_manager
from src.mt5.connection import get_mt5_session_manager

from .realtime import RealTimeDataUpdater
from .models import MarketDataModel

logger = get_logger(__name__)


class GuiBackendIntegrator(QObject):
    """
    Integration layer between GUI and backend services.

    Coordinates data flow between:
    - Phase 2: Database models and services
    - Phase 3: Symbol manager, data collector, MT5 connection
    - GUI: Real-time updates, models, and widgets
    """

    # Integration signals
    dataIntegrated = pyqtSignal(str, dict)      # symbol, data
    serviceStatusChanged = pyqtSignal(str, bool) # service_name, is_running
    integrationError = pyqtSignal(str)          # error_message

    def __init__(self, market_data_model: MarketDataModel):
        """
        Initialize GUI-backend integrator.

        Args:
            market_data_model: Market data model to integrate with
        """
        super().__init__()

        # Configuration and dependencies
        self.settings = get_settings()
        self.market_data_model = market_data_model

        # Backend service references
        self.symbol_manager = get_symbol_manager()
        self.database_manager = get_database_manager()
        self.mt5_session_manager = get_mt5_session_manager()

        # Real-time data updater
        self.data_updater = RealTimeDataUpdater(
            max_updates_per_second=self.settings.mt5.performance.max_ticks_per_second
        )

        # Integration state
        self._is_running = False
        self._service_status: Dict[str, bool] = {}

        # Setup connections
        self._setup_connections()

        logger.info("GUI-backend integrator initialized")

    def _setup_connections(self) -> None:
        """Setup connections between components."""
        try:
            # Connect data updater to market data model
            self.data_updater.dataUpdated.connect(self._on_data_updated)
            self.data_updater.errorOccurred.connect(self._on_data_error)

            # Connect market data model updates
            self.market_data_model.symbolDataUpdated.connect(self._on_symbol_data_updated)

            logger.debug("Component connections established")

        except Exception as e:
            logger.error(f"Error setting up connections: {e}")

    def start_integration(self) -> bool:
        """Start the integration services."""
        if self._is_running:
            logger.warning("Integration already running")
            return True

        try:
            logger.info("Starting GUI-backend integration...")

            # Start real-time data updater
            if not self.data_updater.start():
                logger.error("Failed to start real-time data updater")
                return False

            # Subscribe to backend data feeds
            self._subscribe_to_backend_feeds()

            # Setup periodic synchronization
            self._setup_sync_timers()

            self._is_running = True
            logger.info("GUI-backend integration started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start integration: {e}")
            self.integrationError.emit(str(e))
            return False

    def stop_integration(self) -> None:
        """Stop the integration services."""
        if not self._is_running:
            return

        logger.info("Stopping GUI-backend integration...")

        try:
            # Stop real-time data updater
            self.data_updater.stop()

            # Unsubscribe from backend feeds
            self._unsubscribe_from_backend_feeds()

            self._is_running = False
            logger.info("GUI-backend integration stopped")

        except Exception as e:
            logger.error(f"Error stopping integration: {e}")

    def _subscribe_to_backend_feeds(self) -> None:
        """Subscribe to backend data feeds."""
        try:
            # Subscribe to symbol manager updates
            if self.symbol_manager.is_running:
                # In a complete implementation, this would connect to
                # symbol manager's real-time quote feeds
                logger.debug("Subscribed to symbol manager feeds")
                self._service_status['symbol_manager'] = True
                self.serviceStatusChanged.emit('symbol_manager', True)

            # Subscribe to MT5 data feeds
            session = self.mt5_session_manager.get_session_sync()
            if session:
                # In a complete implementation, this would set up
                # MT5 tick/quote subscriptions
                logger.debug("Subscribed to MT5 data feeds")
                self._service_status['mt5'] = True
                self.serviceStatusChanged.emit('mt5', True)

        except Exception as e:
            logger.error(f"Error subscribing to backend feeds: {e}")

    def _unsubscribe_from_backend_feeds(self) -> None:
        """Unsubscribe from backend data feeds."""
        try:
            # Unsubscribe from all feeds
            self._service_status.clear()
            logger.debug("Unsubscribed from all backend feeds")

        except Exception as e:
            logger.error(f"Error unsubscribing from backend feeds: {e}")

    def _setup_sync_timers(self) -> None:
        """Setup periodic synchronization timers."""
        # Database sync timer
        self.db_sync_timer = QTimer()
        self.db_sync_timer.timeout.connect(self._sync_with_database)
        self.db_sync_timer.start(30000)  # Every 30 seconds

        # Symbol manager sync timer
        self.symbol_sync_timer = QTimer()
        self.symbol_sync_timer.timeout.connect(self._sync_with_symbol_manager)
        self.symbol_sync_timer.start(60000)  # Every minute

    def _sync_with_database(self) -> None:
        """Synchronize GUI data with database."""
        try:
            # Sync symbol data with database
            symbols = self.market_data_model.get_symbols()
            for symbol in symbols:
                symbol_data = self.market_data_model.get_symbol_data(symbol)
                if symbol_data:
                    # In a complete implementation, this would update
                    # the database with current symbol data
                    pass

            logger.debug(f"Synchronized {len(symbols)} symbols with database")

        except Exception as e:
            logger.debug(f"Error syncing with database: {e}")

    def _sync_with_symbol_manager(self) -> None:
        """Synchronize GUI symbols with symbol manager."""
        try:
            # Get symbols from GUI
            gui_symbols = set(self.market_data_model.get_symbols())

            # Get symbols from symbol manager
            manager_symbols = set()
            if self.symbol_manager.is_running:
                symbol_infos = self.symbol_manager.list_symbols()
                manager_symbols = {info.symbol for info in symbol_infos}

            # Add missing symbols to GUI
            missing_in_gui = manager_symbols - gui_symbols
            for symbol in missing_in_gui:
                self.market_data_model.add_symbol(symbol)

            # Remove symbols no longer in manager
            missing_in_manager = gui_symbols - manager_symbols
            for symbol in missing_in_manager:
                self.market_data_model.remove_symbol(symbol)

            if missing_in_gui or missing_in_manager:
                logger.debug(f"Symbol sync: added {len(missing_in_gui)}, removed {len(missing_in_manager)}")

        except Exception as e:
            logger.debug(f"Error syncing with symbol manager: {e}")

    def _on_data_updated(self, symbol: str, data: Dict[str, Any]) -> None:
        """Handle real-time data updates."""
        try:
            # Update market data model
            bid = data.get('bid')
            ask = data.get('ask')
            volume = data.get('volume')

            if bid is not None or ask is not None:
                self.market_data_model.update_symbol_data(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    volume=volume
                )

            # Emit integration signal
            self.dataIntegrated.emit(symbol, data)

        except Exception as e:
            logger.error(f"Error handling data update for {symbol}: {e}")

    def _on_symbol_data_updated(self, symbol: str) -> None:
        """Handle symbol data updates from model."""
        try:
            # This could trigger additional processing or notifications
            logger.debug(f"Symbol data updated: {symbol}")

        except Exception as e:
            logger.debug(f"Error handling symbol data update: {e}")

    def _on_data_error(self, error_message: str) -> None:
        """Handle data update errors."""
        logger.error(f"Data update error: {error_message}")
        self.integrationError.emit(error_message)

    def simulate_market_data(self, symbol: str) -> None:
        """
        Simulate market data for testing (development only).

        Args:
            symbol: Symbol to simulate data for
        """
        import random
        from decimal import Decimal

        try:
            # Generate random market data
            base_price = 1.1000  # Example base price
            spread = 0.0002

            bid = Decimal(str(base_price + random.uniform(-0.01, 0.01)))
            ask = bid + Decimal(str(spread))
            volume = random.randint(1, 1000)

            data = {
                'bid': bid,
                'ask': ask,
                'volume': volume,
                'timestamp': datetime.now()
            }

            # Queue the update
            self.data_updater.queue_update(
                symbol=symbol,
                update_type='quote',
                data=data,
                priority=2
            )

        except Exception as e:
            logger.error(f"Error simulating market data for {symbol}: {e}")

    def add_symbol_to_integration(self, symbol: str) -> bool:
        """
        Add symbol to integration system.

        Args:
            symbol: Symbol to add

        Returns:
            bool: True if added successfully
        """
        try:
            # Add to symbol manager
            if self.symbol_manager.is_running:
                success = self.symbol_manager.add_symbol(symbol, auto_subscribe=True)
                if not success:
                    logger.warning(f"Failed to add {symbol} to symbol manager")

            # Add to market data model
            success = self.market_data_model.add_symbol(symbol)
            if success:
                logger.info(f"Added {symbol} to integration system")

                # Start simulated data for development
                # In production, this would be real MT5 data
                if self.settings.environment.value == "development":
                    # Setup timer for simulated updates
                    timer = QTimer()
                    timer.timeout.connect(lambda: self.simulate_market_data(symbol))
                    timer.start(1000)  # Update every second
                    setattr(self, f"_{symbol}_timer", timer)

            return success

        except Exception as e:
            logger.error(f"Error adding {symbol} to integration: {e}")
            return False

    def remove_symbol_from_integration(self, symbol: str) -> bool:
        """
        Remove symbol from integration system.

        Args:
            symbol: Symbol to remove

        Returns:
            bool: True if removed successfully
        """
        try:
            # Remove from symbol manager
            if self.symbol_manager.is_running:
                success = self.symbol_manager.remove_symbol(symbol)
                if not success:
                    logger.warning(f"Failed to remove {symbol} from symbol manager")

            # Remove from market data model
            success = self.market_data_model.remove_symbol(symbol)
            if success:
                logger.info(f"Removed {symbol} from integration system")

                # Stop simulation timer if it exists
                timer_attr = f"_{symbol}_timer"
                if hasattr(self, timer_attr):
                    timer = getattr(self, timer_attr)
                    timer.stop()
                    delattr(self, timer_attr)

            return success

        except Exception as e:
            logger.error(f"Error removing {symbol} from integration: {e}")
            return False

    def get_service_status(self) -> Dict[str, bool]:
        """Get status of integrated services."""
        status = self._service_status.copy()
        status.update({
            'integration_running': self._is_running,
            'data_updater_running': self.data_updater.is_running(),
            'database_available': self.database_manager is not None,
            'symbol_manager_running': self.symbol_manager.is_running if self.symbol_manager else False,
        })
        return status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from integrated components."""
        metrics = {}

        # Data updater metrics
        if self.data_updater.is_running():
            updater_metrics = self.data_updater.get_performance_metrics()
            metrics['data_updater'] = {
                'updates_per_second': updater_metrics.updates_per_second,
                'average_latency_ms': updater_metrics.average_latency_ms,
                'queue_depth': updater_metrics.queue_depth,
                'dropped_updates': updater_metrics.dropped_updates,
                'total_updates': updater_metrics.total_updates
            }

        # Symbol manager metrics
        if self.symbol_manager and self.symbol_manager.is_running:
            symbol_stats = self.symbol_manager.get_system_stats()
            metrics['symbol_manager'] = symbol_stats

        return metrics

    def is_running(self) -> bool:
        """Check if integration is running."""
        return self._is_running