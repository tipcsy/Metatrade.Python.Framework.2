"""
Real-time data collection system for MT5 market data.

This module provides high-performance data collectors for tick data
and quote streams with automatic reconnection, error handling, and
performance optimization.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.config import get_settings
from src.core.exceptions import Mt5ConnectionError, Mt5DataTimeoutError, PerformanceError
from src.core.logging import get_logger
from src.mt5.connection import get_mt5_session_manager
from .models import TickData, MarketEvent, MarketEventType, ProcessingState
from .buffer import get_buffer_manager
from .events import get_event_publisher

logger = get_logger(__name__)
settings = get_settings()


class DataCollector:
    """
    Base class for real-time data collectors.

    Provides common functionality for collecting market data
    from MT5 with error handling and performance monitoring.
    """

    def __init__(
        self,
        name: str,
        collection_interval: float = 0.1,
        max_retries: int = 5,
        retry_delay: float = 1.0
    ):
        """
        Initialize data collector.

        Args:
            name: Collector name
            collection_interval: Data collection interval in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.name = name
        self.collection_interval = collection_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # State management
        self._is_running = False
        self._is_connected = False
        self._last_error = None
        self._error_count = 0

        # Performance tracking
        self._items_collected = 0
        self._start_time = time.time()
        self._last_collection_time = 0
        self._collection_latencies = []

        # Threading
        self._collection_thread = None
        self._stop_event = threading.Event()

        # Dependencies
        self.buffer_manager = get_buffer_manager()
        self.event_publisher = get_event_publisher()
        self.session_manager = get_mt5_session_manager()

        logger.info(f"Initialized {self.name} collector")

    def start(self) -> bool:
        """Start the data collector."""
        if self._is_running:
            logger.warning(f"{self.name} collector already running")
            return True

        try:
            self._stop_event.clear()
            self._collection_thread = threading.Thread(
                target=self._collection_loop,
                name=f"{self.name}-collector"
            )
            self._collection_thread.daemon = True
            self._collection_thread.start()

            self._is_running = True
            self._start_time = time.time()

            logger.info(f"Started {self.name} collector")
            return True

        except Exception as e:
            logger.error(f"Failed to start {self.name} collector: {e}")
            return False

    def stop(self) -> None:
        """Stop the data collector."""
        if not self._is_running:
            return

        logger.info(f"Stopping {self.name} collector...")

        self._is_running = False
        self._stop_event.set()

        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=5.0)

        logger.info(f"Stopped {self.name} collector")

    def _collection_loop(self) -> None:
        """Main collection loop running in separate thread."""
        logger.debug(f"Started collection loop for {self.name}")

        while not self._stop_event.is_set():
            try:
                start_time = time.time()

                # Collect data
                success = self._collect_data()

                # Calculate latency
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                # Track performance
                self._track_performance(success, latency_ms)

                # Sleep until next collection
                elapsed = end_time - start_time
                sleep_time = max(0, self.collection_interval - elapsed)

                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)

            except Exception as e:
                logger.error(f"Error in {self.name} collection loop: {e}")
                self._last_error = str(e)
                self._error_count += 1

                # Exponential backoff on errors
                retry_delay = min(self.retry_delay * (2 ** min(self._error_count, 5)), 30)
                self._stop_event.wait(retry_delay)

    def _collect_data(self) -> bool:
        """
        Collect data from MT5. Override in subclasses.

        Returns:
            bool: True if collection was successful
        """
        raise NotImplementedError("Subclasses must implement _collect_data")

    def _track_performance(self, success: bool, latency_ms: float) -> None:
        """Track collection performance metrics."""
        if success:
            self._items_collected += 1
            self._error_count = max(0, self._error_count - 1)  # Decay error count
        else:
            self._error_count += 1

        # Track latency
        self._collection_latencies.append(latency_ms)
        if len(self._collection_latencies) > 1000:
            self._collection_latencies = self._collection_latencies[-500:]  # Keep last 500

        # Check performance thresholds
        if latency_ms > settings.mt5.performance.max_tick_latency:
            logger.warning(
                f"High collection latency for {self.name}: {latency_ms:.2f}ms"
            )

        self._last_collection_time = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """Get collector performance statistics."""
        current_time = time.time()
        runtime = current_time - self._start_time

        avg_latency = (
            sum(self._collection_latencies) / len(self._collection_latencies)
            if self._collection_latencies else 0
        )

        return {
            "name": self.name,
            "is_running": self._is_running,
            "is_connected": self._is_connected,
            "runtime_seconds": runtime,
            "items_collected": self._items_collected,
            "collection_rate": self._items_collected / runtime if runtime > 0 else 0,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "avg_latency_ms": avg_latency,
            "last_collection_time": self._last_collection_time
        }

    @property
    def is_running(self) -> bool:
        """Check if collector is running."""
        return self._is_running

    @property
    def is_healthy(self) -> bool:
        """Check if collector is healthy."""
        return (
            self._is_running and
            self._error_count < self.max_retries and
            (time.time() - self._last_collection_time) < (self.collection_interval * 5)
        )


class TickCollector(DataCollector):
    """
    High-performance tick data collector.

    Collects real-time tick data from MT5 with optimized buffering,
    quality checking, and latency monitoring.
    """

    def __init__(
        self,
        symbols: List[str],
        collection_interval: float = 0.2,  # 200ms as per architecture
        enable_quality_check: bool = True
    ):
        """
        Initialize tick collector.

        Args:
            symbols: List of symbols to collect
            collection_interval: Collection interval in seconds
            enable_quality_check: Enable data quality checking
        """
        super().__init__(
            name="TickCollector",
            collection_interval=collection_interval
        )

        self.symbols = set(symbols)
        self.enable_quality_check = enable_quality_check

        # Tick-specific tracking
        self._ticks_per_symbol = {symbol: 0 for symbol in symbols}
        self._last_tick_time = {symbol: 0 for symbol in symbols}

        logger.info(f"Initialized tick collector for {len(symbols)} symbols")

    def add_symbol(self, symbol: str) -> bool:
        """Add symbol to collection."""
        if symbol in self.symbols:
            return True

        try:
            self.symbols.add(symbol)
            self._ticks_per_symbol[symbol] = 0
            self._last_tick_time[symbol] = 0

            # Ensure buffer exists
            self.buffer_manager.get_tick_buffer(symbol)

            logger.info(f"Added symbol {symbol} to tick collector")
            return True

        except Exception as e:
            logger.error(f"Failed to add symbol {symbol}: {e}")
            return False

    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from collection."""
        if symbol not in self.symbols:
            return True

        try:
            self.symbols.discard(symbol)
            self._ticks_per_symbol.pop(symbol, None)
            self._last_tick_time.pop(symbol, None)

            logger.info(f"Removed symbol {symbol} from tick collector")
            return True

        except Exception as e:
            logger.error(f"Failed to remove symbol {symbol}: {e}")
            return False

    def _collect_data(self) -> bool:
        """Collect tick data from MT5."""
        try:
            # Get active session
            session = self.session_manager.get_active_session()
            if not session:
                if not self._is_connected:
                    logger.error("No active MT5 session available")
                    self._is_connected = False
                return False

            self._is_connected = True
            collection_time = datetime.now(timezone.utc)
            ticks_collected = 0

            # Collect ticks for all symbols
            for symbol in self.symbols:
                try:
                    # Get latest tick from MT5
                    tick_info = session.symbol_info_tick(symbol)
                    if tick_info is None:
                        continue

                    # Create tick data object
                    tick = TickData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(tick_info.time, timezone.utc),
                        bid=tick_info.bid,
                        ask=tick_info.ask,
                        volume=getattr(tick_info, 'volume', 0),
                        source_account=session.account_name
                    )

                    # Add to buffer
                    tick_buffer = self.buffer_manager.get_tick_buffer(symbol)
                    if tick_buffer and tick_buffer.add_tick(tick):
                        self._ticks_per_symbol[symbol] += 1
                        self._last_tick_time[symbol] = time.time()
                        ticks_collected += 1

                        # Publish tick event
                        await self._publish_tick_event(tick)

                except Exception as e:
                    logger.debug(f"Error collecting tick for {symbol}: {e}")
                    continue

            # Log performance metrics
            if ticks_collected > 0:
                logger.debug(f"Collected {ticks_collected} ticks")

            return ticks_collected > 0

        except Exception as e:
            logger.error(f"Error in tick collection: {e}")
            self._is_connected = False
            return False

    async def _publish_tick_event(self, tick: TickData) -> None:
        """Publish tick received event."""
        try:
            event = MarketEvent(
                event_id=f"tick_{tick.symbol}_{int(time.time() * 1000)}",
                event_type=MarketEventType.TICK_RECEIVED,
                symbol=tick.symbol,
                data={
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "volume": tick.volume,
                    "mid_price": float(tick.mid_price),
                    "spread": float(tick.spread),
                    "latency_ms": tick.latency_ms
                }
            )

            await self.event_publisher.publish_async(event)

        except Exception as e:
            logger.debug(f"Error publishing tick event: {e}")

    def get_tick_stats(self) -> Dict[str, Any]:
        """Get tick-specific statistics."""
        base_stats = self.get_stats()

        base_stats.update({
            "symbols": list(self.symbols),
            "symbol_count": len(self.symbols),
            "ticks_per_symbol": dict(self._ticks_per_symbol),
            "total_ticks": sum(self._ticks_per_symbol.values()),
            "active_symbols": sum(
                1 for symbol in self.symbols
                if (time.time() - self._last_tick_time.get(symbol, 0)) < 60
            )
        })

        return base_stats


class QuoteCollector(DataCollector):
    """
    Quote data collector for symbols not subscribed to ticks.

    Provides periodic quote updates for symbol monitoring
    and market state tracking.
    """

    def __init__(
        self,
        symbols: List[str],
        collection_interval: float = 1.0
    ):
        """
        Initialize quote collector.

        Args:
            symbols: List of symbols to collect quotes for
            collection_interval: Collection interval in seconds
        """
        super().__init__(
            name="QuoteCollector",
            collection_interval=collection_interval
        )

        self.symbols = set(symbols)
        self._quotes_per_symbol = {symbol: 0 for symbol in symbols}

        logger.info(f"Initialized quote collector for {len(symbols)} symbols")

    def _collect_data(self) -> bool:
        """Collect quote data from MT5."""
        try:
            session = self.session_manager.get_active_session()
            if not session:
                self._is_connected = False
                return False

            self._is_connected = True
            quotes_collected = 0

            for symbol in self.symbols:
                try:
                    # Get symbol info with current prices
                    symbol_info = session.symbol_info(symbol)
                    if symbol_info is None:
                        continue

                    # Create quote data (simplified tick)
                    quote = TickData(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc),
                        bid=symbol_info.bid,
                        ask=symbol_info.ask,
                        volume=0,  # Quotes don't have volume
                        source_account=session.account_name
                    )

                    # Add to buffer
                    tick_buffer = self.buffer_manager.get_tick_buffer(symbol)
                    if tick_buffer and tick_buffer.add_tick(quote):
                        self._quotes_per_symbol[symbol] += 1
                        quotes_collected += 1

                except Exception as e:
                    logger.debug(f"Error collecting quote for {symbol}: {e}")
                    continue

            return quotes_collected > 0

        except Exception as e:
            logger.error(f"Error in quote collection: {e}")
            self._is_connected = False
            return False


class DataCollectionManager:
    """
    Centralized manager for all data collectors.

    Coordinates tick and quote collectors with automatic
    symbol management and performance monitoring.
    """

    def __init__(self):
        """Initialize data collection manager."""
        self._tick_collector: Optional[TickCollector] = None
        self._quote_collector: Optional[QuoteCollector] = None

        # Symbol management
        self._tick_symbols: Set[str] = set()
        self._quote_symbols: Set[str] = set()

        # State tracking
        self._is_running = False
        self._monitor_task = None

        logger.info("Data collection manager initialized")

    def start_collection(
        self,
        tick_symbols: List[str] = None,
        quote_symbols: List[str] = None
    ) -> bool:
        """
        Start data collection for specified symbols.

        Args:
            tick_symbols: Symbols for tick collection
            quote_symbols: Symbols for quote collection

        Returns:
            bool: True if started successfully
        """
        try:
            # Initialize collectors
            if tick_symbols:
                self._tick_symbols = set(tick_symbols)
                self._tick_collector = TickCollector(
                    symbols=tick_symbols,
                    collection_interval=0.2  # 200ms as per architecture
                )

            if quote_symbols:
                self._quote_symbols = set(quote_symbols)
                self._quote_collector = QuoteCollector(
                    symbols=quote_symbols,
                    collection_interval=1.0
                )

            # Start collectors
            success = True

            if self._tick_collector:
                success &= self._tick_collector.start()

            if self._quote_collector:
                success &= self._quote_collector.start()

            if success:
                self._is_running = True
                self._start_monitoring()
                logger.info(
                    f"Started data collection: {len(tick_symbols or [])} tick symbols, "
                    f"{len(quote_symbols or [])} quote symbols"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to start data collection: {e}")
            return False

    def stop_collection(self) -> None:
        """Stop all data collection."""
        logger.info("Stopping data collection...")

        self._is_running = False

        # Stop monitoring
        if self._monitor_task:
            self._monitor_task.cancel()

        # Stop collectors
        if self._tick_collector:
            self._tick_collector.stop()

        if self._quote_collector:
            self._quote_collector.stop()

        logger.info("Data collection stopped")

    def add_tick_symbol(self, symbol: str) -> bool:
        """Add symbol to tick collection."""
        try:
            if symbol in self._tick_symbols:
                return True

            # Initialize collector if needed
            if not self._tick_collector and self._is_running:
                self._tick_collector = TickCollector([symbol])
                self._tick_collector.start()
            elif self._tick_collector:
                self._tick_collector.add_symbol(symbol)

            self._tick_symbols.add(symbol)
            logger.info(f"Added {symbol} to tick collection")
            return True

        except Exception as e:
            logger.error(f"Failed to add tick symbol {symbol}: {e}")
            return False

    def remove_tick_symbol(self, symbol: str) -> bool:
        """Remove symbol from tick collection."""
        try:
            if symbol not in self._tick_symbols:
                return True

            if self._tick_collector:
                self._tick_collector.remove_symbol(symbol)

            self._tick_symbols.discard(symbol)
            logger.info(f"Removed {symbol} from tick collection")
            return True

        except Exception as e:
            logger.error(f"Failed to remove tick symbol {symbol}: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        stats = {
            "is_running": self._is_running,
            "tick_symbols": list(self._tick_symbols),
            "quote_symbols": list(self._quote_symbols),
            "collectors": {}
        }

        if self._tick_collector:
            stats["collectors"]["tick"] = self._tick_collector.get_tick_stats()

        if self._quote_collector:
            stats["collectors"]["quote"] = self._quote_collector.get_stats()

        return stats

    def _start_monitoring(self) -> None:
        """Start performance monitoring."""
        async def monitor():
            while self._is_running:
                try:
                    # Check collector health
                    self._check_collector_health()

                    # Log performance metrics
                    stats = self.get_collection_stats()
                    logger.debug(f"Collection stats: {stats}")

                    await asyncio.sleep(30)  # Monitor every 30 seconds

                except Exception as e:
                    logger.error(f"Collection monitoring error: {e}")
                    await asyncio.sleep(30)

        self._monitor_task = asyncio.create_task(monitor())

    def _check_collector_health(self) -> None:
        """Check and report collector health."""
        issues = []

        if self._tick_collector and not self._tick_collector.is_healthy:
            issues.append("tick collector unhealthy")

        if self._quote_collector and not self._quote_collector.is_healthy:
            issues.append("quote collector unhealthy")

        if issues:
            logger.warning(f"Collection health issues: {', '.join(issues)}")

    @property
    def is_running(self) -> bool:
        """Check if collection is running."""
        return self._is_running


# Global data collection manager instance
_collection_manager: Optional[DataCollectionManager] = None


def get_data_collection_manager() -> DataCollectionManager:
    """Get the global data collection manager instance."""
    global _collection_manager

    if _collection_manager is None:
        _collection_manager = DataCollectionManager()

    return _collection_manager