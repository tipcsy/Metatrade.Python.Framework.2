"""
Real-time data updater for GUI components.

This module provides thread-safe real-time data updates for GUI components
with performance optimization and batching for handling high-frequency data.
"""

from __future__ import annotations

import threading
import queue
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from collections import defaultdict

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, QThread, QMutex, QMutexLocker

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataUpdate:
    """Data update message."""
    symbol: str
    update_type: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=high, 2=normal, 3=low


@dataclass
class PerformanceMetrics:
    """Performance metrics for data updates."""
    updates_per_second: float = 0.0
    average_latency_ms: float = 0.0
    peak_latency_ms: float = 0.0
    queue_depth: int = 0
    dropped_updates: int = 0
    total_updates: int = 0


class RealTimeDataUpdater(QObject):
    """
    Thread-safe real-time data updater for GUI components.

    Features:
    - High-performance batched updates
    - Priority-based update processing
    - Automatic rate limiting and throttling
    - Performance monitoring and metrics
    - Thread-safe GUI updates via signals
    - Memory-efficient update queuing
    """

    # Signals for GUI updates
    dataUpdated = pyqtSignal(str, dict)           # symbol, data
    batchUpdated = pyqtSignal(list)               # list of updates
    performanceMetrics = pyqtSignal(object)       # PerformanceMetrics
    errorOccurred = pyqtSignal(str)               # error message

    def __init__(self, max_updates_per_second: int = 1000):
        """
        Initialize real-time data updater.

        Args:
            max_updates_per_second: Maximum updates to process per second
        """
        super().__init__()

        # Configuration
        self.settings = get_settings()
        self.max_updates_per_second = max_updates_per_second

        # Update queue and processing
        self._update_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=10000)
        self._processing_thread: Optional[QThread] = None
        self._is_running = False
        self._shutdown_requested = False

        # Thread safety
        self._mutex = QMutex()

        # Performance tracking
        self._performance_metrics = PerformanceMetrics()
        self._update_timestamps: List[datetime] = []
        self._latency_samples: List[float] = []

        # Batching configuration
        self._batch_size = 50
        self._batch_timeout_ms = 10
        self._last_batch_time = datetime.now()

        # Rate limiting
        self._rate_limiter_window = timedelta(seconds=1)
        self._rate_limiter_updates: Dict[str, List[datetime]] = defaultdict(list)

        # Update subscribers
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

        # Timers for performance monitoring
        self._setup_timers()

        logger.info(f"Real-time data updater initialized (max {max_updates_per_second} updates/sec)")

    def _setup_timers(self) -> None:
        """Setup performance monitoring timers."""
        # Performance metrics timer
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self._calculate_performance_metrics)
        self.metrics_timer.start(1000)  # Every second

        # Cleanup timer for old data
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._cleanup_old_data)
        self.cleanup_timer.start(10000)  # Every 10 seconds

    def start(self) -> bool:
        """Start the real-time data updater."""
        if self._is_running:
            logger.warning("Data updater already running")
            return True

        try:
            self._is_running = True
            self._shutdown_requested = False

            # Start processing in a separate thread
            self._processing_thread = ProcessingThread(self)
            self._processing_thread.start()

            logger.info("Real-time data updater started")
            return True

        except Exception as e:
            logger.error(f"Failed to start data updater: {e}")
            self._is_running = False
            return False

    def stop(self) -> None:
        """Stop the real-time data updater."""
        if not self._is_running:
            return

        logger.info("Stopping real-time data updater...")

        self._shutdown_requested = True
        self._is_running = False

        # Stop processing thread
        if self._processing_thread:
            self._processing_thread.quit()
            self._processing_thread.wait(5000)  # Wait up to 5 seconds

        # Stop timers
        if hasattr(self, 'metrics_timer'):
            self.metrics_timer.stop()
        if hasattr(self, 'cleanup_timer'):
            self.cleanup_timer.stop()

        logger.info("Real-time data updater stopped")

    def queue_update(
        self,
        symbol: str,
        update_type: str,
        data: Dict[str, Any],
        priority: int = 2
    ) -> bool:
        """
        Queue a data update for processing.

        Args:
            symbol: Symbol identifier
            update_type: Type of update (tick, quote, trade, etc.)
            data: Update data
            priority: Update priority (1=high, 2=normal, 3=low)

        Returns:
            bool: True if update was queued successfully
        """
        if not self._is_running:
            return False

        try:
            # Check rate limiting
            if not self._check_rate_limit(symbol):
                self._performance_metrics.dropped_updates += 1
                return False

            # Create update object
            update = DataUpdate(
                symbol=symbol,
                update_type=update_type,
                data=data,
                timestamp=datetime.now(),
                priority=priority
            )

            # Queue the update (lower number = higher priority)
            self._update_queue.put((priority, update), block=False)

            # Update rate limiter
            self._update_rate_limiter(symbol)

            return True

        except queue.Full:
            self._performance_metrics.dropped_updates += 1
            logger.warning(f"Update queue full, dropping update for {symbol}")
            return False

        except Exception as e:
            logger.error(f"Error queuing update for {symbol}: {e}")
            return False

    def _check_rate_limit(self, symbol: str) -> bool:
        """Check if symbol is within rate limits."""
        current_time = datetime.now()
        cutoff_time = current_time - self._rate_limiter_window

        # Clean old timestamps
        symbol_updates = self._rate_limiter_updates[symbol]
        symbol_updates[:] = [ts for ts in symbol_updates if ts > cutoff_time]

        # Check rate limit
        max_updates_per_symbol = max(1, self.max_updates_per_second // 100)  # Limit per symbol
        return len(symbol_updates) < max_updates_per_symbol

    def _update_rate_limiter(self, symbol: str) -> None:
        """Update rate limiter for symbol."""
        self._rate_limiter_updates[symbol].append(datetime.now())

    def subscribe_to_updates(self, update_type: str, callback: Callable) -> None:
        """
        Subscribe to specific update types.

        Args:
            update_type: Type of updates to subscribe to
            callback: Callback function to call on updates
        """
        with QMutexLocker(self._mutex):
            self._subscribers[update_type].append(callback)

        logger.debug(f"Subscribed to {update_type} updates")

    def unsubscribe_from_updates(self, update_type: str, callback: Callable) -> bool:
        """
        Unsubscribe from specific update types.

        Args:
            update_type: Type of updates to unsubscribe from
            callback: Callback function to remove

        Returns:
            bool: True if unsubscribed successfully
        """
        try:
            with QMutexLocker(self._mutex):
                if callback in self._subscribers[update_type]:
                    self._subscribers[update_type].remove(callback)
                    return True

        except Exception as e:
            logger.error(f"Error unsubscribing from {update_type}: {e}")

        return False

    def _process_updates(self) -> None:
        """Process queued updates (runs in separate thread)."""
        batch = []
        batch_start_time = datetime.now()

        while self._is_running and not self._shutdown_requested:
            try:
                # Get update with timeout
                try:
                    priority, update = self._update_queue.get(timeout=0.001)  # 1ms timeout
                except queue.Empty:
                    # Process batch if timeout reached
                    if batch and (datetime.now() - batch_start_time).total_seconds() * 1000 >= self._batch_timeout_ms:
                        self._process_batch(batch)
                        batch.clear()
                        batch_start_time = datetime.now()
                    continue

                # Add to batch
                batch.append(update)

                # Process batch if it's full or timeout reached
                batch_duration_ms = (datetime.now() - batch_start_time).total_seconds() * 1000

                if len(batch) >= self._batch_size or batch_duration_ms >= self._batch_timeout_ms:
                    self._process_batch(batch)
                    batch.clear()
                    batch_start_time = datetime.now()

            except Exception as e:
                logger.error(f"Error processing updates: {e}")
                self.errorOccurred.emit(str(e))

        # Process remaining batch
        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: List[DataUpdate]) -> None:
        """Process a batch of updates."""
        try:
            current_time = datetime.now()

            # Group updates by symbol for efficiency
            symbol_updates = defaultdict(list)
            for update in batch:
                symbol_updates[update.symbol].append(update)

            # Process each symbol's updates
            for symbol, updates in symbol_updates.items():
                # Get the most recent update for each symbol
                latest_update = max(updates, key=lambda u: u.timestamp)

                # Calculate latency
                latency_ms = (current_time - latest_update.timestamp).total_seconds() * 1000
                self._latency_samples.append(latency_ms)

                # Emit signal for GUI update
                self.dataUpdated.emit(symbol, latest_update.data)

                # Call subscribers
                self._notify_subscribers(latest_update.update_type, latest_update)

                # Update metrics
                self._performance_metrics.total_updates += 1

            # Emit batch update signal
            self.batchUpdated.emit(batch)

            # Track update timestamps for rate calculation
            self._update_timestamps.append(current_time)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.errorOccurred.emit(str(e))

    def _notify_subscribers(self, update_type: str, update: DataUpdate) -> None:
        """Notify subscribers of updates."""
        try:
            with QMutexLocker(self._mutex):
                subscribers = self._subscribers.get(update_type, [])

            for callback in subscribers:
                try:
                    callback(update)
                except Exception as e:
                    logger.debug(f"Error in subscriber callback: {e}")

        except Exception as e:
            logger.debug(f"Error notifying subscribers: {e}")

    def _calculate_performance_metrics(self) -> None:
        """Calculate and emit performance metrics."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(seconds=5)  # 5-second window

            # Clean old timestamps
            self._update_timestamps = [ts for ts in self._update_timestamps if ts > cutoff_time]
            self._latency_samples = self._latency_samples[-1000:]  # Keep last 1000 samples

            # Calculate metrics
            self._performance_metrics.updates_per_second = len(self._update_timestamps) / 5.0
            self._performance_metrics.queue_depth = self._update_queue.qsize()

            if self._latency_samples:
                self._performance_metrics.average_latency_ms = sum(self._latency_samples) / len(self._latency_samples)
                self._performance_metrics.peak_latency_ms = max(self._latency_samples)
            else:
                self._performance_metrics.average_latency_ms = 0.0
                self._performance_metrics.peak_latency_ms = 0.0

            # Emit metrics
            self.performanceMetrics.emit(self._performance_metrics)

        except Exception as e:
            logger.debug(f"Error calculating performance metrics: {e}")

    def _cleanup_old_data(self) -> None:
        """Cleanup old rate limiter data."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - self._rate_limiter_window * 2

            # Clean rate limiter data
            for symbol in list(self._rate_limiter_updates.keys()):
                timestamps = self._rate_limiter_updates[symbol]
                timestamps[:] = [ts for ts in timestamps if ts > cutoff_time]

                # Remove empty entries
                if not timestamps:
                    del self._rate_limiter_updates[symbol]

        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self._performance_metrics

    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        return self._update_queue.qsize()

    def is_running(self) -> bool:
        """Check if updater is running."""
        return self._is_running

    def set_batch_size(self, size: int) -> None:
        """Set batch processing size."""
        self._batch_size = max(1, min(size, 1000))
        logger.debug(f"Batch size set to {self._batch_size}")

    def set_batch_timeout(self, timeout_ms: int) -> None:
        """Set batch processing timeout."""
        self._batch_timeout_ms = max(1, min(timeout_ms, 1000))
        logger.debug(f"Batch timeout set to {self._batch_timeout_ms}ms")


class ProcessingThread(QThread):
    """Thread for processing data updates."""

    def __init__(self, updater: RealTimeDataUpdater):
        """Initialize processing thread."""
        super().__init__()
        self.updater = updater

    def run(self) -> None:
        """Run the update processing loop."""
        try:
            self.updater._process_updates()
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
            self.updater.errorOccurred.emit(str(e))