"""
OHLC data processor for converting tick data to candlestick bars.

This module provides high-performance OHLC generation from tick streams
with support for multiple timeframes and automatic bar completion.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Callable
import threading
import uuid

from src.core.config.settings import Settings, Mt5TimeFrame
from src.core.exceptions import DataValidationError, Mt5PerformanceError
from src.core.logging import get_logger
from .models import TickData, OHLCData, MarketEvent, MarketEventType
from .buffer import BufferManager
from .events import DataEventPublisher

logger = get_logger(__name__)


class TimeframeManager:
    """Manages timeframe calculations and bar boundaries."""

    TIMEFRAME_SECONDS = {
        'M1': 60,
        'M2': 120,
        'M3': 180,
        'M4': 240,
        'M5': 300,
        'M6': 360,
        'M10': 600,
        'M12': 720,
        'M15': 900,
        'M20': 1200,
        'M30': 1800,
        'H1': 3600,
        'H2': 7200,
        'H3': 10800,
        'H4': 14400,
        'H6': 21600,
        'H8': 28800,
        'H12': 43200,
        'D1': 86400,
        'W1': 604800,
        'MN1': 2592000,  # Approximate month
    }

    @classmethod
    def get_bar_start_time(cls, timestamp: datetime, timeframe: str) -> datetime:
        """Calculate bar start time for given timestamp and timeframe."""
        seconds = cls.TIMEFRAME_SECONDS.get(timeframe)
        if not seconds:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        # Convert to Unix timestamp for easier calculation
        unix_timestamp = timestamp.timestamp()

        # Calculate bar start
        if timeframe.startswith('M'):
            # Minutes: align to minute boundaries
            bar_start = unix_timestamp - (unix_timestamp % seconds)
        elif timeframe.startswith('H'):
            # Hours: align to hour boundaries
            bar_start = unix_timestamp - (unix_timestamp % seconds)
        elif timeframe == 'D1':
            # Daily: align to midnight UTC
            dt = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            bar_start = dt.timestamp()
        elif timeframe == 'W1':
            # Weekly: align to Monday
            days_since_monday = timestamp.weekday()
            start_of_week = timestamp - timedelta(days=days_since_monday)
            dt = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            bar_start = dt.timestamp()
        else:
            # Default to simple modulo
            bar_start = unix_timestamp - (unix_timestamp % seconds)

        return datetime.fromtimestamp(bar_start, tz=timezone.utc)

    @classmethod
    def get_bar_end_time(cls, bar_start: datetime, timeframe: str) -> datetime:
        """Calculate bar end time."""
        seconds = cls.TIMEFRAME_SECONDS.get(timeframe)
        if not seconds:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        return bar_start + timedelta(seconds=seconds)

    @classmethod
    def is_bar_complete(cls, current_time: datetime, bar_start: datetime, timeframe: str) -> bool:
        """Check if a bar is complete based on current time."""
        bar_end = cls.get_bar_end_time(bar_start, timeframe)
        return current_time >= bar_end


class OHLCProcessor:
    """
    High-performance OHLC processor for converting ticks to candlestick bars.

    Features:
    - Multi-timeframe support
    - Real-time bar updates
    - Automatic bar completion
    - Gap detection and handling
    - Performance monitoring
    """

    def __init__(
        self,
        settings: Settings,
        buffer_manager: BufferManager,
        event_publisher: DataEventPublisher,
    ):
        self.settings = settings
        self.buffer_manager = buffer_manager
        self.event_publisher = event_publisher

        # Active bars for each symbol/timeframe
        self.active_bars: Dict[str, Dict[str, OHLCData]] = defaultdict(dict)

        # Timeframes to process
        self.timeframes = [tf.value.replace('TIMEFRAME_', '') for tf in settings.mt5.default_timeframes]

        # Processing state
        self.is_processing = False
        self.processing_lock = threading.RLock()

        # Performance metrics
        self.metrics = {
            'bars_created': 0,
            'bars_completed': 0,
            'ticks_processed': 0,
            'processing_errors': 0,
            'avg_processing_time_ms': 0.0,
        }

        # Subscribers for bar updates
        self.bar_subscribers: Dict[str, Set[Callable]] = defaultdict(set)

        logger.info(f"OHLC processor initialized for timeframes: {self.timeframes}")

    async def start_processing(self, symbols: List[str]) -> None:
        """
        Start OHLC processing for specified symbols.

        Args:
            symbols: List of symbols to process
        """
        if self.is_processing:
            logger.warning("OHLC processing already running")
            return

        try:
            self.is_processing = True

            # Subscribe to tick data from buffer manager
            for symbol in symbols:
                buffer = self.buffer_manager.get_buffer()
                buffer.subscribe(symbol, self._process_tick)

            # Start bar completion monitoring
            asyncio.create_task(self._monitor_bar_completion())

            # Publish start event
            await self.event_publisher.publish(MarketEvent(
                event_id=str(uuid.uuid4()),
                event_type=MarketEventType.OHLC_UPDATED,
                data={
                    'action': 'processing_started',
                    'symbols': symbols,
                    'timeframes': self.timeframes,
                }
            ))

            logger.info(f"Started OHLC processing for {len(symbols)} symbols")

        except Exception as e:
            self.is_processing = False
            logger.error(f"Failed to start OHLC processing: {e}")
            raise

    async def stop_processing(self) -> None:
        """Stop OHLC processing."""
        if not self.is_processing:
            return

        try:
            self.is_processing = False

            # Complete any pending bars
            await self._complete_pending_bars()

            # Clear state
            self.active_bars.clear()
            self.bar_subscribers.clear()

            logger.info("Stopped OHLC processing")

        except Exception as e:
            logger.error(f"Error stopping OHLC processing: {e}")

    def subscribe_to_bars(self, symbol: str, timeframe: str, callback: Callable[[OHLCData], None]) -> None:
        """Subscribe to bar updates for specific symbol and timeframe."""
        key = f"{symbol}:{timeframe}"
        self.bar_subscribers[key].add(callback)

    def unsubscribe_from_bars(self, symbol: str, timeframe: str, callback: Callable[[OHLCData], None]) -> None:
        """Unsubscribe from bar updates."""
        key = f"{symbol}:{timeframe}"
        self.bar_subscribers[key].discard(callback)

    def _process_tick(self, tick: TickData) -> None:
        """
        Process a single tick and update OHLC bars.

        Args:
            tick: Tick data to process
        """
        start_time = time.perf_counter()

        try:
            with self.processing_lock:
                # Process tick for each timeframe
                for timeframe in self.timeframes:
                    self._update_bar_from_tick(tick, timeframe)

                self.metrics['ticks_processed'] += 1

        except Exception as e:
            self.metrics['processing_errors'] += 1
            logger.error(f"Error processing tick for {tick.symbol}: {e}")

        finally:
            # Update performance metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_processing_time_average(processing_time)

    def _update_bar_from_tick(self, tick: TickData, timeframe: str) -> None:
        """Update OHLC bar with new tick data."""
        symbol = tick.symbol
        bar_start = TimeframeManager.get_bar_start_time(tick.timestamp, timeframe)

        # Get or create active bar
        if timeframe not in self.active_bars[symbol]:
            self.active_bars[symbol][timeframe] = None

        current_bar = self.active_bars[symbol][timeframe]

        # Check if we need a new bar
        if (current_bar is None or
            current_bar.timestamp != bar_start or
            TimeframeManager.is_bar_complete(tick.timestamp, current_bar.timestamp, timeframe)):

            # Complete previous bar if it exists
            if current_bar is not None and not current_bar.is_complete:
                self._complete_bar(current_bar, timeframe)

            # Create new bar
            current_bar = OHLCData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=bar_start,
                open=tick.mid_price,
                high=tick.mid_price,
                low=tick.mid_price,
                close=tick.mid_price,
                volume=tick.volume,
                tick_count=0,
            )

            self.active_bars[symbol][timeframe] = current_bar
            self.metrics['bars_created'] += 1

        # Update bar with tick data
        current_bar.update_from_tick(tick)

        # Add to buffer
        buffer = self.buffer_manager.get_buffer()
        buffer.push_ohlc(current_bar)

        # Notify subscribers
        self._notify_bar_subscribers(current_bar)

    def _complete_bar(self, bar: OHLCData, timeframe: str) -> None:
        """Mark bar as complete and perform final processing."""
        try:
            bar.is_complete = True
            self.metrics['bars_completed'] += 1

            # Add to buffer again with completion flag
            buffer = self.buffer_manager.get_buffer()
            buffer.push_ohlc(bar)

            # Notify subscribers of completion
            self._notify_bar_subscribers(bar)

            # Publish completion event
            asyncio.create_task(self._publish_bar_completion_event(bar))

            logger.debug(f"Completed bar: {bar.symbol} {bar.timeframe} at {bar.timestamp}")

        except Exception as e:
            logger.error(f"Error completing bar: {e}")

    def _notify_bar_subscribers(self, bar: OHLCData) -> None:
        """Notify subscribers of bar updates."""
        key = f"{bar.symbol}:{bar.timeframe}"
        for callback in self.bar_subscribers.get(key, set()):
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Error in bar subscriber callback: {e}")

    async def _publish_bar_completion_event(self, bar: OHLCData) -> None:
        """Publish bar completion event."""
        try:
            event = MarketEvent(
                event_id=str(uuid.uuid4()),
                event_type=MarketEventType.OHLC_UPDATED,
                symbol=bar.symbol,
                data={
                    'action': 'bar_completed',
                    'timeframe': bar.timeframe,
                    'timestamp': bar.timestamp.isoformat(),
                    'ohlc': bar.to_dict(),
                }
            )

            await self.event_publisher.publish(event)

        except Exception as e:
            logger.error(f"Error publishing bar completion event: {e}")

    async def _monitor_bar_completion(self) -> None:
        """Monitor and auto-complete bars based on time."""
        while self.is_processing:
            try:
                current_time = datetime.now(timezone.utc)

                with self.processing_lock:
                    # Check all active bars for completion
                    for symbol, timeframe_bars in self.active_bars.items():
                        for timeframe, bar in list(timeframe_bars.items()):
                            if (bar is not None and
                                not bar.is_complete and
                                TimeframeManager.is_bar_complete(current_time, bar.timestamp, timeframe)):

                                self._complete_bar(bar, timeframe)

                # Check every 0.6 seconds as per architecture (OHLC processing interval)
                await asyncio.sleep(0.6)

            except Exception as e:
                logger.error(f"Error in bar completion monitoring: {e}")
                await asyncio.sleep(1.0)

    async def _complete_pending_bars(self) -> None:
        """Complete all pending bars during shutdown."""
        try:
            with self.processing_lock:
                for symbol, timeframe_bars in self.active_bars.items():
                    for timeframe, bar in timeframe_bars.items():
                        if bar is not None and not bar.is_complete:
                            self._complete_bar(bar, timeframe)

        except Exception as e:
            logger.error(f"Error completing pending bars: {e}")

    def _update_processing_time_average(self, processing_time_ms: float) -> None:
        """Update running average of processing time."""
        if self.metrics['ticks_processed'] <= 1:
            self.metrics['avg_processing_time_ms'] = processing_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['avg_processing_time_ms'] = (
                alpha * processing_time_ms +
                (1 - alpha) * self.metrics['avg_processing_time_ms']
            )

    def get_active_bars(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get active bars information.

        Args:
            symbol: Optional symbol filter

        Returns:
            Dict containing active bars information
        """
        with self.processing_lock:
            if symbol:
                return {
                    symbol: {
                        timeframe: bar.to_dict() if bar else None
                        for timeframe, bar in self.active_bars.get(symbol, {}).items()
                    }
                }
            else:
                return {
                    sym: {
                        tf: bar.to_dict() if bar else None
                        for tf, bar in bars.items()
                    }
                    for sym, bars in self.active_bars.items()
                }

    def get_metrics(self) -> Dict[str, Any]:
        """Get OHLC processing metrics."""
        return {
            'is_processing': self.is_processing,
            'timeframes': self.timeframes,
            'active_symbols': len(self.active_bars),
            'total_active_bars': sum(
                len(bars) for bars in self.active_bars.values()
            ),
            'metrics': self.metrics.copy(),
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on OHLC processor."""
        health = {
            'status': 'healthy' if self.is_processing else 'stopped',
            'is_processing': self.is_processing,
            'active_symbols': len(self.active_bars),
            'avg_processing_time_ms': self.metrics['avg_processing_time_ms'],
            'error_rate': 0.0,
        }

        # Calculate error rate
        total_processed = self.metrics['ticks_processed']
        if total_processed > 0:
            error_rate = self.metrics['processing_errors'] / total_processed
            health['error_rate'] = error_rate

            # Set status based on error rate
            if error_rate > 0.05:  # 5% error rate threshold
                health['status'] = 'degraded'
            elif error_rate > 0.10:  # 10% error rate threshold
                health['status'] = 'unhealthy'

        # Check processing performance
        if self.metrics['avg_processing_time_ms'] > 10.0:  # 10ms threshold
            health['status'] = 'high_latency'

        return health