"""
High-performance data buffering system for real-time market data.

This module provides circular buffers and memory-efficient storage
for tick data and OHLC bars with automatic overflow handling and
performance optimization.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

from src.core.config import get_settings
from src.core.exceptions import BaseFrameworkError, Mt5PerformanceError
from src.core.logging import get_logger
from .models import TickData, OHLCData, MarketEvent, ProcessingState

logger = get_logger(__name__)
settings = get_settings()


class DataBuffer:
    """
    Base class for high-performance data buffers.

    Provides thread-safe, memory-efficient circular buffering
    with automatic overflow handling and performance monitoring.
    """

    def __init__(
        self,
        max_size: int,
        name: str = "DataBuffer",
        enable_persistence: bool = False,
        overflow_callback: Optional[Callable] = None
    ):
        """
        Initialize data buffer.

        Args:
            max_size: Maximum buffer size
            name: Buffer name for logging
            enable_persistence: Enable data persistence
            overflow_callback: Callback for buffer overflow events
        """
        self.max_size = max_size
        self.name = name
        self.enable_persistence = enable_persistence
        self.overflow_callback = overflow_callback

        # Thread-safe storage
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.RLock()

        # Performance metrics
        self._items_added = 0
        self._items_removed = 0
        self._overflows = 0
        self._start_time = time.time()

        # Async support
        self._async_lock = asyncio.Lock()

        logger.info(f"Initialized {self.name} with max_size={max_size}")

    def add(self, item: Any) -> bool:
        """
        Add item to buffer.

        Args:
            item: Item to add

        Returns:
            bool: True if added successfully
        """
        try:
            with self._lock:
                was_full = len(self._buffer) >= self.max_size

                self._buffer.append(item)
                self._items_added += 1

                if was_full:
                    self._overflows += 1
                    if self.overflow_callback:
                        try:
                            self.overflow_callback(self, item)
                        except Exception as e:
                            logger.error(f"Buffer overflow callback error: {e}")

                return True

        except Exception as e:
            logger.error(f"Error adding item to {self.name}: {e}")
            return False

    async def add_async(self, item: Any) -> bool:
        """
        Add item to buffer asynchronously.

        Args:
            item: Item to add

        Returns:
            bool: True if added successfully
        """
        try:
            async with self._async_lock:
                return self.add(item)
        except Exception as e:
            logger.error(f"Error adding item async to {self.name}: {e}")
            return False

    def get_latest(self, count: int = 1) -> List[Any]:
        """
        Get latest N items from buffer.

        Args:
            count: Number of items to retrieve

        Returns:
            List of items
        """
        try:
            with self._lock:
                if count == 1:
                    return [self._buffer[-1]] if self._buffer else []

                return list(self._buffer)[-count:]

        except Exception as e:
            logger.error(f"Error getting latest from {self.name}: {e}")
            return []

    def get_range(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> List[Any]:
        """
        Get items in specified range.

        Args:
            start_idx: Start index
            end_idx: End index (None for end of buffer)

        Returns:
            List of items in range
        """
        try:
            with self._lock:
                buffer_list = list(self._buffer)
                return buffer_list[start_idx:end_idx]

        except Exception as e:
            logger.error(f"Error getting range from {self.name}: {e}")
            return []

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            logger.debug(f"Cleared {self.name}")

    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0

    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return len(self._buffer) >= self.max_size

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer performance statistics."""
        with self._lock:
            current_time = time.time()
            runtime = current_time - self._start_time

            return {
                "name": self.name,
                "size": len(self._buffer),
                "max_size": self.max_size,
                "items_added": self._items_added,
                "items_removed": self._items_removed,
                "overflows": self._overflows,
                "runtime_seconds": runtime,
                "add_rate_per_second": self._items_added / runtime if runtime > 0 else 0,
                "utilization_percent": (len(self._buffer) / self.max_size) * 100,
                "overflow_rate_percent": (self._overflows / self._items_added * 100) if self._items_added > 0 else 0
            }


class TickBuffer(DataBuffer):
    """
    Specialized buffer for tick data with performance optimization.

    Provides high-throughput tick storage with automatic quality
    assessment and latency tracking.
    """

    def __init__(
        self,
        symbol: str,
        max_size: int = None,
        enable_quality_check: bool = True,
        max_latency_ms: float = 100.0
    ):
        """
        Initialize tick buffer.

        Args:
            symbol: Trading symbol
            max_size: Maximum buffer size (uses settings if None)
            enable_quality_check: Enable data quality checking
            max_latency_ms: Maximum acceptable latency
        """
        if max_size is None:
            max_size = settings.mt5.performance.max_ticks_per_second * 10

        super().__init__(
            max_size=max_size,
            name=f"TickBuffer-{symbol}",
            enable_persistence=True
        )

        self.symbol = symbol
        self.enable_quality_check = enable_quality_check
        self.max_latency_ms = max_latency_ms

        # Tick-specific metrics
        self._total_volume = 0
        self._quality_issues = 0
        self._latency_violations = 0

        logger.info(f"Initialized tick buffer for {symbol}")

    def add_tick(self, tick: TickData) -> bool:
        """
        Add tick data with validation and quality checking.

        Args:
            tick: Tick data to add

        Returns:
            bool: True if added successfully
        """
        try:
            # Validate symbol
            if tick.symbol != self.symbol:
                logger.warning(f"Symbol mismatch: expected {self.symbol}, got {tick.symbol}")
                return False

            # Calculate latency if not set
            if tick.latency_ms is None:
                tick.latency_ms = tick.calculate_latency()

            # Quality checking
            if self.enable_quality_check:
                self._check_tick_quality(tick)

            # Track latency violations
            if tick.latency_ms > self.max_latency_ms:
                self._latency_violations += 1
                logger.warning(
                    f"Latency violation for {self.symbol}: "
                    f"{tick.latency_ms:.2f}ms > {self.max_latency_ms}ms"
                )

            # Update metrics
            self._total_volume += tick.volume

            # Add to buffer
            return self.add(tick)

        except Exception as e:
            logger.error(f"Error adding tick to {self.name}: {e}")
            return False

    def _check_tick_quality(self, tick: TickData) -> None:
        """Check and assess tick data quality."""
        issues = []

        # Check for invalid prices
        if tick.bid <= 0 or tick.ask <= 0:
            issues.append("invalid_prices")

        # Check spread
        spread = tick.spread
        if spread < 0:
            issues.append("negative_spread")
        elif spread > tick.mid_price * 0.01:  # 1% spread threshold
            issues.append("excessive_spread")

        # Check timestamp
        now = datetime.now(timezone.utc)
        time_diff = (now - tick.timestamp).total_seconds()
        if time_diff > 60:  # More than 1 minute old
            issues.append("stale_data")

        # Update quality based on issues
        if issues:
            self._quality_issues += 1

            if len(issues) >= 3:
                tick.quality = "invalid"
            elif len(issues) >= 2:
                tick.quality = "suspect"
            else:
                tick.quality = "low"

            logger.debug(f"Quality issues for {self.symbol}: {issues}")

    def get_latest_tick(self) -> Optional[TickData]:
        """Get the latest tick data."""
        latest = self.get_latest(1)
        return latest[0] if latest else None

    def get_price_history(self, count: int = 100) -> List[float]:
        """Get recent mid prices."""
        ticks = self.get_latest(count)
        return [float(tick.mid_price) for tick in ticks if isinstance(tick, TickData)]

    def get_spread_history(self, count: int = 100) -> List[float]:
        """Get recent spreads."""
        ticks = self.get_latest(count)
        return [float(tick.spread) for tick in ticks if isinstance(tick, TickData)]

    def get_tick_stats(self) -> Dict[str, Any]:
        """Get tick-specific statistics."""
        base_stats = self.get_stats()

        # Add tick-specific metrics
        base_stats.update({
            "symbol": self.symbol,
            "total_volume": self._total_volume,
            "quality_issues": self._quality_issues,
            "latency_violations": self._latency_violations,
            "avg_volume_per_tick": self._total_volume / max(self._items_added, 1),
            "quality_issue_rate": (self._quality_issues / max(self._items_added, 1)) * 100,
            "latency_violation_rate": (self._latency_violations / max(self._items_added, 1)) * 100
        })

        # Add current market state
        latest_tick = self.get_latest_tick()
        if latest_tick:
            base_stats.update({
                "current_bid": float(latest_tick.bid),
                "current_ask": float(latest_tick.ask),
                "current_mid": float(latest_tick.mid_price),
                "current_spread": float(latest_tick.spread),
                "last_update": latest_tick.timestamp.isoformat()
            })

        return base_stats


class OHLCBuffer(DataBuffer):
    """
    Specialized buffer for OHLC data with timeframe management.

    Provides efficient storage of candlestick data with automatic
    bar completion and trend analysis support.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        max_size: int = 10000,
        enable_trend_analysis: bool = True
    ):
        """
        Initialize OHLC buffer.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, H1, etc.)
            max_size: Maximum buffer size
            enable_trend_analysis: Enable trend analysis
        """
        super().__init__(
            max_size=max_size,
            name=f"OHLCBuffer-{symbol}-{timeframe}",
            enable_persistence=True
        )

        self.symbol = symbol
        self.timeframe = timeframe
        self.enable_trend_analysis = enable_trend_analysis

        # OHLC-specific metrics
        self._complete_bars = 0
        self._incomplete_bars = 0

        logger.info(f"Initialized OHLC buffer for {symbol} {timeframe}")

    def add_bar(self, bar: OHLCData) -> bool:
        """
        Add OHLC bar with validation.

        Args:
            bar: OHLC bar data

        Returns:
            bool: True if added successfully
        """
        try:
            # Validate symbol and timeframe
            if bar.symbol != self.symbol:
                logger.warning(f"Symbol mismatch: expected {self.symbol}, got {bar.symbol}")
                return False

            if bar.timeframe != self.timeframe:
                logger.warning(
                    f"Timeframe mismatch: expected {self.timeframe}, got {bar.timeframe}"
                )
                return False

            # Update metrics
            if bar.is_complete:
                self._complete_bars += 1
            else:
                self._incomplete_bars += 1

            # Add to buffer
            return self.add(bar)

        except Exception as e:
            logger.error(f"Error adding OHLC bar to {self.name}: {e}")
            return False

    def get_latest_bar(self) -> Optional[OHLCData]:
        """Get the latest OHLC bar."""
        latest = self.get_latest(1)
        return latest[0] if latest else None

    def get_complete_bars(self, count: int = 100) -> List[OHLCData]:
        """Get recent complete bars."""
        all_bars = self.get_latest(count * 2)  # Get more to filter complete ones
        complete_bars = [bar for bar in all_bars
                        if isinstance(bar, OHLCData) and bar.is_complete]
        return complete_bars[-count:]  # Return latest N complete bars

    def get_price_series(self, price_type: str = "close", count: int = 100) -> List[float]:
        """
        Get price series for analysis.

        Args:
            price_type: Price type (open, high, low, close)
            count: Number of bars

        Returns:
            List of prices
        """
        bars = self.get_complete_bars(count)

        price_attr_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "typical": "typical_price",
            "weighted": "weighted_price"
        }

        attr = price_attr_map.get(price_type, "close")
        return [float(getattr(bar, attr)) for bar in bars]

    def get_ohlc_stats(self) -> Dict[str, Any]:
        """Get OHLC-specific statistics."""
        base_stats = self.get_stats()

        # Add OHLC-specific metrics
        base_stats.update({
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "complete_bars": self._complete_bars,
            "incomplete_bars": self._incomplete_bars,
            "completion_rate": (self._complete_bars / max(self._items_added, 1)) * 100
        })

        # Add current bar state
        latest_bar = self.get_latest_bar()
        if latest_bar:
            base_stats.update({
                "current_open": float(latest_bar.open),
                "current_high": float(latest_bar.high),
                "current_low": float(latest_bar.low),
                "current_close": float(latest_bar.close),
                "current_volume": latest_bar.volume,
                "current_tick_count": latest_bar.tick_count,
                "is_complete": latest_bar.is_complete,
                "last_update": latest_bar.timestamp.isoformat()
            })

        return base_stats


class BufferManager:
    """
    Centralized manager for all data buffers.

    Provides unified access to tick and OHLC buffers with
    performance monitoring and resource management.
    """

    def __init__(self):
        """Initialize buffer manager."""
        self._tick_buffers: Dict[str, TickBuffer] = {}
        self._ohlc_buffers: Dict[str, Dict[str, OHLCBuffer]] = {}
        self._lock = threading.RLock()

        # Performance monitoring
        self._monitoring_enabled = True
        self._monitor_interval = 10  # seconds
        self._monitor_task = None

        logger.info("Buffer manager initialized")

    def get_tick_buffer(
        self,
        symbol: str,
        create_if_missing: bool = True
    ) -> Optional[TickBuffer]:
        """Get or create tick buffer for symbol."""
        with self._lock:
            if symbol not in self._tick_buffers and create_if_missing:
                self._tick_buffers[symbol] = TickBuffer(symbol)
                logger.info(f"Created tick buffer for {symbol}")

            return self._tick_buffers.get(symbol)

    def get_ohlc_buffer(
        self,
        symbol: str,
        timeframe: str,
        create_if_missing: bool = True
    ) -> Optional[OHLCBuffer]:
        """Get or create OHLC buffer for symbol and timeframe."""
        with self._lock:
            if symbol not in self._ohlc_buffers:
                self._ohlc_buffers[symbol] = {}

            if timeframe not in self._ohlc_buffers[symbol] and create_if_missing:
                self._ohlc_buffers[symbol][timeframe] = OHLCBuffer(
                    symbol, timeframe
                )
                logger.info(f"Created OHLC buffer for {symbol} {timeframe}")

            return self._ohlc_buffers[symbol].get(timeframe)

    def get_all_buffers(self) -> Dict[str, Any]:
        """Get all managed buffers."""
        with self._lock:
            return {
                "tick_buffers": dict(self._tick_buffers),
                "ohlc_buffers": dict(self._ohlc_buffers)
            }

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics."""
        with self._lock:
            stats = {
                "tick_buffers": {},
                "ohlc_buffers": {},
                "summary": {
                    "total_tick_buffers": len(self._tick_buffers),
                    "total_ohlc_buffers": sum(
                        len(timeframes) for timeframes in self._ohlc_buffers.values()
                    ),
                    "total_symbols": len(set(
                        list(self._tick_buffers.keys()) +
                        list(self._ohlc_buffers.keys())
                    ))
                }
            }

            # Collect tick buffer stats
            for symbol, buffer in self._tick_buffers.items():
                stats["tick_buffers"][symbol] = buffer.get_tick_stats()

            # Collect OHLC buffer stats
            for symbol, timeframes in self._ohlc_buffers.items():
                stats["ohlc_buffers"][symbol] = {}
                for timeframe, buffer in timeframes.items():
                    stats["ohlc_buffers"][symbol][timeframe] = buffer.get_ohlc_stats()

            return stats

    def clear_all_buffers(self) -> None:
        """Clear all buffers."""
        with self._lock:
            for buffer in self._tick_buffers.values():
                buffer.clear()

            for timeframes in self._ohlc_buffers.values():
                for buffer in timeframes.values():
                    buffer.clear()

            logger.info("Cleared all buffers")

    def start_monitoring(self) -> None:
        """Start buffer performance monitoring."""
        if self._monitoring_enabled and self._monitor_task is None:
            async def monitor():
                while self._monitoring_enabled:
                    try:
                        stats = self.get_buffer_stats()
                        self._log_performance_metrics(stats)
                        await asyncio.sleep(self._monitor_interval)
                    except Exception as e:
                        logger.error(f"Buffer monitoring error: {e}")
                        await asyncio.sleep(self._monitor_interval)

            self._monitor_task = asyncio.create_task(monitor())
            logger.info("Buffer monitoring started")

    def stop_monitoring(self) -> None:
        """Stop buffer performance monitoring."""
        self._monitoring_enabled = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        logger.info("Buffer monitoring stopped")

    def _log_performance_metrics(self, stats: Dict[str, Any]) -> None:
        """Log buffer performance metrics."""
        summary = stats["summary"]

        logger.info(
            f"Buffer Status: {summary['total_tick_buffers']} tick buffers, "
            f"{summary['total_ohlc_buffers']} OHLC buffers, "
            f"{summary['total_symbols']} symbols"
        )

        # Log performance warnings
        for symbol, buffer_stats in stats["tick_buffers"].items():
            if buffer_stats["latency_violation_rate"] > 5:
                logger.warning(
                    f"High latency violations for {symbol}: "
                    f"{buffer_stats['latency_violation_rate']:.1f}%"
                )

            if buffer_stats["overflow_rate_percent"] > 1:
                logger.warning(
                    f"Buffer overflows for {symbol}: "
                    f"{buffer_stats['overflow_rate_percent']:.1f}%"
                )


# Global buffer manager instance
_buffer_manager: Optional[BufferManager] = None


def get_buffer_manager() -> BufferManager:
    """Get the global buffer manager instance."""
    global _buffer_manager

    if _buffer_manager is None:
        _buffer_manager = BufferManager()

    return _buffer_manager