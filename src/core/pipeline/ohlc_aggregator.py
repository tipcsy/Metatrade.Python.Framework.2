"""
OHLC Aggregation Engine.

This module provides high-performance real-time OHLC (Open, High, Low, Close)
aggregation from tick data with multiple timeframe support, volume profiling,
and advanced analytics capabilities.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd

from src.core.config.settings import MarketDataSettings
from src.core.exceptions import AggregationError
from src.core.logging import get_logger
from src.core.data.models import TickData, ProcessedTick, OHLCBar, VolumeProfile, AggregationMetrics

logger = get_logger(__name__)


class TimeFrame(Enum):
    """Supported timeframes for OHLC aggregation."""
    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_10 = "10s"
    SECOND_15 = "15s"
    SECOND_30 = "30s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class AggregationMode(Enum):
    """OHLC aggregation modes."""
    BID = "bid"
    ASK = "ask"
    MID = "mid"
    LAST = "last"
    WEIGHTED = "weighted"


@dataclass
class TimeFrameConfig:
    """Configuration for a specific timeframe."""
    timeframe: TimeFrame
    enabled: bool = True
    buffer_size: int = 1000
    callback: Optional[Callable[[OHLCBar], None]] = None
    store_volume_profile: bool = False
    volume_profile_bins: int = 20


@dataclass
class AggregatorConfig:
    """OHLC aggregator configuration."""
    timeframes: List[TimeFrameConfig] = field(default_factory=list)
    aggregation_mode: AggregationMode = AggregationMode.MID
    min_tick_size: float = 0.00001
    enable_volume_profile: bool = True
    enable_advanced_metrics: bool = True
    max_bars_per_symbol: int = 10000


@dataclass
class BarBuilder:
    """Real-time OHLC bar builder."""
    symbol: str
    timeframe: TimeFrame
    start_time: float
    end_time: float
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    volume: float = 0.0
    tick_count: int = 0
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    price_sum: float = 0.0
    vwap: Optional[float] = None
    volume_profile: Optional[VolumeProfile] = None
    first_tick_time: Optional[float] = None
    last_tick_time: Optional[float] = None

    def add_tick(self, tick: Union[TickData, ProcessedTick], price: float, volume: float) -> None:
        """Add tick data to the bar."""
        current_time = time.time()

        # Initialize prices on first tick
        if self.open_price is None:
            self.open_price = price
            self.high_price = price
            self.low_price = price
            self.first_tick_time = tick.time

        # Update OHLC
        if price > (self.high_price or 0):
            self.high_price = price
        if price < (self.low_price or float('inf')):
            self.low_price = price
        self.close_price = price

        # Update volume and counts
        self.volume += volume
        self.tick_count += 1
        self.price_sum += price * volume
        self.last_tick_time = tick.time

        # Update bid/ask volumes if available
        if hasattr(tick, 'bid') and hasattr(tick, 'ask'):
            if tick.bid > 0 and tick.ask > 0:
                mid = (tick.bid + tick.ask) / 2
                if price >= mid:
                    self.ask_volume += volume
                else:
                    self.bid_volume += volume

        # Update VWAP
        if self.volume > 0:
            self.vwap = self.price_sum / self.volume

        # Update volume profile if enabled
        if self.volume_profile is not None:
            self.volume_profile.add_trade(price, volume)

    def to_ohlc_bar(self) -> OHLCBar:
        """Convert builder to OHLC bar."""
        return OHLCBar(
            symbol=self.symbol,
            timeframe=self.timeframe.value,
            start_time=self.start_time,
            end_time=self.end_time,
            open_price=self.open_price or 0.0,
            high_price=self.high_price or 0.0,
            low_price=self.low_price or 0.0,
            close_price=self.close_price or 0.0,
            volume=self.volume,
            tick_count=self.tick_count,
            bid_volume=self.bid_volume,
            ask_volume=self.ask_volume,
            vwap=self.vwap,
            volume_profile=self.volume_profile,
            first_tick_time=self.first_tick_time,
            last_tick_time=self.last_tick_time
        )


class OHLCAggregator:
    """
    High-performance OHLC aggregation engine.

    Features:
    - Multiple timeframe support
    - Real-time bar building
    - Volume profile calculation
    - Advanced analytics
    - Configurable aggregation modes
    """

    def __init__(self, config: AggregatorConfig) -> None:
        """Initialize OHLC aggregator.

        Args:
            config: Aggregator configuration
        """
        self.config = config

        # Bar builders organized by symbol and timeframe
        self._bar_builders: Dict[str, Dict[TimeFrame, BarBuilder]] = defaultdict(dict)

        # Completed bars storage
        self._completed_bars: Dict[str, Dict[TimeFrame, deque]] = defaultdict(lambda: defaultdict(deque))

        # Timeframe configurations
        self._timeframe_configs: Dict[TimeFrame, TimeFrameConfig] = {
            tf_config.timeframe: tf_config for tf_config in config.timeframes
        }

        # Metrics
        self._metrics: Dict[str, AggregationMetrics] = defaultdict(AggregationMetrics)
        self._metrics_lock = threading.RLock()

        # Callbacks
        self._bar_callbacks: Dict[TimeFrame, List[Callable[[OHLCBar], None]]] = defaultdict(list)

        # Threading
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Time calculation helpers
        self._timeframe_seconds = self._calculate_timeframe_seconds()

        logger.info(
            "OHLC aggregator initialized",
            extra={
                "timeframes": [tf.value for tf in self._timeframe_configs.keys()],
                "aggregation_mode": config.aggregation_mode.value,
                "volume_profile_enabled": config.enable_volume_profile,
            }
        )

    def start(self) -> None:
        """Start the OHLC aggregator."""
        logger.info("Starting OHLC aggregator")

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="OHLC-Cleanup",
            daemon=True
        )
        self._cleanup_thread.start()

        logger.info("OHLC aggregator started")

    def stop(self) -> None:
        """Stop the OHLC aggregator."""
        logger.info("Stopping OHLC aggregator")

        # Signal stop
        self._stop_event.set()

        # Wait for cleanup thread
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)

        logger.info("OHLC aggregator stopped")

    def process_tick(self, tick: Union[TickData, ProcessedTick]) -> None:
        """Process a single tick for all configured timeframes.

        Args:
            tick: Tick data to process
        """
        try:
            # Extract price based on aggregation mode
            price = self._extract_price(tick)
            if price is None or price <= 0:
                return

            volume = getattr(tick, 'volume', 1.0)
            if volume <= 0:
                volume = 1.0

            current_time = time.time()

            # Process for each configured timeframe
            for timeframe, config in self._timeframe_configs.items():
                if not config.enabled:
                    continue

                try:
                    self._process_tick_for_timeframe(tick, timeframe, price, volume, current_time)
                except Exception as e:
                    logger.error(
                        "Error processing tick for timeframe",
                        extra={
                            "symbol": tick.symbol,
                            "timeframe": timeframe.value,
                            "error": str(e),
                        },
                        exc_info=True
                    )

            # Update metrics
            with self._metrics_lock:
                self._metrics[tick.symbol].ticks_processed += 1

        except Exception as e:
            logger.error(
                "Error processing tick",
                extra={"tick": tick, "error": str(e)},
                exc_info=True
            )

    def process_batch(self, ticks: List[Union[TickData, ProcessedTick]]) -> None:
        """Process a batch of ticks.

        Args:
            ticks: List of ticks to process
        """
        for tick in ticks:
            self.process_tick(tick)

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        count: Optional[int] = None
    ) -> List[OHLCBar]:
        """Get completed bars for a symbol and timeframe.

        Args:
            symbol: Symbol name
            timeframe: Timeframe
            count: Number of bars to return (None for all)

        Returns:
            List of OHLC bars
        """
        bars_deque = self._completed_bars[symbol][timeframe]

        if count is None:
            return list(bars_deque)
        else:
            return list(bars_deque)[-count:]

    def get_current_bar(self, symbol: str, timeframe: TimeFrame) -> Optional[OHLCBar]:
        """Get current (incomplete) bar for a symbol and timeframe.

        Args:
            symbol: Symbol name
            timeframe: Timeframe

        Returns:
            Current bar or None if not available
        """
        builder = self._bar_builders[symbol].get(timeframe)
        if builder is None:
            return None

        return builder.to_ohlc_bar()

    def add_bar_callback(self, timeframe: TimeFrame, callback: Callable[[OHLCBar], None]) -> None:
        """Add callback for completed bars.

        Args:
            timeframe: Timeframe to subscribe to
            callback: Callback function
        """
        self._bar_callbacks[timeframe].append(callback)

    def get_metrics(self, symbol: Optional[str] = None) -> Union[AggregationMetrics, Dict[str, AggregationMetrics]]:
        """Get aggregation metrics.

        Args:
            symbol: Specific symbol or None for all

        Returns:
            Metrics for symbol(s)
        """
        with self._metrics_lock:
            if symbol:
                return self._metrics.get(symbol, AggregationMetrics())
            else:
                return dict(self._metrics)

    def get_status(self) -> Dict[str, Any]:
        """Get aggregator status.

        Returns:
            Status dictionary
        """
        with self._metrics_lock:
            active_symbols = len(self._bar_builders)
            total_builders = sum(len(builders) for builders in self._bar_builders.values())
            total_bars = sum(
                sum(len(bars) for bars in symbol_bars.values())
                for symbol_bars in self._completed_bars.values()
            )

            return {
                "active_symbols": active_symbols,
                "active_builders": total_builders,
                "completed_bars": total_bars,
                "enabled_timeframes": [tf.value for tf in self._timeframe_configs.keys()],
                "aggregation_mode": self.config.aggregation_mode.value,
                "metrics_summary": {
                    symbol: {
                        "ticks_processed": metrics.ticks_processed,
                        "bars_created": metrics.bars_created,
                    }
                    for symbol, metrics in self._metrics.items()
                }
            }

    def _extract_price(self, tick: Union[TickData, ProcessedTick]) -> Optional[float]:
        """Extract price from tick based on aggregation mode."""
        try:
            if self.config.aggregation_mode == AggregationMode.BID:
                return getattr(tick, 'bid', None)
            elif self.config.aggregation_mode == AggregationMode.ASK:
                return getattr(tick, 'ask', None)
            elif self.config.aggregation_mode == AggregationMode.LAST:
                return getattr(tick, 'last', None)
            elif self.config.aggregation_mode == AggregationMode.MID:
                bid = getattr(tick, 'bid', None)
                ask = getattr(tick, 'ask', None)
                if bid and ask and bid > 0 and ask > 0:
                    return (bid + ask) / 2
                return getattr(tick, 'last', None)
            elif self.config.aggregation_mode == AggregationMode.WEIGHTED:
                # Use mid price with volume weighting
                bid = getattr(tick, 'bid', None)
                ask = getattr(tick, 'ask', None)
                if bid and ask and bid > 0 and ask > 0:
                    return (bid + ask) / 2
                return getattr(tick, 'last', None)
            else:
                return getattr(tick, 'last', None) or getattr(tick, 'bid', None)

        except Exception as e:
            logger.warning(
                "Error extracting price from tick",
                extra={"tick": tick, "error": str(e)}
            )
            return None

    def _process_tick_for_timeframe(
        self,
        tick: Union[TickData, ProcessedTick],
        timeframe: TimeFrame,
        price: float,
        volume: float,
        current_time: float
    ) -> None:
        """Process tick for a specific timeframe."""
        symbol = tick.symbol

        # Get or create bar builder
        builder = self._bar_builders[symbol].get(timeframe)

        # Determine if we need a new bar
        tick_time = getattr(tick, 'time', current_time)
        bar_start_time, bar_end_time = self._calculate_bar_times(tick_time, timeframe)

        # Check if current builder is for a different time period
        if builder is None or builder.start_time != bar_start_time:
            # Complete existing bar if it exists
            if builder is not None:
                completed_bar = builder.to_ohlc_bar()
                self._store_completed_bar(completed_bar)
                self._notify_bar_callbacks(completed_bar)

            # Create new bar builder
            config = self._timeframe_configs[timeframe]
            volume_profile = None

            if config.store_volume_profile and self.config.enable_volume_profile:
                volume_profile = VolumeProfile(bins=config.volume_profile_bins)

            builder = BarBuilder(
                symbol=symbol,
                timeframe=timeframe,
                start_time=bar_start_time,
                end_time=bar_end_time,
                volume_profile=volume_profile
            )
            self._bar_builders[symbol][timeframe] = builder

        # Add tick to builder
        builder.add_tick(tick, price, volume)

    def _calculate_bar_times(self, tick_time: float, timeframe: TimeFrame) -> Tuple[float, float]:
        """Calculate bar start and end times for a tick."""
        dt = datetime.fromtimestamp(tick_time)

        if timeframe == TimeFrame.SECOND_1:
            start_dt = dt.replace(microsecond=0)
            end_dt = start_dt + timedelta(seconds=1)
        elif timeframe == TimeFrame.SECOND_5:
            second = (dt.second // 5) * 5
            start_dt = dt.replace(second=second, microsecond=0)
            end_dt = start_dt + timedelta(seconds=5)
        elif timeframe == TimeFrame.SECOND_10:
            second = (dt.second // 10) * 10
            start_dt = dt.replace(second=second, microsecond=0)
            end_dt = start_dt + timedelta(seconds=10)
        elif timeframe == TimeFrame.SECOND_15:
            second = (dt.second // 15) * 15
            start_dt = dt.replace(second=second, microsecond=0)
            end_dt = start_dt + timedelta(seconds=15)
        elif timeframe == TimeFrame.SECOND_30:
            second = (dt.second // 30) * 30
            start_dt = dt.replace(second=second, microsecond=0)
            end_dt = start_dt + timedelta(seconds=30)
        elif timeframe == TimeFrame.MINUTE_1:
            start_dt = dt.replace(second=0, microsecond=0)
            end_dt = start_dt + timedelta(minutes=1)
        elif timeframe == TimeFrame.MINUTE_5:
            minute = (dt.minute // 5) * 5
            start_dt = dt.replace(minute=minute, second=0, microsecond=0)
            end_dt = start_dt + timedelta(minutes=5)
        elif timeframe == TimeFrame.MINUTE_15:
            minute = (dt.minute // 15) * 15
            start_dt = dt.replace(minute=minute, second=0, microsecond=0)
            end_dt = start_dt + timedelta(minutes=15)
        elif timeframe == TimeFrame.MINUTE_30:
            minute = (dt.minute // 30) * 30
            start_dt = dt.replace(minute=minute, second=0, microsecond=0)
            end_dt = start_dt + timedelta(minutes=30)
        elif timeframe == TimeFrame.HOUR_1:
            start_dt = dt.replace(minute=0, second=0, microsecond=0)
            end_dt = start_dt + timedelta(hours=1)
        elif timeframe == TimeFrame.HOUR_4:
            hour = (dt.hour // 4) * 4
            start_dt = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
            end_dt = start_dt + timedelta(hours=4)
        elif timeframe == TimeFrame.DAY_1:
            start_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end_dt = start_dt + timedelta(days=1)
        else:
            # Default to 1-minute bars
            start_dt = dt.replace(second=0, microsecond=0)
            end_dt = start_dt + timedelta(minutes=1)

        return start_dt.timestamp(), end_dt.timestamp()

    def _store_completed_bar(self, bar: OHLCBar) -> None:
        """Store a completed bar."""
        symbol = bar.symbol
        timeframe = TimeFrame(bar.timeframe)

        # Get configuration for this timeframe
        config = self._timeframe_configs.get(timeframe)
        if config is None:
            return

        # Add to storage with size limit
        bars_deque = self._completed_bars[symbol][timeframe]
        bars_deque.append(bar)

        # Trim if exceeding maximum size
        max_bars = getattr(config, 'buffer_size', self.config.max_bars_per_symbol)
        while len(bars_deque) > max_bars:
            bars_deque.popleft()

        # Update metrics
        with self._metrics_lock:
            self._metrics[symbol].bars_created += 1
            self._metrics[symbol].last_bar_time = bar.end_time

    def _notify_bar_callbacks(self, bar: OHLCBar) -> None:
        """Notify callbacks about completed bar."""
        timeframe = TimeFrame(bar.timeframe)
        callbacks = self._bar_callbacks.get(timeframe, [])

        for callback in callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(
                    "Error in bar callback",
                    extra={
                        "symbol": bar.symbol,
                        "timeframe": bar.timeframe,
                        "error": str(e),
                    },
                    exc_info=True
                )

    def _calculate_timeframe_seconds(self) -> Dict[TimeFrame, int]:
        """Calculate seconds for each timeframe."""
        return {
            TimeFrame.TICK: 0,
            TimeFrame.SECOND_1: 1,
            TimeFrame.SECOND_5: 5,
            TimeFrame.SECOND_10: 10,
            TimeFrame.SECOND_15: 15,
            TimeFrame.SECOND_30: 30,
            TimeFrame.MINUTE_1: 60,
            TimeFrame.MINUTE_5: 300,
            TimeFrame.MINUTE_15: 900,
            TimeFrame.MINUTE_30: 1800,
            TimeFrame.HOUR_1: 3600,
            TimeFrame.HOUR_4: 14400,
            TimeFrame.DAY_1: 86400,
            TimeFrame.WEEK_1: 604800,
            TimeFrame.MONTH_1: 2592000,  # Approximate
        }

    def _cleanup_loop(self) -> None:
        """Cleanup loop for old data."""
        logger.debug("OHLC cleanup loop started")

        while not self._stop_event.is_set():
            try:
                self._stop_event.wait(timeout=300)  # 5 minutes

                if self._stop_event.is_set():
                    break

                # Perform cleanup
                self._cleanup_old_data()

            except Exception as e:
                logger.error(
                    "Error in cleanup loop",
                    extra={"error": str(e)},
                    exc_info=True
                )

        logger.debug("OHLC cleanup loop ended")

    def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory leaks."""
        try:
            current_time = time.time()
            cleaned_symbols = 0
            cleaned_bars = 0

            # Clean up completed bars older than retention period
            retention_period = 86400 * 7  # 7 days

            for symbol in list(self._completed_bars.keys()):
                symbol_bars = self._completed_bars[symbol]

                for timeframe in list(symbol_bars.keys()):
                    bars_deque = symbol_bars[timeframe]
                    original_size = len(bars_deque)

                    # Remove old bars
                    while bars_deque and current_time - bars_deque[0].end_time > retention_period:
                        bars_deque.popleft()
                        cleaned_bars += 1

                    # Remove empty timeframe entries
                    if len(bars_deque) == 0:
                        del symbol_bars[timeframe]

                # Remove empty symbol entries
                if len(symbol_bars) == 0:
                    del self._completed_bars[symbol]
                    cleaned_symbols += 1

            if cleaned_bars > 0 or cleaned_symbols > 0:
                logger.debug(
                    "Cleaned up old OHLC data",
                    extra={
                        "cleaned_symbols": cleaned_symbols,
                        "cleaned_bars": cleaned_bars,
                    }
                )

        except Exception as e:
            logger.error(
                "Error cleaning up old data",
                extra={"error": str(e)},
                exc_info=True
            )