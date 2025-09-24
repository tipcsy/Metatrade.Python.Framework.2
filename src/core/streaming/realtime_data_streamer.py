"""
Real-time Data Streaming Service.

This module provides high-performance real-time data streaming capabilities
for MetaTrader 5 market data with optimized throughput and minimal latency.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Set, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import threading
import uuid
from datetime import datetime, timezone

from src.core.config.settings import Settings
from src.core.exceptions import (
    StreamingError,
    BufferOverflowError,
    PerformanceError,
)
from src.core.logging import get_logger
from src.core.data.models import TickData, OHLCData, MarketEvent, MarketEventType
from src.mt5.connection import Mt5ConnectionManager

logger = get_logger(__name__)


class StreamingState(Enum):
    """Real-time streaming states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class StreamingMetrics:
    """Real-time streaming performance metrics."""
    # Throughput metrics
    ticks_per_second: float = 0.0
    ohlc_per_second: float = 0.0
    events_per_second: float = 0.0
    total_ticks_streamed: int = 0
    total_ohlc_streamed: int = 0
    total_events_published: int = 0

    # Latency metrics
    avg_tick_latency_ms: float = 0.0
    max_tick_latency_ms: float = 0.0
    avg_processing_latency_ms: float = 0.0
    max_processing_latency_ms: float = 0.0

    # Buffer metrics
    tick_buffer_usage: float = 0.0
    ohlc_buffer_usage: float = 0.0
    max_tick_buffer_usage: float = 0.0
    max_ohlc_buffer_usage: float = 0.0

    # Error metrics
    streaming_errors: int = 0
    buffer_overflows: int = 0
    connection_errors: int = 0
    data_quality_issues: int = 0

    # Performance metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    network_latency_ms: float = 0.0

    # Session metrics
    session_start_time: Optional[datetime] = None
    session_duration_seconds: float = 0.0
    uptime_percentage: float = 100.0


@dataclass
class StreamingConfig:
    """Real-time streaming configuration."""
    # Buffer settings
    tick_buffer_size: int = 10000
    ohlc_buffer_size: int = 5000
    event_buffer_size: int = 1000

    # Performance settings
    max_ticks_per_second: int = 1000
    max_processing_latency_ms: float = 50.0
    tick_fetch_interval_ms: int = 200  # 0.2 seconds
    ohlc_fetch_interval_ms: int = 600  # 0.6 seconds

    # Threading settings
    tick_worker_threads: int = 2
    ohlc_worker_threads: int = 1
    processing_worker_threads: int = 4

    # Quality settings
    enable_data_validation: bool = True
    enable_duplicate_detection: bool = True
    max_duplicate_cache_size: int = 1000

    # Failover settings
    enable_failover: bool = True
    max_connection_retries: int = 3
    connection_retry_delay_ms: int = 1000

    # Monitoring settings
    metrics_collection_interval_ms: int = 1000
    health_check_interval_ms: int = 5000


class RealTimeDataStreamer:
    """
    High-performance real-time market data streaming service.

    Features:
    - Ultra-low latency tick streaming
    - Optimized OHLC data streaming
    - Intelligent buffering and batching
    - Connection failover and recovery
    - Real-time performance monitoring
    - Data quality validation
    - Event-driven architecture
    """

    def __init__(
        self,
        settings: Settings,
        mt5_manager: Mt5ConnectionManager,
        config: Optional[StreamingConfig] = None
    ) -> None:
        """Initialize real-time data streamer.

        Args:
            settings: Application settings
            mt5_manager: MT5 connection manager
            config: Streaming configuration
        """
        self.settings = settings
        self.mt5_manager = mt5_manager
        self.config = config or StreamingConfig()

        # Streaming state
        self.state = StreamingState.STOPPED
        self.is_shutting_down = False
        self._state_lock = threading.Lock()

        # Subscriptions
        self.subscribed_symbols: Set[str] = set()
        self.subscribed_timeframes: Dict[str, Set[str]] = defaultdict(set)
        self.subscription_lock = threading.Lock()

        # Data buffers
        self.tick_buffer: deque = deque(maxlen=self.config.tick_buffer_size)
        self.ohlc_buffer: deque = deque(maxlen=self.config.ohlc_buffer_size)
        self.event_buffer: deque = deque(maxlen=self.config.event_buffer_size)

        # Buffer locks
        self.tick_buffer_lock = threading.Lock()
        self.ohlc_buffer_lock = threading.Lock()
        self.event_buffer_lock = threading.Lock()

        # Duplicate detection
        self.tick_cache: deque = deque(maxlen=self.config.max_duplicate_cache_size)
        self.ohlc_cache: deque = deque(maxlen=self.config.max_duplicate_cache_size)

        # Performance tracking
        self.metrics = StreamingMetrics()
        self.latency_samples: deque = deque(maxlen=1000)
        self.processing_samples: deque = deque(maxlen=1000)

        # Event subscribers
        self.tick_subscribers: List[Callable] = []
        self.ohlc_subscribers: List[Callable] = []
        self.event_subscribers: List[Callable] = []

        # Worker tasks
        self.worker_tasks: List[asyncio.Task] = []

        # Last fetch timestamps
        self.last_tick_fetch: Dict[str, datetime] = {}
        self.last_ohlc_fetch: Dict[str, Dict[str, datetime]] = defaultdict(dict)

        logger.info(
            "Real-time data streamer initialized",
            extra={
                "tick_buffer_size": self.config.tick_buffer_size,
                "ohlc_buffer_size": self.config.ohlc_buffer_size,
                "max_ticks_per_second": self.config.max_ticks_per_second,
            }
        )

    async def start_streaming(self) -> None:
        """Start real-time data streaming."""
        with self._state_lock:
            if self.state in [StreamingState.RUNNING, StreamingState.STARTING]:
                logger.warning("Streaming already started or starting")
                return

            self.state = StreamingState.STARTING

        try:
            logger.info("Starting real-time data streaming")

            # Initialize metrics
            self.metrics.session_start_time = datetime.now(timezone.utc)

            # Start worker tasks
            self.worker_tasks = [
                asyncio.create_task(self._tick_streaming_worker(), name="tick-streamer"),
                asyncio.create_task(self._ohlc_streaming_worker(), name="ohlc-streamer"),
                asyncio.create_task(self._event_processing_worker(), name="event-processor"),
                asyncio.create_task(self._metrics_collector(), name="metrics-collector"),
                asyncio.create_task(self._health_monitor(), name="health-monitor"),
            ]

            # Add processing workers
            for i in range(self.config.processing_worker_threads):
                task = asyncio.create_task(
                    self._data_processing_worker(),
                    name=f"data-processor-{i}"
                )
                self.worker_tasks.append(task)

            with self._state_lock:
                self.state = StreamingState.RUNNING

            logger.info(
                "Real-time data streaming started successfully",
                extra={
                    "worker_tasks": len(self.worker_tasks),
                    "subscribed_symbols": len(self.subscribed_symbols),
                }
            )

        except Exception as e:
            with self._state_lock:
                self.state = StreamingState.ERROR
            logger.error(
                "Failed to start real-time data streaming",
                extra={"error": str(e)},
                exc_info=True
            )
            await self._cleanup_workers()
            raise StreamingError(f"Failed to start streaming: {e}") from e

    async def stop_streaming(self) -> None:
        """Stop real-time data streaming."""
        with self._state_lock:
            if self.state == StreamingState.STOPPED:
                return
            self.state = StreamingState.STOPPING

        self.is_shutting_down = True
        logger.info("Stopping real-time data streaming")

        try:
            # Stop all worker tasks
            await self._cleanup_workers()

            # Clear buffers
            with self.tick_buffer_lock:
                self.tick_buffer.clear()
            with self.ohlc_buffer_lock:
                self.ohlc_buffer.clear()
            with self.event_buffer_lock:
                self.event_buffer.clear()

            with self._state_lock:
                self.state = StreamingState.STOPPED

            logger.info("Real-time data streaming stopped successfully")

        except Exception as e:
            logger.error(
                "Error stopping real-time data streaming",
                extra={"error": str(e)},
                exc_info=True
            )

    async def subscribe_symbol(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None
    ) -> None:
        """Subscribe to symbol for real-time data.

        Args:
            symbol: Symbol to subscribe to
            timeframes: Optional timeframes for OHLC data
        """
        with self.subscription_lock:
            self.subscribed_symbols.add(symbol)

            if timeframes:
                self.subscribed_timeframes[symbol].update(timeframes)
            else:
                # Use default timeframes from settings
                default_timeframes = [tf.value for tf in self.settings.mt5.default_timeframes]
                self.subscribed_timeframes[symbol].update(default_timeframes)

        logger.info(
            "Subscribed to symbol",
            extra={
                "symbol": symbol,
                "timeframes": list(self.subscribed_timeframes.get(symbol, [])),
                "total_subscriptions": len(self.subscribed_symbols),
            }
        )

    async def unsubscribe_symbol(self, symbol: str) -> None:
        """Unsubscribe from symbol.

        Args:
            symbol: Symbol to unsubscribe from
        """
        with self.subscription_lock:
            self.subscribed_symbols.discard(symbol)
            self.subscribed_timeframes.pop(symbol, None)

        # Clean up last fetch timestamps
        self.last_tick_fetch.pop(symbol, None)
        self.last_ohlc_fetch.pop(symbol, None)

        logger.info(
            "Unsubscribed from symbol",
            extra={
                "symbol": symbol,
                "remaining_subscriptions": len(self.subscribed_symbols),
            }
        )

    def subscribe_to_ticks(self, callback: Callable[[TickData], None]) -> str:
        """Subscribe to tick data events.

        Args:
            callback: Callback function for tick data

        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())
        callback._subscription_id = subscription_id
        self.tick_subscribers.append(callback)

        logger.debug(
            "Added tick subscriber",
            extra={
                "subscription_id": subscription_id,
                "total_subscribers": len(self.tick_subscribers),
            }
        )

        return subscription_id

    def subscribe_to_ohlc(self, callback: Callable[[OHLCData], None]) -> str:
        """Subscribe to OHLC data events.

        Args:
            callback: Callback function for OHLC data

        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())
        callback._subscription_id = subscription_id
        self.ohlc_subscribers.append(callback)

        logger.debug(
            "Added OHLC subscriber",
            extra={
                "subscription_id": subscription_id,
                "total_subscribers": len(self.ohlc_subscribers),
            }
        )

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if subscription was found and removed
        """
        removed = False

        # Remove from tick subscribers
        self.tick_subscribers = [
            cb for cb in self.tick_subscribers
            if getattr(cb, '_subscription_id', None) != subscription_id
        ]

        # Remove from OHLC subscribers
        original_ohlc_count = len(self.ohlc_subscribers)
        self.ohlc_subscribers = [
            cb for cb in self.ohlc_subscribers
            if getattr(cb, '_subscription_id', None) != subscription_id
        ]

        removed = len(self.ohlc_subscribers) < original_ohlc_count

        if removed:
            logger.debug(
                "Removed subscription",
                extra={"subscription_id": subscription_id}
            )

        return removed

    async def _tick_streaming_worker(self) -> None:
        """Worker for streaming tick data."""
        logger.debug("Started tick streaming worker")

        while not self.is_shutting_down:
            try:
                # Wait for next fetch interval
                await asyncio.sleep(self.config.tick_fetch_interval_ms / 1000.0)

                if not self.subscribed_symbols:
                    continue

                # Fetch ticks for all subscribed symbols
                for symbol in list(self.subscribed_symbols):
                    try:
                        await self._fetch_symbol_ticks(symbol)
                    except Exception as e:
                        logger.warning(
                            "Error fetching ticks for symbol",
                            extra={
                                "symbol": symbol,
                                "error": str(e),
                            }
                        )
                        self.metrics.streaming_errors += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in tick streaming worker",
                    extra={"error": str(e)},
                    exc_info=True
                )
                self.metrics.streaming_errors += 1
                await asyncio.sleep(1.0)  # Brief pause on error

    async def _ohlc_streaming_worker(self) -> None:
        """Worker for streaming OHLC data."""
        logger.debug("Started OHLC streaming worker")

        while not self.is_shutting_down:
            try:
                # Wait for next fetch interval
                await asyncio.sleep(self.config.ohlc_fetch_interval_ms / 1000.0)

                if not self.subscribed_symbols:
                    continue

                # Fetch OHLC for all subscribed symbols and timeframes
                for symbol in list(self.subscribed_symbols):
                    timeframes = list(self.subscribed_timeframes.get(symbol, []))
                    for timeframe in timeframes:
                        try:
                            await self._fetch_symbol_ohlc(symbol, timeframe)
                        except Exception as e:
                            logger.warning(
                                "Error fetching OHLC for symbol",
                                extra={
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "error": str(e),
                                }
                            )
                            self.metrics.streaming_errors += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in OHLC streaming worker",
                    extra={"error": str(e)},
                    exc_info=True
                )
                self.metrics.streaming_errors += 1
                await asyncio.sleep(1.0)

    async def _fetch_symbol_ticks(self, symbol: str) -> None:
        """Fetch ticks for a specific symbol."""
        try:
            session = await self.mt5_manager.get_session()

            # Import MT5 here to avoid import issues
            import MetaTrader5 as mt5

            # Get last tick fetch time for this symbol
            from_date = self.last_tick_fetch.get(symbol)
            if from_date is None:
                # First fetch - get recent ticks
                from_date = datetime.now(timezone.utc).replace(microsecond=0) - \
                           timedelta(seconds=60)  # Last minute

            # Fetch ticks from MT5
            ticks = mt5.copy_ticks_from(
                symbol,
                from_date,
                0,  # count (0 = all available)
                mt5.COPY_TICKS_ALL
            )

            if ticks is not None and len(ticks) > 0:
                fetch_time = datetime.now(timezone.utc)

                for tick_info in ticks:
                    # Create TickData object
                    tick = self._create_tick_data(tick_info, symbol, fetch_time)

                    # Check for duplicates if enabled
                    if self.config.enable_duplicate_detection:
                        if self._is_duplicate_tick(tick):
                            continue

                    # Add to buffer
                    await self._buffer_tick_data(tick)

                self.last_tick_fetch[symbol] = fetch_time
                self.metrics.total_ticks_streamed += len(ticks)

                logger.debug(
                    "Fetched ticks for symbol",
                    extra={
                        "symbol": symbol,
                        "tick_count": len(ticks),
                        "from_date": from_date.isoformat(),
                    }
                )

            await self.mt5_manager.return_session(session)

        except Exception as e:
            logger.error(
                "Error fetching ticks",
                extra={
                    "symbol": symbol,
                    "error": str(e),
                },
                exc_info=True
            )
            self.metrics.connection_errors += 1
            raise

    async def _fetch_symbol_ohlc(self, symbol: str, timeframe: str) -> None:
        """Fetch OHLC data for a specific symbol and timeframe."""
        try:
            session = await self.mt5_manager.get_session()

            import MetaTrader5 as mt5

            # Get last OHLC fetch time
            from_date = self.last_ohlc_fetch[symbol].get(timeframe)
            if from_date is None:
                # First fetch - get recent bars
                from_date = datetime.now(timezone.utc).replace(microsecond=0) - \
                           timedelta(hours=1)  # Last hour

            # Convert timeframe string to MT5 constant
            mt5_timeframe = self._get_mt5_timeframe(timeframe)
            if mt5_timeframe is None:
                logger.warning(f"Unsupported timeframe: {timeframe}")
                return

            # Fetch OHLC from MT5
            rates = mt5.copy_rates_from(
                symbol,
                mt5_timeframe,
                from_date,
                100  # max bars
            )

            if rates is not None and len(rates) > 0:
                fetch_time = datetime.now(timezone.utc)

                for rate_info in rates:
                    # Create OHLCData object
                    ohlc = self._create_ohlc_data(rate_info, symbol, timeframe, fetch_time)

                    # Check for duplicates
                    if self.config.enable_duplicate_detection:
                        if self._is_duplicate_ohlc(ohlc):
                            continue

                    # Add to buffer
                    await self._buffer_ohlc_data(ohlc)

                self.last_ohlc_fetch[symbol][timeframe] = fetch_time
                self.metrics.total_ohlc_streamed += len(rates)

                logger.debug(
                    "Fetched OHLC for symbol",
                    extra={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "bar_count": len(rates),
                        "from_date": from_date.isoformat(),
                    }
                )

            await self.mt5_manager.return_session(session)

        except Exception as e:
            logger.error(
                "Error fetching OHLC",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "error": str(e),
                },
                exc_info=True
            )
            self.metrics.connection_errors += 1
            raise

    def _create_tick_data(
        self,
        tick_info: Any,
        symbol: str,
        fetch_time: datetime
    ) -> TickData:
        """Create TickData object from MT5 tick info."""
        # Calculate latency
        tick_time = datetime.fromtimestamp(tick_info.time, tz=timezone.utc)
        latency_ms = (fetch_time - tick_time).total_seconds() * 1000

        # Create and return TickData
        from src.core.data.models import TickData, DataQuality

        return TickData(
            symbol=symbol,
            timestamp=tick_time,
            bid=float(tick_info.bid),
            ask=float(tick_info.ask),
            volume=int(tick_info.volume_real) if hasattr(tick_info, 'volume_real') else 0,
            flags=int(tick_info.flags) if hasattr(tick_info, 'flags') else 0,
            quality=DataQuality.HIGH,
            latency_ms=latency_ms,
        )

    def _create_ohlc_data(
        self,
        rate_info: Any,
        symbol: str,
        timeframe: str,
        fetch_time: datetime
    ) -> OHLCData:
        """Create OHLCData object from MT5 rate info."""
        # Calculate latency
        bar_time = datetime.fromtimestamp(rate_info.time, tz=timezone.utc)
        latency_ms = (fetch_time - bar_time).total_seconds() * 1000

        from src.core.data.models import OHLCData, DataQuality

        return OHLCData(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=bar_time,
            open=float(rate_info.open),
            high=float(rate_info.high),
            low=float(rate_info.low),
            close=float(rate_info.close),
            volume=int(rate_info.tick_volume) if hasattr(rate_info, 'tick_volume') else 0,
            tick_count=int(rate_info.tick_volume) if hasattr(rate_info, 'tick_volume') else 0,
            quality=DataQuality.HIGH,
            latency_ms=latency_ms,
        )

    def _get_mt5_timeframe(self, timeframe: str) -> Optional[int]:
        """Convert timeframe string to MT5 constant."""
        import MetaTrader5 as mt5

        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M2': mt5.TIMEFRAME_M2,
            'M3': mt5.TIMEFRAME_M3,
            'M4': mt5.TIMEFRAME_M4,
            'M5': mt5.TIMEFRAME_M5,
            'M6': mt5.TIMEFRAME_M6,
            'M10': mt5.TIMEFRAME_M10,
            'M12': mt5.TIMEFRAME_M12,
            'M15': mt5.TIMEFRAME_M15,
            'M20': mt5.TIMEFRAME_M20,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H2': mt5.TIMEFRAME_H2,
            'H3': mt5.TIMEFRAME_H3,
            'H4': mt5.TIMEFRAME_H4,
            'H6': mt5.TIMEFRAME_H6,
            'H8': mt5.TIMEFRAME_H8,
            'H12': mt5.TIMEFRAME_H12,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1,
        }

        return timeframe_map.get(timeframe)

    def _is_duplicate_tick(self, tick: TickData) -> bool:
        """Check if tick is a duplicate."""
        tick_key = (tick.symbol, tick.timestamp, tick.bid, tick.ask)

        if tick_key in self.tick_cache:
            return True

        self.tick_cache.append(tick_key)
        return False

    def _is_duplicate_ohlc(self, ohlc: OHLCData) -> bool:
        """Check if OHLC is a duplicate."""
        ohlc_key = (ohlc.symbol, ohlc.timeframe, ohlc.timestamp)

        if ohlc_key in self.ohlc_cache:
            return True

        self.ohlc_cache.append(ohlc_key)
        return False

    async def _buffer_tick_data(self, tick: TickData) -> None:
        """Add tick data to buffer."""
        try:
            with self.tick_buffer_lock:
                if len(self.tick_buffer) >= self.config.tick_buffer_size:
                    self.metrics.buffer_overflows += 1
                    logger.warning("Tick buffer overflow, dropping oldest data")

                self.tick_buffer.append(tick)

            # Update buffer usage metrics
            usage = len(self.tick_buffer) / self.config.tick_buffer_size
            self.metrics.tick_buffer_usage = usage
            self.metrics.max_tick_buffer_usage = max(
                self.metrics.max_tick_buffer_usage, usage
            )

        except Exception as e:
            logger.error(f"Error buffering tick data: {e}")
            raise BufferOverflowError(f"Failed to buffer tick data: {e}") from e

    async def _buffer_ohlc_data(self, ohlc: OHLCData) -> None:
        """Add OHLC data to buffer."""
        try:
            with self.ohlc_buffer_lock:
                if len(self.ohlc_buffer) >= self.config.ohlc_buffer_size:
                    self.metrics.buffer_overflows += 1
                    logger.warning("OHLC buffer overflow, dropping oldest data")

                self.ohlc_buffer.append(ohlc)

            # Update buffer usage metrics
            usage = len(self.ohlc_buffer) / self.config.ohlc_buffer_size
            self.metrics.ohlc_buffer_usage = usage
            self.metrics.max_ohlc_buffer_usage = max(
                self.metrics.max_ohlc_buffer_usage, usage
            )

        except Exception as e:
            logger.error(f"Error buffering OHLC data: {e}")
            raise BufferOverflowError(f"Failed to buffer OHLC data: {e}") from e

    async def _data_processing_worker(self) -> None:
        """Worker for processing buffered data."""
        while not self.is_shutting_down:
            try:
                # Process tick data
                tick_batch = []
                with self.tick_buffer_lock:
                    while self.tick_buffer and len(tick_batch) < 10:
                        tick_batch.append(self.tick_buffer.popleft())

                if tick_batch:
                    await self._process_tick_batch(tick_batch)

                # Process OHLC data
                ohlc_batch = []
                with self.ohlc_buffer_lock:
                    while self.ohlc_buffer and len(ohlc_batch) < 5:
                        ohlc_batch.append(self.ohlc_buffer.popleft())

                if ohlc_batch:
                    await self._process_ohlc_batch(ohlc_batch)

                # Brief pause if no data to process
                if not tick_batch and not ohlc_batch:
                    await asyncio.sleep(0.01)  # 10ms

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in data processing worker",
                    extra={"error": str(e)},
                    exc_info=True
                )
                await asyncio.sleep(0.1)

    async def _process_tick_batch(self, tick_batch: List[TickData]) -> None:
        """Process batch of tick data."""
        start_time = time.perf_counter()

        try:
            for tick in tick_batch:
                # Notify subscribers
                for callback in self.tick_subscribers:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(tick)
                        else:
                            callback(tick)
                    except Exception as e:
                        logger.warning(
                            "Error in tick subscriber callback",
                            extra={"error": str(e)}
                        )

            # Update metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_samples.append(processing_time)

            self.metrics.avg_processing_latency_ms = (
                sum(self.processing_samples) / len(self.processing_samples)
            )
            self.metrics.max_processing_latency_ms = max(
                self.metrics.max_processing_latency_ms,
                processing_time
            )

        except Exception as e:
            logger.error(f"Error processing tick batch: {e}")
            self.metrics.streaming_errors += 1

    async def _process_ohlc_batch(self, ohlc_batch: List[OHLCData]) -> None:
        """Process batch of OHLC data."""
        start_time = time.perf_counter()

        try:
            for ohlc in ohlc_batch:
                # Notify subscribers
                for callback in self.ohlc_subscribers:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(ohlc)
                        else:
                            callback(ohlc)
                    except Exception as e:
                        logger.warning(
                            "Error in OHLC subscriber callback",
                            extra={"error": str(e)}
                        )

            # Update metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_samples.append(processing_time)

        except Exception as e:
            logger.error(f"Error processing OHLC batch: {e}")
            self.metrics.streaming_errors += 1

    async def _event_processing_worker(self) -> None:
        """Worker for processing events."""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(0.1)  # Basic event processing placeholder
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing worker: {e}")
                await asyncio.sleep(1.0)

    async def _metrics_collector(self) -> None:
        """Worker for collecting performance metrics."""
        last_tick_count = 0
        last_ohlc_count = 0

        while not self.is_shutting_down:
            try:
                await asyncio.sleep(self.config.metrics_collection_interval_ms / 1000.0)

                # Calculate throughput
                tick_diff = self.metrics.total_ticks_streamed - last_tick_count
                ohlc_diff = self.metrics.total_ohlc_streamed - last_ohlc_count
                interval_seconds = self.config.metrics_collection_interval_ms / 1000.0

                self.metrics.ticks_per_second = tick_diff / interval_seconds
                self.metrics.ohlc_per_second = ohlc_diff / interval_seconds

                last_tick_count = self.metrics.total_ticks_streamed
                last_ohlc_count = self.metrics.total_ohlc_streamed

                # Update session duration
                if self.metrics.session_start_time:
                    self.metrics.session_duration_seconds = (
                        datetime.now(timezone.utc) - self.metrics.session_start_time
                    ).total_seconds()

                logger.debug(
                    "Streaming metrics updated",
                    extra={
                        "ticks_per_second": self.metrics.ticks_per_second,
                        "ohlc_per_second": self.metrics.ohlc_per_second,
                        "tick_buffer_usage": self.metrics.tick_buffer_usage,
                        "ohlc_buffer_usage": self.metrics.ohlc_buffer_usage,
                    }
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(1.0)

    async def _health_monitor(self) -> None:
        """Worker for monitoring streaming health."""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(self.config.health_check_interval_ms / 1000.0)

                # Check for performance issues
                if self.metrics.avg_processing_latency_ms > self.config.max_processing_latency_ms:
                    logger.warning(
                        "High processing latency detected",
                        extra={
                            "current_latency_ms": self.metrics.avg_processing_latency_ms,
                            "threshold_ms": self.config.max_processing_latency_ms,
                        }
                    )

                # Check buffer usage
                if (self.metrics.tick_buffer_usage > 0.8 or
                    self.metrics.ohlc_buffer_usage > 0.8):
                    logger.warning(
                        "High buffer usage detected",
                        extra={
                            "tick_buffer_usage": self.metrics.tick_buffer_usage,
                            "ohlc_buffer_usage": self.metrics.ohlc_buffer_usage,
                        }
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(1.0)

    async def _cleanup_workers(self) -> None:
        """Cleanup all worker tasks."""
        if not self.worker_tasks:
            return

        logger.debug("Cleaning up streaming worker tasks")

        # Cancel all tasks
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()

        # Wait for cancellation
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        self.worker_tasks.clear()
        logger.debug("Streaming worker tasks cleaned up")

    def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming status."""
        with self._state_lock:
            return {
                "state": self.state.value,
                "is_running": self.state == StreamingState.RUNNING,
                "subscribed_symbols": list(self.subscribed_symbols),
                "subscribed_timeframes": {
                    symbol: list(timeframes)
                    for symbol, timeframes in self.subscribed_timeframes.items()
                },
                "worker_tasks": len(self.worker_tasks),
                "tick_subscribers": len(self.tick_subscribers),
                "ohlc_subscribers": len(self.ohlc_subscribers),
                "metrics": self.get_streaming_metrics(),
            }

    def get_streaming_metrics(self) -> Dict[str, Any]:
        """Get streaming performance metrics."""
        return {
            "throughput": {
                "ticks_per_second": self.metrics.ticks_per_second,
                "ohlc_per_second": self.metrics.ohlc_per_second,
                "total_ticks_streamed": self.metrics.total_ticks_streamed,
                "total_ohlc_streamed": self.metrics.total_ohlc_streamed,
            },
            "latency": {
                "avg_processing_latency_ms": self.metrics.avg_processing_latency_ms,
                "max_processing_latency_ms": self.metrics.max_processing_latency_ms,
            },
            "buffers": {
                "tick_buffer_usage": self.metrics.tick_buffer_usage,
                "ohlc_buffer_usage": self.metrics.ohlc_buffer_usage,
                "max_tick_buffer_usage": self.metrics.max_tick_buffer_usage,
                "max_ohlc_buffer_usage": self.metrics.max_ohlc_buffer_usage,
            },
            "errors": {
                "streaming_errors": self.metrics.streaming_errors,
                "buffer_overflows": self.metrics.buffer_overflows,
                "connection_errors": self.metrics.connection_errors,
            },
            "session": {
                "session_start_time": (
                    self.metrics.session_start_time.isoformat()
                    if self.metrics.session_start_time else None
                ),
                "session_duration_seconds": self.metrics.session_duration_seconds,
                "uptime_percentage": self.metrics.uptime_percentage,
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on streaming service."""
        with self._state_lock:
            is_healthy = (
                self.state == StreamingState.RUNNING and
                self.metrics.avg_processing_latency_ms < self.config.max_processing_latency_ms and
                self.metrics.tick_buffer_usage < 0.9 and
                self.metrics.ohlc_buffer_usage < 0.9
            )

            return {
                "is_healthy": is_healthy,
                "state": self.state.value,
                "issues": self._get_health_issues(),
                "recommendations": self._get_health_recommendations(),
                "metrics_summary": {
                    "ticks_per_second": self.metrics.ticks_per_second,
                    "avg_latency_ms": self.metrics.avg_processing_latency_ms,
                    "buffer_usage": max(
                        self.metrics.tick_buffer_usage,
                        self.metrics.ohlc_buffer_usage
                    ),
                    "error_count": (
                        self.metrics.streaming_errors +
                        self.metrics.buffer_overflows +
                        self.metrics.connection_errors
                    ),
                },
            }

    def _get_health_issues(self) -> List[str]:
        """Get current health issues."""
        issues = []

        if self.state != StreamingState.RUNNING:
            issues.append(f"Streaming not running (state: {self.state.value})")

        if self.metrics.avg_processing_latency_ms > self.config.max_processing_latency_ms:
            issues.append(
                f"High processing latency: {self.metrics.avg_processing_latency_ms:.1f}ms"
            )

        if self.metrics.tick_buffer_usage > 0.9:
            issues.append(f"High tick buffer usage: {self.metrics.tick_buffer_usage:.1%}")

        if self.metrics.ohlc_buffer_usage > 0.9:
            issues.append(f"High OHLC buffer usage: {self.metrics.ohlc_buffer_usage:.1%}")

        if self.metrics.streaming_errors > 0:
            issues.append(f"Streaming errors detected: {self.metrics.streaming_errors}")

        return issues

    def _get_health_recommendations(self) -> List[str]:
        """Get health recommendations."""
        recommendations = []

        if self.metrics.avg_processing_latency_ms > self.config.max_processing_latency_ms:
            recommendations.append("Consider increasing processing worker threads")

        if max(self.metrics.tick_buffer_usage, self.metrics.ohlc_buffer_usage) > 0.8:
            recommendations.append("Consider increasing buffer sizes")

        if self.metrics.connection_errors > 0:
            recommendations.append("Check MT5 connection stability")

        if len(self.subscribed_symbols) > 10:
            recommendations.append("Consider reducing symbol subscriptions for better performance")

        return recommendations

    async def __aenter__(self) -> RealTimeDataStreamer:
        """Async context manager entry."""
        await self.start_streaming()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop_streaming()


# Global streamer instance
_realtime_streamer: Optional[RealTimeDataStreamer] = None


def get_realtime_streamer() -> Optional[RealTimeDataStreamer]:
    """Get the global real-time streamer instance."""
    return _realtime_streamer


def initialize_realtime_streamer(
    settings: Settings,
    mt5_manager: Mt5ConnectionManager,
    config: Optional[StreamingConfig] = None
) -> RealTimeDataStreamer:
    """Initialize the global real-time streamer."""
    global _realtime_streamer

    if _realtime_streamer is not None:
        raise RuntimeError("Real-time streamer already initialized")

    _realtime_streamer = RealTimeDataStreamer(settings, mt5_manager, config)
    return _realtime_streamer