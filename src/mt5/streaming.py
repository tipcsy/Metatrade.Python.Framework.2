"""
MT5 Real-time Data Streaming Service.

This module provides high-performance real-time data streaming from MetaTrader 5,
including tick data, price updates, and market events with advanced buffering
and distribution mechanisms.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

import MetaTrader5 as mt5
import numpy as np

from src.core.config.settings import Mt5Settings, MarketDataSettings
from src.core.exceptions import Mt5StreamingError, Mt5ConnectionError
from src.core.logging import get_logger
from src.core.data.models import TickData, MarketEvent, StreamEvent
from .connection.manager import Mt5ConnectionManager

logger = get_logger(__name__)


class StreamType(Enum):
    """Types of data streams."""
    TICK = "tick"
    PRICE = "price"
    ORDER = "order"
    TRADE = "trade"
    MARKET_DEPTH = "market_depth"
    ACCOUNT = "account"


class StreamStatus(Enum):
    """Stream status states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class StreamSubscription:
    """Data stream subscription configuration."""
    stream_id: str
    stream_type: StreamType
    symbols: Set[str] = field(default_factory=set)
    callback: Optional[Callable[[StreamEvent], None]] = None
    buffer_size: int = 1000
    batch_size: int = 1
    last_update: float = field(default_factory=time.time)
    active: bool = True


@dataclass
class StreamMetrics:
    """Stream performance metrics."""
    messages_received: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0
    bytes_received: int = 0
    last_message_time: float = 0.0
    processing_time_avg: float = 0.0
    queue_size: int = 0
    error_count: int = 0


class Mt5StreamingService:
    """
    High-performance MT5 real-time data streaming service.

    Features:
    - Multiple simultaneous data streams
    - Intelligent buffering and batching
    - Configurable rate limiting
    - Stream health monitoring
    - Automatic reconnection
    - Multi-threading support
    """

    def __init__(
        self,
        connection_manager: Mt5ConnectionManager,
        market_data_settings: MarketDataSettings,
        max_buffer_size: int = 10000,
        worker_threads: int = 4
    ) -> None:
        """Initialize MT5 streaming service.

        Args:
            connection_manager: MT5 connection manager
            market_data_settings: Market data configuration
            max_buffer_size: Maximum buffer size per stream
            worker_threads: Number of worker threads
        """
        self.connection_manager = connection_manager
        self.settings = market_data_settings
        self.max_buffer_size = max_buffer_size

        # Stream management
        self._subscriptions: Dict[str, StreamSubscription] = {}
        self._stream_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_buffer_size))
        self._stream_metrics: Dict[str, StreamMetrics] = defaultdict(StreamMetrics)
        self._stream_status = StreamStatus.STOPPED

        # Threading
        self._thread_pool = ThreadPoolExecutor(max_workers=worker_threads, thread_name_prefix="MT5-Stream")
        self._streaming_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Rate limiting
        self._rate_limiters: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Event distribution
        self._event_handlers: Dict[StreamType, List[Callable]] = defaultdict(list)

        # Monitoring
        self._last_health_check = 0.0
        self._health_check_interval = 30.0

        # Locks
        self._subscription_lock = threading.RLock()
        self._metrics_lock = threading.RLock()

        logger.info(
            "MT5 streaming service initialized",
            extra={
                "max_buffer_size": max_buffer_size,
                "worker_threads": worker_threads,
                "tick_rate_limit": self.settings.tick_rate_limit,
            }
        )

    async def start_streaming(self) -> None:
        """Start the streaming service."""
        if self._stream_status == StreamStatus.RUNNING:
            logger.warning("Streaming service already running")
            return

        try:
            self._stream_status = StreamStatus.STARTING
            logger.info("Starting MT5 streaming service")

            # Ensure connection manager is initialized
            if not self.connection_manager.is_initialized:
                await self.connection_manager.initialize()

            # Reset stop event
            self._stop_event.clear()

            # Start streaming thread
            self._streaming_thread = threading.Thread(
                target=self._streaming_loop,
                name="MT5-StreamingLoop",
                daemon=True
            )
            self._streaming_thread.start()

            self._stream_status = StreamStatus.RUNNING
            logger.info("MT5 streaming service started successfully")

        except Exception as e:
            self._stream_status = StreamStatus.ERROR
            logger.error(
                "Failed to start streaming service",
                extra={"error": str(e)},
                exc_info=True
            )
            raise Mt5StreamingError(f"Failed to start streaming: {e}") from e

    async def stop_streaming(self) -> None:
        """Stop the streaming service."""
        if self._stream_status == StreamStatus.STOPPED:
            return

        logger.info("Stopping MT5 streaming service")

        try:
            # Signal stop
            self._stop_event.set()

            # Wait for streaming thread to finish
            if self._streaming_thread and self._streaming_thread.is_alive():
                self._streaming_thread.join(timeout=10.0)
                if self._streaming_thread.is_alive():
                    logger.warning("Streaming thread did not stop within timeout")

            # Clear subscriptions and buffers
            with self._subscription_lock:
                self._subscriptions.clear()
                self._stream_buffers.clear()

            self._stream_status = StreamStatus.STOPPED
            logger.info("MT5 streaming service stopped")

        except Exception as e:
            logger.error(
                "Error stopping streaming service",
                extra={"error": str(e)},
                exc_info=True
            )

    def subscribe_ticks(
        self,
        symbols: List[str],
        callback: Optional[Callable[[List[TickData]], None]] = None,
        buffer_size: int = 1000,
        batch_size: int = 1
    ) -> str:
        """Subscribe to tick data stream.

        Args:
            symbols: List of symbols to subscribe to
            callback: Optional callback function for tick data
            buffer_size: Buffer size for the subscription
            batch_size: Number of ticks to batch together

        Returns:
            Subscription ID
        """
        subscription_id = f"tick_{int(time.time() * 1000000)}"

        subscription = StreamSubscription(
            stream_id=subscription_id,
            stream_type=StreamType.TICK,
            symbols=set(symbols),
            callback=callback,
            buffer_size=buffer_size,
            batch_size=batch_size
        )

        with self._subscription_lock:
            self._subscriptions[subscription_id] = subscription
            self._stream_buffers[subscription_id] = deque(maxlen=buffer_size)

        logger.info(
            "Tick subscription created",
            extra={
                "subscription_id": subscription_id,
                "symbols": symbols,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
            }
        )

        return subscription_id

    def subscribe_prices(
        self,
        symbols: List[str],
        callback: Optional[Callable[[Dict[str, Dict]], None]] = None
    ) -> str:
        """Subscribe to price updates.

        Args:
            symbols: List of symbols to subscribe to
            callback: Optional callback function for price updates

        Returns:
            Subscription ID
        """
        subscription_id = f"price_{int(time.time() * 1000000)}"

        subscription = StreamSubscription(
            stream_id=subscription_id,
            stream_type=StreamType.PRICE,
            symbols=set(symbols),
            callback=callback
        )

        with self._subscription_lock:
            self._subscriptions[subscription_id] = subscription

        logger.info(
            "Price subscription created",
            extra={
                "subscription_id": subscription_id,
                "symbols": symbols,
            }
        )

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a data stream.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if subscription was removed
        """
        with self._subscription_lock:
            subscription = self._subscriptions.pop(subscription_id, None)
            if subscription:
                self._stream_buffers.pop(subscription_id, None)
                self._stream_metrics.pop(subscription_id, None)

                logger.info(
                    "Subscription removed",
                    extra={
                        "subscription_id": subscription_id,
                        "stream_type": subscription.stream_type.value,
                    }
                )
                return True

        return False

    def get_buffered_data(self, subscription_id: str, max_items: Optional[int] = None) -> List[Any]:
        """Get buffered data for a subscription.

        Args:
            subscription_id: Subscription ID
            max_items: Maximum number of items to return

        Returns:
            List of buffered data items
        """
        buffer = self._stream_buffers.get(subscription_id)
        if buffer is None:
            return []

        if max_items is None:
            return list(buffer)
        else:
            return list(buffer)[-max_items:]

    def get_stream_metrics(self, subscription_id: Optional[str] = None) -> Union[StreamMetrics, Dict[str, StreamMetrics]]:
        """Get streaming metrics.

        Args:
            subscription_id: Specific subscription ID, or None for all

        Returns:
            Metrics for subscription(s)
        """
        with self._metrics_lock:
            if subscription_id:
                return self._stream_metrics.get(subscription_id, StreamMetrics())
            else:
                return dict(self._stream_metrics)

    def get_service_status(self) -> Dict[str, Any]:
        """Get streaming service status.

        Returns:
            Dictionary containing service status
        """
        with self._subscription_lock, self._metrics_lock:
            total_messages = sum(m.messages_received for m in self._stream_metrics.values())
            total_processed = sum(m.messages_processed for m in self._stream_metrics.values())
            total_dropped = sum(m.messages_dropped for m in self._stream_metrics.values())

            return {
                "status": self._stream_status.value,
                "subscriptions_count": len(self._subscriptions),
                "total_messages_received": total_messages,
                "total_messages_processed": total_processed,
                "total_messages_dropped": total_dropped,
                "processing_rate": total_processed / max(time.time() - self._last_health_check, 1) if self._last_health_check > 0 else 0,
                "active_subscriptions": [s.stream_id for s in self._subscriptions.values() if s.active],
                "buffer_usage": {
                    sid: len(buffer) for sid, buffer in self._stream_buffers.items()
                }
            }

    def add_event_handler(self, stream_type: StreamType, handler: Callable[[StreamEvent], None]) -> None:
        """Add an event handler for a stream type.

        Args:
            stream_type: Type of stream to handle
            handler: Handler function
        """
        self._event_handlers[stream_type].append(handler)
        logger.debug(
            "Event handler added",
            extra={
                "stream_type": stream_type.value,
                "handler_count": len(self._event_handlers[stream_type]),
            }
        )

    def remove_event_handler(self, stream_type: StreamType, handler: Callable[[StreamEvent], None]) -> bool:
        """Remove an event handler.

        Args:
            stream_type: Type of stream
            handler: Handler function to remove

        Returns:
            True if handler was removed
        """
        try:
            self._event_handlers[stream_type].remove(handler)
            logger.debug(
                "Event handler removed",
                extra={"stream_type": stream_type.value}
            )
            return True
        except ValueError:
            return False

    def _streaming_loop(self) -> None:
        """Main streaming loop running in separate thread."""
        logger.info("Streaming loop started")

        try:
            while not self._stop_event.is_set():
                try:
                    # Process tick subscriptions
                    self._process_tick_subscriptions()

                    # Process price subscriptions
                    self._process_price_subscriptions()

                    # Health check
                    if time.time() - self._last_health_check > self._health_check_interval:
                        self._perform_health_check()

                    # Brief sleep to prevent CPU overload
                    time.sleep(0.001)  # 1ms

                except Exception as e:
                    logger.error(
                        "Error in streaming loop",
                        extra={"error": str(e)},
                        exc_info=True
                    )
                    time.sleep(1.0)  # Back off on error

        except Exception as e:
            logger.error(
                "Fatal error in streaming loop",
                extra={"error": str(e)},
                exc_info=True
            )
            self._stream_status = StreamStatus.ERROR

        logger.info("Streaming loop ended")

    def _process_tick_subscriptions(self) -> None:
        """Process tick data subscriptions."""
        tick_subscriptions = [
            s for s in self._subscriptions.values()
            if s.stream_type == StreamType.TICK and s.active
        ]

        if not tick_subscriptions:
            return

        # Collect all unique symbols
        all_symbols = set()
        for subscription in tick_subscriptions:
            all_symbols.update(subscription.symbols)

        if not all_symbols:
            return

        try:
            # Get tick data from MT5
            current_time = time.time()

            for symbol in all_symbols:
                # Rate limiting check
                last_request = self._rate_limiters["tick"].get(symbol, 0)
                if current_time - last_request < (1.0 / self.settings.tick_rate_limit):
                    continue

                # Get latest ticks
                ticks = mt5.copy_ticks_from(symbol, current_time - 1, 100, mt5.COPY_TICKS_ALL)
                if ticks is None or len(ticks) == 0:
                    continue

                self._rate_limiters["tick"][symbol] = current_time

                # Process ticks for each subscription
                for subscription in tick_subscriptions:
                    if symbol in subscription.symbols:
                        self._process_ticks_for_subscription(subscription, symbol, ticks)

        except Exception as e:
            logger.error(
                "Error processing tick subscriptions",
                extra={"error": str(e)},
                exc_info=True
            )

    def _process_price_subscriptions(self) -> None:
        """Process price update subscriptions."""
        price_subscriptions = [
            s for s in self._subscriptions.values()
            if s.stream_type == StreamType.PRICE and s.active
        ]

        if not price_subscriptions:
            return

        # Collect all unique symbols
        all_symbols = set()
        for subscription in price_subscriptions:
            all_symbols.update(subscription.symbols)

        if not all_symbols:
            return

        try:
            # Get price data for all symbols
            price_data = {}
            for symbol in all_symbols:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    price_data[symbol] = {
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'last': tick.last,
                        'volume': tick.volume,
                        'time': tick.time,
                        'spread': tick.ask - tick.bid if tick.ask and tick.bid else 0
                    }

            # Distribute to subscriptions
            for subscription in price_subscriptions:
                relevant_prices = {
                    symbol: data for symbol, data in price_data.items()
                    if symbol in subscription.symbols
                }

                if relevant_prices and subscription.callback:
                    try:
                        subscription.callback(relevant_prices)
                        self._update_metrics(subscription.stream_id, len(relevant_prices))
                    except Exception as e:
                        logger.error(
                            "Error in price callback",
                            extra={
                                "subscription_id": subscription.stream_id,
                                "error": str(e),
                            },
                            exc_info=True
                        )

        except Exception as e:
            logger.error(
                "Error processing price subscriptions",
                extra={"error": str(e)},
                exc_info=True
            )

    def _process_ticks_for_subscription(self, subscription: StreamSubscription, symbol: str, ticks: Any) -> None:
        """Process ticks for a specific subscription."""
        try:
            # Convert ticks to TickData objects
            tick_data_list = []
            for tick in ticks:
                tick_data = TickData(
                    symbol=symbol,
                    time=tick.time,
                    bid=tick.bid,
                    ask=tick.ask,
                    last=tick.last,
                    volume=tick.volume,
                    flags=tick.flags
                )
                tick_data_list.append(tick_data)

            # Add to buffer
            buffer = self._stream_buffers[subscription.stream_id]
            buffer.extend(tick_data_list)

            # Call callback if configured and batch size reached
            if subscription.callback and len(tick_data_list) >= subscription.batch_size:
                try:
                    subscription.callback(tick_data_list)
                except Exception as e:
                    logger.error(
                        "Error in tick callback",
                        extra={
                            "subscription_id": subscription.stream_id,
                            "symbol": symbol,
                            "error": str(e),
                        },
                        exc_info=True
                    )

            # Update metrics
            self._update_metrics(subscription.stream_id, len(tick_data_list))
            subscription.last_update = time.time()

        except Exception as e:
            logger.error(
                "Error processing ticks for subscription",
                extra={
                    "subscription_id": subscription.stream_id,
                    "symbol": symbol,
                    "error": str(e),
                },
                exc_info=True
            )

    def _update_metrics(self, subscription_id: str, message_count: int) -> None:
        """Update metrics for a subscription."""
        with self._metrics_lock:
            metrics = self._stream_metrics[subscription_id]
            metrics.messages_received += message_count
            metrics.messages_processed += message_count
            metrics.last_message_time = time.time()

            # Update queue size
            buffer = self._stream_buffers.get(subscription_id)
            if buffer:
                metrics.queue_size = len(buffer)

    def _perform_health_check(self) -> None:
        """Perform periodic health check."""
        try:
            current_time = time.time()

            with self._subscription_lock:
                # Check for inactive subscriptions
                inactive_count = 0
                for subscription in self._subscriptions.values():
                    if current_time - subscription.last_update > 60:  # 60 seconds
                        inactive_count += 1

                logger.debug(
                    "Streaming service health check",
                    extra={
                        "active_subscriptions": len([s for s in self._subscriptions.values() if s.active]),
                        "inactive_subscriptions": inactive_count,
                        "total_buffers": len(self._stream_buffers),
                    }
                )

            self._last_health_check = current_time

        except Exception as e:
            logger.error(
                "Error in health check",
                extra={"error": str(e)},
                exc_info=True
            )

    async def __aenter__(self) -> Mt5StreamingService:
        """Async context manager entry."""
        await self.start_streaming()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop_streaming()
        self._thread_pool.shutdown(wait=True)