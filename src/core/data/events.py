"""
Real-time event publishing and subscription system.

This module provides high-performance event handling for market data
events, system notifications, and GUI updates with async support.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

from src.core.config import get_settings
from src.core.logging import get_logger
from .models import MarketEvent, MarketEventType

logger = get_logger(__name__)
settings = get_settings()


class EventSubscriber:
    """
    Event subscriber with filtering and async support.

    Provides flexible event subscription with type filtering,
    symbol filtering, and both sync/async callback support.
    """

    def __init__(
        self,
        subscriber_id: str,
        callback: Callable[[MarketEvent], Union[None, Any]],
        event_types: Optional[List[MarketEventType]] = None,
        symbols: Optional[List[str]] = None,
        is_async: bool = False
    ):
        """
        Initialize event subscriber.

        Args:
            subscriber_id: Unique subscriber ID
            callback: Callback function for events
            event_types: Event types to subscribe to (None for all)
            symbols: Symbols to filter by (None for all)
            is_async: Whether callback is async
        """
        self.subscriber_id = subscriber_id
        self.callback = callback
        self.event_types = set(event_types) if event_types else None
        self.symbols = set(symbols) if symbols else None
        self.is_async = is_async

        # Performance tracking
        self.events_received = 0
        self.last_event_time = 0
        self.created_at = time.time()

        logger.debug(f"Created event subscriber {subscriber_id}")

    def should_receive_event(self, event: MarketEvent) -> bool:
        """Check if this subscriber should receive the event."""
        # Check event type filter
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check symbol filter
        if self.symbols and event.symbol and event.symbol not in self.symbols:
            return False

        return True

    def deliver_event(self, event: MarketEvent) -> bool:
        """
        Deliver event to subscriber.

        Returns:
            bool: True if delivered successfully
        """
        try:
            if not self.should_receive_event(event):
                return True

            self.events_received += 1
            self.last_event_time = time.time()

            if self.is_async:
                # Schedule async callback
                asyncio.create_task(self.callback(event))
            else:
                # Call sync callback
                self.callback(event)

            return True

        except Exception as e:
            logger.error(f"Error delivering event to {self.subscriber_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get subscriber statistics."""
        return {
            "subscriber_id": self.subscriber_id,
            "event_types": list(self.event_types) if self.event_types else None,
            "symbols": list(self.symbols) if self.symbols else None,
            "is_async": self.is_async,
            "events_received": self.events_received,
            "last_event_time": self.last_event_time,
            "created_at": self.created_at,
            "events_per_second": (
                self.events_received / (time.time() - self.created_at)
                if time.time() > self.created_at else 0
            )
        }


class DataEventPublisher:
    """
    High-performance event publisher with async support.

    Provides efficient event broadcasting to multiple subscribers
    with filtering, buffering, and performance monitoring.
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize event publisher.

        Args:
            max_queue_size: Maximum event queue size
        """
        self.max_queue_size = max_queue_size

        # Subscriber management
        self._subscribers: Dict[str, EventSubscriber] = {}
        self._subscribers_lock = threading.RLock()

        # Event queue for async processing
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._processing_task: Optional[asyncio.Task] = None

        # Performance metrics
        self._events_published = 0
        self._events_dropped = 0
        self._subscribers_notified = 0
        self._start_time = time.time()

        # State management
        self._is_running = False

        logger.info("Event publisher initialized")

    def start(self) -> bool:
        """Start the event publisher."""
        if self._is_running:
            return True

        try:
            self._processing_task = asyncio.create_task(self._process_events())
            self._is_running = True

            logger.info("Event publisher started")
            return True

        except Exception as e:
            logger.error(f"Failed to start event publisher: {e}")
            return False

    def stop(self) -> None:
        """Stop the event publisher."""
        if not self._is_running:
            return

        logger.info("Stopping event publisher...")

        self._is_running = False

        if self._processing_task:
            self._processing_task.cancel()

        logger.info("Event publisher stopped")

    def subscribe(
        self,
        subscriber_id: str,
        callback: Callable[[MarketEvent], Union[None, Any]],
        event_types: Optional[List[MarketEventType]] = None,
        symbols: Optional[List[str]] = None,
        is_async: bool = False
    ) -> bool:
        """
        Subscribe to events.

        Args:
            subscriber_id: Unique subscriber ID
            callback: Callback function
            event_types: Event types to subscribe to
            symbols: Symbols to filter by
            is_async: Whether callback is async

        Returns:
            bool: True if subscribed successfully
        """
        try:
            with self._subscribers_lock:
                if subscriber_id in self._subscribers:
                    logger.warning(f"Subscriber {subscriber_id} already exists")
                    return False

                subscriber = EventSubscriber(
                    subscriber_id=subscriber_id,
                    callback=callback,
                    event_types=event_types,
                    symbols=symbols,
                    is_async=is_async
                )

                self._subscribers[subscriber_id] = subscriber

            logger.info(
                f"Subscribed {subscriber_id} to events: "
                f"types={event_types}, symbols={symbols}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe {subscriber_id}: {e}")
            return False

    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscriber_id: Subscriber ID to remove

        Returns:
            bool: True if unsubscribed successfully
        """
        try:
            with self._subscribers_lock:
                if subscriber_id in self._subscribers:
                    del self._subscribers[subscriber_id]
                    logger.info(f"Unsubscribed {subscriber_id}")
                    return True

            logger.warning(f"Subscriber {subscriber_id} not found")
            return False

        except Exception as e:
            logger.error(f"Failed to unsubscribe {subscriber_id}: {e}")
            return False

    def publish(self, event: MarketEvent) -> bool:
        """
        Publish event synchronously.

        Args:
            event: Event to publish

        Returns:
            bool: True if published successfully
        """
        try:
            # Direct delivery to subscribers
            with self._subscribers_lock:
                subscribers = list(self._subscribers.values())

            delivered = 0
            for subscriber in subscribers:
                if subscriber.deliver_event(event):
                    delivered += 1

            self._events_published += 1
            self._subscribers_notified += delivered

            logger.debug(f"Published event {event.event_type} to {delivered} subscribers")
            return True

        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return False

    async def publish_async(self, event: MarketEvent) -> bool:
        """
        Publish event asynchronously.

        Args:
            event: Event to publish

        Returns:
            bool: True if queued successfully
        """
        try:
            if not self._is_running:
                return self.publish(event)  # Fallback to sync

            # Add to queue for async processing
            try:
                self._event_queue.put_nowait(event)
                return True
            except asyncio.QueueFull:
                # Queue is full, drop event
                self._events_dropped += 1
                logger.warning("Event queue full, dropping event")
                return False

        except Exception as e:
            logger.error(f"Error publishing event async: {e}")
            return False

    async def _process_events(self) -> None:
        """Process events from the async queue."""
        logger.debug("Started async event processing")

        while self._is_running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )

                # Deliver to subscribers
                await self._deliver_event_async(event)

                # Mark task done
                self._event_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")

        logger.debug("Stopped async event processing")

    async def _deliver_event_async(self, event: MarketEvent) -> None:
        """Deliver event to all subscribers asynchronously."""
        try:
            with self._subscribers_lock:
                subscribers = list(self._subscribers.values())

            # Create delivery tasks
            tasks = []
            for subscriber in subscribers:
                if subscriber.should_receive_event(event):
                    if subscriber.is_async:
                        task = asyncio.create_task(self._deliver_to_async_subscriber(subscriber, event))
                    else:
                        task = asyncio.create_task(self._deliver_to_sync_subscriber(subscriber, event))
                    tasks.append(task)

            # Wait for all deliveries to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Count successful deliveries
                delivered = sum(1 for result in results if result is True)
                self._events_published += 1
                self._subscribers_notified += delivered

        except Exception as e:
            logger.error(f"Error delivering event async: {e}")

    async def _deliver_to_async_subscriber(
        self,
        subscriber: EventSubscriber,
        event: MarketEvent
    ) -> bool:
        """Deliver event to async subscriber."""
        try:
            subscriber.events_received += 1
            subscriber.last_event_time = time.time()
            await subscriber.callback(event)
            return True
        except Exception as e:
            logger.error(f"Error in async callback for {subscriber.subscriber_id}: {e}")
            return False

    async def _deliver_to_sync_subscriber(
        self,
        subscriber: EventSubscriber,
        event: MarketEvent
    ) -> bool:
        """Deliver event to sync subscriber in thread pool."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                subscriber.deliver_event,
                event
            )
            return True
        except Exception as e:
            logger.error(f"Error in sync callback for {subscriber.subscriber_id}: {e}")
            return False

    def get_subscribers(self) -> Dict[str, EventSubscriber]:
        """Get all subscribers."""
        with self._subscribers_lock:
            return dict(self._subscribers)

    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        runtime = time.time() - self._start_time

        with self._subscribers_lock:
            subscriber_stats = [
                subscriber.get_stats()
                for subscriber in self._subscribers.values()
            ]

        return {
            "is_running": self._is_running,
            "runtime_seconds": runtime,
            "events_published": self._events_published,
            "events_dropped": self._events_dropped,
            "subscribers_notified": self._subscribers_notified,
            "queue_size": self._event_queue.qsize() if self._event_queue else 0,
            "max_queue_size": self.max_queue_size,
            "subscriber_count": len(self._subscribers),
            "events_per_second": self._events_published / runtime if runtime > 0 else 0,
            "subscribers": subscriber_stats
        }


class DataEventSubscriber:
    """
    Convenience class for easy event subscription.

    Provides simplified interface for subscribing to specific
    event types with automatic callback handling.
    """

    def __init__(self, name: str):
        """
        Initialize data event subscriber.

        Args:
            name: Subscriber name
        """
        self.name = name
        self.subscriber_id = f"{name}_{uuid4().hex[:8]}"

        # Get publisher
        self.publisher = get_event_publisher()

        # Callback registry
        self._callbacks: Dict[MarketEventType, List[Callable]] = {}

        logger.debug(f"Initialized event subscriber {self.name}")

    def on_tick_received(self, callback: Callable[[MarketEvent], Any]) -> bool:
        """Subscribe to tick received events."""
        return self._register_callback(MarketEventType.TICK_RECEIVED, callback)

    def on_ohlc_updated(self, callback: Callable[[MarketEvent], Any]) -> bool:
        """Subscribe to OHLC updated events."""
        return self._register_callback(MarketEventType.OHLC_UPDATED, callback)

    def on_trend_change(self, callback: Callable[[MarketEvent], Any]) -> bool:
        """Subscribe to trend change events."""
        return self._register_callback(MarketEventType.TREND_CHANGE, callback)

    def on_connection_lost(self, callback: Callable[[MarketEvent], Any]) -> bool:
        """Subscribe to connection lost events."""
        return self._register_callback(MarketEventType.CONNECTION_LOST, callback)

    def on_connection_restored(self, callback: Callable[[MarketEvent], Any]) -> bool:
        """Subscribe to connection restored events."""
        return self._register_callback(MarketEventType.CONNECTION_RESTORED, callback)

    def _register_callback(
        self,
        event_type: MarketEventType,
        callback: Callable[[MarketEvent], Any]
    ) -> bool:
        """Register callback for specific event type."""
        try:
            if event_type not in self._callbacks:
                self._callbacks[event_type] = []

            self._callbacks[event_type].append(callback)

            # Subscribe to publisher
            return self.publisher.subscribe(
                subscriber_id=f"{self.subscriber_id}_{event_type}",
                callback=self._event_dispatcher,
                event_types=[event_type]
            )

        except Exception as e:
            logger.error(f"Failed to register callback for {event_type}: {e}")
            return False

    def _event_dispatcher(self, event: MarketEvent) -> None:
        """Dispatch events to registered callbacks."""
        try:
            callbacks = self._callbacks.get(event.event_type, [])

            for callback in callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in callback for {event.event_type}: {e}")

        except Exception as e:
            logger.error(f"Error dispatching event: {e}")

    def subscribe_to_symbols(
        self,
        symbols: List[str],
        event_types: List[MarketEventType] = None
    ) -> bool:
        """
        Subscribe to events for specific symbols.

        Args:
            symbols: List of symbols to subscribe to
            event_types: Event types to subscribe to (None for all)

        Returns:
            bool: True if subscribed successfully
        """
        return self.publisher.subscribe(
            subscriber_id=f"{self.subscriber_id}_symbols",
            callback=self._event_dispatcher,
            event_types=event_types,
            symbols=symbols
        )

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all events."""
        # Unsubscribe all related subscriber IDs
        for event_type in self._callbacks:
            self.publisher.unsubscribe(f"{self.subscriber_id}_{event_type}")

        self.publisher.unsubscribe(f"{self.subscriber_id}_symbols")
        self._callbacks.clear()


# Global event publisher instance
_event_publisher: Optional[DataEventPublisher] = None


def get_event_publisher() -> DataEventPublisher:
    """Get the global event publisher instance."""
    global _event_publisher

    if _event_publisher is None:
        _event_publisher = DataEventPublisher()
        # Auto-start the publisher
        _event_publisher.start()

    return _event_publisher


def create_event_subscriber(name: str) -> DataEventSubscriber:
    """Create a new event subscriber."""
    return DataEventSubscriber(name)