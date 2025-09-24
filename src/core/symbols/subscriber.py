"""
Symbol subscription management for data collection.

This module provides subscription management for symbols with different
data types and collection strategies.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.data.events import create_event_subscriber, DataEventSubscriber
from src.core.data.models import MarketEventType
from .models import SymbolInfo

logger = get_logger(__name__)
settings = get_settings()


class SubscriptionType(str, Enum):
    """Types of symbol subscriptions."""

    TICK = "tick"
    QUOTE = "quote"
    OHLC = "ohlc"
    DEPTH = "depth"
    NEWS = "news"


class SubscriptionStatus(str, Enum):
    """Subscription status."""

    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class SymbolSubscription:
    """Individual symbol subscription with configuration."""

    def __init__(
        self,
        symbol: str,
        subscription_type: SubscriptionType,
        callback: Optional[Callable] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize symbol subscription.

        Args:
            symbol: Symbol name
            subscription_type: Type of subscription
            callback: Optional callback function
            config: Subscription configuration
        """
        self.symbol = symbol
        self.subscription_type = subscription_type
        self.callback = callback
        self.config = config or {}

        # Subscription state
        self.status = SubscriptionStatus.ACTIVE
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.data_count = 0

    def on_data(self, data: Any) -> None:
        """Handle incoming data for this subscription."""
        try:
            self.data_count += 1

            if self.callback:
                self.callback(self.symbol, self.subscription_type, data)

            # Reset error count on successful data
            if self.error_count > 0:
                self.error_count = 0
                self.last_error = None

        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)

            if self.error_count >= 5:  # Pause after 5 errors
                self.status = SubscriptionStatus.ERROR

            logger.error(f"Error processing data for {self.symbol} {self.subscription_type}: {e}")

    def pause(self) -> None:
        """Pause subscription."""
        self.status = SubscriptionStatus.PAUSED

    def resume(self) -> None:
        """Resume subscription."""
        if self.status == SubscriptionStatus.PAUSED:
            self.status = SubscriptionStatus.ACTIVE

    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self.status == SubscriptionStatus.ACTIVE

    def get_info(self) -> Dict[str, Any]:
        """Get subscription information."""
        return {
            "symbol": self.symbol,
            "subscription_type": self.subscription_type.value,
            "status": self.status.value,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "data_count": self.data_count,
            "config": self.config
        }


class SymbolSubscriber:
    """
    Symbol subscription manager for data collection.

    Manages subscriptions for multiple symbols with different data types
    and provides unified callback handling.
    """

    def __init__(self, name: str = "SymbolSubscriber"):
        """
        Initialize symbol subscriber.

        Args:
            name: Subscriber name
        """
        self.name = name

        # Subscription storage
        self._subscriptions: Dict[str, Dict[SubscriptionType, SymbolSubscription]] = {}

        # Event subscriber for market data
        self._event_subscriber = create_event_subscriber(name)

        # Set up event handlers
        self._setup_event_handlers()

        logger.info(f"Symbol subscriber '{name}' initialized")

    def subscribe(
        self,
        symbol: str,
        subscription_type: SubscriptionType,
        callback: Optional[Callable] = None,
        config: Dict[str, Any] = None
    ) -> bool:
        """
        Subscribe to symbol data.

        Args:
            symbol: Symbol name
            subscription_type: Type of subscription
            callback: Optional callback function
            config: Subscription configuration

        Returns:
            bool: True if subscribed successfully
        """
        try:
            # Initialize symbol subscriptions if needed
            if symbol not in self._subscriptions:
                self._subscriptions[symbol] = {}

            # Check if already subscribed
            if subscription_type in self._subscriptions[symbol]:
                logger.warning(f"Already subscribed to {symbol} {subscription_type}")
                return True

            # Create subscription
            subscription = SymbolSubscription(
                symbol=symbol,
                subscription_type=subscription_type,
                callback=callback,
                config=config
            )

            self._subscriptions[symbol][subscription_type] = subscription

            # Subscribe to relevant events based on type
            self._subscribe_to_events(symbol, subscription_type)

            logger.info(f"Subscribed to {symbol} {subscription_type}")
            return True

        except Exception as e:
            logger.error(f"Error subscribing to {symbol} {subscription_type}: {e}")
            return False

    def unsubscribe(
        self,
        symbol: str,
        subscription_type: Optional[SubscriptionType] = None
    ) -> bool:
        """
        Unsubscribe from symbol data.

        Args:
            symbol: Symbol name
            subscription_type: Type to unsubscribe (all if None)

        Returns:
            bool: True if unsubscribed successfully
        """
        try:
            if symbol not in self._subscriptions:
                return True

            if subscription_type:
                # Unsubscribe from specific type
                self._subscriptions[symbol].pop(subscription_type, None)

                # Clean up empty symbol entry
                if not self._subscriptions[symbol]:
                    del self._subscriptions[symbol]

                logger.info(f"Unsubscribed from {symbol} {subscription_type}")
            else:
                # Unsubscribe from all types for symbol
                del self._subscriptions[symbol]
                logger.info(f"Unsubscribed from all {symbol} subscriptions")

            return True

        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {e}")
            return False

    def subscribe_to_symbols(
        self,
        symbols: List[str],
        subscription_types: List[SubscriptionType],
        callback: Optional[Callable] = None
    ) -> int:
        """
        Subscribe to multiple symbols and types.

        Args:
            symbols: List of symbols
            subscription_types: List of subscription types
            callback: Optional callback function

        Returns:
            int: Number of successful subscriptions
        """
        success_count = 0

        for symbol in symbols:
            for subscription_type in subscription_types:
                if self.subscribe(symbol, subscription_type, callback):
                    success_count += 1

        return success_count

    def pause_subscription(self, symbol: str, subscription_type: SubscriptionType) -> bool:
        """Pause specific subscription."""
        subscription = self._get_subscription(symbol, subscription_type)
        if subscription:
            subscription.pause()
            return True
        return False

    def resume_subscription(self, symbol: str, subscription_type: SubscriptionType) -> bool:
        """Resume specific subscription."""
        subscription = self._get_subscription(symbol, subscription_type)
        if subscription:
            subscription.resume()
            return True
        return False

    def pause_symbol(self, symbol: str) -> bool:
        """Pause all subscriptions for symbol."""
        if symbol in self._subscriptions:
            for subscription in self._subscriptions[symbol].values():
                subscription.pause()
            return True
        return False

    def resume_symbol(self, symbol: str) -> bool:
        """Resume all subscriptions for symbol."""
        if symbol in self._subscriptions:
            for subscription in self._subscriptions[symbol].values():
                subscription.resume()
            return True
        return False

    def get_subscriptions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get subscription information.

        Args:
            symbol: Specific symbol (all if None)

        Returns:
            List of subscription information
        """
        subscriptions = []

        if symbol:
            # Get subscriptions for specific symbol
            if symbol in self._subscriptions:
                for subscription in self._subscriptions[symbol].values():
                    subscriptions.append(subscription.get_info())
        else:
            # Get all subscriptions
            for symbol_subs in self._subscriptions.values():
                for subscription in symbol_subs.values():
                    subscriptions.append(subscription.get_info())

        return subscriptions

    def get_subscribed_symbols(self) -> Set[str]:
        """Get set of subscribed symbols."""
        return set(self._subscriptions.keys())

    def get_subscription_count(self) -> Dict[str, int]:
        """Get subscription counts by type."""
        counts = {}

        for symbol_subs in self._subscriptions.values():
            for subscription_type in symbol_subs.keys():
                type_name = subscription_type.value
                counts[type_name] = counts.get(type_name, 0) + 1

        return counts

    def get_stats(self) -> Dict[str, Any]:
        """Get subscriber statistics."""
        total_subscriptions = sum(
            len(symbol_subs) for symbol_subs in self._subscriptions.values()
        )

        active_count = 0
        error_count = 0
        total_data_count = 0

        for symbol_subs in self._subscriptions.values():
            for subscription in symbol_subs.values():
                if subscription.is_active():
                    active_count += 1
                if subscription.status == SubscriptionStatus.ERROR:
                    error_count += 1
                total_data_count += subscription.data_count

        return {
            "name": self.name,
            "total_subscriptions": total_subscriptions,
            "active_subscriptions": active_count,
            "error_subscriptions": error_count,
            "subscribed_symbols": len(self._subscriptions),
            "total_data_received": total_data_count,
            "subscription_counts": self.get_subscription_count()
        }

    def _get_subscription(
        self,
        symbol: str,
        subscription_type: SubscriptionType
    ) -> Optional[SymbolSubscription]:
        """Get specific subscription."""
        return self._subscriptions.get(symbol, {}).get(subscription_type)

    def _subscribe_to_events(self, symbol: str, subscription_type: SubscriptionType) -> None:
        """Subscribe to relevant market events."""
        if subscription_type == SubscriptionType.TICK:
            self._event_subscriber.subscribe_to_symbols(
                symbols=[symbol],
                event_types=[MarketEventType.TICK_RECEIVED]
            )
        elif subscription_type == SubscriptionType.OHLC:
            self._event_subscriber.subscribe_to_symbols(
                symbols=[symbol],
                event_types=[MarketEventType.OHLC_UPDATED]
            )

    def _setup_event_handlers(self) -> None:
        """Set up event handlers for market data."""

        def handle_tick_event(event):
            """Handle tick received event."""
            symbol = event.symbol
            if not symbol or symbol not in self._subscriptions:
                return

            subscription = self._get_subscription(symbol, SubscriptionType.TICK)
            if subscription and subscription.is_active():
                subscription.on_data(event.data)

        def handle_ohlc_event(event):
            """Handle OHLC updated event."""
            symbol = event.symbol
            if not symbol or symbol not in self._subscriptions:
                return

            subscription = self._get_subscription(symbol, SubscriptionType.OHLC)
            if subscription and subscription.is_active():
                subscription.on_data(event.data)

        # Register event handlers
        self._event_subscriber.on_tick_received(handle_tick_event)
        self._event_subscriber.on_ohlc_updated(handle_ohlc_event)