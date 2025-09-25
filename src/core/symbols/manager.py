"""
Centralized symbol management system.

This module provides comprehensive symbol management with MT5 integration,
database persistence, and real-time monitoring capabilities.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set

from src.core.config import get_settings
from src.core.logging import get_logger
from src.database.database import get_database_manager
from src.database.services.symbols import SymbolService
from src.mt5.connection import get_mt5_session_manager
from src.core.data.collector import get_data_collection_manager
from src.core.tasks import get_task_manager, background_task, scheduled_task

from .models import SymbolInfo, SymbolStatus, SymbolGroup, SymbolType, SymbolStats, MarketSession

logger = get_logger(__name__)
settings = get_settings()


class SymbolManager:
    """
    Centralized symbol management system.

    Provides comprehensive symbol management with MT5 integration,
    database persistence, real-time monitoring, and subscription management.
    """

    def __init__(self):
        """Initialize symbol manager."""
        # Core dependencies
        self.db_manager = get_database_manager()
        self.symbol_service = SymbolService()
        self.mt5_session_manager = get_mt5_session_manager()
        self.data_collector = get_data_collection_manager()
        self.task_manager = get_task_manager()

        # Symbol storage
        self._symbols: Dict[str, SymbolInfo] = {}
        self._symbol_groups: Dict[str, SymbolGroup] = {}
        self._symbol_stats: Dict[str, SymbolStats] = {}
        self._lock = threading.RLock()

        # Subscription management
        self._subscribed_symbols: Set[str] = set()
        self._tick_symbols: Set[str] = set()
        self._quote_symbols: Set[str] = set()

        # State management
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Configuration
        self._sync_interval = 300  # 5 minutes
        self._stats_interval = 600  # 10 minutes
        self._cleanup_interval = 3600  # 1 hour

        logger.info("Symbol manager initialized")

    def start(self) -> bool:
        """Start the symbol management system."""
        if self._is_running:
            logger.warning("Symbol manager already running")
            return True

        try:
            # Load symbols from database
            self._load_symbols_from_database()

            # Load symbol groups
            self._load_groups_from_database()

            # Start monitoring tasks
            self._start_monitoring()

            self._is_running = True
            logger.info("Symbol manager started")
            return True

        except Exception as e:
            logger.error(f"Failed to start symbol manager: {e}")
            return False

    def stop(self) -> None:
        """Stop the symbol management system."""
        if not self._is_running:
            return

        logger.info("Stopping symbol manager...")

        self._is_running = False

        # Stop monitoring
        if self._monitor_task:
            self._monitor_task.cancel()

        # Save current state to database
        try:
            self._save_symbols_to_database()
            self._save_groups_to_database()
        except Exception as e:
            logger.error(f"Error saving symbols during shutdown: {e}")

        logger.info("Symbol manager stopped")

    def add_symbol(
        self,
        symbol: str,
        symbol_info: SymbolInfo = None,
        auto_subscribe: bool = False,
        subscription_type: str = "quote"
    ) -> bool:
        """
        Add symbol to management.

        Args:
            symbol: Symbol name
            symbol_info: Symbol information (fetched from MT5 if None)
            auto_subscribe: Automatically subscribe to data
            subscription_type: Type of subscription (tick/quote)

        Returns:
            bool: True if added successfully
        """
        try:
            with self._lock:
                # Check if already exists
                if symbol in self._symbols:
                    logger.warning(f"Symbol {symbol} already exists")
                    return False

                # Get symbol info from MT5 if not provided
                if symbol_info is None:
                    symbol_info = self._fetch_symbol_info_from_mt5(symbol)
                    if symbol_info is None:
                        logger.error(f"Failed to fetch symbol info for {symbol}")
                        return False

                # Store symbol
                self._symbols[symbol] = symbol_info

                # Save to database
                self._save_symbol_to_database(symbol_info)

                # Auto-subscribe if requested
                if auto_subscribe:
                    self.subscribe_to_symbol(symbol, subscription_type)

                logger.info(f"Added symbol {symbol}")
                return True

        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
            return False

    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove symbol from management.

        Args:
            symbol: Symbol name to remove

        Returns:
            bool: True if removed successfully
        """
        try:
            with self._lock:
                if symbol not in self._symbols:
                    logger.warning(f"Symbol {symbol} not found")
                    return False

                # Unsubscribe from data
                self.unsubscribe_from_symbol(symbol)

                # Remove from memory
                del self._symbols[symbol]
                self._symbol_stats.pop(symbol, None)

                # Remove from groups
                for group in self._symbol_groups.values():
                    group.remove_symbol(symbol)

                # Remove from database
                try:
                    with self.db_manager.get_session() as session:
                        self.symbol_service.delete_by_symbol(session, symbol)
                except Exception as e:
                    logger.error(f"Error removing symbol from database: {e}")

                logger.info(f"Removed symbol {symbol}")
                return True

        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {e}")
            return False

    def get_symbol(self, symbol: str) -> Optional[SymbolInfo]:
        """Get symbol information."""
        with self._lock:
            return self._symbols.get(symbol)

    def list_symbols(
        self,
        status: Optional[SymbolStatus] = None,
        symbol_type: Optional[SymbolType] = None,
        tradable_only: bool = False
    ) -> List[SymbolInfo]:
        """
        List symbols with optional filtering.

        Args:
            status: Filter by symbol status
            symbol_type: Filter by symbol type
            tradable_only: Only include tradable symbols

        Returns:
            List of symbol information
        """
        with self._lock:
            symbols = list(self._symbols.values())

        # Apply filters
        if status:
            symbols = [s for s in symbols if s.status == status]

        if symbol_type:
            symbols = [s for s in symbols if s.symbol_type == symbol_type]

        if tradable_only:
            symbols = [s for s in symbols if s.is_tradable and s.is_market_open()]

        return symbols

    def search_symbols(self, query: str, limit: int = 50) -> List[SymbolInfo]:
        """
        Search symbols by name or description.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching symbols
        """
        query = query.lower()
        matches = []

        with self._lock:
            for symbol_info in self._symbols.values():
                if (query in symbol_info.symbol.lower() or
                    query in symbol_info.description.lower()):
                    matches.append(symbol_info)

                if len(matches) >= limit:
                    break

        return matches

    def subscribe_to_symbol(
        self,
        symbol: str,
        subscription_type: str = "quote"
    ) -> bool:
        """
        Subscribe to symbol data collection.

        Args:
            symbol: Symbol name
            subscription_type: Type of subscription (tick/quote)

        Returns:
            bool: True if subscribed successfully
        """
        try:
            with self._lock:
                if symbol not in self._symbols:
                    logger.error(f"Symbol {symbol} not found for subscription")
                    return False

                # Check if already subscribed
                if symbol in self._subscribed_symbols:
                    logger.debug(f"Symbol {symbol} already subscribed")
                    return True

                # Add to appropriate collector
                if subscription_type == "tick":
                    success = self.data_collector.add_tick_symbol(symbol)
                    if success:
                        self._tick_symbols.add(symbol)
                else:
                    # Quote subscription (default)
                    success = True  # Quote collector handles this automatically
                    self._quote_symbols.add(symbol)

                if success:
                    self._subscribed_symbols.add(symbol)
                    logger.info(f"Subscribed to {symbol} ({subscription_type})")

                return success

        except Exception as e:
            logger.error(f"Error subscribing to symbol {symbol}: {e}")
            return False

    def unsubscribe_from_symbol(self, symbol: str) -> bool:
        """
        Unsubscribe from symbol data collection.

        Args:
            symbol: Symbol name

        Returns:
            bool: True if unsubscribed successfully
        """
        try:
            with self._lock:
                if symbol not in self._subscribed_symbols:
                    return True  # Already unsubscribed

                # Remove from collectors
                if symbol in self._tick_symbols:
                    self.data_collector.remove_tick_symbol(symbol)
                    self._tick_symbols.discard(symbol)

                self._quote_symbols.discard(symbol)
                self._subscribed_symbols.discard(symbol)

                logger.info(f"Unsubscribed from {symbol}")
                return True

        except Exception as e:
            logger.error(f"Error unsubscribing from symbol {symbol}: {e}")
            return False

    def create_symbol_group(
        self,
        group_id: str,
        name: str,
        description: str = "",
        symbols: List[str] = None
    ) -> bool:
        """
        Create symbol group.

        Args:
            group_id: Group identifier
            name: Group name
            description: Group description
            symbols: Initial symbols

        Returns:
            bool: True if created successfully
        """
        try:
            with self._lock:
                if group_id in self._symbol_groups:
                    logger.warning(f"Symbol group {group_id} already exists")
                    return False

                group = SymbolGroup(
                    group_id=group_id,
                    name=name,
                    description=description,
                    symbols=symbols or []
                )

                self._symbol_groups[group_id] = group

                # Save to database
                self._save_group_to_database(group)

                logger.info(f"Created symbol group {group_id}")
                return True

        except Exception as e:
            logger.error(f"Error creating symbol group {group_id}: {e}")
            return False

    def delete_symbol_group(self, group_id: str) -> bool:
        """Delete symbol group."""
        try:
            with self._lock:
                if group_id not in self._symbol_groups:
                    logger.warning(f"Symbol group {group_id} not found")
                    return False

                del self._symbol_groups[group_id]

                # Remove from database
                try:
                    with self.db_manager.get_session() as session:
                        # Assuming we have a group service
                        pass  # TODO: Implement group deletion from DB
                except Exception as e:
                    logger.error(f"Error removing group from database: {e}")

                logger.info(f"Deleted symbol group {group_id}")
                return True

        except Exception as e:
            logger.error(f"Error deleting symbol group {group_id}: {e}")
            return False

    def add_symbol_to_group(self, group_id: str, symbol: str) -> bool:
        """Add symbol to group."""
        with self._lock:
            group = self._symbol_groups.get(group_id)
            if group and symbol in self._symbols:
                return group.add_symbol(symbol)
            return False

    def remove_symbol_from_group(self, group_id: str, symbol: str) -> bool:
        """Remove symbol from group."""
        with self._lock:
            group = self._symbol_groups.get(group_id)
            if group:
                return group.remove_symbol(symbol)
            return False

    def get_symbol_group(self, group_id: str) -> Optional[SymbolGroup]:
        """Get symbol group."""
        with self._lock:
            return self._symbol_groups.get(group_id)

    def list_symbol_groups(self) -> List[SymbolGroup]:
        """List all symbol groups."""
        with self._lock:
            return list(self._symbol_groups.values())

    def update_symbol_quote(
        self,
        symbol: str,
        bid: float = None,
        ask: float = None,
        last_price: float = None,
        volume: int = None
    ) -> bool:
        """
        Update symbol quote information.

        Args:
            symbol: Symbol name
            bid: Bid price
            ask: Ask price
            last_price: Last trade price
            volume: Volume

        Returns:
            bool: True if updated successfully
        """
        try:
            with self._lock:
                symbol_info = self._symbols.get(symbol)
                if not symbol_info:
                    return False

                symbol_info.update_quote(bid, ask, last_price, volume)

                # Update in database (async)
                self._schedule_symbol_update(symbol_info)

                return True

        except Exception as e:
            logger.error(f"Error updating quote for {symbol}: {e}")
            return False

    def get_symbol_stats(self, symbol: str) -> Optional[SymbolStats]:
        """Get symbol statistics."""
        with self._lock:
            return self._symbol_stats.get(symbol)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get symbol system statistics."""
        with self._lock:
            total_symbols = len(self._symbols)
            active_symbols = len([s for s in self._symbols.values() if s.status == SymbolStatus.ACTIVE])
            subscribed_symbols = len(self._subscribed_symbols)
            tick_symbols = len(self._tick_symbols)
            quote_symbols = len(self._quote_symbols)

            # Count by type
            type_counts = {}
            for symbol_info in self._symbols.values():
                symbol_type = symbol_info.symbol_type.value
                type_counts[symbol_type] = type_counts.get(symbol_type, 0) + 1

            return {
                "total_symbols": total_symbols,
                "active_symbols": active_symbols,
                "subscribed_symbols": subscribed_symbols,
                "tick_symbols": tick_symbols,
                "quote_symbols": quote_symbols,
                "symbol_groups": len(self._symbol_groups),
                "type_counts": type_counts,
                "is_running": self._is_running
            }

    def _fetch_symbol_info_from_mt5(self, symbol: str) -> Optional[SymbolInfo]:
        """Fetch symbol information from MT5."""
        try:
            session = self.mt5_session_manager.get_active_session()
            if not session:
                logger.error("No active MT5 session for symbol fetch")
                return None

            # Get symbol info from MT5
            mt5_symbol_info = session.symbol_info(symbol)
            if mt5_symbol_info is None:
                logger.warning(f"Symbol {symbol} not found in MT5")
                return None

            # Get current tick
            tick_info = session.symbol_info_tick(symbol)

            # Create SymbolInfo object
            symbol_info = SymbolInfo(
                symbol=symbol,
                description=getattr(mt5_symbol_info, 'description', ''),
                base_currency=getattr(mt5_symbol_info, 'currency_base', None),
                quote_currency=getattr(mt5_symbol_info, 'currency_profit', None),
                symbol_type=self._determine_symbol_type(symbol, mt5_symbol_info),
                tick_size=getattr(mt5_symbol_info, 'point', None),
                tick_value=getattr(mt5_symbol_info, 'trade_tick_value', None),
                contract_size=getattr(mt5_symbol_info, 'trade_contract_size', None),
                margin_required=getattr(mt5_symbol_info, 'margin_initial', None),
                is_tradable=getattr(mt5_symbol_info, 'visible', True),
                mt5_symbol=symbol,
                mt5_digits=getattr(mt5_symbol_info, 'digits', None),
                mt5_point=getattr(mt5_symbol_info, 'point', None)
            )

            # Update with current tick if available
            if tick_info:
                symbol_info.update_quote(
                    bid=tick_info.bid,
                    ask=tick_info.ask,
                    volume=getattr(tick_info, 'volume', 0)
                )

            return symbol_info

        except Exception as e:
            logger.error(f"Error fetching symbol info for {symbol}: {e}")
            return None

    def _determine_symbol_type(self, symbol: str, mt5_info: Any) -> SymbolType:
        """Determine symbol type from MT5 information."""
        try:
            # Check by symbol name patterns
            symbol_upper = symbol.upper()

            # Forex pairs
            if len(symbol) == 6 and not any(char.isdigit() for char in symbol):
                return SymbolType.FOREX

            # Stocks (usually have exchange suffix or specific patterns)
            if '.' in symbol or symbol_upper.endswith(('_STOCK', '_EQ')):
                return SymbolType.STOCK

            # Indices
            if any(term in symbol_upper for term in ['INDEX', 'IND', 'SPX', 'NDX', 'DJI']):
                return SymbolType.INDEX

            # Commodities
            if any(term in symbol_upper for term in ['GOLD', 'SILVER', 'OIL', 'GAS', 'WHEAT']):
                return SymbolType.COMMODITY

            # Crypto
            if any(term in symbol_upper for term in ['BTC', 'ETH', 'CRYPTO', 'COIN']):
                return SymbolType.CRYPTO

            # CFDs
            if 'CFD' in symbol_upper:
                return SymbolType.CFD

            # Default to unknown
            return SymbolType.UNKNOWN

        except Exception:
            return SymbolType.UNKNOWN

    def _map_market_to_symbol_type(self, market: str) -> SymbolType:
        """Map database market field to SymbolType enum."""
        try:
            market_upper = market.upper()

            if market_upper in ['FOREX', 'FX']:
                return SymbolType.FOREX
            elif market_upper in ['STOCK', 'EQUITY', 'SHARES']:
                return SymbolType.STOCK
            elif market_upper in ['COMMODITY', 'COMMODITIES']:
                return SymbolType.COMMODITY
            elif market_upper in ['INDEX', 'INDICES']:
                return SymbolType.INDEX
            elif market_upper in ['CRYPTO', 'CRYPTOCURRENCY', 'BITCOIN']:
                return SymbolType.CRYPTO
            elif market_upper in ['CFD']:
                return SymbolType.CFD
            else:
                return SymbolType.UNKNOWN
        except Exception:
            return SymbolType.UNKNOWN

    def _load_symbols_from_database(self) -> None:
        """Load symbols from database."""
        try:
            # Get symbols using the service's built-in session management
            symbols = self.symbol_service.get_all()

            logger.info(f"Found {len(symbols)} symbols in database")

            for symbol_data in symbols:
                # Convert database model to SymbolInfo
                symbol_info = SymbolInfo(
                    symbol=symbol_data.symbol,
                    description=symbol_data.name or "",
                    symbol_type=self._map_market_to_symbol_type(symbol_data.market),
                    status=SymbolStatus.ACTIVE if symbol_data.is_tradeable else SymbolStatus.INACTIVE,
                    is_tradable=symbol_data.is_tradeable,
                    is_visible=True,  # Default value since not in DB
                    created_at=symbol_data.created_at,
                    updated_at=symbol_data.updated_at
                )
                self._symbols[symbol_data.symbol] = symbol_info

            logger.info(f"Loaded {len(self._symbols)} symbols from database")

        except Exception as e:
            logger.error(f"Error loading symbols from database: {e}")

    def _load_groups_from_database(self) -> None:
        """Load symbol groups from database."""
        try:
            # TODO: Implement group loading from database
            logger.debug("Symbol group loading from database not yet implemented")

        except Exception as e:
            logger.error(f"Error loading groups from database: {e}")

    def _save_symbols_to_database(self) -> None:
        """Save all symbols to database."""
        try:
            with self.db_manager.get_session() as session:
                for symbol_info in self._symbols.values():
                    self.symbol_service.create_or_update(session, symbol_info)

            logger.debug("Saved symbols to database")

        except Exception as e:
            logger.error(f"Error saving symbols to database: {e}")

    def _save_symbol_to_database(self, symbol_info: SymbolInfo) -> None:
        """Save single symbol to database."""
        try:
            with self.db_manager.get_session() as session:
                self.symbol_service.create_or_update(session, symbol_info)

        except Exception as e:
            logger.error(f"Error saving symbol to database: {e}")

    def _save_groups_to_database(self) -> None:
        """Save all symbol groups to database."""
        try:
            # TODO: Implement group saving to database
            logger.debug("Symbol group saving to database not yet implemented")

        except Exception as e:
            logger.error(f"Error saving groups to database: {e}")

    def _save_group_to_database(self, group: SymbolGroup) -> None:
        """Save single symbol group to database."""
        try:
            # TODO: Implement group saving to database
            logger.debug("Symbol group saving to database not yet implemented")

        except Exception as e:
            logger.error(f"Error saving group to database: {e}")

    def _schedule_symbol_update(self, symbol_info: SymbolInfo) -> None:
        """Schedule asynchronous symbol update to database."""
        @background_task(name=f"update_symbol_{symbol_info.symbol}")
        def update_symbol():
            self._save_symbol_to_database(symbol_info)

        update_symbol()

    def _start_monitoring(self) -> None:
        """Start symbol monitoring tasks."""
        # Schedule periodic symbol synchronization
        @scheduled_task(
            interval_seconds=self._sync_interval,
            name="sync_symbols_with_mt5"
        )
        def sync_symbols():
            self._sync_symbols_with_mt5()

        # Schedule statistics calculation
        @scheduled_task(
            interval_seconds=self._stats_interval,
            name="calculate_symbol_stats"
        )
        def calculate_stats():
            self._calculate_symbol_stats()

        # Schedule cleanup
        @scheduled_task(
            interval_seconds=self._cleanup_interval,
            name="cleanup_symbol_data"
        )
        def cleanup():
            self._cleanup_symbol_data()

        logger.info("Symbol monitoring tasks scheduled")

    @background_task(name="sync_symbols_with_mt5")
    def _sync_symbols_with_mt5(self) -> None:
        """Synchronize symbols with MT5."""
        try:
            logger.debug("Synchronizing symbols with MT5...")

            session = self.mt5_session_manager.get_active_session()
            if not session:
                logger.warning("No active MT5 session for synchronization")
                return

            # Update existing symbols
            updated_count = 0
            with self._lock:
                for symbol in list(self._symbols.keys()):
                    try:
                        # Get fresh data from MT5
                        updated_info = self._fetch_symbol_info_from_mt5(symbol)
                        if updated_info:
                            # Update symbol info
                            self._symbols[symbol] = updated_info
                            self._schedule_symbol_update(updated_info)
                            updated_count += 1

                    except Exception as e:
                        logger.debug(f"Error updating symbol {symbol}: {e}")

            logger.info(f"Synchronized {updated_count} symbols with MT5")

        except Exception as e:
            logger.error(f"Error synchronizing symbols with MT5: {e}")

    @background_task(name="calculate_symbol_stats")
    def _calculate_symbol_stats(self) -> None:
        """Calculate symbol statistics."""
        try:
            logger.debug("Calculating symbol statistics...")

            # TODO: Implement comprehensive statistics calculation
            # This would involve analyzing historical data, calculating
            # volatility, volume changes, etc.

            logger.debug("Symbol statistics calculation completed")

        except Exception as e:
            logger.error(f"Error calculating symbol statistics: {e}")

    @background_task(name="cleanup_symbol_data")
    def _cleanup_symbol_data(self) -> None:
        """Cleanup old symbol data."""
        try:
            logger.debug("Cleaning up symbol data...")

            # Remove inactive symbols older than 30 days
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=30)
            to_remove = []

            with self._lock:
                for symbol, symbol_info in self._symbols.items():
                    if (symbol_info.status == SymbolStatus.INACTIVE and
                        symbol_info.updated_at < cutoff_time):
                        to_remove.append(symbol)

                for symbol in to_remove:
                    self.remove_symbol(symbol)

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} inactive symbols")

        except Exception as e:
            logger.error(f"Error cleaning up symbol data: {e}")

    @property
    def is_running(self) -> bool:
        """Check if symbol manager is running."""
        return self._is_running


# Global symbol manager instance
_symbol_manager: Optional[SymbolManager] = None


def get_symbol_manager() -> SymbolManager:
    """Get the global symbol manager instance."""
    global _symbol_manager

    if _symbol_manager is None:
        _symbol_manager = SymbolManager()

    return _symbol_manager