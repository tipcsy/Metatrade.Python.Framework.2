"""
Symbol and instrument services for the MetaTrader Python Framework.

This module provides business logic for symbol management including
symbol groups, symbols, and trading sessions.
"""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, List, Optional, Type, Union

from sqlalchemy import and_, or_
from sqlalchemy.orm import joinedload

from src.core.exceptions import ValidationError, NotFoundError, BusinessLogicError
from src.core.logging import get_logger
from src.database.models.symbols import Symbol, SymbolGroup, SymbolSession
from src.database.schemas.symbols import (
    SymbolGroupCreateSchema,
    SymbolGroupUpdateSchema,
    SymbolGroupResponseSchema,
    SymbolCreateSchema,
    SymbolUpdateSchema,
    SymbolResponseSchema,
    SymbolSessionCreateSchema,
    SymbolSessionResponseSchema,
)
from .base import CachedService

logger = get_logger(__name__)


class SymbolGroupService(CachedService[SymbolGroup, SymbolGroupCreateSchema, SymbolGroupUpdateSchema, SymbolGroupResponseSchema]):
    """Service for symbol group management."""

    def __init__(self):
        super().__init__(SymbolGroup, cache_ttl=3600)  # Cache for 1 hour

    @property
    def create_schema(self) -> Type[SymbolGroupCreateSchema]:
        return SymbolGroupCreateSchema

    @property
    def update_schema(self) -> Type[SymbolGroupUpdateSchema]:
        return SymbolGroupUpdateSchema

    @property
    def response_schema(self) -> Type[SymbolGroupResponseSchema]:
        return SymbolGroupResponseSchema

    def _apply_business_rules(self, instance: SymbolGroup, operation: str) -> None:
        """Apply business rules for symbol groups."""
        if operation in ('create', 'update'):
            # Validate market hours
            if (instance.market_open_hour is not None and
                instance.market_close_hour is not None):
                if instance.market_open_hour == instance.market_close_hour:
                    raise BusinessLogicError("Market open and close hours cannot be the same")

            # Ensure unique name per group type
            existing = self.get_by_name_and_type(instance.name, instance.group_type)
            if existing and existing.id != instance.id:
                raise BusinessLogicError(f"Symbol group '{instance.name}' already exists for type '{instance.group_type}'")

    def get_by_name_and_type(self, name: str, group_type: str) -> Optional[SymbolGroup]:
        """
        Get symbol group by name and type.

        Args:
            name: Group name
            group_type: Group type

        Returns:
            Symbol group or None
        """
        with self._db_manager.get_session() as session:
            return session.query(SymbolGroup).filter(
                and_(
                    SymbolGroup.name == name,
                    SymbolGroup.group_type == group_type,
                    SymbolGroup.deleted_at.is_(None)
                )
            ).first()

    def get_by_type(self, group_type: str, active_only: bool = True) -> List[SymbolGroup]:
        """
        Get symbol groups by type.

        Args:
            group_type: Group type
            active_only: Whether to return only active groups

        Returns:
            List of symbol groups
        """
        filters = {'group_type': group_type}
        if active_only:
            filters['is_active'] = True

        return self.get_all(filters=filters, order_by='display_order')

    def get_all_types(self) -> List[str]:
        """
        Get all available group types.

        Returns:
            List of group types
        """
        with self._db_manager.get_session() as session:
            result = session.query(SymbolGroup.group_type).distinct().all()
            return [row[0] for row in result]

    def reorder_groups(self, group_orders: Dict[str, int]) -> bool:
        """
        Reorder symbol groups.

        Args:
            group_orders: Mapping of group IDs to display orders

        Returns:
            True if successful
        """
        try:
            with self._db_manager.get_session() as session:
                for group_id, display_order in group_orders.items():
                    session.query(SymbolGroup).filter(
                        SymbolGroup.id == group_id
                    ).update({'display_order': display_order})

                session.commit()

            # Invalidate cache
            self.invalidate_all_cache()
            logger.info(f"Reordered {len(group_orders)} symbol groups")
            return True

        except Exception as e:
            logger.error(f"Error reordering symbol groups: {e}")
            return False


class SymbolService(CachedService[Symbol, SymbolCreateSchema, SymbolUpdateSchema, SymbolResponseSchema]):
    """Service for symbol management."""

    def __init__(self):
        super().__init__(Symbol, cache_ttl=1800)  # Cache for 30 minutes

    @property
    def create_schema(self) -> Type[SymbolCreateSchema]:
        return SymbolCreateSchema

    @property
    def update_schema(self) -> Type[SymbolUpdateSchema]:
        return SymbolUpdateSchema

    @property
    def response_schema(self) -> Type[SymbolResponseSchema]:
        return SymbolResponseSchema

    def _apply_business_rules(self, instance: Symbol, operation: str) -> None:
        """Apply business rules for symbols."""
        if operation in ('create', 'update'):
            # Validate symbol uniqueness
            if operation == 'create':
                existing = self.get_by_symbol(instance.symbol)
                if existing:
                    raise BusinessLogicError(f"Symbol '{instance.symbol}' already exists")

            # Validate lot size constraints
            if instance.min_lot >= instance.max_lot:
                raise BusinessLogicError("Minimum lot size must be less than maximum lot size")

            if instance.lot_step > instance.min_lot:
                raise BusinessLogicError("Lot step cannot be greater than minimum lot size")

            # Validate margin requirements
            if (instance.margin_initial is not None and
                instance.margin_maintenance is not None):
                if instance.margin_maintenance > instance.margin_initial:
                    raise BusinessLogicError("Maintenance margin cannot exceed initial margin")

            # Validate currency pair
            if instance.base_currency == instance.quote_currency:
                raise BusinessLogicError("Base and quote currencies must be different")

    def get_by_symbol(self, symbol: str) -> Optional[Symbol]:
        """
        Get symbol by symbol name.

        Args:
            symbol: Symbol name

        Returns:
            Symbol or None
        """
        cache_key = f"symbol:{symbol}"

        # Check cache first
        if self.cache_enabled:
            cached_data = self._cache_manager.get(self.cache_type, cache_key)
            if cached_data:
                return Symbol(**cached_data)

        with self._db_manager.get_session() as session:
            instance = session.query(Symbol).filter(
                and_(
                    Symbol.symbol == symbol.upper(),
                    Symbol.deleted_at.is_(None)
                )
            ).first()

            if instance and self.cache_enabled:
                self._cache_manager.set(
                    self.cache_type,
                    cache_key,
                    self._model_to_dict(instance),
                    self.cache_ttl
                )

            return instance

    def get_by_market(self, market: str, active_only: bool = True) -> List[Symbol]:
        """
        Get symbols by market.

        Args:
            market: Market identifier
            active_only: Whether to return only tradeable symbols

        Returns:
            List of symbols
        """
        filters = {'market': market.upper()}
        if active_only:
            filters['is_tradeable'] = True

        return self.get_all(filters=filters, order_by='symbol')

    def get_by_currency_pair(self, base_currency: str, quote_currency: str) -> List[Symbol]:
        """
        Get symbols by currency pair.

        Args:
            base_currency: Base currency
            quote_currency: Quote currency

        Returns:
            List of symbols
        """
        filters = {
            'base_currency': base_currency.upper(),
            'quote_currency': quote_currency.upper(),
            'is_tradeable': True
        }
        return self.get_all(filters=filters, order_by='symbol')

    def get_by_group(self, group_id: str, active_only: bool = True) -> List[Symbol]:
        """
        Get symbols by group.

        Args:
            group_id: Symbol group ID
            active_only: Whether to return only tradeable symbols

        Returns:
            List of symbols
        """
        filters = {'symbol_group_id': group_id}
        if active_only:
            filters['is_tradeable'] = True

        return self.get_all(filters=filters, order_by='symbol')

    def search_symbols(
        self,
        query: str,
        limit: int = 50,
        markets: Optional[List[str]] = None,
        active_only: bool = True
    ) -> List[Symbol]:
        """
        Search symbols by name or description.

        Args:
            query: Search query
            limit: Maximum results
            markets: Filter by markets
            active_only: Whether to return only tradeable symbols

        Returns:
            List of matching symbols
        """
        with self._db_manager.get_session() as session:
            search_query = session.query(Symbol).filter(
                or_(
                    Symbol.symbol.ilike(f"%{query.upper()}%"),
                    Symbol.name.ilike(f"%{query}%")
                )
            )

            if active_only:
                search_query = search_query.filter(Symbol.is_tradeable == True)

            if markets:
                search_query = search_query.filter(Symbol.market.in_(markets))

            search_query = search_query.filter(Symbol.deleted_at.is_(None))
            search_query = search_query.order_by(Symbol.symbol).limit(limit)

            return search_query.all()

    def update_quote(
        self,
        symbol_id: str,
        bid: Decimal,
        ask: Decimal,
        volume: Optional[Decimal] = None
    ) -> bool:
        """
        Update symbol quote information.

        Args:
            symbol_id: Symbol ID
            bid: Bid price
            ask: Ask price
            volume: Trade volume

        Returns:
            True if successful

        Raises:
            ValidationError: If prices are invalid
            NotFoundError: If symbol not found
        """
        if bid <= 0 or ask <= 0:
            raise ValidationError("Bid and ask prices must be positive")

        if ask < bid:
            raise ValidationError("Ask price cannot be less than bid price")

        try:
            with self._db_manager.get_session() as session:
                symbol = session.query(Symbol).filter(Symbol.id == symbol_id).first()
                if not symbol:
                    raise NotFoundError(f"Symbol with ID {symbol_id} not found")

                symbol.update_quote(bid, ask, volume)
                session.commit()

                # Update cache
                if self.cache_enabled:
                    cache_key = self._get_cache_key(symbol_id)
                    self._cache_manager.set(
                        self.cache_type,
                        cache_key,
                        self._model_to_dict(symbol),
                        self.cache_ttl
                    )

                    # Also update symbol name cache
                    symbol_cache_key = f"symbol:{symbol.symbol}"
                    self._cache_manager.set(
                        self.cache_type,
                        symbol_cache_key,
                        self._model_to_dict(symbol),
                        self.cache_ttl
                    )

                logger.debug(f"Updated quote for symbol {symbol.symbol}: {bid}/{ask}")
                return True

        except (ValidationError, NotFoundError):
            raise
        except Exception as e:
            logger.error(f"Error updating quote for symbol {symbol_id}: {e}")
            return False

    def validate_lot_size(self, symbol_id: str, lot_size: Decimal) -> bool:
        """
        Validate lot size for symbol.

        Args:
            symbol_id: Symbol ID
            lot_size: Lot size to validate

        Returns:
            True if valid

        Raises:
            NotFoundError: If symbol not found
            ValidationError: If lot size is invalid
        """
        symbol = self.get_by_id(symbol_id)
        if not symbol:
            raise NotFoundError(f"Symbol with ID {symbol_id} not found")

        if not symbol.validate_lot_size(lot_size):
            raise ValidationError(
                f"Invalid lot size {lot_size} for symbol {symbol.symbol}. "
                f"Must be between {symbol.min_lot} and {symbol.max_lot} "
                f"in steps of {symbol.lot_step}"
            )

        return True

    def calculate_pip_value(
        self,
        symbol_id: str,
        lot_size: Decimal,
        account_currency: str = "USD"
    ) -> Decimal:
        """
        Calculate pip value for symbol and lot size.

        Args:
            symbol_id: Symbol ID
            lot_size: Lot size
            account_currency: Account currency

        Returns:
            Pip value in account currency

        Raises:
            NotFoundError: If symbol not found
        """
        symbol = self.get_by_id(symbol_id)
        if not symbol:
            raise NotFoundError(f"Symbol with ID {symbol_id} not found")

        pip_value = symbol.calculate_pip_value(lot_size)

        # TODO: Implement currency conversion if account currency differs
        # For now, return the pip value as-is
        return pip_value

    def get_tradeable_symbols(
        self,
        market: Optional[str] = None,
        group_id: Optional[str] = None
    ) -> List[Symbol]:
        """
        Get all tradeable symbols with optional filtering.

        Args:
            market: Filter by market
            group_id: Filter by group

        Returns:
            List of tradeable symbols
        """
        filters = {'is_tradeable': True, 'trade_mode': 'FULL'}

        if market:
            filters['market'] = market.upper()
        if group_id:
            filters['symbol_group_id'] = group_id

        return self.get_all(filters=filters, order_by='symbol')

    def get_market_summary(self) -> Dict[str, Dict[str, Union[int, List[str]]]]:
        """
        Get market summary statistics.

        Returns:
            Dictionary with market statistics
        """
        with self._db_manager.get_session() as session:
            from sqlalchemy import func

            # Get symbol counts by market
            market_counts = session.query(
                Symbol.market,
                func.count(Symbol.id).label('total'),
                func.sum(
                    func.case([(Symbol.is_tradeable == True, 1)], else_=0)
                ).label('tradeable')
            ).filter(
                Symbol.deleted_at.is_(None)
            ).group_by(Symbol.market).all()

            # Get unique currencies
            currencies = session.query(Symbol.base_currency).distinct().all()
            currencies.extend(session.query(Symbol.quote_currency).distinct().all())
            unique_currencies = sorted(set(row[0] for row in currencies))

            summary = {
                'markets': {},
                'total_symbols': 0,
                'total_tradeable': 0,
                'currencies': unique_currencies
            }

            for market, total, tradeable in market_counts:
                summary['markets'][market] = {
                    'total': total,
                    'tradeable': tradeable or 0
                }
                summary['total_symbols'] += total
                summary['total_tradeable'] += tradeable or 0

            return summary

    def create_or_update(self, session, symbol_info) -> Symbol:
        """
        Create or update symbol from SymbolInfo model.

        Args:
            session: Database session
            symbol_info: SymbolInfo instance

        Returns:
            Symbol database model
        """
        # Check if symbol exists
        existing = session.query(Symbol).filter(
            and_(
                Symbol.symbol == symbol_info.symbol.upper(),
                Symbol.deleted_at.is_(None)
            )
        ).first()

        if existing:
            # Update existing symbol
            existing.description = symbol_info.description
            existing.is_tradeable = symbol_info.is_tradable
            existing.is_visible = symbol_info.is_visible
            existing.status = symbol_info.status.value
            existing.symbol_type = symbol_info.symbol_type.value
            existing.updated_at = symbol_info.updated_at

            if symbol_info.bid is not None:
                existing.bid = symbol_info.bid
            if symbol_info.ask is not None:
                existing.ask = symbol_info.ask
            if symbol_info.last_price is not None:
                existing.last_price = symbol_info.last_price

            session.commit()
            return existing
        else:
            # Create new symbol
            new_symbol = Symbol(
                symbol=symbol_info.symbol.upper(),
                name=symbol_info.description,
                description=symbol_info.description,
                symbol_type=symbol_info.symbol_type.value,
                market=getattr(symbol_info, 'exchange', 'UNKNOWN') or 'UNKNOWN',
                base_currency=symbol_info.base_currency or 'USD',
                quote_currency=symbol_info.quote_currency or 'USD',
                is_tradeable=symbol_info.is_tradable,
                is_visible=symbol_info.is_visible,
                status=symbol_info.status.value,
                bid=symbol_info.bid,
                ask=symbol_info.ask,
                last_price=symbol_info.last_price,
                created_at=symbol_info.created_at,
                updated_at=symbol_info.updated_at
            )

            session.add(new_symbol)
            session.commit()
            return new_symbol

    def delete_by_symbol(self, session, symbol: str) -> bool:
        """
        Delete symbol by symbol name.

        Args:
            session: Database session
            symbol: Symbol name

        Returns:
            True if deleted successfully
        """
        try:
            existing = session.query(Symbol).filter(
                and_(
                    Symbol.symbol == symbol.upper(),
                    Symbol.deleted_at.is_(None)
                )
            ).first()

            if existing:
                existing.deleted_at = datetime.now(timezone.utc)
                session.commit()
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting symbol {symbol}: {e}")
            session.rollback()
            return False


class SymbolSessionService(CachedService[SymbolSession, SymbolSessionCreateSchema, SymbolSessionCreateSchema, SymbolSessionResponseSchema]):
    """Service for symbol trading session management."""

    def __init__(self):
        super().__init__(SymbolSession, cache_ttl=3600)  # Cache for 1 hour

    @property
    def create_schema(self) -> Type[SymbolSessionCreateSchema]:
        return SymbolSessionCreateSchema

    @property
    def update_schema(self) -> Type[SymbolSessionCreateSchema]:
        return SymbolSessionCreateSchema

    @property
    def response_schema(self) -> Type[SymbolSessionResponseSchema]:
        return SymbolSessionResponseSchema

    def _apply_business_rules(self, instance: SymbolSession, operation: str) -> None:
        """Apply business rules for symbol sessions."""
        if operation in ('create', 'update'):
            # Validate session times
            if instance.session_start >= instance.session_end:
                raise BusinessLogicError("Session start time must be before end time")

            # Validate time range (0-10080 minutes in a week)
            if not (0 <= instance.session_start <= 10080):
                raise BusinessLogicError("Session start time must be between 0 and 10080 minutes")

            if not (0 <= instance.session_end <= 10080):
                raise BusinessLogicError("Session end time must be between 0 and 10080 minutes")

            # Check for overlapping sessions for the same symbol
            existing_sessions = self.get_by_symbol(instance.symbol_id)
            for session in existing_sessions:
                if session.id == instance.id:
                    continue

                # Check for overlap
                if (instance.session_start < session.session_end and
                    instance.session_end > session.session_start):
                    raise BusinessLogicError(
                        f"Session overlaps with existing session '{session.session_name}'"
                    )

    def get_by_symbol(self, symbol_id: str, active_only: bool = True) -> List[SymbolSession]:
        """
        Get sessions by symbol.

        Args:
            symbol_id: Symbol ID
            active_only: Whether to return only active sessions

        Returns:
            List of symbol sessions
        """
        filters = {'symbol_id': symbol_id}
        if active_only:
            filters['is_active'] = True

        return self.get_all(filters=filters, order_by='session_start')

    def get_active_sessions(self, current_time_minutes: int) -> List[SymbolSession]:
        """
        Get currently active sessions.

        Args:
            current_time_minutes: Current time in minutes from Sunday 00:00 UTC

        Returns:
            List of active sessions
        """
        with self._db_manager.get_session() as session:
            return session.query(SymbolSession).filter(
                and_(
                    SymbolSession.is_active == True,
                    SymbolSession.session_start <= current_time_minutes,
                    SymbolSession.session_end > current_time_minutes,
                    SymbolSession.deleted_at.is_(None)
                )
            ).all()

    def is_symbol_tradeable_now(self, symbol_id: str, current_time_minutes: int) -> bool:
        """
        Check if symbol is tradeable at current time.

        Args:
            symbol_id: Symbol ID
            current_time_minutes: Current time in minutes from Sunday 00:00 UTC

        Returns:
            True if tradeable
        """
        sessions = self.get_by_symbol(symbol_id, active_only=True)

        for session in sessions:
            if (session.session_start <= current_time_minutes < session.session_end and
                session.trade_mode in ('FULL', 'LONG_ONLY', 'SHORT_ONLY')):
                return True

        return False

    def get_next_session(self, symbol_id: str, current_time_minutes: int) -> Optional[SymbolSession]:
        """
        Get next trading session for symbol.

        Args:
            symbol_id: Symbol ID
            current_time_minutes: Current time in minutes from Sunday 00:00 UTC

        Returns:
            Next session or None
        """
        sessions = self.get_by_symbol(symbol_id, active_only=True)

        # Find next session
        next_session = None
        min_wait_time = float('inf')

        for session in sessions:
            if session.session_start > current_time_minutes:
                wait_time = session.session_start - current_time_minutes
                if wait_time < min_wait_time:
                    min_wait_time = wait_time
                    next_session = session

        return next_session

    def create_default_sessions(self, symbol_id: str) -> List[SymbolSession]:
        """
        Create default trading sessions for a symbol.

        Args:
            symbol_id: Symbol ID

        Returns:
            List of created sessions
        """
        # Default Forex sessions (in minutes from Sunday 00:00 UTC)
        default_sessions = [
            {
                'symbol_id': symbol_id,
                'session_name': 'SYDNEY',
                'session_start': 1380,  # Monday 23:00 UTC (Sydney open)
                'session_end': 1980,   # Tuesday 09:00 UTC (Sydney close)
                'is_active': True,
                'trade_mode': 'FULL'
            },
            {
                'symbol_id': symbol_id,
                'session_name': 'TOKYO',
                'session_start': 1440,  # Monday 00:00 UTC (Tokyo open)
                'session_end': 1980,   # Tuesday 09:00 UTC (Tokyo close)
                'is_active': True,
                'trade_mode': 'FULL'
            },
            {
                'symbol_id': symbol_id,
                'session_name': 'LONDON',
                'session_start': 2280,  # Tuesday 08:00 UTC (London open)
                'session_end': 2760,   # Tuesday 17:00 UTC (London close)
                'is_active': True,
                'trade_mode': 'FULL'
            },
            {
                'symbol_id': symbol_id,
                'session_name': 'NEW_YORK',
                'session_start': 2520,  # Tuesday 13:00 UTC (NY open)
                'session_end': 3000,   # Tuesday 22:00 UTC (NY close)
                'is_active': True,
                'trade_mode': 'FULL'
            }
        ]

        created_sessions = []
        for session_data in default_sessions:
            try:
                session = self.create(session_data)
                created_sessions.append(session)
            except Exception as e:
                logger.error(f"Failed to create default session {session_data['session_name']}: {e}")

        logger.info(f"Created {len(created_sessions)} default sessions for symbol {symbol_id}")
        return created_sessions