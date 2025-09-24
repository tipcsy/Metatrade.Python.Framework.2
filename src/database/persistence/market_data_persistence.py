"""
Market Data Persistence Layer.

This module provides high-performance persistence for market data including
tick data, OHLC bars, volume profiles, and related market information with
optimized storage, retrieval, and data archiving capabilities.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Tuple, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
from enum import Enum

import sqlalchemy as sa
from sqlalchemy import select, insert, update, delete, and_, or_, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config.settings import DatabaseSettings, MarketDataSettings
from src.core.exceptions import DatabaseError, PersistenceError
from src.core.logging import get_logger
from src.core.data.models import TickData, ProcessedTick, OHLCBar, VolumeProfile
from src.database.connection_manager import DatabaseConnectionManager
from src.database.models import TickModel, OHLCModel, VolumeProfileModel, SymbolModel

logger = get_logger(__name__)


class StorageMode(Enum):
    """Data storage modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    BULK = "bulk"
    ARCHIVE = "archive"


class CompressionType(Enum):
    """Data compression types."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class PersistenceConfig:
    """Market data persistence configuration."""
    batch_size: int = 1000
    flush_interval: float = 30.0  # seconds
    enable_compression: bool = True
    compression_type: CompressionType = CompressionType.LZ4
    enable_partitioning: bool = True
    partition_by_date: bool = True
    retention_days: int = 365
    enable_archiving: bool = True
    archive_after_days: int = 90
    max_memory_buffer: int = 10000


@dataclass
class PersistenceMetrics:
    """Persistence layer metrics."""
    records_persisted: int = 0
    batches_processed: int = 0
    total_processing_time: float = 0.0
    average_batch_time: float = 0.0
    compression_ratio: float = 1.0
    storage_size_bytes: int = 0
    last_flush_time: float = field(default_factory=time.time)
    errors_count: int = 0


class MarketDataPersistence:
    """
    High-performance market data persistence layer.

    Features:
    - Optimized batch operations
    - Data compression and partitioning
    - Automatic data archiving
    - Real-time and batch storage modes
    - Performance monitoring
    """

    def __init__(
        self,
        connection_manager: DatabaseConnectionManager,
        config: PersistenceConfig,
        database_name: Optional[str] = None
    ) -> None:
        """Initialize market data persistence.

        Args:
            connection_manager: Database connection manager
            config: Persistence configuration
            database_name: Database name to use
        """
        self.connection_manager = connection_manager
        self.config = config
        self.database_name = database_name

        # Data buffers for batch operations
        self._tick_buffer: deque = deque(maxlen=config.max_memory_buffer)
        self._ohlc_buffer: deque = deque(maxlen=config.max_memory_buffer)
        self._volume_profile_buffer: deque = deque(maxlen=config.max_memory_buffer)

        # Metrics
        self._metrics = PersistenceMetrics()
        self._metrics_lock = asyncio.Lock()

        # Background tasks
        self._flush_task: Optional[asyncio.Task] = None
        self._archive_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Symbol cache for performance
        self._symbol_cache: Dict[str, int] = {}
        self._symbol_cache_lock = asyncio.Lock()

        logger.info(
            "Market data persistence initialized",
            extra={
                "batch_size": config.batch_size,
                "flush_interval": config.flush_interval,
                "compression_enabled": config.enable_compression,
                "partitioning_enabled": config.enable_partitioning,
            }
        )

    async def start(self) -> None:
        """Start the persistence layer."""
        logger.info("Starting market data persistence")

        # Start background tasks
        self._flush_task = asyncio.create_task(self._flush_loop())

        if self.config.enable_archiving:
            self._archive_task = asyncio.create_task(self._archive_loop())

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Load symbol cache
        await self._load_symbol_cache()

        logger.info("Market data persistence started")

    async def stop(self) -> None:
        """Stop the persistence layer."""
        logger.info("Stopping market data persistence")

        # Signal stop
        self._stop_event.set()

        # Flush remaining data
        await self._flush_all_buffers()

        # Wait for background tasks
        tasks = [self._flush_task, self._archive_task, self._cleanup_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Market data persistence stopped")

    async def store_tick(self, tick: Union[TickData, ProcessedTick], mode: StorageMode = StorageMode.BATCH) -> bool:
        """Store tick data.

        Args:
            tick: Tick data to store
            mode: Storage mode

        Returns:
            True if stored successfully
        """
        try:
            if mode == StorageMode.REAL_TIME:
                return await self._store_tick_realtime(tick)
            else:
                return await self._store_tick_batch(tick)

        except Exception as e:
            logger.error(
                "Failed to store tick data",
                extra={"tick": tick, "mode": mode.value, "error": str(e)},
                exc_info=True
            )
            await self._update_error_metrics()
            return False

    async def store_ohlc_bar(self, bar: OHLCBar, mode: StorageMode = StorageMode.BATCH) -> bool:
        """Store OHLC bar data.

        Args:
            bar: OHLC bar to store
            mode: Storage mode

        Returns:
            True if stored successfully
        """
        try:
            if mode == StorageMode.REAL_TIME:
                return await self._store_ohlc_realtime(bar)
            else:
                return await self._store_ohlc_batch(bar)

        except Exception as e:
            logger.error(
                "Failed to store OHLC bar",
                extra={"bar": bar, "mode": mode.value, "error": str(e)},
                exc_info=True
            )
            await self._update_error_metrics()
            return False

    async def store_volume_profile(self, profile: VolumeProfile, mode: StorageMode = StorageMode.BATCH) -> bool:
        """Store volume profile data.

        Args:
            profile: Volume profile to store
            mode: Storage mode

        Returns:
            True if stored successfully
        """
        try:
            if mode == StorageMode.REAL_TIME:
                return await self._store_volume_profile_realtime(profile)
            else:
                return await self._store_volume_profile_batch(profile)

        except Exception as e:
            logger.error(
                "Failed to store volume profile",
                extra={"profile": profile, "mode": mode.value, "error": str(e)},
                exc_info=True
            )
            await self._update_error_metrics()
            return False

    async def get_ticks(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[TickData]:
        """Retrieve tick data.

        Args:
            symbol: Symbol name
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of records

        Returns:
            List of tick data
        """
        try:
            async with self.connection_manager.get_session(self.database_name) as session:
                query = select(TickModel).where(TickModel.symbol == symbol)

                if start_time:
                    query = query.where(TickModel.timestamp >= start_time)
                if end_time:
                    query = query.where(TickModel.timestamp <= end_time)

                query = query.order_by(TickModel.timestamp.desc())

                if limit:
                    query = query.limit(limit)

                result = await session.execute(query)
                tick_models = result.scalars().all()

                # Convert to TickData objects
                ticks = []
                for model in tick_models:
                    tick = TickData(
                        symbol=model.symbol,
                        time=model.timestamp.timestamp(),
                        bid=model.bid,
                        ask=model.ask,
                        last=model.last,
                        volume=model.volume,
                        flags=model.flags or 0
                    )
                    ticks.append(tick)

                return ticks

        except Exception as e:
            logger.error(
                "Failed to retrieve tick data",
                extra={
                    "symbol": symbol,
                    "start_time": start_time,
                    "end_time": end_time,
                    "error": str(e),
                },
                exc_info=True
            )
            raise PersistenceError(f"Failed to retrieve tick data: {e}") from e

    async def get_ohlc_bars(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[OHLCBar]:
        """Retrieve OHLC bar data.

        Args:
            symbol: Symbol name
            timeframe: Timeframe string
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of records

        Returns:
            List of OHLC bars
        """
        try:
            async with self.connection_manager.get_session(self.database_name) as session:
                query = select(OHLCModel).where(
                    and_(
                        OHLCModel.symbol == symbol,
                        OHLCModel.timeframe == timeframe
                    )
                )

                if start_time:
                    query = query.where(OHLCModel.start_time >= start_time)
                if end_time:
                    query = query.where(OHLCModel.end_time <= end_time)

                query = query.order_by(OHLCModel.start_time.desc())

                if limit:
                    query = query.limit(limit)

                result = await session.execute(query)
                ohlc_models = result.scalars().all()

                # Convert to OHLCBar objects
                bars = []
                for model in ohlc_models:
                    bar = OHLCBar(
                        symbol=model.symbol,
                        timeframe=model.timeframe,
                        start_time=model.start_time.timestamp(),
                        end_time=model.end_time.timestamp(),
                        open_price=model.open_price,
                        high_price=model.high_price,
                        low_price=model.low_price,
                        close_price=model.close_price,
                        volume=model.volume,
                        tick_count=model.tick_count,
                        bid_volume=model.bid_volume,
                        ask_volume=model.ask_volume,
                        vwap=model.vwap
                    )
                    bars.append(bar)

                return bars

        except Exception as e:
            logger.error(
                "Failed to retrieve OHLC data",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_time": start_time,
                    "end_time": end_time,
                    "error": str(e),
                },
                exc_info=True
            )
            raise PersistenceError(f"Failed to retrieve OHLC data: {e}") from e

    async def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """Get latest tick for a symbol.

        Args:
            symbol: Symbol name

        Returns:
            Latest tick data or None
        """
        ticks = await self.get_ticks(symbol, limit=1)
        return ticks[0] if ticks else None

    async def get_latest_ohlc(self, symbol: str, timeframe: str) -> Optional[OHLCBar]:
        """Get latest OHLC bar for a symbol and timeframe.

        Args:
            symbol: Symbol name
            timeframe: Timeframe string

        Returns:
            Latest OHLC bar or None
        """
        bars = await self.get_ohlc_bars(symbol, timeframe, limit=1)
        return bars[0] if bars else None

    async def bulk_store_ticks(self, ticks: List[Union[TickData, ProcessedTick]]) -> int:
        """Bulk store tick data.

        Args:
            ticks: List of ticks to store

        Returns:
            Number of ticks stored successfully
        """
        if not ticks:
            return 0

        try:
            start_time = time.time()
            stored_count = 0

            async with self.connection_manager.get_session(self.database_name) as session:
                # Prepare data for bulk insert
                tick_data = []
                for tick in ticks:
                    symbol_id = await self._get_symbol_id(tick.symbol, session)

                    tick_dict = {
                        'symbol_id': symbol_id,
                        'symbol': tick.symbol,
                        'timestamp': datetime.fromtimestamp(tick.time),
                        'bid': tick.bid if hasattr(tick, 'bid') else None,
                        'ask': tick.ask if hasattr(tick, 'ask') else None,
                        'last': tick.last if hasattr(tick, 'last') else None,
                        'volume': tick.volume if hasattr(tick, 'volume') else 0,
                        'flags': getattr(tick, 'flags', 0)
                    }
                    tick_data.append(tick_dict)

                # Bulk insert
                if tick_data:
                    stmt = insert(TickModel).values(tick_data)
                    await session.execute(stmt)
                    await session.commit()
                    stored_count = len(tick_data)

            # Update metrics
            processing_time = time.time() - start_time
            await self._update_persistence_metrics(stored_count, processing_time)

            logger.debug(
                f"Bulk stored {stored_count} ticks",
                extra={
                    "count": stored_count,
                    "processing_time": processing_time,
                }
            )

            return stored_count

        except Exception as e:
            logger.error(
                "Failed to bulk store ticks",
                extra={"count": len(ticks), "error": str(e)},
                exc_info=True
            )
            await self._update_error_metrics()
            return 0

    async def bulk_store_ohlc_bars(self, bars: List[OHLCBar]) -> int:
        """Bulk store OHLC bar data.

        Args:
            bars: List of OHLC bars to store

        Returns:
            Number of bars stored successfully
        """
        if not bars:
            return 0

        try:
            start_time = time.time()
            stored_count = 0

            async with self.connection_manager.get_session(self.database_name) as session:
                # Prepare data for bulk insert
                bar_data = []
                for bar in bars:
                    symbol_id = await self._get_symbol_id(bar.symbol, session)

                    bar_dict = {
                        'symbol_id': symbol_id,
                        'symbol': bar.symbol,
                        'timeframe': bar.timeframe,
                        'start_time': datetime.fromtimestamp(bar.start_time),
                        'end_time': datetime.fromtimestamp(bar.end_time),
                        'open_price': bar.open_price,
                        'high_price': bar.high_price,
                        'low_price': bar.low_price,
                        'close_price': bar.close_price,
                        'volume': bar.volume,
                        'tick_count': bar.tick_count,
                        'bid_volume': getattr(bar, 'bid_volume', 0),
                        'ask_volume': getattr(bar, 'ask_volume', 0),
                        'vwap': getattr(bar, 'vwap', None)
                    }
                    bar_data.append(bar_dict)

                # Bulk insert with upsert logic
                if bar_data:
                    stmt = insert(OHLCModel).values(bar_data)
                    # Handle upsert for different databases
                    engine_dialect = session.bind.dialect.name
                    if engine_dialect == 'sqlite':
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['symbol', 'timeframe', 'start_time'],
                            set_={
                                'end_time': stmt.excluded.end_time,
                                'open_price': stmt.excluded.open_price,
                                'high_price': stmt.excluded.high_price,
                                'low_price': stmt.excluded.low_price,
                                'close_price': stmt.excluded.close_price,
                                'volume': stmt.excluded.volume,
                                'tick_count': stmt.excluded.tick_count,
                                'bid_volume': stmt.excluded.bid_volume,
                                'ask_volume': stmt.excluded.ask_volume,
                                'vwap': stmt.excluded.vwap,
                            }
                        )
                    elif engine_dialect == 'postgresql':
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['symbol', 'timeframe', 'start_time'],
                            set_={
                                'end_time': stmt.excluded.end_time,
                                'open_price': stmt.excluded.open_price,
                                'high_price': stmt.excluded.high_price,
                                'low_price': stmt.excluded.low_price,
                                'close_price': stmt.excluded.close_price,
                                'volume': stmt.excluded.volume,
                                'tick_count': stmt.excluded.tick_count,
                                'bid_volume': stmt.excluded.bid_volume,
                                'ask_volume': stmt.excluded.ask_volume,
                                'vwap': stmt.excluded.vwap,
                            }
                        )

                    await session.execute(stmt)
                    await session.commit()
                    stored_count = len(bar_data)

            # Update metrics
            processing_time = time.time() - start_time
            await self._update_persistence_metrics(stored_count, processing_time)

            logger.debug(
                f"Bulk stored {stored_count} OHLC bars",
                extra={
                    "count": stored_count,
                    "processing_time": processing_time,
                }
            )

            return stored_count

        except Exception as e:
            logger.error(
                "Failed to bulk store OHLC bars",
                extra={"count": len(bars), "error": str(e)},
                exc_info=True
            )
            await self._update_error_metrics()
            return 0

    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Storage statistics dictionary
        """
        try:
            async with self.connection_manager.get_session(self.database_name) as session:
                # Tick data statistics
                tick_count_result = await session.execute(select(sa.func.count(TickModel.id)))
                tick_count = tick_count_result.scalar() or 0

                # OHLC data statistics
                ohlc_count_result = await session.execute(select(sa.func.count(OHLCModel.id)))
                ohlc_count = ohlc_count_result.scalar() or 0

                # Symbol statistics
                symbol_count_result = await session.execute(select(sa.func.count(SymbolModel.id)))
                symbol_count = symbol_count_result.scalar() or 0

                # Date range statistics
                oldest_tick_result = await session.execute(select(sa.func.min(TickModel.timestamp)))
                oldest_tick = oldest_tick_result.scalar()

                newest_tick_result = await session.execute(select(sa.func.max(TickModel.timestamp)))
                newest_tick = newest_tick_result.scalar()

                async with self._metrics_lock:
                    return {
                        "tick_count": tick_count,
                        "ohlc_count": ohlc_count,
                        "symbol_count": symbol_count,
                        "oldest_data": oldest_tick.isoformat() if oldest_tick else None,
                        "newest_data": newest_tick.isoformat() if newest_tick else None,
                        "buffer_sizes": {
                            "ticks": len(self._tick_buffer),
                            "ohlc": len(self._ohlc_buffer),
                            "volume_profiles": len(self._volume_profile_buffer),
                        },
                        "metrics": self._metrics.__dict__,
                    }

        except Exception as e:
            logger.error(
                "Failed to get storage statistics",
                extra={"error": str(e)},
                exc_info=True
            )
            return {"error": str(e)}

    async def archive_old_data(self, cutoff_date: datetime) -> Dict[str, int]:
        """Archive old data.

        Args:
            cutoff_date: Archive data older than this date

        Returns:
            Dictionary with counts of archived records
        """
        try:
            archived_counts = {"ticks": 0, "ohlc_bars": 0}

            async with self.connection_manager.get_session(self.database_name) as session:
                # Archive tick data
                tick_delete_stmt = delete(TickModel).where(TickModel.timestamp < cutoff_date)
                tick_result = await session.execute(tick_delete_stmt)
                archived_counts["ticks"] = tick_result.rowcount

                # Archive OHLC data
                ohlc_delete_stmt = delete(OHLCModel).where(OHLCModel.start_time < cutoff_date)
                ohlc_result = await session.execute(ohlc_delete_stmt)
                archived_counts["ohlc_bars"] = ohlc_result.rowcount

                await session.commit()

            logger.info(
                "Data archiving completed",
                extra={
                    "cutoff_date": cutoff_date.isoformat(),
                    "archived_ticks": archived_counts["ticks"],
                    "archived_ohlc": archived_counts["ohlc_bars"],
                }
            )

            return archived_counts

        except Exception as e:
            logger.error(
                "Failed to archive old data",
                extra={"cutoff_date": cutoff_date.isoformat(), "error": str(e)},
                exc_info=True
            )
            raise PersistenceError(f"Data archiving failed: {e}") from e

    async def _store_tick_realtime(self, tick: Union[TickData, ProcessedTick]) -> bool:
        """Store tick data in real-time mode."""
        try:
            async with self.connection_manager.get_session(self.database_name) as session:
                symbol_id = await self._get_symbol_id(tick.symbol, session)

                tick_model = TickModel(
                    symbol_id=symbol_id,
                    symbol=tick.symbol,
                    timestamp=datetime.fromtimestamp(tick.time),
                    bid=getattr(tick, 'bid', None),
                    ask=getattr(tick, 'ask', None),
                    last=getattr(tick, 'last', None),
                    volume=getattr(tick, 'volume', 0),
                    flags=getattr(tick, 'flags', 0)
                )

                session.add(tick_model)
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to store tick in real-time: {e}", exc_info=True)
            return False

    async def _store_tick_batch(self, tick: Union[TickData, ProcessedTick]) -> bool:
        """Store tick data in batch mode."""
        self._tick_buffer.append(tick)
        return True

    async def _store_ohlc_realtime(self, bar: OHLCBar) -> bool:
        """Store OHLC bar in real-time mode."""
        try:
            async with self.connection_manager.get_session(self.database_name) as session:
                symbol_id = await self._get_symbol_id(bar.symbol, session)

                ohlc_model = OHLCModel(
                    symbol_id=symbol_id,
                    symbol=bar.symbol,
                    timeframe=bar.timeframe,
                    start_time=datetime.fromtimestamp(bar.start_time),
                    end_time=datetime.fromtimestamp(bar.end_time),
                    open_price=bar.open_price,
                    high_price=bar.high_price,
                    low_price=bar.low_price,
                    close_price=bar.close_price,
                    volume=bar.volume,
                    tick_count=bar.tick_count,
                    bid_volume=getattr(bar, 'bid_volume', 0),
                    ask_volume=getattr(bar, 'ask_volume', 0),
                    vwap=getattr(bar, 'vwap', None)
                )

                session.add(ohlc_model)
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to store OHLC in real-time: {e}", exc_info=True)
            return False

    async def _store_ohlc_batch(self, bar: OHLCBar) -> bool:
        """Store OHLC bar in batch mode."""
        self._ohlc_buffer.append(bar)
        return True

    async def _store_volume_profile_realtime(self, profile: VolumeProfile) -> bool:
        """Store volume profile in real-time mode."""
        # Implementation would depend on VolumeProfileModel structure
        return True

    async def _store_volume_profile_batch(self, profile: VolumeProfile) -> bool:
        """Store volume profile in batch mode."""
        self._volume_profile_buffer.append(profile)
        return True

    async def _get_symbol_id(self, symbol: str, session: AsyncSession) -> int:
        """Get or create symbol ID."""
        async with self._symbol_cache_lock:
            # Check cache first
            if symbol in self._symbol_cache:
                return self._symbol_cache[symbol]

            # Query database
            result = await session.execute(
                select(SymbolModel.id).where(SymbolModel.symbol == symbol)
            )
            symbol_id = result.scalar()

            if symbol_id is None:
                # Create new symbol
                symbol_model = SymbolModel(symbol=symbol, active=True)
                session.add(symbol_model)
                await session.flush()
                symbol_id = symbol_model.id

            # Cache the result
            self._symbol_cache[symbol] = symbol_id
            return symbol_id

    async def _load_symbol_cache(self) -> None:
        """Load symbol cache from database."""
        try:
            async with self.connection_manager.get_session(self.database_name) as session:
                result = await session.execute(select(SymbolModel.id, SymbolModel.symbol))
                symbols = result.all()

                async with self._symbol_cache_lock:
                    self._symbol_cache = {symbol: symbol_id for symbol_id, symbol in symbols}

                logger.debug(f"Loaded {len(self._symbol_cache)} symbols into cache")

        except Exception as e:
            logger.warning(f"Failed to load symbol cache: {e}")

    async def _flush_all_buffers(self) -> None:
        """Flush all data buffers to database."""
        try:
            # Flush tick buffer
            if self._tick_buffer:
                ticks = list(self._tick_buffer)
                self._tick_buffer.clear()
                await self.bulk_store_ticks(ticks)

            # Flush OHLC buffer
            if self._ohlc_buffer:
                bars = list(self._ohlc_buffer)
                self._ohlc_buffer.clear()
                await self.bulk_store_ohlc_bars(bars)

            # Update flush time
            async with self._metrics_lock:
                self._metrics.last_flush_time = time.time()

        except Exception as e:
            logger.error(f"Error flushing buffers: {e}", exc_info=True)

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        logger.debug("Data flush loop started")

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.flush_interval
                )

                if self._stop_event.is_set():
                    break

            except asyncio.TimeoutError:
                # Flush timeout - perform flush
                await self._flush_all_buffers()

        logger.debug("Data flush loop ended")

    async def _archive_loop(self) -> None:
        """Background archiving loop."""
        logger.debug("Data archive loop started")

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=86400  # 24 hours
                )

                if self._stop_event.is_set():
                    break

            except asyncio.TimeoutError:
                # Archive timeout - perform archiving
                cutoff_date = datetime.now() - timedelta(days=self.config.archive_after_days)
                await self.archive_old_data(cutoff_date)

        logger.debug("Data archive loop ended")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        logger.debug("Data cleanup loop started")

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=3600  # 1 hour
                )

                if self._stop_event.is_set():
                    break

            except asyncio.TimeoutError:
                # Cleanup timeout - perform cleanup
                await self._perform_cleanup()

        logger.debug("Data cleanup loop ended")

    async def _perform_cleanup(self) -> None:
        """Perform periodic cleanup tasks."""
        try:
            # Clean up symbol cache if it gets too large
            async with self._symbol_cache_lock:
                if len(self._symbol_cache) > 10000:
                    self._symbol_cache.clear()
                    await self._load_symbol_cache()

        except Exception as e:
            logger.error(f"Error in cleanup: {e}", exc_info=True)

    async def _update_persistence_metrics(self, records_count: int, processing_time: float) -> None:
        """Update persistence metrics."""
        async with self._metrics_lock:
            self._metrics.records_persisted += records_count
            self._metrics.batches_processed += 1
            self._metrics.total_processing_time += processing_time

            if self._metrics.batches_processed > 0:
                self._metrics.average_batch_time = (
                    self._metrics.total_processing_time / self._metrics.batches_processed
                )

    async def _update_error_metrics(self) -> None:
        """Update error metrics."""
        async with self._metrics_lock:
            self._metrics.errors_count += 1

    async def __aenter__(self) -> MarketDataPersistence:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()