"""
Historical data synchronization manager with gap filling capabilities.

This module provides comprehensive historical data synchronization, gap detection,
and intelligent backfill strategies for maintaining data continuity.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import threading
from enum import Enum

import MetaTrader5 as mt5

from core.config.settings import Settings, Mt5TimeFrame
from core.exceptions import (
    Mt5DataNotAvailableError,
    DataSynchronizationError,
    PerformanceError,
)
from core.logging import get_logger
from core.data.models import OHLCData, MarketEvent, MarketEventType, DataQuality
from core.data import get_buffer_manager, get_event_publisher
from core.tasks import get_task_manager
from database.services.symbols import SymbolService

logger = get_logger(__name__)


class SyncStatus(str, Enum):
    """Synchronization status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class GapType(str, Enum):
    """Types of data gaps."""
    MISSING_PERIOD = "missing_period"
    PARTIAL_DATA = "partial_data"
    QUALITY_ISSUE = "quality_issue"
    CONNECTION_LOSS = "connection_loss"


@dataclass
class DataGap:
    """Represents a gap in historical data."""
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    gap_type: GapType
    priority: int
    detected_at: datetime
    filled_at: Optional[datetime] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def duration_minutes(self) -> float:
        """Gap duration in minutes."""
        return (self.end_time - self.start_time).total_seconds() / 60

    @property
    def is_critical(self) -> bool:
        """Check if gap is critical (affects trading decisions)."""
        return self.duration_minutes > 60 or self.priority >= 8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'gap_type': self.gap_type.value,
            'priority': self.priority,
            'duration_minutes': self.duration_minutes,
            'is_critical': self.is_critical,
            'detected_at': self.detected_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'retry_count': self.retry_count,
            'metadata': self.metadata,
        }


@dataclass
class SyncTask:
    """Historical data synchronization task."""
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    priority: int
    task_type: str  # 'initial', 'gap_fill', 'update'
    created_at: datetime
    status: SyncStatus = SyncStatus.PENDING
    progress_percentage: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    @property
    def duration_hours(self) -> float:
        """Task duration in hours."""
        return (self.end_time - self.start_time).total_seconds() / 3600

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'priority': self.priority,
            'task_type': self.task_type,
            'status': self.status.value,
            'progress_percentage': self.progress_percentage,
            'duration_hours': self.duration_hours,
            'created_at': self.created_at.isoformat(),
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
        }


class GapDetector:
    """
    Gap detection system for identifying missing historical data.

    Features:
    - Real-time gap detection
    - Pattern-based gap identification
    - Quality-based gap assessment
    - Priority scoring
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # Detection parameters
        self.max_expected_gap_minutes = {
            'M1': 2, 'M3': 6, 'M5': 10, 'M15': 30,
            'M30': 60, 'H1': 120, 'H4': 480, 'D1': 1440
        }

        # Detected gaps
        self.detected_gaps: Dict[str, List[DataGap]] = defaultdict(list)
        self.gap_history: deque = deque(maxlen=1000)

        # Detection state
        self.last_data_times: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        self.detection_lock = threading.RLock()

    def detect_gaps(
        self,
        symbol: str,
        timeframe: str,
        recent_data: List[OHLCData]
    ) -> List[DataGap]:
        """
        Detect gaps in recent OHLC data.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            recent_data: Recent OHLC data points

        Returns:
            List of detected gaps
        """
        gaps = []

        if not recent_data or len(recent_data) < 2:
            return gaps

        try:
            # Sort data by timestamp
            sorted_data = sorted(recent_data, key=lambda x: x.timestamp)

            # Expected interval for timeframe
            expected_interval = self._get_timeframe_interval_minutes(timeframe)
            if not expected_interval:
                return gaps

            # Check for gaps between consecutive periods
            for i in range(1, len(sorted_data)):
                prev_data = sorted_data[i - 1]
                current_data = sorted_data[i]

                # Calculate actual interval
                actual_interval = (current_data.timestamp - prev_data.timestamp).total_seconds() / 60

                # Check if gap exists
                if actual_interval > expected_interval * 1.5:  # 50% tolerance
                    gap = DataGap(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=prev_data.timestamp + timedelta(minutes=expected_interval),
                        end_time=current_data.timestamp,
                        gap_type=GapType.MISSING_PERIOD,
                        priority=self._calculate_gap_priority(actual_interval, expected_interval),
                        detected_at=datetime.now(timezone.utc),
                        metadata={
                            'expected_interval_minutes': expected_interval,
                            'actual_interval_minutes': actual_interval,
                            'prev_timestamp': prev_data.timestamp.isoformat(),
                            'current_timestamp': current_data.timestamp.isoformat(),
                        }
                    )
                    gaps.append(gap)

            # Check for quality-based gaps
            quality_gaps = self._detect_quality_gaps(symbol, timeframe, sorted_data)
            gaps.extend(quality_gaps)

            # Store detected gaps
            with self.detection_lock:
                self.detected_gaps[f"{symbol}:{timeframe}"].extend(gaps)
                self.gap_history.extend(gaps)

            return gaps

        except Exception as e:
            logger.error(f"Error detecting gaps for {symbol} {timeframe}: {e}")
            return []

    def _detect_quality_gaps(
        self,
        symbol: str,
        timeframe: str,
        data: List[OHLCData]
    ) -> List[DataGap]:
        """Detect quality-based gaps (poor data quality periods)."""
        gaps = []

        try:
            # Find consecutive periods with poor quality
            poor_quality_start = None
            poor_quality_count = 0

            for ohlc in data:
                if ohlc.quality in [DataQuality.LOW, DataQuality.SUSPECT, DataQuality.INVALID]:
                    if poor_quality_start is None:
                        poor_quality_start = ohlc.timestamp
                    poor_quality_count += 1
                else:
                    # End of poor quality period
                    if poor_quality_start and poor_quality_count >= 3:  # 3+ consecutive poor quality periods
                        gap = DataGap(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_time=poor_quality_start,
                            end_time=ohlc.timestamp,
                            gap_type=GapType.QUALITY_ISSUE,
                            priority=min(poor_quality_count, 10),
                            detected_at=datetime.now(timezone.utc),
                            metadata={
                                'poor_quality_count': poor_quality_count,
                                'quality_issue': True,
                            }
                        )
                        gaps.append(gap)

                    poor_quality_start = None
                    poor_quality_count = 0

        except Exception as e:
            logger.error(f"Error detecting quality gaps: {e}")

        return gaps

    def _get_timeframe_interval_minutes(self, timeframe: str) -> Optional[int]:
        """Get expected interval in minutes for timeframe."""
        intervals = {
            'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4, 'M5': 5,
            'M6': 6, 'M10': 10, 'M12': 12, 'M15': 15, 'M20': 20,
            'M30': 30, 'H1': 60, 'H2': 120, 'H3': 180, 'H4': 240,
            'H6': 360, 'H8': 480, 'H12': 720, 'D1': 1440,
        }
        return intervals.get(timeframe)

    def _calculate_gap_priority(self, actual_interval: float, expected_interval: float) -> int:
        """Calculate gap priority based on severity."""
        ratio = actual_interval / expected_interval

        if ratio >= 10:
            return 10  # Critical
        elif ratio >= 5:
            return 8   # High
        elif ratio >= 3:
            return 6   # Medium
        elif ratio >= 2:
            return 4   # Low
        else:
            return 2   # Very Low

    def get_gaps_for_symbol(self, symbol: str, timeframe: str) -> List[DataGap]:
        """Get detected gaps for symbol and timeframe."""
        with self.detection_lock:
            key = f"{symbol}:{timeframe}"
            return list(self.detected_gaps.get(key, []))

    def clear_filled_gaps(self, symbol: str, timeframe: str) -> None:
        """Clear gaps that have been filled."""
        with self.detection_lock:
            key = f"{symbol}:{timeframe}"
            if key in self.detected_gaps:
                self.detected_gaps[key] = [
                    gap for gap in self.detected_gaps[key]
                    if gap.filled_at is None
                ]


class HistoricalDataFetcher:
    """
    Historical data fetcher with MT5 integration.

    Features:
    - Efficient batch data retrieval
    - Rate limiting and connection management
    - Error handling and retry logic
    - Data quality validation
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # Fetching parameters
        self.batch_size = 1000  # Records per batch
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.max_retries = 3

        # MT5 session management
        self.session_manager = None  # Will be injected

        # Fetching state
        self.fetch_lock = threading.RLock()

    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[OHLCData]:
        """
        Fetch historical OHLC data from MT5.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_time: Start time for data
            end_time: End time for data
            progress_callback: Optional progress callback

        Returns:
            List of OHLC data
        """
        all_data = []

        try:
            # Convert timeframe
            mt5_timeframe = self._convert_timeframe(timeframe)
            if not mt5_timeframe:
                raise ValueError(f"Invalid timeframe: {timeframe}")

            # Calculate total periods needed
            interval_minutes = self._get_timeframe_interval_minutes(timeframe)
            total_minutes = (end_time - start_time).total_seconds() / 60
            total_periods = int(total_minutes / interval_minutes) if interval_minutes else 0

            logger.debug(
                f"Fetching {total_periods} periods of {symbol} {timeframe} "
                f"from {start_time} to {end_time}"
            )

            # Fetch data in batches
            current_time = start_time
            fetched_periods = 0

            while current_time < end_time:
                # Calculate batch end time
                batch_end = min(
                    current_time + timedelta(minutes=self.batch_size * interval_minutes),
                    end_time
                )

                # Fetch batch
                batch_data = await self._fetch_batch(
                    symbol, mt5_timeframe, current_time, batch_end
                )

                if batch_data:
                    all_data.extend(batch_data)
                    fetched_periods += len(batch_data)

                    # Update progress
                    if progress_callback and total_periods > 0:
                        progress = min((fetched_periods / total_periods) * 100, 100)
                        progress_callback(progress)

                # Move to next batch
                current_time = batch_end

                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)

            # Sort by timestamp
            all_data.sort(key=lambda x: x.timestamp)

            logger.info(
                f"Fetched {len(all_data)} periods of historical data for {symbol} {timeframe}"
            )

            return all_data

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise DataSynchronizationError(f"Failed to fetch historical data: {e}")

    async def _fetch_batch(
        self,
        symbol: str,
        mt5_timeframe: int,
        start_time: datetime,
        end_time: datetime
    ) -> List[OHLCData]:
        """Fetch a single batch of data from MT5."""
        batch_data = []
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                with self.fetch_lock:
                    # Get rates from MT5
                    rates = mt5.copy_rates_range(
                        symbol,
                        mt5_timeframe,
                        start_time,
                        end_time
                    )

                    if rates is None or len(rates) == 0:
                        logger.debug(f"No data available for {symbol} from {start_time} to {end_time}")
                        return batch_data

                    # Convert to OHLCData objects
                    for rate in rates:
                        ohlc = OHLCData(
                            symbol=symbol,
                            timeframe=self._convert_mt5_timeframe_back(mt5_timeframe),
                            timestamp=datetime.fromtimestamp(rate['time'], tz=timezone.utc),
                            open=rate['open'],
                            high=rate['high'],
                            low=rate['low'],
                            close=rate['close'],
                            volume=int(rate['tick_volume']),
                            tick_count=int(rate.get('real_volume', 0)),
                            quality=DataQuality.HIGH,  # Assume high quality for historical data
                            is_complete=True,
                        )
                        batch_data.append(ohlc)

                return batch_data

            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"Error fetching batch (attempt {retry_count}/{self.max_retries}): {e}"
                )

                if retry_count < self.max_retries:
                    await asyncio.sleep(retry_count * 2)  # Exponential backoff
                else:
                    raise

        return batch_data

    def _convert_timeframe(self, timeframe: str) -> Optional[int]:
        """Convert string timeframe to MT5 constant."""
        mapping = {
            'M1': mt5.TIMEFRAME_M1, 'M2': mt5.TIMEFRAME_M2, 'M3': mt5.TIMEFRAME_M3,
            'M4': mt5.TIMEFRAME_M4, 'M5': mt5.TIMEFRAME_M5, 'M6': mt5.TIMEFRAME_M6,
            'M10': mt5.TIMEFRAME_M10, 'M12': mt5.TIMEFRAME_M12, 'M15': mt5.TIMEFRAME_M15,
            'M20': mt5.TIMEFRAME_M20, 'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
            'H2': mt5.TIMEFRAME_H2, 'H3': mt5.TIMEFRAME_H3, 'H4': mt5.TIMEFRAME_H4,
            'H6': mt5.TIMEFRAME_H6, 'H8': mt5.TIMEFRAME_H8, 'H12': mt5.TIMEFRAME_H12,
            'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1,
        }
        return mapping.get(timeframe)

    def _convert_mt5_timeframe_back(self, mt5_timeframe: int) -> str:
        """Convert MT5 timeframe constant back to string."""
        reverse_mapping = {
            mt5.TIMEFRAME_M1: 'M1', mt5.TIMEFRAME_M2: 'M2', mt5.TIMEFRAME_M3: 'M3',
            mt5.TIMEFRAME_M4: 'M4', mt5.TIMEFRAME_M5: 'M5', mt5.TIMEFRAME_M6: 'M6',
            mt5.TIMEFRAME_M10: 'M10', mt5.TIMEFRAME_M12: 'M12', mt5.TIMEFRAME_M15: 'M15',
            mt5.TIMEFRAME_M20: 'M20', mt5.TIMEFRAME_M30: 'M30', mt5.TIMEFRAME_H1: 'H1',
            mt5.TIMEFRAME_H2: 'H2', mt5.TIMEFRAME_H3: 'H3', mt5.TIMEFRAME_H4: 'H4',
            mt5.TIMEFRAME_H6: 'H6', mt5.TIMEFRAME_H8: 'H8', mt5.TIMEFRAME_H12: 'H12',
            mt5.TIMEFRAME_D1: 'D1', mt5.TIMEFRAME_W1: 'W1', mt5.TIMEFRAME_MN1: 'MN1',
        }
        return reverse_mapping.get(mt5_timeframe, 'M1')

    def _get_timeframe_interval_minutes(self, timeframe: str) -> int:
        """Get interval in minutes for timeframe."""
        intervals = {
            'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4, 'M5': 5,
            'M6': 6, 'M10': 10, 'M12': 12, 'M15': 15, 'M20': 20,
            'M30': 30, 'H1': 60, 'H2': 120, 'H3': 180, 'H4': 240,
            'H6': 360, 'H8': 480, 'H12': 720, 'D1': 1440,
        }
        return intervals.get(timeframe, 1)


class HistoricalSyncManager:
    """
    Historical data synchronization manager.

    Features:
    - Automated gap detection and filling
    - Priority-based synchronization
    - Progress tracking and reporting
    - Database integration
    - Performance optimization
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.buffer_manager = get_buffer_manager()
        self.event_publisher = get_event_publisher()
        self.task_manager = get_task_manager()

        # Components
        self.gap_detector = GapDetector(settings)
        self.data_fetcher = HistoricalDataFetcher(settings)
        self.symbol_service = SymbolService()

        # Synchronization state
        self.active_tasks: Dict[str, SyncTask] = {}
        self.completed_tasks: Dict[str, SyncTask] = {}
        self.sync_queue: List[SyncTask] = []

        # Configuration
        self.max_concurrent_syncs = 3
        self.sync_history_days = 30  # Days of history to maintain
        self.auto_sync_enabled = True
        self.gap_check_interval = 300  # 5 minutes

        # State management
        self.is_running = False
        self.sync_lock = threading.RLock()
        self.worker_tasks: List[asyncio.Task] = []

        logger.info("Historical sync manager initialized")

    async def start_sync_manager(self) -> None:
        """Start the historical synchronization manager."""
        if self.is_running:
            logger.warning("Sync manager already running")
            return

        try:
            self.is_running = True

            # Start worker tasks
            self.worker_tasks = [
                asyncio.create_task(self._sync_worker(), name="sync-worker"),
                asyncio.create_task(self._gap_detection_worker(), name="gap-detector"),
                asyncio.create_task(self._monitoring_worker(), name="sync-monitor"),
            ]

            logger.info("Historical sync manager started")

        except Exception as e:
            self.is_running = False
            logger.error(f"Failed to start sync manager: {e}")
            raise

    async def stop_sync_manager(self) -> None:
        """Stop the synchronization manager."""
        if not self.is_running:
            return

        try:
            self.is_running = False

            # Cancel worker tasks
            for task in self.worker_tasks:
                task.cancel()

            # Wait for tasks to complete
            if self.worker_tasks:
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)

            self.worker_tasks.clear()

            # Clear state
            self.active_tasks.clear()
            self.sync_queue.clear()

            logger.info("Historical sync manager stopped")

        except Exception as e:
            logger.error(f"Error stopping sync manager: {e}")

    async def sync_symbol_history(
        self,
        symbol: str,
        timeframes: List[str],
        days_back: int = 30,
        priority: int = 5
    ) -> List[str]:
        """
        Synchronize historical data for a symbol.

        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to sync
            days_back: Number of days to sync back
            priority: Task priority (1-10, higher = more priority)

        Returns:
            List of task IDs
        """
        task_ids = []

        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days_back)

            for timeframe in timeframes:
                task = SyncTask(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    priority=priority,
                    task_type='initial',
                    created_at=datetime.now(timezone.utc),
                )

                await self._queue_sync_task(task)
                task_ids.append(f"{symbol}:{timeframe}")

            logger.info(
                f"Queued {len(task_ids)} sync tasks for {symbol} "
                f"({days_back} days, timeframes: {timeframes})"
            )

            return task_ids

        except Exception as e:
            logger.error(f"Error queueing sync tasks for {symbol}: {e}")
            return []

    async def fill_detected_gaps(self, symbol: str, timeframe: str) -> int:
        """
        Fill detected gaps for symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            Number of gaps queued for filling
        """
        gaps = self.gap_detector.get_gaps_for_symbol(symbol, timeframe)
        if not gaps:
            return 0

        # Sort gaps by priority (highest first)
        gaps.sort(key=lambda g: g.priority, reverse=True)

        queued_count = 0

        for gap in gaps:
            if gap.filled_at is None:  # Not yet filled
                task = SyncTask(
                    symbol=gap.symbol,
                    timeframe=gap.timeframe,
                    start_time=gap.start_time,
                    end_time=gap.end_time,
                    priority=gap.priority,
                    task_type='gap_fill',
                    created_at=datetime.now(timezone.utc),
                )

                await self._queue_sync_task(task)
                queued_count += 1

        logger.info(f"Queued {queued_count} gap fill tasks for {symbol} {timeframe}")
        return queued_count

    async def _queue_sync_task(self, task: SyncTask) -> None:
        """Queue a synchronization task."""
        with self.sync_lock:
            # Check for existing task
            task_key = f"{task.symbol}:{task.timeframe}"
            if task_key in self.active_tasks:
                logger.debug(f"Sync task already active for {task_key}")
                return

            # Add to queue
            self.sync_queue.append(task)
            self.sync_queue.sort(key=lambda t: t.priority, reverse=True)

            logger.debug(f"Queued sync task: {task_key} (priority: {task.priority})")

    async def _sync_worker(self) -> None:
        """Main synchronization worker."""
        logger.debug("Started sync worker")

        while self.is_running:
            try:
                # Check for available tasks
                if not self._can_start_new_sync():
                    await asyncio.sleep(1.0)
                    continue

                # Get next task
                task = self._get_next_task()
                if not task:
                    await asyncio.sleep(1.0)
                    continue

                # Start task
                await self._execute_sync_task(task)

            except Exception as e:
                logger.error(f"Error in sync worker: {e}")
                await asyncio.sleep(5.0)

    async def _gap_detection_worker(self) -> None:
        """Gap detection worker."""
        logger.debug("Started gap detection worker")

        while self.is_running:
            try:
                if self.auto_sync_enabled:
                    await self._check_for_gaps()

                await asyncio.sleep(self.gap_check_interval)

            except Exception as e:
                logger.error(f"Error in gap detection worker: {e}")
                await asyncio.sleep(60.0)

    async def _monitoring_worker(self) -> None:
        """Monitoring and reporting worker."""
        while self.is_running:
            try:
                await self._report_sync_status()
                await asyncio.sleep(60.0)  # Report every minute

            except Exception as e:
                logger.error(f"Error in monitoring worker: {e}")
                await asyncio.sleep(60.0)

    def _can_start_new_sync(self) -> bool:
        """Check if new sync task can be started."""
        with self.sync_lock:
            return len(self.active_tasks) < self.max_concurrent_syncs

    def _get_next_task(self) -> Optional[SyncTask]:
        """Get next task from queue."""
        with self.sync_lock:
            if self.sync_queue:
                return self.sync_queue.pop(0)
            return None

    async def _execute_sync_task(self, task: SyncTask) -> None:
        """Execute a synchronization task."""
        task_key = f"{task.symbol}:{task.timeframe}"

        try:
            with self.sync_lock:
                self.active_tasks[task_key] = task
                task.status = SyncStatus.RUNNING

            logger.info(f"Starting sync task: {task_key}")

            # Progress callback
            def update_progress(progress: float):
                task.progress_percentage = progress

            # Fetch historical data
            historical_data = await self.data_fetcher.fetch_historical_data(
                symbol=task.symbol,
                timeframe=task.timeframe,
                start_time=task.start_time,
                end_time=task.end_time,
                progress_callback=update_progress
            )

            # Store data in buffer
            if historical_data:
                buffer = self.buffer_manager.get_ohlc_buffer(task.symbol, task.timeframe)
                if buffer:
                    for ohlc_data in historical_data:
                        buffer.add_bar(ohlc_data)

            # Update task status
            task.status = SyncStatus.COMPLETED
            task.progress_percentage = 100.0

            # Publish completion event
            await self._publish_sync_event(task, 'completed')

            logger.info(
                f"Completed sync task: {task_key} "
                f"({len(historical_data)} records synced)"
            )

        except Exception as e:
            # Handle task failure
            task.status = SyncStatus.FAILED
            task.error_message = str(e)
            task.retry_count += 1

            logger.error(f"Sync task failed: {task_key} - {e}")

            # Retry if possible
            if task.retry_count < task.max_retries:
                task.status = SyncStatus.PENDING
                await self._queue_sync_task(task)

            await self._publish_sync_event(task, 'failed')

        finally:
            # Move to completed tasks
            with self.sync_lock:
                self.active_tasks.pop(task_key, None)
                self.completed_tasks[task_key] = task

    async def _check_for_gaps(self) -> None:
        """Check for gaps in recent data."""
        try:
            # Get active symbols from buffer manager
            buffer_stats = self.buffer_manager.get_buffer_stats()
            symbols = buffer_stats.get('ohlc_buffers', {}).keys()

            for symbol in symbols:
                timeframes = buffer_stats['ohlc_buffers'][symbol].keys()

                for timeframe in timeframes:
                    # Get recent data
                    buffer = self.buffer_manager.get_ohlc_buffer(symbol, timeframe)
                    if not buffer:
                        continue

                    recent_data = buffer.get_complete_bars(count=50)
                    if len(recent_data) < 10:  # Need minimum data for gap detection
                        continue

                    # Detect gaps
                    gaps = self.gap_detector.detect_gaps(symbol, timeframe, recent_data)

                    # Queue gap fill tasks for critical gaps
                    for gap in gaps:
                        if gap.is_critical:
                            await self.fill_detected_gaps(symbol, timeframe)
                            break  # Only queue once per symbol/timeframe

        except Exception as e:
            logger.error(f"Error checking for gaps: {e}")

    async def _report_sync_status(self) -> None:
        """Report synchronization status."""
        with self.sync_lock:
            active_count = len(self.active_tasks)
            queue_count = len(self.sync_queue)
            completed_count = len(self.completed_tasks)

            if active_count > 0 or queue_count > 0:
                logger.info(
                    f"Sync Status: {active_count} active, "
                    f"{queue_count} queued, {completed_count} completed"
                )

    async def _publish_sync_event(self, task: SyncTask, event_type: str) -> None:
        """Publish synchronization event."""
        try:
            event = MarketEvent(
                event_id=f"sync_{task.symbol}_{task.timeframe}_{int(time.time())}",
                event_type=MarketEventType.DATA_GAP_DETECTED,
                symbol=task.symbol,
                data={
                    'sync_event': event_type,
                    'task': task.to_dict(),
                }
            )

            await self.event_publisher.publish(event)

        except Exception as e:
            logger.debug(f"Error publishing sync event: {e}")

    def get_sync_status(self) -> Dict[str, Any]:
        """Get comprehensive synchronization status."""
        with self.sync_lock:
            # Calculate metrics
            total_gaps = sum(len(gaps) for gaps in self.gap_detector.detected_gaps.values())
            filled_gaps = sum(
                len([g for g in gaps if g.filled_at is not None])
                for gaps in self.gap_detector.detected_gaps.values()
            )

            return {
                'is_running': self.is_running,
                'active_tasks': len(self.active_tasks),
                'queued_tasks': len(self.sync_queue),
                'completed_tasks': len(self.completed_tasks),
                'max_concurrent_syncs': self.max_concurrent_syncs,
                'auto_sync_enabled': self.auto_sync_enabled,
                'gap_detection': {
                    'total_gaps_detected': total_gaps,
                    'gaps_filled': filled_gaps,
                    'gap_fill_rate': (filled_gaps / max(total_gaps, 1)) * 100,
                },
                'active_task_details': [task.to_dict() for task in self.active_tasks.values()],
                'recent_gaps': [gap.to_dict() for gap in list(self.gap_detector.gap_history)[-10:]],
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on sync manager."""
        health = {
            'status': 'healthy' if self.is_running else 'stopped',
            'is_running': self.is_running,
            'worker_tasks_running': len([t for t in self.worker_tasks if not t.done()]),
            'sync_performance': 'good',
        }

        # Check for stuck tasks
        if self.is_running:
            stuck_tasks = 0
            current_time = datetime.now(timezone.utc)

            for task in self.active_tasks.values():
                if task.status == SyncStatus.RUNNING:
                    runtime = (current_time - task.created_at).total_seconds()
                    if runtime > 3600:  # 1 hour timeout
                        stuck_tasks += 1

            if stuck_tasks > 0:
                health['sync_performance'] = 'degraded'
                health['status'] = 'degraded'
                health['stuck_tasks'] = stuck_tasks

        return health


# Global sync manager instance
_historical_sync_manager: Optional[HistoricalSyncManager] = None


def get_historical_sync_manager() -> HistoricalSyncManager:
    """Get the global historical sync manager instance."""
    global _historical_sync_manager

    if _historical_sync_manager is None:
        from core.config.settings import Settings
        settings = Settings()
        _historical_sync_manager = HistoricalSyncManager(settings)

    return _historical_sync_manager