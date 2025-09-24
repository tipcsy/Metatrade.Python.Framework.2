"""
Data backfill processing for historical data synchronization.

This module provides comprehensive backfill capabilities to fill data gaps
with intelligent strategy selection and progress tracking.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Dict, Optional, Any, AsyncGenerator, Callable

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.data.models import TickData, OHLCData
from src.mt5.connection.manager import get_connection_manager
from src.database import get_session
from src.database.models import TickData as DBTickData, OHLCData as DBOHLCData
from .detector import DataGap, GapType

logger = get_logger(__name__)
settings = get_settings()


class BackfillStrategy(Enum):
    """Backfill strategies."""
    MT5_HISTORICAL = "mt5_historical"      # Use MT5 historical data
    INTERPOLATION = "interpolation"        # Interpolate missing values
    EXTERNAL_SOURCE = "external_source"    # External data provider
    SKIP = "skip"                         # Skip backfill for this gap


class BackfillResult(BaseModel):
    """Result of backfill operation."""

    gap_id: str = Field(description="Gap identifier")
    symbol: str = Field(description="Trading symbol")
    strategy_used: BackfillStrategy = Field(description="Strategy used for backfill")
    start_time: datetime = Field(description="Backfill start time")
    end_time: datetime = Field(description="Backfill end time")
    points_requested: int = Field(description="Number of points requested")
    points_retrieved: int = Field(description="Number of points actually retrieved")
    points_saved: int = Field(description="Number of points saved to database")
    success: bool = Field(description="Whether backfill was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    duration_seconds: float = Field(description="Backfill duration in seconds")
    data_quality_score: float = Field(default=1.0, description="Quality score 0-1")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def completion_rate(self) -> float:
        """Get completion rate as percentage."""
        if self.points_requested == 0:
            return 100.0
        return (self.points_retrieved / self.points_requested) * 100

    @property
    def save_rate(self) -> float:
        """Get save rate as percentage."""
        if self.points_retrieved == 0:
            return 100.0
        return (self.points_saved / self.points_retrieved) * 100


class BackfillProcessor:
    """
    Processes data backfill operations to fill historical data gaps.

    Provides intelligent strategy selection, parallel processing,
    and comprehensive progress tracking.
    """

    def __init__(self):
        """Initialize backfill processor."""
        self.connection_manager = get_connection_manager()

        self._backfill_stats = {
            "total_gaps_processed": 0,
            "successful_backfills": 0,
            "failed_backfills": 0,
            "points_retrieved": 0,
            "points_saved": 0,
            "total_duration": 0.0
        }

        # Strategy configuration
        self._strategy_config = {
            BackfillStrategy.MT5_HISTORICAL: {
                "enabled": True,
                "priority": 1,
                "max_points_per_request": 10000,
                "retry_attempts": 3,
                "timeout_seconds": 30
            },
            BackfillStrategy.INTERPOLATION: {
                "enabled": True,
                "priority": 2,
                "max_gap_minutes": 60,  # Only for small gaps
                "confidence_threshold": 0.8
            },
            BackfillStrategy.EXTERNAL_SOURCE: {
                "enabled": False,  # Disabled by default
                "priority": 3,
                "api_key": None,
                "base_url": None
            }
        }

        # Progress tracking
        self._active_backfills: Dict[str, Dict[str, Any]] = {}
        self._progress_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        logger.info("Backfill processor initialized")

    async def process_gap(self, gap: DataGap) -> BackfillResult:
        """
        Process a single data gap with backfill.

        Args:
            gap: Data gap to process

        Returns:
            BackfillResult with processing outcome
        """
        gap_id = f"{gap.symbol}_{gap.start_time.isoformat()}_{gap.end_time.isoformat()}"
        start_time = datetime.now(timezone.utc)

        try:
            # Select appropriate strategy
            strategy = self._select_strategy(gap)

            logger.info(f"Processing gap {gap_id} with strategy {strategy.value}")

            # Track active backfill
            self._active_backfills[gap_id] = {
                "gap": gap,
                "strategy": strategy,
                "start_time": start_time,
                "status": "processing",
                "progress": 0.0
            }

            # Process based on strategy
            if strategy == BackfillStrategy.MT5_HISTORICAL:
                result = await self._backfill_from_mt5(gap, gap_id)
            elif strategy == BackfillStrategy.INTERPOLATION:
                result = await self._backfill_with_interpolation(gap, gap_id)
            elif strategy == BackfillStrategy.EXTERNAL_SOURCE:
                result = await self._backfill_from_external(gap, gap_id)
            else:
                result = BackfillResult(
                    gap_id=gap_id,
                    symbol=gap.symbol,
                    strategy_used=strategy,
                    start_time=gap.start_time,
                    end_time=gap.end_time,
                    points_requested=gap.expected_points,
                    points_retrieved=0,
                    points_saved=0,
                    success=False,
                    error_message=f"Strategy {strategy.value} not implemented",
                    duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds()
                )

            # Update statistics
            self._update_stats(result)

            # Clean up tracking
            self._active_backfills.pop(gap_id, None)

            logger.info(f"Completed gap {gap_id}: {result.completion_rate:.1f}% success")
            return result

        except Exception as e:
            error_msg = f"Error processing gap {gap_id}: {e}"
            logger.error(error_msg)

            # Clean up tracking
            self._active_backfills.pop(gap_id, None)

            return BackfillResult(
                gap_id=gap_id,
                symbol=gap.symbol,
                strategy_used=BackfillStrategy.SKIP,
                start_time=gap.start_time,
                end_time=gap.end_time,
                points_requested=gap.expected_points,
                points_retrieved=0,
                points_saved=0,
                success=False,
                error_message=error_msg,
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds()
            )

    async def process_gaps_batch(
        self,
        gaps: List[DataGap],
        max_concurrent: int = 5,
        priority_order: bool = True
    ) -> List[BackfillResult]:
        """
        Process multiple gaps concurrently.

        Args:
            gaps: List of gaps to process
            max_concurrent: Maximum concurrent backfill operations
            priority_order: Whether to process by priority (critical first)

        Returns:
            List of backfill results
        """
        if not gaps:
            return []

        # Sort gaps by priority if requested
        if priority_order:
            gaps_sorted = sorted(gaps, key=lambda g: (
                0 if g.severity == "critical" else
                1 if g.severity == "high" else
                2 if g.severity == "medium" else 3,
                -g.expected_points  # More missing points = higher priority
            ))
        else:
            gaps_sorted = gaps

        logger.info(f"Processing {len(gaps_sorted)} gaps with {max_concurrent} concurrent operations")

        # Process gaps in batches
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(gap):
            async with semaphore:
                return await self.process_gap(gap)

        # Create tasks for all gaps
        tasks = [process_with_semaphore(gap) for gap in gaps_sorted]

        # Execute with progress reporting
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)

            # Report overall progress
            progress = (i + 1) / len(tasks) * 100
            logger.info(f"Batch progress: {progress:.1f}% ({i + 1}/{len(tasks)})")

        logger.info(f"Completed batch processing: {len(results)} results")
        return results

    def get_active_backfills(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active backfill operations."""
        return self._active_backfills.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get backfill processor statistics."""
        stats = self._backfill_stats.copy()

        if stats["total_gaps_processed"] > 0:
            stats["success_rate"] = (stats["successful_backfills"] / stats["total_gaps_processed"]) * 100
            stats["average_points_per_gap"] = stats["points_retrieved"] / stats["total_gaps_processed"]
            stats["average_duration_per_gap"] = stats["total_duration"] / stats["total_gaps_processed"]

        stats["active_backfills"] = len(self._active_backfills)
        return stats

    def add_progress_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add progress callback function."""
        self._progress_callbacks.append(callback)

    def _select_strategy(self, gap: DataGap) -> BackfillStrategy:
        """Select appropriate backfill strategy for gap."""
        # Skip market closure gaps
        if gap.gap_type in [GapType.WEEKEND, GapType.MARKET_CLOSED, GapType.HOLIDAY]:
            return BackfillStrategy.SKIP

        # Prefer MT5 historical for most gaps
        if (self._strategy_config[BackfillStrategy.MT5_HISTORICAL]["enabled"] and
            gap.gap_duration_minutes <= 24 * 60 * 7):  # Within a week
            return BackfillStrategy.MT5_HISTORICAL

        # Use interpolation for small gaps
        if (self._strategy_config[BackfillStrategy.INTERPOLATION]["enabled"] and
            gap.gap_duration_minutes <= self._strategy_config[BackfillStrategy.INTERPOLATION]["max_gap_minutes"] and
            gap.expected_points <= 100):
            return BackfillStrategy.INTERPOLATION

        # External source as fallback
        if self._strategy_config[BackfillStrategy.EXTERNAL_SOURCE]["enabled"]:
            return BackfillStrategy.EXTERNAL_SOURCE

        return BackfillStrategy.SKIP

    async def _backfill_from_mt5(self, gap: DataGap, gap_id: str) -> BackfillResult:
        """Backfill gap using MT5 historical data."""
        start_time = datetime.now(timezone.utc)
        points_retrieved = 0
        points_saved = 0
        error_message = None

        try:
            # Update progress
            self._update_progress(gap_id, "connecting", 10.0)

            # Ensure MT5 connection
            if not await self.connection_manager.ensure_connection():
                raise Exception("MT5 connection failed")

            # Determine data type and retrieve
            if gap.gap_type == GapType.MISSING_TICK:
                data_points = await self._retrieve_mt5_ticks(gap)
            else:  # MISSING_OHLC
                timeframe = gap.metadata.get("timeframe", "M1")
                data_points = await self._retrieve_mt5_ohlc(gap, timeframe)

            points_retrieved = len(data_points)

            # Update progress
            self._update_progress(gap_id, "saving", 70.0)

            # Save to database
            if data_points:
                points_saved = await self._save_data_points(data_points, gap.gap_type)

            success = points_saved > 0

            # Update progress
            self._update_progress(gap_id, "completed", 100.0)

        except Exception as e:
            error_message = str(e)
            success = False
            logger.error(f"MT5 backfill failed for gap {gap_id}: {e}")

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        return BackfillResult(
            gap_id=gap_id,
            symbol=gap.symbol,
            strategy_used=BackfillStrategy.MT5_HISTORICAL,
            start_time=gap.start_time,
            end_time=gap.end_time,
            points_requested=gap.expected_points,
            points_retrieved=points_retrieved,
            points_saved=points_saved,
            success=success,
            error_message=error_message,
            duration_seconds=duration,
            data_quality_score=0.9 if success else 0.0
        )

    async def _backfill_with_interpolation(self, gap: DataGap, gap_id: str) -> BackfillResult:
        """Backfill gap using data interpolation."""
        start_time = datetime.now(timezone.utc)
        points_generated = 0
        points_saved = 0
        error_message = None

        try:
            self._update_progress(gap_id, "interpolating", 20.0)

            # Get surrounding data points for interpolation
            before_data, after_data = await self._get_surrounding_data(gap)

            if not before_data or not after_data:
                raise Exception("Insufficient surrounding data for interpolation")

            # Generate interpolated points
            interpolated_points = self._interpolate_data_points(gap, before_data, after_data)
            points_generated = len(interpolated_points)

            self._update_progress(gap_id, "saving", 80.0)

            # Save interpolated data
            if interpolated_points:
                points_saved = await self._save_data_points(interpolated_points, gap.gap_type)

            success = points_saved > 0
            self._update_progress(gap_id, "completed", 100.0)

        except Exception as e:
            error_message = str(e)
            success = False
            logger.error(f"Interpolation backfill failed for gap {gap_id}: {e}")

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        return BackfillResult(
            gap_id=gap_id,
            symbol=gap.symbol,
            strategy_used=BackfillStrategy.INTERPOLATION,
            start_time=gap.start_time,
            end_time=gap.end_time,
            points_requested=gap.expected_points,
            points_retrieved=points_generated,
            points_saved=points_saved,
            success=success,
            error_message=error_message,
            duration_seconds=duration,
            data_quality_score=0.6 if success else 0.0  # Lower quality for interpolated data
        )

    async def _backfill_from_external(self, gap: DataGap, gap_id: str) -> BackfillResult:
        """Backfill gap using external data source."""
        # Placeholder for external data source implementation
        return BackfillResult(
            gap_id=gap_id,
            symbol=gap.symbol,
            strategy_used=BackfillStrategy.EXTERNAL_SOURCE,
            start_time=gap.start_time,
            end_time=gap.end_time,
            points_requested=gap.expected_points,
            points_retrieved=0,
            points_saved=0,
            success=False,
            error_message="External source not implemented",
            duration_seconds=0.0
        )

    async def _retrieve_mt5_ticks(self, gap: DataGap) -> List[TickData]:
        """Retrieve tick data from MT5."""
        # This would use the MT5 connection to get tick data
        # Placeholder implementation
        return []

    async def _retrieve_mt5_ohlc(self, gap: DataGap, timeframe: str) -> List[OHLCData]:
        """Retrieve OHLC data from MT5."""
        # This would use the MT5 connection to get OHLC data
        # Placeholder implementation
        return []

    async def _get_surrounding_data(self, gap: DataGap) -> tuple[List[Any], List[Any]]:
        """Get data points before and after gap for interpolation."""
        # This would query the database for surrounding data points
        # Placeholder implementation
        return [], []

    def _interpolate_data_points(self, gap: DataGap, before_data: List[Any], after_data: List[Any]) -> List[Any]:
        """Generate interpolated data points."""
        # This would implement linear or more sophisticated interpolation
        # Placeholder implementation
        return []

    async def _save_data_points(self, data_points: List[Any], gap_type: GapType) -> int:
        """Save data points to database."""
        saved_count = 0

        try:
            with get_session() as session:
                for point in data_points:
                    if gap_type == GapType.MISSING_TICK:
                        db_tick = DBTickData(
                            symbol=point.symbol,
                            timestamp=point.timestamp,
                            bid=point.bid,
                            ask=point.ask,
                            volume=point.volume
                        )
                        session.add(db_tick)
                    else:  # OHLC
                        db_ohlc = DBOHLCData(
                            symbol=point.symbol,
                            timeframe=point.timeframe,
                            bar_timestamp=point.timestamp,
                            open_price=point.open,
                            high_price=point.high,
                            low_price=point.low,
                            close_price=point.close,
                            volume=point.volume
                        )
                        session.add(db_ohlc)

                session.commit()
                saved_count = len(data_points)

        except Exception as e:
            logger.error(f"Error saving data points: {e}")
            saved_count = 0

        return saved_count

    def _update_progress(self, gap_id: str, status: str, progress: float) -> None:
        """Update backfill progress."""
        if gap_id in self._active_backfills:
            self._active_backfills[gap_id].update({
                "status": status,
                "progress": progress,
                "last_update": datetime.now(timezone.utc)
            })

            # Notify callbacks
            for callback in self._progress_callbacks:
                try:
                    callback(gap_id, self._active_backfills[gap_id])
                except Exception as e:
                    logger.error(f"Progress callback failed: {e}")

    def _update_stats(self, result: BackfillResult) -> None:
        """Update backfill statistics."""
        self._backfill_stats["total_gaps_processed"] += 1
        self._backfill_stats["points_retrieved"] += result.points_retrieved
        self._backfill_stats["points_saved"] += result.points_saved
        self._backfill_stats["total_duration"] += result.duration_seconds

        if result.success:
            self._backfill_stats["successful_backfills"] += 1
        else:
            self._backfill_stats["failed_backfills"] += 1