"""
Historical data synchronization manager.

This module provides comprehensive historical data synchronization
with gap detection, backfill processing, and quality validation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Dict, Optional, Any, Callable

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.tasks import background_task, scheduled_task
from .detector import GapDetector, DataGap
from .backfill import BackfillProcessor, BackfillResult
from .validator import HistoricalDataValidator, DataQualityReport

logger = get_logger(__name__)
settings = get_settings()


class SyncStatus(Enum):
    """Synchronization status."""
    IDLE = "idle"
    SCANNING = "scanning"
    BACKFILLING = "backfilling"
    VALIDATING = "validating"
    COMPLETED = "completed"
    ERROR = "error"


class SyncConfig(BaseModel):
    """Configuration for synchronization operations."""

    symbols: List[str] = Field(description="Symbols to synchronize")
    timeframes: List[str] = Field(default=["M1", "M5", "M15", "H1"], description="OHLC timeframes")
    include_ticks: bool = Field(default=False, description="Whether to synchronize tick data")
    start_time: datetime = Field(description="Synchronization start time")
    end_time: datetime = Field(description="Synchronization end time")

    # Gap detection settings
    enable_gap_detection: bool = Field(default=True, description="Enable gap detection")
    gap_detection_interval: timedelta = Field(default=timedelta(hours=1), description="Gap detection interval")

    # Backfill settings
    enable_backfill: bool = Field(default=True, description="Enable automatic backfill")
    max_concurrent_backfills: int = Field(default=5, description="Maximum concurrent backfill operations")
    backfill_batch_size: int = Field(default=50, description="Backfill batch size")

    # Validation settings
    enable_validation: bool = Field(default=True, description="Enable data quality validation")
    validation_sample_size: int = Field(default=1000, description="Validation sample size")
    min_quality_threshold: float = Field(default=0.85, description="Minimum acceptable quality score")

    # Scheduling
    auto_sync_enabled: bool = Field(default=False, description="Enable automatic scheduled sync")
    sync_interval: timedelta = Field(default=timedelta(hours=6), description="Automatic sync interval")


class SyncResult(BaseModel):
    """Result of synchronization operation."""

    sync_id: str = Field(description="Synchronization operation ID")
    config: SyncConfig = Field(description="Synchronization configuration")
    status: SyncStatus = Field(description="Final status")
    start_time: datetime = Field(description="Operation start time")
    end_time: datetime = Field(description="Operation end time")
    duration_seconds: float = Field(description="Total operation duration")

    # Gap detection results
    gaps_detected: int = Field(description="Total gaps detected")
    critical_gaps: int = Field(description="Critical gaps detected")
    gaps_by_severity: Dict[str, int] = Field(default_factory=dict, description="Gaps by severity")

    # Backfill results
    backfill_attempts: int = Field(description="Backfill attempts made")
    backfill_successes: int = Field(description="Successful backfills")
    backfill_failures: int = Field(description="Failed backfills")
    points_retrieved: int = Field(description="Total points retrieved")
    points_saved: int = Field(description="Total points saved")

    # Validation results
    validation_reports: int = Field(description="Validation reports generated")
    average_quality_score: float = Field(description="Average quality score")
    quality_issues: int = Field(description="Quality issues found")

    # Summary
    success: bool = Field(description="Whether sync was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")

    @property
    def backfill_success_rate(self) -> float:
        """Get backfill success rate as percentage."""
        if self.backfill_attempts == 0:
            return 100.0
        return (self.backfill_successes / self.backfill_attempts) * 100

    @property
    def data_recovery_rate(self) -> float:
        """Get data recovery rate as percentage."""
        if self.gaps_detected == 0:
            return 100.0
        return (self.backfill_successes / self.gaps_detected) * 100


class HistoricalDataManager:
    """
    Manages historical data synchronization operations.

    Provides comprehensive synchronization including gap detection,
    intelligent backfill, and quality validation with progress tracking.
    """

    def __init__(self):
        """Initialize historical data manager."""
        self.gap_detector = GapDetector()
        self.backfill_processor = BackfillProcessor()
        self.data_validator = HistoricalDataValidator()

        self._active_syncs: Dict[str, Dict[str, Any]] = {}
        self._sync_history: List[SyncResult] = []
        self._progress_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        self._manager_stats = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "total_gaps_processed": 0,
            "total_points_recovered": 0
        }

        # Auto-sync configuration
        self._auto_sync_task: Optional[asyncio.Task] = None
        self._is_auto_sync_enabled = False

        logger.info("Historical data manager initialized")

    async def start_sync(self, config: SyncConfig) -> str:
        """
        Start a new synchronization operation.

        Args:
            config: Synchronization configuration

        Returns:
            Synchronization operation ID
        """
        sync_id = f"sync_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now(timezone.utc)

        logger.info(f"Starting synchronization {sync_id}")

        # Track active sync
        self._active_syncs[sync_id] = {
            "config": config,
            "status": SyncStatus.IDLE,
            "start_time": start_time,
            "progress": 0.0,
            "current_step": "initializing",
            "gaps_detected": [],
            "backfill_results": [],
            "validation_reports": []
        }

        # Start sync in background
        asyncio.create_task(self._execute_sync(sync_id))

        return sync_id

    async def get_sync_status(self, sync_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active synchronization operation."""
        return self._active_syncs.get(sync_id)

    async def cancel_sync(self, sync_id: str) -> bool:
        """Cancel active synchronization operation."""
        if sync_id in self._active_syncs:
            self._active_syncs[sync_id]["status"] = SyncStatus.ERROR
            self._active_syncs[sync_id]["error_message"] = "Cancelled by user"
            logger.info(f"Cancelled synchronization {sync_id}")
            return True
        return False

    def get_sync_history(self, limit: int = 10) -> List[SyncResult]:
        """Get synchronization history."""
        return self._sync_history[-limit:] if self._sync_history else []

    def get_active_syncs(self) -> Dict[str, Dict[str, Any]]:
        """Get all active synchronization operations."""
        return self._active_syncs.copy()

    def add_progress_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add progress callback function."""
        self._progress_callbacks.append(callback)

    async def enable_auto_sync(self, config: SyncConfig) -> bool:
        """
        Enable automatic synchronization.

        Args:
            config: Auto-sync configuration

        Returns:
            True if enabled successfully
        """
        try:
            if self._is_auto_sync_enabled:
                logger.warning("Auto-sync already enabled")
                return True

            # Validate configuration
            if not config.auto_sync_enabled:
                config.auto_sync_enabled = True

            # Start auto-sync task
            @scheduled_task(
                interval_seconds=int(config.sync_interval.total_seconds()),
                name="auto_historical_sync"
            )
            async def auto_sync_task():
                """Automatic synchronization task."""
                try:
                    # Create auto-sync configuration
                    auto_config = config.model_copy()
                    auto_config.end_time = datetime.now(timezone.utc)

                    # Start sync
                    sync_id = await self.start_sync(auto_config)
                    logger.info(f"Started automatic synchronization: {sync_id}")

                except Exception as e:
                    logger.error(f"Auto-sync task failed: {e}")

            self._is_auto_sync_enabled = True
            logger.info("Auto-sync enabled")
            return True

        except Exception as e:
            logger.error(f"Failed to enable auto-sync: {e}")
            return False

    async def disable_auto_sync(self) -> bool:
        """Disable automatic synchronization."""
        try:
            if not self._is_auto_sync_enabled:
                return True

            self._is_auto_sync_enabled = False
            logger.info("Auto-sync disabled")
            return True

        except Exception as e:
            logger.error(f"Failed to disable auto-sync: {e}")
            return False

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        stats = self._manager_stats.copy()
        stats["active_syncs"] = len(self._active_syncs)
        stats["auto_sync_enabled"] = self._is_auto_sync_enabled

        if stats["total_syncs"] > 0:
            stats["success_rate"] = (stats["successful_syncs"] / stats["total_syncs"]) * 100

        return stats

    async def _execute_sync(self, sync_id: str) -> None:
        """Execute synchronization operation."""
        sync_info = self._active_syncs[sync_id]
        config = sync_info["config"]
        start_time = sync_info["start_time"]

        try:
            # Phase 1: Gap Detection
            if config.enable_gap_detection:
                await self._execute_gap_detection(sync_id)

            # Phase 2: Backfill Processing
            if config.enable_backfill and sync_info["gaps_detected"]:
                await self._execute_backfill(sync_id)

            # Phase 3: Data Validation
            if config.enable_validation:
                await self._execute_validation(sync_id)

            # Complete sync
            await self._complete_sync(sync_id, True)

        except Exception as e:
            error_msg = f"Sync {sync_id} failed: {e}"
            logger.error(error_msg)
            sync_info["error_message"] = error_msg
            await self._complete_sync(sync_id, False)

    async def _execute_gap_detection(self, sync_id: str) -> None:
        """Execute gap detection phase."""
        sync_info = self._active_syncs[sync_id]
        config = sync_info["config"]

        sync_info["status"] = SyncStatus.SCANNING
        sync_info["current_step"] = "detecting gaps"
        self._notify_progress(sync_id)

        logger.info(f"Starting gap detection for sync {sync_id}")

        all_gaps = self.gap_detector.detect_all_gaps(
            config.symbols,
            config.timeframes,
            config.start_time,
            config.end_time,
            config.include_ticks
        )

        # Flatten gaps list
        gaps = []
        for gap_list in all_gaps.values():
            gaps.extend(gap_list)

        sync_info["gaps_detected"] = gaps
        sync_info["progress"] = 30.0

        logger.info(f"Gap detection completed for sync {sync_id}: {len(gaps)} gaps found")
        self._notify_progress(sync_id)

    async def _execute_backfill(self, sync_id: str) -> None:
        """Execute backfill processing phase."""
        sync_info = self._active_syncs[sync_id]
        config = sync_info["config"]
        gaps = sync_info["gaps_detected"]

        sync_info["status"] = SyncStatus.BACKFILLING
        sync_info["current_step"] = "processing backfills"
        self._notify_progress(sync_id)

        logger.info(f"Starting backfill processing for sync {sync_id}: {len(gaps)} gaps")

        # Filter gaps that require backfill
        backfill_gaps = [gap for gap in gaps if gap.requires_backfill]

        if backfill_gaps:
            # Process gaps in batches
            results = await self.backfill_processor.process_gaps_batch(
                backfill_gaps,
                max_concurrent=config.max_concurrent_backfills,
                priority_order=True
            )

            sync_info["backfill_results"] = results
        else:
            sync_info["backfill_results"] = []

        sync_info["progress"] = 70.0
        logger.info(f"Backfill processing completed for sync {sync_id}")
        self._notify_progress(sync_id)

    async def _execute_validation(self, sync_id: str) -> None:
        """Execute data validation phase."""
        sync_info = self._active_syncs[sync_id]
        config = sync_info["config"]

        sync_info["status"] = SyncStatus.VALIDATING
        sync_info["current_step"] = "validating data quality"
        self._notify_progress(sync_id)

        logger.info(f"Starting data validation for sync {sync_id}")

        # Validate data quality for all symbols and timeframes
        validation_reports = await self.data_validator.validate_multiple_symbols(
            config.symbols,
            config.timeframes,
            config.start_time,
            config.end_time,
            config.include_ticks
        )

        sync_info["validation_reports"] = validation_reports
        sync_info["progress"] = 90.0

        logger.info(f"Data validation completed for sync {sync_id}")
        self._notify_progress(sync_id)

    async def _complete_sync(self, sync_id: str, success: bool) -> None:
        """Complete synchronization operation."""
        sync_info = self._active_syncs[sync_id]
        config = sync_info["config"]
        end_time = datetime.now(timezone.utc)

        # Calculate summary statistics
        gaps = sync_info.get("gaps_detected", [])
        backfill_results = sync_info.get("backfill_results", [])
        validation_reports = sync_info.get("validation_reports", {})

        gaps_by_severity = {}
        for gap in gaps:
            severity = gap.severity
            gaps_by_severity[severity] = gaps_by_severity.get(severity, 0) + 1

        successful_backfills = sum(1 for r in backfill_results if r.success)
        total_points_retrieved = sum(r.points_retrieved for r in backfill_results)
        total_points_saved = sum(r.points_saved for r in backfill_results)

        avg_quality_score = 0.0
        quality_issues = 0
        if validation_reports:
            scores = [report.overall_score for report in validation_reports.values()]
            avg_quality_score = sum(scores) / len(scores) if scores else 0.0
            quality_issues = sum(len(report.critical_issues) for report in validation_reports.values())

        # Create result
        result = SyncResult(
            sync_id=sync_id,
            config=config,
            status=SyncStatus.COMPLETED if success else SyncStatus.ERROR,
            start_time=sync_info["start_time"],
            end_time=end_time,
            duration_seconds=(end_time - sync_info["start_time"]).total_seconds(),
            gaps_detected=len(gaps),
            critical_gaps=len([g for g in gaps if g.severity == "critical"]),
            gaps_by_severity=gaps_by_severity,
            backfill_attempts=len(backfill_results),
            backfill_successes=successful_backfills,
            backfill_failures=len(backfill_results) - successful_backfills,
            points_retrieved=total_points_retrieved,
            points_saved=total_points_saved,
            validation_reports=len(validation_reports),
            average_quality_score=avg_quality_score,
            quality_issues=quality_issues,
            success=success,
            error_message=sync_info.get("error_message")
        )

        # Add to history
        self._sync_history.append(result)
        if len(self._sync_history) > 100:  # Keep last 100 results
            self._sync_history = self._sync_history[-100:]

        # Update statistics
        self._manager_stats["total_syncs"] += 1
        if success:
            self._manager_stats["successful_syncs"] += 1
        else:
            self._manager_stats["failed_syncs"] += 1

        self._manager_stats["total_gaps_processed"] += len(gaps)
        self._manager_stats["total_points_recovered"] += total_points_saved

        # Clean up active sync
        sync_info["status"] = result.status
        sync_info["progress"] = 100.0
        self._notify_progress(sync_id)

        # Remove from active syncs after a delay
        await asyncio.sleep(60)  # Keep for 1 minute for final status check
        self._active_syncs.pop(sync_id, None)

        logger.info(f"Synchronization {sync_id} completed: {result.backfill_success_rate:.1f}% success rate")

    def _notify_progress(self, sync_id: str) -> None:
        """Notify progress callbacks."""
        if sync_id in self._active_syncs:
            sync_info = self._active_syncs[sync_id]
            for callback in self._progress_callbacks:
                try:
                    callback(sync_id, sync_info)
                except Exception as e:
                    logger.error(f"Progress callback failed: {e}")


# Global manager instance
_historical_data_manager: Optional[HistoricalDataManager] = None


def get_historical_data_manager() -> HistoricalDataManager:
    """Get the global historical data manager instance."""
    global _historical_data_manager

    if _historical_data_manager is None:
        _historical_data_manager = HistoricalDataManager()

    return _historical_data_manager