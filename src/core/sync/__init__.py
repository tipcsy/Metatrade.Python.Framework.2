"""
Historical data synchronization for the MetaTrader Python Framework.

This module provides comprehensive historical data synchronization capabilities
with gap detection, backfill processing, and incremental updates.
"""

from .manager import HistoricalDataManager, SyncConfig, SyncResult, SyncStatus
from .detector import GapDetector, DataGap, GapType
from .backfill import BackfillProcessor, BackfillStrategy, BackfillResult
from .validator import HistoricalDataValidator, DataQualityReport
from .historical_sync_manager import (
    HistoricalSyncManager,
    GapDetector as Phase3GapDetector,
    HistoricalDataFetcher,
    SyncTask,
    DataGap as Phase3DataGap,
    SyncStatus as Phase3SyncStatus,
    GapType as Phase3GapType,
    get_historical_sync_manager,
)

__all__ = [
    # Core synchronization
    "HistoricalDataManager",
    "SyncConfig",
    "SyncResult",
    "SyncStatus",

    # Gap detection
    "GapDetector",
    "DataGap",
    "GapType",

    # Backfill processing
    "BackfillProcessor",
    "BackfillStrategy",
    "BackfillResult",

    # Validation
    "HistoricalDataValidator",
    "DataQualityReport",

    # Phase 3 Components
    "HistoricalSyncManager",
    "Phase3GapDetector",
    "HistoricalDataFetcher",
    "SyncTask",
    "Phase3DataGap",
    "Phase3SyncStatus",
    "Phase3GapType",
    "get_historical_sync_manager",
]