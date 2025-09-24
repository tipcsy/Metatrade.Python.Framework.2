"""
Core data processing components for the MetaTrader Python Framework.

This module provides comprehensive data processing capabilities including:
- High-performance tick data collection
- Real-time OHLC generation
- Efficient data buffering
- Event-driven architecture
- Performance monitoring
"""

from .buffer import DataBuffer, TickBuffer, OHLCBuffer, BufferManager, get_buffer_manager
from .collector import TickCollector, DataCollector, DataCollectionManager, get_data_collection_manager
from .ohlc_processor import OHLCProcessor, TimeframeManager
from .models import (
    TickData,
    OHLCData,
    MarketEvent,
    MarketEventType,
    ProcessingState,
    TrendAnalysis,
    DataQuality,
    TrendDirection,
)
from .events import get_event_publisher

__all__ = [
    # Buffer classes
    "DataBuffer",
    "TickBuffer",
    "OHLCBuffer",
    "BufferManager",
    "get_buffer_manager",

    # Collector classes
    "TickCollector",
    "DataCollector",
    "DataCollectionManager",
    "get_data_collection_manager",

    # Processor classes
    "OHLCProcessor",
    "TimeframeManager",

    # Data models
    "TickData",
    "OHLCData",
    "MarketEvent",
    "MarketEventType",
    "ProcessingState",
    "TrendAnalysis",
    "DataQuality",
    "TrendDirection",

    # Event system
    "get_event_publisher",
]