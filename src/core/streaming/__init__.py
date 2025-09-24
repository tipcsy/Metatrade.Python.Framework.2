"""
Real-time Data Streaming Module.

This module provides high-performance real-time data streaming capabilities
for MetaTrader 5 market data.
"""

from .realtime_data_streamer import (
    RealTimeDataStreamer,
    StreamingConfig,
    StreamingState,
    StreamingMetrics,
    get_realtime_streamer,
    initialize_realtime_streamer,
)

__all__ = [
    "RealTimeDataStreamer",
    "StreamingConfig",
    "StreamingState",
    "StreamingMetrics",
    "get_realtime_streamer",
    "initialize_realtime_streamer",
]