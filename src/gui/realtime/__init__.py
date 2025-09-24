"""
Real-time data update system for the MetaTrader Python Framework GUI.

This package provides thread-safe real-time data updates, event handling,
and performance optimization for high-frequency market data display.
"""

from __future__ import annotations

from .data_updater import RealTimeDataUpdater
from .update_manager import UpdateManager

__all__ = [
    "RealTimeDataUpdater",
    "UpdateManager",
]