"""
Technical indicators for the MetaTrader Python Framework.

This module provides high-performance technical indicators with
multi-timeframe support, trend analysis, and signal generation.
"""

from .macd import MACDIndicator, MACDSignal, MACDAnalysis
from .macd_analyzer import (
    MACDAnalyzer,
    MACDCalculator,
    MACDSignalGenerator,
    MACDValues,
    MACDConfig,
    TrendSignal,
    get_macd_analyzer,
)
from .trend import TrendAnalyzer, TrendSignal, TrendDirection
from .base import BaseIndicator, IndicatorConfig, IndicatorResult
from .manager import IndicatorManager, get_indicator_manager

__all__ = [
    # MACD
    "MACDIndicator",
    "MACDSignal",
    "MACDAnalysis",

    # MACD Analyzer (Phase 3)
    "MACDAnalyzer",
    "MACDCalculator",
    "MACDSignalGenerator",
    "MACDValues",
    "MACDConfig",
    "TrendSignal",
    "get_macd_analyzer",

    # Trend Analysis
    "TrendAnalyzer",
    "TrendSignal",
    "TrendDirection",

    # Base Classes
    "BaseIndicator",
    "IndicatorConfig",
    "IndicatorResult",

    # Management
    "IndicatorManager",
    "get_indicator_manager",
]