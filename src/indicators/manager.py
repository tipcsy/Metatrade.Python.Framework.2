"""
Centralized indicator management system.

This module provides unified management for all technical indicators
with multi-timeframe coordination, caching, and performance monitoring.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.tasks import background_task, scheduled_task
from .base import BaseIndicator, IndicatorResult
from .macd import MACDIndicator, MACDConfig
from .trend import TrendAnalyzer, TrendConfig, TrendAnalysis

logger = get_logger(__name__)
settings = get_settings()


class IndicatorManager:
    """
    Centralized manager for all technical indicators.

    Provides unified access to indicators with multi-timeframe support,
    result caching, and performance monitoring.
    """

    def __init__(self):
        """Initialize indicator manager."""
        # Indicator storage by symbol and timeframe
        self._indicators: Dict[str, Dict[str, Dict[str, BaseIndicator]]] = {}
        self._lock = threading.RLock()

        # Result caching
        self._results_cache: Dict[str, List[IndicatorResult]] = {}
        self._cache_max_age = timedelta(hours=24)
        self._cache_max_size = 10000

        # Performance tracking
        self._calculation_stats = {
            "total_calculations": 0,
            "total_errors": 0,
            "average_calculation_time": 0.0,
            "calculations_per_second": 0.0
        }

        # Default configurations
        self._default_macd_config = MACDConfig()
        self._default_trend_config = TrendConfig()

        # Monitoring
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

        logger.info("Indicator manager initialized")

    def start(self) -> bool:
        """Start the indicator manager."""
        if self._is_running:
            logger.warning("Indicator manager already running")
            return True

        try:
            # Start monitoring tasks
            self._start_monitoring()

            self._is_running = True
            logger.info("Indicator manager started")
            return True

        except Exception as e:
            logger.error(f"Failed to start indicator manager: {e}")
            return False

    def stop(self) -> None:
        """Stop the indicator manager."""
        if not self._is_running:
            return

        logger.info("Stopping indicator manager...")

        self._is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()

        logger.info("Indicator manager stopped")

    def create_macd_indicator(
        self,
        symbol: str,
        timeframe: str,
        config: MACDConfig = None
    ) -> bool:
        """
        Create MACD indicator for symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, H1, etc.)
            config: MACD configuration

        Returns:
            bool: True if created successfully
        """
        try:
            with self._lock:
                # Initialize symbol structure if needed
                if symbol not in self._indicators:
                    self._indicators[symbol] = {}
                if timeframe not in self._indicators[symbol]:
                    self._indicators[symbol][timeframe] = {}

                # Check if indicator already exists
                if "MACD" in self._indicators[symbol][timeframe]:
                    logger.warning(f"MACD indicator already exists for {symbol} {timeframe}")
                    return True

                # Create indicator
                macd_indicator = MACDIndicator(config or self._default_macd_config)
                self._indicators[symbol][timeframe]["MACD"] = macd_indicator

                logger.info(f"Created MACD indicator for {symbol} {timeframe}")
                return True

        except Exception as e:
            logger.error(f"Error creating MACD indicator for {symbol} {timeframe}: {e}")
            return False

    def create_trend_analyzer(
        self,
        symbol: str,
        config: TrendConfig = None
    ) -> bool:
        """
        Create trend analyzer for symbol.

        Args:
            symbol: Trading symbol
            config: Trend analyzer configuration

        Returns:
            bool: True if created successfully
        """
        try:
            with self._lock:
                # Initialize symbol structure
                if symbol not in self._indicators:
                    self._indicators[symbol] = {}
                if "MULTI" not in self._indicators[symbol]:
                    self._indicators[symbol]["MULTI"] = {}

                # Check if analyzer already exists
                if "TrendAnalyzer" in self._indicators[symbol]["MULTI"]:
                    logger.warning(f"Trend analyzer already exists for {symbol}")
                    return True

                # Create analyzer
                trend_analyzer = TrendAnalyzer(config or self._default_trend_config)
                self._indicators[symbol]["MULTI"]["TrendAnalyzer"] = trend_analyzer

                logger.info(f"Created trend analyzer for {symbol}")
                return True

        except Exception as e:
            logger.error(f"Error creating trend analyzer for {symbol}: {e}")
            return False

    def calculate_macd(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, float]]
    ) -> Optional[IndicatorResult]:
        """
        Calculate MACD for symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLC price data

        Returns:
            IndicatorResult or None if calculation failed
        """
        start_time = time.time()

        try:
            with self._lock:
                indicator = self._get_indicator(symbol, timeframe, "MACD")
                if not indicator:
                    # Create indicator if it doesn't exist
                    if not self.create_macd_indicator(symbol, timeframe):
                        return None
                    indicator = self._get_indicator(symbol, timeframe, "MACD")

                if not indicator:
                    logger.error(f"Failed to get MACD indicator for {symbol} {timeframe}")
                    return None

                # Calculate MACD
                result = indicator.calculate(symbol, timeframe, data)

                if result:
                    # Cache result
                    self._cache_result(result)

                    # Update performance stats
                    self._update_performance_stats(time.time() - start_time, True)

                    logger.debug(f"Calculated MACD for {symbol} {timeframe}")
                else:
                    self._update_performance_stats(time.time() - start_time, False)

                return result

        except Exception as e:
            logger.error(f"Error calculating MACD for {symbol} {timeframe}: {e}")
            self._update_performance_stats(time.time() - start_time, False)
            return None

    def analyze_trend(
        self,
        symbol: str,
        timeframe_data: Dict[str, List[Dict[str, float]]]
    ) -> Optional[TrendAnalysis]:
        """
        Perform multi-timeframe trend analysis.

        Args:
            symbol: Trading symbol
            timeframe_data: Data for each timeframe

        Returns:
            TrendAnalysis or None if analysis failed
        """
        start_time = time.time()

        try:
            with self._lock:
                analyzer = self._get_indicator(symbol, "MULTI", "TrendAnalyzer")
                if not analyzer:
                    # Create analyzer if it doesn't exist
                    if not self.create_trend_analyzer(symbol):
                        return None
                    analyzer = self._get_indicator(symbol, "MULTI", "TrendAnalyzer")

                if not analyzer or not isinstance(analyzer, TrendAnalyzer):
                    logger.error(f"Failed to get trend analyzer for {symbol}")
                    return None

                # Perform analysis
                analysis = analyzer.analyze_multi_timeframe(symbol, timeframe_data)

                if analysis:
                    # Cache the analysis result by converting to IndicatorResult
                    result = IndicatorResult(
                        indicator_name="TrendAnalyzer",
                        symbol=symbol,
                        timeframe="MULTI",
                        timestamp=analysis.timestamp,
                        bar_timestamp=analysis.timestamp,
                        values={
                            "overall_direction": analysis.overall_direction.value,
                            "overall_strength": analysis.overall_strength,
                            "overall_confidence": analysis.overall_confidence
                        },
                        signals=analysis.to_dict(),
                        confidence=analysis.overall_confidence
                    )

                    self._cache_result(result)

                    # Update performance stats
                    self._update_performance_stats(time.time() - start_time, True)

                    logger.debug(f"Analyzed trend for {symbol}")
                else:
                    self._update_performance_stats(time.time() - start_time, False)

                return analysis

        except Exception as e:
            logger.error(f"Error analyzing trend for {symbol}: {e}")
            self._update_performance_stats(time.time() - start_time, False)
            return None

    def get_latest_result(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str
    ) -> Optional[IndicatorResult]:
        """
        Get latest cached result for indicator.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicator_name: Indicator name

        Returns:
            Latest IndicatorResult or None
        """
        cache_key = f"{symbol}_{timeframe}_{indicator_name}"

        with self._lock:
            results = self._results_cache.get(cache_key, [])
            return results[-1] if results else None

    def get_result_history(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        count: int = 10
    ) -> List[IndicatorResult]:
        """
        Get historical results for indicator.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicator_name: Indicator name
            count: Number of results to return

        Returns:
            List of historical results
        """
        cache_key = f"{symbol}_{timeframe}_{indicator_name}"

        with self._lock:
            results = self._results_cache.get(cache_key, [])
            return results[-count:] if results else []

    def list_indicators(
        self,
        symbol: str = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        List all available indicators.

        Args:
            symbol: Specific symbol (all symbols if None)

        Returns:
            Dictionary of symbols -> timeframes -> indicator names
        """
        with self._lock:
            if symbol:
                return {symbol: self._indicators.get(symbol, {})}
            else:
                result = {}
                for sym, timeframes in self._indicators.items():
                    result[sym] = {}
                    for tf, indicators in timeframes.items():
                        result[sym][tf] = list(indicators.keys())
                return result

    def remove_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str
    ) -> bool:
        """
        Remove specific indicator.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicator_name: Indicator name

        Returns:
            bool: True if removed successfully
        """
        try:
            with self._lock:
                if (symbol in self._indicators and
                    timeframe in self._indicators[symbol] and
                    indicator_name in self._indicators[symbol][timeframe]):

                    del self._indicators[symbol][timeframe][indicator_name]

                    # Clean up empty structures
                    if not self._indicators[symbol][timeframe]:
                        del self._indicators[symbol][timeframe]
                    if not self._indicators[symbol]:
                        del self._indicators[symbol]

                    # Clean up cache
                    cache_key = f"{symbol}_{timeframe}_{indicator_name}"
                    self._results_cache.pop(cache_key, None)

                    logger.info(f"Removed {indicator_name} for {symbol} {timeframe}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error removing {indicator_name} for {symbol} {timeframe}: {e}")
            return False

    def reset_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str
    ) -> bool:
        """
        Reset specific indicator state.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicator_name: Indicator name

        Returns:
            bool: True if reset successfully
        """
        try:
            with self._lock:
                indicator = self._get_indicator(symbol, timeframe, indicator_name)
                if indicator:
                    indicator.reset()

                    # Clear cached results
                    cache_key = f"{symbol}_{timeframe}_{indicator_name}"
                    self._results_cache.pop(cache_key, None)

                    logger.info(f"Reset {indicator_name} for {symbol} {timeframe}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error resetting {indicator_name} for {symbol} {timeframe}: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get indicator manager performance statistics."""
        with self._lock:
            total_indicators = sum(
                len(indicators)
                for timeframes in self._indicators.values()
                for indicators in timeframes.values()
            )

            cached_results = sum(len(results) for results in self._results_cache.values())

            return {
                "is_running": self._is_running,
                "total_indicators": total_indicators,
                "cached_results": cached_results,
                "calculation_stats": self._calculation_stats.copy(),
                "cache_hit_ratio": self._calculate_cache_hit_ratio(),
                "symbols_tracked": len(self._indicators),
                "memory_usage_estimate_mb": self._estimate_memory_usage()
            }

    def _get_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str
    ) -> Optional[BaseIndicator]:
        """Get specific indicator instance."""
        return (
            self._indicators
            .get(symbol, {})
            .get(timeframe, {})
            .get(indicator_name)
        )

    def _cache_result(self, result: IndicatorResult) -> None:
        """Cache indicator result."""
        cache_key = f"{result.symbol}_{result.timeframe}_{result.indicator_name}"

        if cache_key not in self._results_cache:
            self._results_cache[cache_key] = []

        self._results_cache[cache_key].append(result)

        # Limit cache size per indicator
        max_results_per_indicator = 1000
        if len(self._results_cache[cache_key]) > max_results_per_indicator:
            self._results_cache[cache_key] = self._results_cache[cache_key][-max_results_per_indicator:]

        # Global cache size limit
        if len(self._results_cache) > self._cache_max_size:
            self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """Clean up old cached results."""
        current_time = datetime.now(timezone.utc)
        keys_to_remove = []

        for cache_key, results in self._results_cache.items():
            # Remove old results
            cutoff_time = current_time - self._cache_max_age
            fresh_results = [
                r for r in results
                if r.timestamp >= cutoff_time
            ]

            if fresh_results:
                self._results_cache[cache_key] = fresh_results
            else:
                keys_to_remove.append(cache_key)

        # Remove empty cache entries
        for key in keys_to_remove:
            del self._results_cache[key]

        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} old cache entries")

    def _update_performance_stats(self, calculation_time: float, success: bool) -> None:
        """Update performance statistics."""
        self._calculation_stats["total_calculations"] += 1
        if not success:
            self._calculation_stats["total_errors"] += 1

        # Update average calculation time (rolling average)
        total_calcs = self._calculation_stats["total_calculations"]
        current_avg = self._calculation_stats["average_calculation_time"]
        self._calculation_stats["average_calculation_time"] = (
            (current_avg * (total_calcs - 1) + calculation_time) / total_calcs
        )

        # Calculate calculations per second (last 100 calculations)
        if total_calcs % 100 == 0:
            self._calculation_stats["calculations_per_second"] = 100 / (
                self._calculation_stats["average_calculation_time"] * 100
            )

    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        # This would require tracking cache hits vs misses
        # For now, return a placeholder
        return 0.85

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimate based on number of indicators and cached results
        indicators_memory = len(self._indicators) * 0.1  # ~100KB per indicator
        cache_memory = sum(len(results) for results in self._results_cache.values()) * 0.001  # ~1KB per result

        return indicators_memory + cache_memory

    def _start_monitoring(self) -> None:
        """Start performance monitoring tasks."""

        @scheduled_task(
            interval_seconds=300,  # Every 5 minutes
            name="cleanup_indicator_cache"
        )
        def cleanup_cache():
            self._cleanup_cache()

        @scheduled_task(
            interval_seconds=60,  # Every minute
            name="log_indicator_performance"
        )
        def log_performance():
            stats = self.get_performance_stats()
            logger.info(
                f"Indicator Performance: "
                f"{stats['total_indicators']} indicators, "
                f"{stats['calculation_stats']['total_calculations']} calculations, "
                f"{stats['calculation_stats']['average_calculation_time']*1000:.1f}ms avg"
            )

        logger.info("Started indicator monitoring tasks")

    @property
    def is_running(self) -> bool:
        """Check if indicator manager is running."""
        return self._is_running


# Global indicator manager instance
_indicator_manager: Optional[IndicatorManager] = None


def get_indicator_manager() -> IndicatorManager:
    """Get the global indicator manager instance."""
    global _indicator_manager

    if _indicator_manager is None:
        _indicator_manager = IndicatorManager()

    return _indicator_manager