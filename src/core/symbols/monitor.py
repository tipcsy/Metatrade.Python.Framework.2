"""
Symbol monitoring and health checking system.

This module provides comprehensive monitoring of symbol data quality,
availability, and performance metrics.
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
from .models import SymbolInfo, SymbolStats, SymbolStatus

logger = get_logger(__name__)
settings = get_settings()


class SymbolHealthCheck:
    """Individual symbol health check result."""

    def __init__(self, symbol: str):
        """
        Initialize health check for symbol.

        Args:
            symbol: Symbol name
        """
        self.symbol = symbol
        self.timestamp = datetime.now(timezone.utc)

        # Health metrics
        self.is_healthy = True
        self.health_score = 100.0  # 0-100 scale
        self.issues: List[str] = []
        self.warnings: List[str] = []

        # Data quality metrics
        self.data_completeness = 100.0
        self.data_freshness_seconds = 0.0
        self.price_stability = True
        self.spread_reasonableness = True

        # Availability metrics
        self.market_open = True
        self.trading_enabled = True
        self.connection_status = "connected"

        # Performance metrics
        self.avg_latency_ms = 0.0
        self.update_frequency_hz = 0.0
        self.error_rate_percent = 0.0

    def add_issue(self, issue: str) -> None:
        """Add health issue."""
        self.issues.append(issue)
        self.is_healthy = False

    def add_warning(self, warning: str) -> None:
        """Add health warning."""
        self.warnings.append(warning)
        # Warnings don't make symbol unhealthy but reduce score
        self.health_score = max(0, self.health_score - 5)

    def calculate_health_score(self) -> float:
        """Calculate overall health score."""
        score = 100.0

        # Deduct for issues
        score -= len(self.issues) * 20  # Major deduction for issues
        score -= len(self.warnings) * 5   # Minor deduction for warnings

        # Deduct for poor data quality
        score -= (100 - self.data_completeness) * 0.3
        score -= min(self.data_freshness_seconds / 60, 20)  # Up to 20 points for staleness

        # Deduct for high error rate
        score -= min(self.error_rate_percent * 2, 30)

        # Deduct for high latency
        score -= min(self.avg_latency_ms / 10, 20)

        self.health_score = max(0.0, score)
        return self.health_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "is_healthy": self.is_healthy,
            "health_score": self.health_score,
            "issues": self.issues,
            "warnings": self.warnings,
            "data_completeness": self.data_completeness,
            "data_freshness_seconds": self.data_freshness_seconds,
            "price_stability": self.price_stability,
            "spread_reasonableness": self.spread_reasonableness,
            "market_open": self.market_open,
            "trading_enabled": self.trading_enabled,
            "connection_status": self.connection_status,
            "avg_latency_ms": self.avg_latency_ms,
            "update_frequency_hz": self.update_frequency_hz,
            "error_rate_percent": self.error_rate_percent
        }


class SymbolMonitor:
    """
    Comprehensive symbol monitoring system.

    Monitors symbol health, data quality, performance, and availability
    with automated alerting and remediation capabilities.
    """

    def __init__(self):
        """Initialize symbol monitor."""
        # Health check storage
        self._health_checks: Dict[str, SymbolHealthCheck] = {}
        self._health_history: Dict[str, List[SymbolHealthCheck]] = {}
        self._lock = threading.RLock()

        # Configuration
        self._check_interval = 60  # seconds
        self._history_retention_hours = 24
        self._max_history_entries = 1000

        # Thresholds
        self._staleness_threshold_seconds = 300  # 5 minutes
        self._high_latency_threshold_ms = 1000   # 1 second
        self._high_error_rate_percent = 5.0      # 5%
        self._min_update_frequency_hz = 0.1      # 6 updates per minute

        # State
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Statistics
        self._total_checks = 0
        self._healthy_symbols = 0
        self._unhealthy_symbols = 0

        logger.info("Symbol monitor initialized")

    def start(self) -> bool:
        """Start symbol monitoring."""
        if self._is_running:
            logger.warning("Symbol monitor already running")
            return True

        try:
            # Schedule monitoring tasks
            self._schedule_monitoring_tasks()

            self._is_running = True
            logger.info("Symbol monitor started")
            return True

        except Exception as e:
            logger.error(f"Failed to start symbol monitor: {e}")
            return False

    def stop(self) -> None:
        """Stop symbol monitoring."""
        if not self._is_running:
            return

        logger.info("Stopping symbol monitor...")

        self._is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()

        logger.info("Symbol monitor stopped")

    def check_symbol_health(self, symbol_info: SymbolInfo) -> SymbolHealthCheck:
        """
        Perform comprehensive health check on symbol.

        Args:
            symbol_info: Symbol information

        Returns:
            SymbolHealthCheck result
        """
        health_check = SymbolHealthCheck(symbol_info.symbol)

        try:
            # Check data freshness
            self._check_data_freshness(symbol_info, health_check)

            # Check price data quality
            self._check_price_quality(symbol_info, health_check)

            # Check market availability
            self._check_market_availability(symbol_info, health_check)

            # Check trading status
            self._check_trading_status(symbol_info, health_check)

            # Calculate final health score
            health_check.calculate_health_score()

            # Store health check
            with self._lock:
                self._health_checks[symbol_info.symbol] = health_check
                self._add_to_history(health_check)

            self._total_checks += 1

            if health_check.is_healthy:
                self._healthy_symbols += 1
            else:
                self._unhealthy_symbols += 1

            logger.debug(
                f"Health check for {symbol_info.symbol}: "
                f"score={health_check.health_score:.1f}, "
                f"healthy={health_check.is_healthy}"
            )

            return health_check

        except Exception as e:
            logger.error(f"Error checking health for {symbol_info.symbol}: {e}")
            health_check.add_issue(f"Health check failed: {e}")
            return health_check

    def get_symbol_health(self, symbol: str) -> Optional[SymbolHealthCheck]:
        """Get latest health check for symbol."""
        with self._lock:
            return self._health_checks.get(symbol)

    def get_all_health_checks(self) -> Dict[str, SymbolHealthCheck]:
        """Get all latest health checks."""
        with self._lock:
            return dict(self._health_checks)

    def get_healthy_symbols(self) -> List[str]:
        """Get list of healthy symbols."""
        with self._lock:
            return [
                symbol for symbol, health in self._health_checks.items()
                if health.is_healthy
            ]

    def get_unhealthy_symbols(self) -> List[str]:
        """Get list of unhealthy symbols."""
        with self._lock:
            return [
                symbol for symbol, health in self._health_checks.items()
                if not health.is_healthy
            ]

    def get_symbol_history(
        self,
        symbol: str,
        hours: int = 24
    ) -> List[SymbolHealthCheck]:
        """
        Get health check history for symbol.

        Args:
            symbol: Symbol name
            hours: Hours of history to retrieve

        Returns:
            List of health checks
        """
        with self._lock:
            history = self._health_history.get(symbol, [])

            if hours > 0:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
                history = [h for h in history if h.timestamp >= cutoff_time]

            return sorted(history, key=lambda h: h.timestamp, reverse=True)

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        with self._lock:
            total_symbols = len(self._health_checks)
            healthy_count = len(self.get_healthy_symbols())
            unhealthy_count = total_symbols - healthy_count

            # Calculate average health score
            avg_health_score = 0.0
            if total_symbols > 0:
                total_score = sum(h.health_score for h in self._health_checks.values())
                avg_health_score = total_score / total_symbols

            # Count issues by type
            issue_counts = {}
            warning_counts = {}

            for health in self._health_checks.values():
                for issue in health.issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                for warning in health.warnings:
                    warning_counts[warning] = warning_counts.get(warning, 0) + 1

            return {
                "is_running": self._is_running,
                "total_symbols": total_symbols,
                "healthy_symbols": healthy_count,
                "unhealthy_symbols": unhealthy_count,
                "health_percentage": (healthy_count / max(total_symbols, 1)) * 100,
                "average_health_score": avg_health_score,
                "total_checks_performed": self._total_checks,
                "common_issues": sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                "common_warnings": sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }

    def _check_data_freshness(
        self,
        symbol_info: SymbolInfo,
        health_check: SymbolHealthCheck
    ) -> None:
        """Check data freshness."""
        if symbol_info.last_quote_time:
            age = (datetime.now(timezone.utc) - symbol_info.last_quote_time).total_seconds()
            health_check.data_freshness_seconds = age

            if age > self._staleness_threshold_seconds:
                if symbol_info.is_market_open():
                    health_check.add_issue(f"Stale data: {age:.0f}s old")
                else:
                    health_check.add_warning(f"Market closed, data age: {age:.0f}s")
        else:
            health_check.add_issue("No quote data available")

    def _check_price_quality(
        self,
        symbol_info: SymbolInfo,
        health_check: SymbolHealthCheck
    ) -> None:
        """Check price data quality."""
        if symbol_info.bid is not None and symbol_info.ask is not None:
            # Check for reasonable spread
            spread = symbol_info.spread
            mid_price = symbol_info.mid_price

            if spread and mid_price and mid_price > 0:
                spread_percentage = (spread / mid_price) * 100

                if spread_percentage > 10:  # More than 10% spread
                    health_check.add_warning(f"High spread: {spread_percentage:.2f}%")
                    health_check.spread_reasonableness = False

                if spread < 0:
                    health_check.add_issue("Negative spread detected")
                    health_check.spread_reasonableness = False

            # Check for zero prices
            if symbol_info.bid <= 0 or symbol_info.ask <= 0:
                health_check.add_issue("Invalid price: zero or negative")
                health_check.price_stability = False

        else:
            health_check.add_issue("Missing bid/ask prices")

    def _check_market_availability(
        self,
        symbol_info: SymbolInfo,
        health_check: SymbolHealthCheck
    ) -> None:
        """Check market availability."""
        health_check.market_open = symbol_info.is_market_open()

        if not health_check.market_open:
            health_check.add_warning("Market is currently closed")

        # Check symbol status
        if symbol_info.status != SymbolStatus.ACTIVE:
            health_check.add_issue(f"Symbol status: {symbol_info.status}")

    def _check_trading_status(
        self,
        symbol_info: SymbolInfo,
        health_check: SymbolHealthCheck
    ) -> None:
        """Check trading status."""
        health_check.trading_enabled = symbol_info.is_tradable

        if not symbol_info.is_tradable:
            health_check.add_warning("Trading is disabled for this symbol")

        if not symbol_info.is_visible:
            health_check.add_warning("Symbol is not visible")

    def _add_to_history(self, health_check: SymbolHealthCheck) -> None:
        """Add health check to history."""
        symbol = health_check.symbol

        if symbol not in self._health_history:
            self._health_history[symbol] = []

        self._health_history[symbol].append(health_check)

        # Limit history size
        if len(self._health_history[symbol]) > self._max_history_entries:
            self._health_history[symbol] = self._health_history[symbol][-self._max_history_entries:]

        # Clean old entries
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self._history_retention_hours)
        self._health_history[symbol] = [
            h for h in self._health_history[symbol]
            if h.timestamp >= cutoff_time
        ]

    def _schedule_monitoring_tasks(self) -> None:
        """Schedule monitoring tasks."""

        @scheduled_task(
            interval_seconds=self._check_interval,
            name="monitor_symbol_health"
        )
        def monitor_symbols():
            self._perform_health_checks()

        @scheduled_task(
            interval_seconds=3600,  # Every hour
            name="cleanup_monitoring_data"
        )
        def cleanup():
            self._cleanup_old_data()

        logger.info("Monitoring tasks scheduled")

    @background_task(name="perform_health_checks")
    def _perform_health_checks(self) -> None:
        """Perform health checks on all symbols."""
        try:
            from .manager import get_symbol_manager
            symbol_manager = get_symbol_manager()

            symbols = symbol_manager.list_symbols()
            healthy_count = 0
            unhealthy_count = 0

            for symbol_info in symbols:
                try:
                    health_check = self.check_symbol_health(symbol_info)

                    if health_check.is_healthy:
                        healthy_count += 1
                    else:
                        unhealthy_count += 1

                        # Log critical issues
                        if health_check.health_score < 50:
                            logger.warning(
                                f"Critical health issue for {symbol_info.symbol}: "
                                f"score={health_check.health_score:.1f}, "
                                f"issues={health_check.issues}"
                            )

                except Exception as e:
                    logger.error(f"Error checking health for {symbol_info.symbol}: {e}")
                    unhealthy_count += 1

            logger.info(
                f"Health checks completed: {healthy_count} healthy, "
                f"{unhealthy_count} unhealthy symbols"
            )

        except Exception as e:
            logger.error(f"Error performing health checks: {e}")

    @background_task(name="cleanup_monitoring_data")
    def _cleanup_old_data(self) -> None:
        """Cleanup old monitoring data."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self._history_retention_hours)
            cleaned_count = 0

            with self._lock:
                for symbol in list(self._health_history.keys()):
                    original_count = len(self._health_history[symbol])

                    self._health_history[symbol] = [
                        h for h in self._health_history[symbol]
                        if h.timestamp >= cutoff_time
                    ]

                    cleaned_count += original_count - len(self._health_history[symbol])

                    # Remove empty histories
                    if not self._health_history[symbol]:
                        del self._health_history[symbol]

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old health check entries")

        except Exception as e:
            logger.error(f"Error cleaning up monitoring data: {e}")

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._is_running


# Global symbol monitor instance
_symbol_monitor: Optional[SymbolMonitor] = None


def get_symbol_monitor() -> SymbolMonitor:
    """Get the global symbol monitor instance."""
    global _symbol_monitor

    if _symbol_monitor is None:
        _symbol_monitor = SymbolMonitor()

    return _symbol_monitor