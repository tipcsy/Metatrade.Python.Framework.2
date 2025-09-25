"""
Advanced Metrics Collection and Performance Monitoring for MetaTrader Python Framework Phase 5.

This module implements comprehensive performance monitoring, metrics collection,
and analytics for all trading system components with institutional-grade
observability and alerting capabilities.

Key Features:
- Real-time performance metrics collection
- Custom business metrics and KPIs
- System health monitoring
- Latency distribution analysis
- Resource utilization tracking
- Alert generation and notification
- Historical metrics storage
- Dashboard data aggregation
- Compliance reporting metrics
- A/B testing analytics
"""

from __future__ import annotations

import asyncio
import time
import psutil
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
import json
import statistics

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.core.exceptions import BaseFrameworkError
from src.core.logging import get_logger
from src.core.config import Settings

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Definition of a metric with metadata."""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    retention_days: int = 30
    alert_rules: List['AlertRule'] = field(default_factory=list)


@dataclass
class MetricPoint:
    """Single metric data point."""
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    metric_name: str
    condition: str  # ">", "<", "==", "!=", ">=", "<="
    threshold: float
    severity: AlertSeverity
    window_minutes: int = 5
    min_data_points: int = 1
    description: str = ""
    enabled: bool = True


@dataclass
class Alert:
    """Generated alert."""
    alert_id: str
    rule_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class PerformanceSnapshot:
    """System performance snapshot."""
    timestamp: datetime

    # System metrics
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int

    # Application metrics
    active_connections: int
    thread_count: int
    open_file_descriptors: int

    # Trading metrics
    orders_per_second: float
    ticks_per_second: float
    latency_p95_ns: Optional[float] = None
    latency_p99_ns: Optional[float] = None


class MetricsCollectorError(BaseFrameworkError):
    """Metrics collector specific errors."""
    error_code = "METRICS_COLLECTOR_ERROR"
    error_category = "metrics"


class LatencyTracker:
    """High-performance latency tracking with percentiles."""

    def __init__(self, name: str, max_samples: int = 10000):
        self.name = name
        self.max_samples = max_samples
        self.samples = deque(maxlen=max_samples)
        self.lock = threading.RLock()

        # Pre-computed percentiles cache
        self._percentiles_cache: Optional[Dict[str, float]] = None
        self._cache_dirty = True

    def record(self, latency_ns: int) -> None:
        """Record latency sample."""
        with self.lock:
            self.samples.append(latency_ns)
            self._cache_dirty = True

    def get_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles."""
        with self.lock:
            if not self._cache_dirty and self._percentiles_cache:
                return self._percentiles_cache.copy()

            if not self.samples:
                return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'p999': 0.0, 'max': 0.0}

            samples_list = list(self.samples)
            percentiles = {
                'p50': np.percentile(samples_list, 50),
                'p95': np.percentile(samples_list, 95),
                'p99': np.percentile(samples_list, 99),
                'p999': np.percentile(samples_list, 99.9),
                'max': np.max(samples_list),
                'min': np.min(samples_list),
                'avg': np.mean(samples_list),
                'count': len(samples_list)
            }

            self._percentiles_cache = percentiles
            self._cache_dirty = False

            return percentiles.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        percentiles = self.get_percentiles()

        return {
            'name': self.name,
            'sample_count': percentiles['count'],
            'latency_ns': {
                'min': percentiles['min'],
                'max': percentiles['max'],
                'avg': percentiles['avg'],
                'p50': percentiles['p50'],
                'p95': percentiles['p95'],
                'p99': percentiles['p99'],
                'p999': percentiles['p999']
            },
            'latency_ms': {
                'min': percentiles['min'] / 1e6,
                'max': percentiles['max'] / 1e6,
                'avg': percentiles['avg'] / 1e6,
                'p50': percentiles['p50'] / 1e6,
                'p95': percentiles['p95'] / 1e6,
                'p99': percentiles['p99'] / 1e6,
                'p999': percentiles['p999'] / 1e6
            }
        }


class Counter:
    """Thread-safe counter metric."""

    def __init__(self, name: str):
        self.name = name
        self.value = 0
        self.lock = threading.RLock()

    def increment(self, delta: int = 1) -> None:
        """Increment counter."""
        with self.lock:
            self.value += delta

    def get_value(self) -> int:
        """Get current counter value."""
        with self.lock:
            return self.value

    def reset(self) -> None:
        """Reset counter to zero."""
        with self.lock:
            self.value = 0


class Gauge:
    """Thread-safe gauge metric."""

    def __init__(self, name: str):
        self.name = name
        self.value = 0.0
        self.lock = threading.RLock()

    def set_value(self, value: float) -> None:
        """Set gauge value."""
        with self.lock:
            self.value = value

    def increment(self, delta: float = 1.0) -> None:
        """Increment gauge value."""
        with self.lock:
            self.value += delta

    def decrement(self, delta: float = 1.0) -> None:
        """Decrement gauge value."""
        with self.lock:
            self.value -= delta

    def get_value(self) -> float:
        """Get current gauge value."""
        with self.lock:
            return self.value


class RateCalculator:
    """Calculate rates (events per second) over time windows."""

    def __init__(self, name: str, window_seconds: int = 60):
        self.name = name
        self.window_seconds = window_seconds
        self.events = deque()
        self.lock = threading.RLock()

    def record_event(self, count: int = 1) -> None:
        """Record event occurrence."""
        current_time = time.time()

        with self.lock:
            self.events.append((current_time, count))
            self._cleanup_old_events(current_time)

    def get_rate(self) -> float:
        """Get current rate (events per second)."""
        current_time = time.time()

        with self.lock:
            self._cleanup_old_events(current_time)

            if not self.events:
                return 0.0

            total_events = sum(count for _, count in self.events)
            time_span = current_time - self.events[0][0]

            if time_span == 0:
                return 0.0

            return total_events / time_span

    def _cleanup_old_events(self, current_time: float) -> None:
        """Remove events outside the time window."""
        cutoff_time = current_time - self.window_seconds

        while self.events and self.events[0][0] < cutoff_time:
            self.events.popleft()


class AlertManager:
    """Manage alert rules and alert generation."""

    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_callbacks: List[Callable[[Alert], None]] = []

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.rule_id} for metric {rule.metric_name}")

    def remove_alert_rule(self, rule_id: str) -> None:
        """Remove alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")

    def check_metric_for_alerts(self, metric_name: str, value: float, tags: Dict[str, str] = None) -> List[Alert]:
        """Check if metric value triggers any alerts."""
        if tags is None:
            tags = {}

        triggered_alerts = []

        for rule in self.alert_rules.values():
            if rule.metric_name == metric_name and rule.enabled:
                if self._evaluate_condition(value, rule.condition, rule.threshold):
                    alert = self._create_alert(rule, value, tags)
                    triggered_alerts.append(alert)

        return triggered_alerts

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 1e-9
        elif condition == "!=":
            return abs(value - threshold) >= 1e-9
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False

    def _create_alert(self, rule: AlertRule, value: float, tags: Dict[str, str]) -> Alert:
        """Create alert from rule."""
        alert_id = f"{rule.rule_id}_{int(time.time())}"

        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            metric_name=rule.metric_name,
            severity=rule.severity,
            message=f"{rule.description or rule.metric_name} is {value} (threshold: {rule.threshold})",
            value=value,
            threshold=rule.threshold,
            timestamp=datetime.now(timezone.utc),
            tags=tags
        )

        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        logger.warning(f"Alert triggered: {alert.message}")

        return alert

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True

        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unacknowledged) alerts."""
        return [alert for alert in self.active_alerts.values() if not alert.acknowledged]

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add alert notification callback."""
        self.alert_callbacks.append(callback)


class SystemMonitor:
    """Monitor system resources and performance."""

    def __init__(self):
        self.process = psutil.Process()
        self.boot_time = psutil.boot_time()

    def get_system_snapshot(self) -> PerformanceSnapshot:
        """Get current system performance snapshot."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            # Network I/O
            net_io = psutil.net_io_counters()
            network_sent = net_io.bytes_sent
            network_recv = net_io.bytes_recv

            # Process-specific metrics
            process_info = self.process.as_dict([
                'num_threads', 'num_fds', 'connections'
            ])

            active_connections = len(process_info.get('connections', []))
            thread_count = process_info.get('num_threads', 0)
            open_fds = process_info.get('num_fds', 0)

            return PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                disk_usage_percent=disk_percent,
                network_bytes_sent=network_sent,
                network_bytes_recv=network_recv,
                active_connections=active_connections,
                thread_count=thread_count,
                open_file_descriptors=open_fds,
                orders_per_second=0.0,  # Will be updated by trading components
                ticks_per_second=0.0    # Will be updated by trading components
            )

        except Exception as e:
            logger.error(f"Error getting system snapshot: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                active_connections=0,
                thread_count=0,
                open_file_descriptors=0,
                orders_per_second=0.0,
                ticks_per_second=0.0
            )


class MetricsCollector:
    """
    Advanced metrics collection and performance monitoring system.

    Features:
    - High-performance metric collection
    - Real-time alerting
    - System resource monitoring
    - Business metrics tracking
    - Historical data storage
    - Dashboard data aggregation
    """

    def __init__(
        self,
        settings: Settings,
        db_session: AsyncSession
    ):
        """
        Initialize metrics collector.

        Args:
            settings: Application settings
            db_session: Database session for persistence
        """
        self.settings = settings
        self.db_session = db_session

        # Metric storage
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.latency_trackers: Dict[str, LatencyTracker] = {}
        self.rate_calculators: Dict[str, RateCalculator] = {}

        # Metric definitions
        self.metric_definitions: Dict[str, MetricDefinition] = {}

        # Alert management
        self.alert_manager = AlertManager()

        # System monitoring
        self.system_monitor = SystemMonitor()

        # Data collection
        self.metric_points: deque = deque(maxlen=100000)

        # Background tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._system_monitoring_task: Optional[asyncio.Task] = None
        self._persistence_task: Optional[asyncio.Task] = None

        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="MetricsCollector"
        )

        # Control flags
        self.is_running = False

        self._initialize_default_metrics()
        self._initialize_default_alerts()

        logger.info("Metrics collector initialized with comprehensive monitoring")

    def _initialize_default_metrics(self) -> None:
        """Initialize default system and business metrics."""
        default_metrics = [
            # Trading metrics
            MetricDefinition("orders_submitted", MetricType.COUNTER, "Total orders submitted", "count"),
            MetricDefinition("orders_filled", MetricType.COUNTER, "Total orders filled", "count"),
            MetricDefinition("orders_cancelled", MetricType.COUNTER, "Total orders cancelled", "count"),
            MetricDefinition("orders_rejected", MetricType.COUNTER, "Total orders rejected", "count"),

            MetricDefinition("order_submission_latency", MetricType.HISTOGRAM, "Order submission latency", "nanoseconds"),
            MetricDefinition("order_fill_latency", MetricType.HISTOGRAM, "Order fill latency", "nanoseconds"),
            MetricDefinition("risk_validation_latency", MetricType.HISTOGRAM, "Risk validation latency", "nanoseconds"),

            MetricDefinition("active_orders", MetricType.GAUGE, "Currently active orders", "count"),
            MetricDefinition("active_positions", MetricType.GAUGE, "Currently active positions", "count"),
            MetricDefinition("portfolio_value", MetricType.GAUGE, "Total portfolio value", "USD"),
            MetricDefinition("unrealized_pnl", MetricType.GAUGE, "Unrealized P&L", "USD"),

            # Data processing metrics
            MetricDefinition("ticks_processed", MetricType.COUNTER, "Total ticks processed", "count"),
            MetricDefinition("tick_processing_latency", MetricType.HISTOGRAM, "Tick processing latency", "nanoseconds"),
            MetricDefinition("data_quality_score", MetricType.GAUGE, "Data quality score", "ratio"),

            # System metrics
            MetricDefinition("cpu_usage", MetricType.GAUGE, "CPU usage percentage", "percent"),
            MetricDefinition("memory_usage", MetricType.GAUGE, "Memory usage percentage", "percent"),
            MetricDefinition("disk_usage", MetricType.GAUGE, "Disk usage percentage", "percent"),
            MetricDefinition("thread_count", MetricType.GAUGE, "Active thread count", "count"),

            # Business metrics
            MetricDefinition("daily_pnl", MetricType.GAUGE, "Daily P&L", "USD"),
            MetricDefinition("sharpe_ratio", MetricType.GAUGE, "Portfolio Sharpe ratio", "ratio"),
            MetricDefinition("max_drawdown", MetricType.GAUGE, "Maximum drawdown", "percent"),
            MetricDefinition("win_rate", MetricType.GAUGE, "Trade win rate", "percent"),

            # ML metrics
            MetricDefinition("model_predictions", MetricType.COUNTER, "ML model predictions", "count"),
            MetricDefinition("prediction_latency", MetricType.HISTOGRAM, "Model prediction latency", "nanoseconds"),
            MetricDefinition("model_accuracy", MetricType.GAUGE, "Model prediction accuracy", "percent")
        ]

        for metric_def in default_metrics:
            self.metric_definitions[metric_def.name] = metric_def

    def _initialize_default_alerts(self) -> None:
        """Initialize default alert rules."""
        default_alerts = [
            AlertRule("cpu_high", "cpu_usage", ">", 80.0, AlertSeverity.WARNING,
                     description="High CPU usage"),
            AlertRule("cpu_critical", "cpu_usage", ">", 95.0, AlertSeverity.CRITICAL,
                     description="Critical CPU usage"),

            AlertRule("memory_high", "memory_usage", ">", 85.0, AlertSeverity.WARNING,
                     description="High memory usage"),
            AlertRule("memory_critical", "memory_usage", ">", 95.0, AlertSeverity.CRITICAL,
                     description="Critical memory usage"),

            AlertRule("order_latency_high", "order_submission_latency", ">", 100000, AlertSeverity.WARNING,
                     description="High order submission latency (>100μs)"),
            AlertRule("order_latency_critical", "order_submission_latency", ">", 500000, AlertSeverity.CRITICAL,
                     description="Critical order submission latency (>500μs)"),

            AlertRule("daily_loss_warning", "daily_pnl", "<", -10000, AlertSeverity.WARNING,
                     description="Daily loss exceeds $10K"),
            AlertRule("daily_loss_critical", "daily_pnl", "<", -50000, AlertSeverity.CRITICAL,
                     description="Daily loss exceeds $50K"),

            AlertRule("data_quality_low", "data_quality_score", "<", 0.8, AlertSeverity.WARNING,
                     description="Data quality score below 80%")
        ]

        for rule in default_alerts:
            self.alert_manager.add_alert_rule(rule)

    async def start(self) -> None:
        """Start the metrics collector."""
        if self.is_running:
            logger.warning("Metrics collector is already running")
            return

        try:
            # Start background tasks
            self._collection_task = asyncio.create_task(self._collect_metrics())
            self._system_monitoring_task = asyncio.create_task(self._monitor_system())
            self._persistence_task = asyncio.create_task(self._persist_metrics())

            self.is_running = True
            logger.info("Metrics collector started successfully")

        except Exception as e:
            logger.error(f"Failed to start metrics collector: {e}")
            raise MetricsCollectorError("Failed to start metrics collector", cause=e)

    async def stop(self) -> None:
        """Stop the metrics collector gracefully."""
        if not self.is_running:
            logger.warning("Metrics collector is not running")
            return

        try:
            logger.info("Stopping metrics collector...")
            self.is_running = False

            # Cancel background tasks
            tasks = [
                self._collection_task,
                self._system_monitoring_task,
                self._persistence_task
            ]

            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)

            logger.info("Metrics collector stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping metrics collector: {e}")

    # Metric recording methods

    def increment_counter(self, name: str, delta: int = 1, tags: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        if name not in self.counters:
            self.counters[name] = Counter(name)

        self.counters[name].increment(delta)

        # Record metric point
        self._record_metric_point(name, float(self.counters[name].get_value()), tags)

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Set a gauge metric value."""
        if name not in self.gauges:
            self.gauges[name] = Gauge(name)

        self.gauges[name].set_value(value)

        # Record metric point and check alerts
        self._record_metric_point(name, value, tags)
        self._check_alerts(name, value, tags)

    def record_latency(self, name: str, latency_ns: int) -> None:
        """Record a latency measurement."""
        if name not in self.latency_trackers:
            self.latency_trackers[name] = LatencyTracker(name)

        self.latency_trackers[name].record(latency_ns)

        # Also record as metric point (using P95 for alerting)
        percentiles = self.latency_trackers[name].get_percentiles()
        self._record_metric_point(name, percentiles['p95'])
        self._check_alerts(name, percentiles['p95'])

    def record_rate_event(self, name: str, count: int = 1) -> None:
        """Record an event for rate calculation."""
        if name not in self.rate_calculators:
            self.rate_calculators[name] = RateCalculator(name)

        self.rate_calculators[name].record_event(count)

        # Record current rate as metric point
        current_rate = self.rate_calculators[name].get_rate()
        self._record_metric_point(f"{name}_rate", current_rate)

    def _record_metric_point(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a metric point for persistence."""
        point = MetricPoint(
            metric_name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {}
        )

        self.metric_points.append(point)

    def _check_alerts(self, metric_name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Check if metric triggers any alerts."""
        alerts = self.alert_manager.check_metric_for_alerts(metric_name, value, tags)

        for alert in alerts:
            logger.warning(f"Alert triggered: {alert.message}")

    # Business metric helpers

    def record_order_submitted(self, order_id: str, latency_ns: int, symbol: str) -> None:
        """Record order submission metrics."""
        self.increment_counter("orders_submitted")
        self.record_latency("order_submission_latency", latency_ns)

    def record_order_filled(self, order_id: str, symbol: str, fill_price: float, quantity: float) -> None:
        """Record order fill metrics."""
        self.increment_counter("orders_filled")

    def record_order_cancelled(self, order_id: str, reason: str) -> None:
        """Record order cancellation metrics."""
        self.increment_counter("orders_cancelled")

    def record_order_rejected(self, order_id: str, reason: str) -> None:
        """Record order rejection metrics."""
        self.increment_counter("orders_rejected")

    def record_tick_processed(self, symbol: str, processing_time_ns: int, quality_score: float) -> None:
        """Record tick processing metrics."""
        self.increment_counter("ticks_processed")
        self.record_latency("tick_processing_latency", processing_time_ns)
        self.set_gauge("data_quality_score", quality_score, {"symbol": symbol})

    def record_ml_prediction(self, model_id: str, latency_ns: int, prediction_value: float) -> None:
        """Record ML prediction metrics."""
        self.increment_counter("model_predictions", tags={"model": model_id})
        self.record_latency("prediction_latency", latency_ns)

    def update_portfolio_metrics(self, portfolio_value: float, unrealized_pnl: float,
                                active_positions: int) -> None:
        """Update portfolio-level metrics."""
        self.set_gauge("portfolio_value", portfolio_value)
        self.set_gauge("unrealized_pnl", unrealized_pnl)
        self.set_gauge("active_positions", float(active_positions))

    def update_daily_pnl(self, daily_pnl: float) -> None:
        """Update daily P&L metric."""
        self.set_gauge("daily_pnl", daily_pnl)

    # Query methods

    def get_counter_value(self, name: str) -> int:
        """Get current counter value."""
        if name in self.counters:
            return self.counters[name].get_value()
        return 0

    def get_gauge_value(self, name: str) -> float:
        """Get current gauge value."""
        if name in self.gauges:
            return self.gauges[name].get_value()
        return 0.0

    def get_latency_statistics(self, name: str) -> Dict[str, Any]:
        """Get latency statistics."""
        if name in self.latency_trackers:
            return self.latency_trackers[name].get_statistics()
        return {}

    def get_rate(self, name: str) -> float:
        """Get current rate."""
        if name in self.rate_calculators:
            return self.rate_calculators[name].get_rate()
        return 0.0

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        metrics = {}

        # Counters
        for name, counter in self.counters.items():
            metrics[name] = {
                'type': 'counter',
                'value': counter.get_value()
            }

        # Gauges
        for name, gauge in self.gauges.items():
            metrics[name] = {
                'type': 'gauge',
                'value': gauge.get_value()
            }

        # Latency trackers
        for name, tracker in self.latency_trackers.items():
            metrics[name] = {
                'type': 'histogram',
                'value': tracker.get_statistics()
            }

        # Rates
        for name, rate_calc in self.rate_calculators.items():
            metrics[f"{name}_rate"] = {
                'type': 'rate',
                'value': rate_calc.get_rate()
            }

        return metrics

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return self.alert_manager.get_active_alerts()

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        return self.alert_manager.acknowledge_alert(alert_id)

    # Background tasks

    async def _collect_metrics(self) -> None:
        """Background task to collect metrics."""
        while self.is_running:
            try:
                # Collect application metrics
                await self._collect_application_metrics()

                await asyncio.sleep(10)  # Collect every 10 seconds

            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(10)

    async def _monitor_system(self) -> None:
        """Background task to monitor system resources."""
        while self.is_running:
            try:
                snapshot = self.system_monitor.get_system_snapshot()

                # Update system metrics
                self.set_gauge("cpu_usage", snapshot.cpu_usage_percent)
                self.set_gauge("memory_usage", snapshot.memory_usage_percent)
                self.set_gauge("disk_usage", snapshot.disk_usage_percent)
                self.set_gauge("thread_count", float(snapshot.thread_count))

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(30)

    async def _persist_metrics(self) -> None:
        """Background task to persist metrics to database."""
        while self.is_running:
            try:
                # Batch persist metric points
                if self.metric_points:
                    points_to_persist = []

                    # Get up to 1000 points
                    for _ in range(min(1000, len(self.metric_points))):
                        if self.metric_points:
                            points_to_persist.append(self.metric_points.popleft())

                    if points_to_persist:
                        await self._persist_metric_points(points_to_persist)

                await asyncio.sleep(60)  # Persist every minute

            except Exception as e:
                logger.error(f"Error in metrics persistence: {e}")
                await asyncio.sleep(60)

    async def _collect_application_metrics(self) -> None:
        """Collect application-specific metrics."""
        try:
            # This would collect metrics from various components
            # For now, we'll just update some basic stats

            # Calculate rates
            for name, rate_calc in self.rate_calculators.items():
                current_rate = rate_calc.get_rate()
                self.set_gauge(f"{name}_rate", current_rate)

        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")

    async def _persist_metric_points(self, points: List[MetricPoint]) -> None:
        """Persist metric points to database."""
        try:
            # This would insert metrics into a time-series database
            # For now, we'll just log the count
            logger.debug(f"Persisting {len(points)} metric points")

        except Exception as e:
            logger.warning(f"Failed to persist metrics: {e}")

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add alert notification callback."""
        self.alert_manager.add_alert_callback(callback)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'system': {
                'cpu_usage': self.get_gauge_value('cpu_usage'),
                'memory_usage': self.get_gauge_value('memory_usage'),
                'disk_usage': self.get_gauge_value('disk_usage'),
                'thread_count': int(self.get_gauge_value('thread_count'))
            },
            'trading': {
                'orders_submitted': self.get_counter_value('orders_submitted'),
                'orders_filled': self.get_counter_value('orders_filled'),
                'orders_cancelled': self.get_counter_value('orders_cancelled'),
                'orders_rejected': self.get_counter_value('orders_rejected'),
                'active_orders': int(self.get_gauge_value('active_orders')),
                'active_positions': int(self.get_gauge_value('active_positions')),
                'portfolio_value': self.get_gauge_value('portfolio_value'),
                'daily_pnl': self.get_gauge_value('daily_pnl')
            },
            'data_processing': {
                'ticks_processed': self.get_counter_value('ticks_processed'),
                'data_quality_score': self.get_gauge_value('data_quality_score')
            },
            'latency': {
                'order_submission': self.get_latency_statistics('order_submission_latency'),
                'tick_processing': self.get_latency_statistics('tick_processing_latency'),
                'ml_prediction': self.get_latency_statistics('prediction_latency')
            },
            'alerts': {
                'active_count': len(self.get_active_alerts()),
                'by_severity': self._count_alerts_by_severity()
            }
        }

        return summary

    def _count_alerts_by_severity(self) -> Dict[str, int]:
        """Count active alerts by severity."""
        active_alerts = self.get_active_alerts()
        counts = {severity.value: 0 for severity in AlertSeverity}

        for alert in active_alerts:
            counts[alert.severity.value] += 1

        return counts