"""
Performance monitoring system.

This module provides comprehensive performance monitoring capabilities
with real-time metrics collection, alerting, and historical tracking.
"""

from __future__ import annotations

import asyncio
import psutil
import threading
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.tasks import background_task, scheduled_task

logger = get_logger(__name__)
settings = get_settings()


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"          # Incremental values
    GAUGE = "gauge"             # Current value snapshots
    HISTOGRAM = "histogram"      # Value distributions
    TIMER = "timer"             # Duration measurements


class PerformanceMetric(BaseModel):
    """Performance metric data point."""

    name: str = Field(description="Metric name")
    type: MetricType = Field(description="Metric type")
    value: Union[int, float] = Field(description="Metric value")
    timestamp: datetime = Field(description="Measurement timestamp")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")
    unit: str = Field(default="", description="Measurement unit")
    description: str = Field(default="", description="Metric description")

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "unit": self.unit,
            "description": self.description
        }


class SystemMetrics(BaseModel):
    """System-level performance metrics."""

    timestamp: datetime = Field(description="Metrics timestamp")

    # CPU metrics
    cpu_percent: float = Field(description="CPU usage percentage")
    cpu_count: int = Field(description="Number of CPU cores")
    load_average: List[float] = Field(description="System load average")

    # Memory metrics
    memory_total: int = Field(description="Total memory bytes")
    memory_available: int = Field(description="Available memory bytes")
    memory_percent: float = Field(description="Memory usage percentage")
    memory_used: int = Field(description="Used memory bytes")

    # Disk metrics
    disk_total: int = Field(description="Total disk space bytes")
    disk_used: int = Field(description="Used disk space bytes")
    disk_free: int = Field(description="Free disk space bytes")
    disk_percent: float = Field(description="Disk usage percentage")

    # Network metrics
    network_bytes_sent: int = Field(description="Network bytes sent")
    network_bytes_recv: int = Field(description="Network bytes received")
    network_packets_sent: int = Field(description="Network packets sent")
    network_packets_recv: int = Field(description="Network packets received")

    # Process metrics
    process_cpu_percent: float = Field(description="Process CPU usage")
    process_memory_percent: float = Field(description="Process memory usage")
    process_memory_rss: int = Field(description="Process RSS memory")
    process_memory_vms: int = Field(description="Process VMS memory")
    process_threads: int = Field(description="Number of process threads")
    process_connections: int = Field(description="Number of network connections")

    @property
    def memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self.memory_used / (1024 * 1024)

    @property
    def disk_usage_gb(self) -> float:
        """Get disk usage in GB."""
        return self.disk_used / (1024 * 1024 * 1024)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.

    Provides real-time metrics collection, alerting, historical tracking,
    and performance analysis capabilities.
    """

    def __init__(self):
        """Initialize performance monitor."""
        self._metrics: Dict[str, List[PerformanceMetric]] = {}
        self._system_metrics: List[SystemMetrics] = []
        self._lock = threading.RLock()

        # Configuration
        self._max_metrics_per_name = 10000
        self._max_system_metrics = 1000
        self._collection_interval = 10  # seconds

        # Monitoring state
        self._is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Alert system
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        # Performance tracking
        self._monitor_stats = {
            "total_metrics_collected": 0,
            "alerts_triggered": 0,
            "collection_errors": 0,
            "start_time": None
        }

        # Initialize system info
        self._process = psutil.Process()

        logger.info("Performance monitor initialized")

    def start_monitoring(self) -> bool:
        """Start performance monitoring."""
        if self._is_monitoring:
            logger.warning("Performance monitoring already started")
            return True

        try:
            self._is_monitoring = True
            self._monitor_stats["start_time"] = datetime.now(timezone.utc)

            # Start monitoring tasks
            self._start_monitoring_tasks()

            logger.info("Performance monitoring started")
            return True

        except Exception as e:
            logger.error(f"Failed to start performance monitoring: {e}")
            self._is_monitoring = False
            return False

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self._is_monitoring:
            return

        logger.info("Stopping performance monitoring...")

        self._is_monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()

        logger.info("Performance monitoring stopped")

    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        labels: Dict[str, str] = None,
        unit: str = "",
        description: str = ""
    ) -> None:
        """
        Record a performance metric.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Optional labels
            unit: Measurement unit
            description: Metric description
        """
        metric = PerformanceMetric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            unit=unit,
            description=description
        )

        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []

            self._metrics[name].append(metric)

            # Limit metric history
            if len(self._metrics[name]) > self._max_metrics_per_name:
                self._metrics[name] = self._metrics[name][-self._max_metrics_per_name:]

            self._monitor_stats["total_metrics_collected"] += 1

        # Check alert rules
        self._check_alerts(metric)

        logger.debug(f"Recorded metric {name}: {value} {unit}")

    def increment_counter(self, name: str, value: Union[int, float] = 1, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, labels)

    def set_gauge(self, name: str, value: Union[int, float], labels: Dict[str, str] = None, unit: str = "") -> None:
        """Set a gauge metric value."""
        self.record_metric(name, value, MetricType.GAUGE, labels, unit)

    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None) -> None:
        """Record a timer metric (duration in seconds)."""
        self.record_metric(name, duration, MetricType.TIMER, labels, "seconds")

    def get_metrics(self, name: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """
        Get metrics by name or all metrics.

        Args:
            name: Specific metric name (all if None)
            limit: Maximum number of metrics to return

        Returns:
            List of performance metrics
        """
        with self._lock:
            if name:
                return self._metrics.get(name, [])[-limit:]
            else:
                all_metrics = []
                for metric_list in self._metrics.values():
                    all_metrics.extend(metric_list[-limit:])
                return sorted(all_metrics, key=lambda m: m.timestamp)[-limit:]

    def get_metric_names(self) -> List[str]:
        """Get all metric names."""
        with self._lock:
            return list(self._metrics.keys())

    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the latest system metrics."""
        with self._lock:
            return self._system_metrics[-1] if self._system_metrics else None

    def get_system_metrics_history(self, limit: int = 100) -> List[SystemMetrics]:
        """Get system metrics history."""
        with self._lock:
            return self._system_metrics[-limit:] if self._system_metrics else []

    def add_alert_rule(
        self,
        rule_name: str,
        metric_name: str,
        condition: str,  # "gt", "lt", "eq", "ge", "le"
        threshold: Union[int, float],
        duration_minutes: int = 1,
        enabled: bool = True
    ) -> None:
        """
        Add performance alert rule.

        Args:
            rule_name: Alert rule name
            metric_name: Metric to monitor
            condition: Comparison condition
            threshold: Alert threshold value
            duration_minutes: Duration to sustain condition
            enabled: Whether rule is enabled
        """
        self._alert_rules[rule_name] = {
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "duration_minutes": duration_minutes,
            "enabled": enabled,
            "last_triggered": None,
            "trigger_count": 0
        }

        logger.info(f"Added alert rule: {rule_name}")

    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove alert rule."""
        if rule_name in self._alert_rules:
            del self._alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False

    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add alert callback function."""
        self._alert_callbacks.append(callback)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            latest_system = self.get_latest_system_metrics()

            summary = {
                "monitoring_enabled": self._is_monitoring,
                "metrics_count": len(self._metrics),
                "total_data_points": sum(len(metrics) for metrics in self._metrics.values()),
                "alert_rules": len(self._alert_rules),
                "stats": self._monitor_stats.copy()
            }

            if latest_system:
                summary["system"] = {
                    "cpu_percent": latest_system.cpu_percent,
                    "memory_percent": latest_system.memory_percent,
                    "disk_percent": latest_system.disk_percent,
                    "process_cpu": latest_system.process_cpu_percent,
                    "process_memory_mb": latest_system.memory_usage_mb,
                    "process_threads": latest_system.process_threads
                }

            return summary

    def get_metric_statistics(self, name: str, minutes: int = 60) -> Dict[str, Any]:
        """
        Get statistical analysis for a metric.

        Args:
            name: Metric name
            minutes: Time window in minutes

        Returns:
            Statistical summary
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        with self._lock:
            metrics = self._metrics.get(name, [])
            recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            if not recent_metrics:
                return {"error": "No recent metrics found"}

            values = [m.value for m in recent_metrics]

            stats = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1],
                "time_window_minutes": minutes
            }

            # Calculate percentiles
            sorted_values = sorted(values)
            n = len(sorted_values)

            stats["p50"] = sorted_values[int(n * 0.5)]
            stats["p95"] = sorted_values[int(n * 0.95)]
            stats["p99"] = sorted_values[int(n * 0.99)]

            return stats

    def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""

        @scheduled_task(
            interval_seconds=self._collection_interval,
            name="collect_system_metrics"
        )
        def collect_system_metrics():
            """Collect system performance metrics."""
            try:
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
                self._monitor_stats["collection_errors"] += 1

        @scheduled_task(
            interval_seconds=60,  # Every minute
            name="cleanup_old_metrics"
        )
        def cleanup_metrics():
            """Clean up old metrics data."""
            self._cleanup_old_metrics()

        logger.info("Started monitoring background tasks")

    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]

            # Memory metrics
            memory = psutil.virtual_memory()

            # Disk metrics
            disk = psutil.disk_usage('/')

            # Network metrics
            network = psutil.net_io_counters()

            # Process metrics
            process_cpu = self._process.cpu_percent()
            process_memory = self._process.memory_info()
            process_memory_percent = self._process.memory_percent()

            # Create system metrics
            system_metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                load_average=list(load_avg),
                memory_total=memory.total,
                memory_available=memory.available,
                memory_percent=memory.percent,
                memory_used=memory.used,
                disk_total=disk.total,
                disk_used=disk.used,
                disk_free=disk.free,
                disk_percent=disk.percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                network_packets_sent=network.packets_sent,
                network_packets_recv=network.packets_recv,
                process_cpu_percent=process_cpu,
                process_memory_percent=process_memory_percent,
                process_memory_rss=process_memory.rss,
                process_memory_vms=process_memory.vms,
                process_threads=self._process.num_threads(),
                process_connections=len(self._process.connections())
            )

            with self._lock:
                self._system_metrics.append(system_metrics)

                # Limit system metrics history
                if len(self._system_metrics) > self._max_system_metrics:
                    self._system_metrics = self._system_metrics[-self._max_system_metrics:]

            # Record as individual metrics for alerting
            self.set_gauge("system.cpu.percent", cpu_percent, unit="%")
            self.set_gauge("system.memory.percent", memory.percent, unit="%")
            self.set_gauge("system.disk.percent", disk.percent, unit="%")
            self.set_gauge("process.cpu.percent", process_cpu, unit="%")
            self.set_gauge("process.memory.mb", system_metrics.memory_usage_mb, unit="MB")

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            raise

    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics data."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

        with self._lock:
            cleaned_count = 0
            for name, metrics in self._metrics.items():
                original_count = len(metrics)
                self._metrics[name] = [m for m in metrics if m.timestamp >= cutoff_time]
                cleaned_count += original_count - len(self._metrics[name])

            if cleaned_count > 0:
                logger.debug(f"Cleaned up {cleaned_count} old metrics")

    def _check_alerts(self, metric: PerformanceMetric) -> None:
        """Check if metric triggers any alert rules."""
        for rule_name, rule in self._alert_rules.items():
            if not rule["enabled"] or rule["metric_name"] != metric.name:
                continue

            try:
                condition = rule["condition"]
                threshold = rule["threshold"]
                triggered = False

                if condition == "gt" and metric.value > threshold:
                    triggered = True
                elif condition == "lt" and metric.value < threshold:
                    triggered = True
                elif condition == "ge" and metric.value >= threshold:
                    triggered = True
                elif condition == "le" and metric.value <= threshold:
                    triggered = True
                elif condition == "eq" and metric.value == threshold:
                    triggered = True

                if triggered:
                    self._trigger_alert(rule_name, rule, metric)

            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")

    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], metric: PerformanceMetric) -> None:
        """Trigger an alert."""
        current_time = datetime.now(timezone.utc)

        # Update rule statistics
        rule["last_triggered"] = current_time
        rule["trigger_count"] += 1
        self._monitor_stats["alerts_triggered"] += 1

        # Create alert data
        alert_data = {
            "rule_name": rule_name,
            "metric_name": metric.name,
            "metric_value": metric.value,
            "threshold": rule["threshold"],
            "condition": rule["condition"],
            "timestamp": current_time,
            "metric": metric.to_dict()
        }

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(rule_name, alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Performance alert triggered: {rule_name} - {metric.name} = {metric.value}")


# Global monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor

    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()

    return _performance_monitor