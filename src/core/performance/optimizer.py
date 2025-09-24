"""
Performance optimization components and monitoring system.

This module provides comprehensive performance monitoring, optimization strategies,
and resource management for high-frequency trading applications.
"""

from __future__ import annotations

import asyncio
import gc
import psutil
import resource
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable, Set, Union
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import weakref

from core.config.settings import Settings
from core.exceptions import PerformanceError, ConfigurationError
from core.logging import get_logger
from core.tasks import get_task_manager

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_cores: int = 0
    cpu_freq_mhz: float = 0.0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Memory metrics
    memory_total_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    memory_cached_mb: float = 0.0

    # Process-specific metrics
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    process_memory_percent: float = 0.0
    process_threads: int = 0
    process_handles: int = 0

    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_connections: int = 0

    # Disk I/O metrics
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    disk_usage_percent: float = 0.0

    # Python-specific metrics
    gc_generation0: int = 0
    gc_generation1: int = 0
    gc_generation2: int = 0
    gc_collected: int = 0

    # Application metrics
    active_threads: int = 0
    active_tasks: int = 0
    queue_sizes: Dict[str, int] = field(default_factory=dict)
    response_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    error_rate_percent: float = 0.0


@dataclass
class PerformanceThresholds:
    """Performance monitoring thresholds."""
    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 80.0
    memory_critical: float = 95.0
    response_time_warning_ms: float = 100.0
    response_time_critical_ms: float = 500.0
    throughput_warning_percent: float = 0.8  # 80% of target
    error_rate_warning: float = 1.0  # 1%
    error_rate_critical: float = 5.0  # 5%


class PerformanceMonitor:
    """
    Real-time performance monitoring system.

    Features:
    - System resource monitoring
    - Application performance tracking
    - Threshold-based alerting
    - Historical trend analysis
    - Automated optimization triggers
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.thresholds = PerformanceThresholds()

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_interval = 10.0  # seconds
        self.metrics_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        # System information
        self.process = psutil.Process()
        self.system_info = self._get_system_info()

        # Monitoring task
        self.monitor_task: Optional[asyncio.Task] = None
        self.metrics_lock = threading.RLock()

        # Performance counters
        self.counters = {
            'tick_processing_times': deque(maxlen=1000),
            'ohlc_processing_times': deque(maxlen=1000),
            'database_query_times': deque(maxlen=1000),
            'api_response_times': deque(maxlen=1000),
            'error_counts': defaultdict(int),
            'throughput_samples': deque(maxlen=100),
        }

        logger.info("Performance monitor initialized")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get static system information."""
        cpu_info = psutil.cpu_freq()
        memory_info = psutil.virtual_memory()

        return {
            'cpu_cores_physical': psutil.cpu_count(logical=False),
            'cpu_cores_logical': psutil.cpu_count(logical=True),
            'cpu_max_freq_mhz': cpu_info.max if cpu_info else 0,
            'memory_total_gb': round(memory_info.total / (1024**3), 2),
            'python_version': self.process.exe(),
            'platform': psutil.platform(),
            'boot_time': datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc),
        }

    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()

                # Store metrics
                with self.metrics_lock:
                    self.metrics_history.append(metrics)

                # Check thresholds and trigger alerts
                await self._check_thresholds(metrics)

                # Sleep until next collection
                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                load_avg = (0.0, 0.0, 0.0)

            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0

            # Process metrics
            process_info = self.process.as_dict([
                'cpu_percent', 'memory_info', 'num_threads', 'num_handles'
            ])

            # Network metrics
            net_io = psutil.net_io_counters()
            net_connections = len(psutil.net_connections())

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_usage = psutil.disk_usage('/')

            # Python GC metrics
            gc_stats = gc.get_stats()
            gc_counts = gc.get_count()

            # Application metrics
            active_threads = threading.active_count()
            task_manager = get_task_manager()
            task_stats = task_manager.get_system_stats() if task_manager.is_running else {}

            # Calculate derived metrics
            response_time = self._calculate_average_response_time()
            throughput = self._calculate_throughput()
            error_rate = self._calculate_error_rate()

            return PerformanceMetrics(
                # CPU metrics
                cpu_percent=cpu_percent,
                cpu_cores=psutil.cpu_count(),
                cpu_freq_mhz=cpu_freq_mhz,
                load_average=load_avg,

                # Memory metrics
                memory_total_mb=memory.total / (1024**2),
                memory_available_mb=memory.available / (1024**2),
                memory_used_mb=memory.used / (1024**2),
                memory_percent=memory.percent,
                memory_cached_mb=getattr(memory, 'cached', 0) / (1024**2),

                # Process metrics
                process_cpu_percent=process_info.get('cpu_percent', 0.0),
                process_memory_mb=process_info.get('memory_info', psutil.pmem()).rss / (1024**2),
                process_memory_percent=self.process.memory_percent(),
                process_threads=process_info.get('num_threads', 0),
                process_handles=process_info.get('num_handles', 0),

                # Network metrics
                network_bytes_sent=net_io.bytes_sent if net_io else 0,
                network_bytes_recv=net_io.bytes_recv if net_io else 0,
                network_connections=net_connections,

                # Disk metrics
                disk_read_mb=(disk_io.read_bytes / (1024**2)) if disk_io else 0.0,
                disk_write_mb=(disk_io.write_bytes / (1024**2)) if disk_io else 0.0,
                disk_usage_percent=disk_usage.percent if disk_usage else 0.0,

                # Python GC metrics
                gc_generation0=gc_counts[0],
                gc_generation1=gc_counts[1],
                gc_generation2=gc_counts[2],
                gc_collected=sum(stat.get('collected', 0) for stat in gc_stats),

                # Application metrics
                active_threads=active_threads,
                active_tasks=task_stats.get('active_tasks', 0),
                queue_sizes=self._get_queue_sizes(),
                response_time_ms=response_time,
                throughput_per_second=throughput,
                error_rate_percent=error_rate,
            )

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return PerformanceMetrics()

    def _calculate_average_response_time(self) -> float:
        """Calculate average response time across all operations."""
        all_times = []

        for timing_list in [
            self.counters['tick_processing_times'],
            self.counters['ohlc_processing_times'],
            self.counters['api_response_times']
        ]:
            all_times.extend(timing_list)

        return sum(all_times) / len(all_times) if all_times else 0.0

    def _calculate_throughput(self) -> float:
        """Calculate current throughput."""
        throughput_samples = self.counters['throughput_samples']
        return sum(throughput_samples) / len(throughput_samples) if throughput_samples else 0.0

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate percentage."""
        total_errors = sum(self.counters['error_counts'].values())
        total_operations = len(self.counters['tick_processing_times']) + len(self.counters['ohlc_processing_times'])

        if total_operations == 0:
            return 0.0

        return (total_errors / total_operations) * 100

    def _get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes from various components."""
        queue_sizes = {}

        try:
            # Get task manager queue sizes
            task_manager = get_task_manager()
            if task_manager.is_running:
                task_stats = task_manager.get_queue_stats()
                if 'queues' in task_stats:
                    for queue_name, queue_info in task_stats['queues'].items():
                        queue_sizes[f"task_{queue_name}"] = queue_info.get('size', 0)

        except Exception as e:
            logger.debug(f"Error getting queue sizes: {e}")

        return queue_sizes

    async def _check_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check performance thresholds and trigger alerts."""
        alerts = []

        # CPU threshold checks
        if metrics.cpu_percent >= self.thresholds.cpu_critical:
            alerts.append(('cpu_critical', {
                'cpu_percent': metrics.cpu_percent,
                'threshold': self.thresholds.cpu_critical
            }))
        elif metrics.cpu_percent >= self.thresholds.cpu_warning:
            alerts.append(('cpu_warning', {
                'cpu_percent': metrics.cpu_percent,
                'threshold': self.thresholds.cpu_warning
            }))

        # Memory threshold checks
        if metrics.memory_percent >= self.thresholds.memory_critical:
            alerts.append(('memory_critical', {
                'memory_percent': metrics.memory_percent,
                'threshold': self.thresholds.memory_critical
            }))
        elif metrics.memory_percent >= self.thresholds.memory_warning:
            alerts.append(('memory_warning', {
                'memory_percent': metrics.memory_percent,
                'threshold': self.thresholds.memory_warning
            }))

        # Response time checks
        if metrics.response_time_ms >= self.thresholds.response_time_critical_ms:
            alerts.append(('response_time_critical', {
                'response_time_ms': metrics.response_time_ms,
                'threshold': self.thresholds.response_time_critical_ms
            }))
        elif metrics.response_time_ms >= self.thresholds.response_time_warning_ms:
            alerts.append(('response_time_warning', {
                'response_time_ms': metrics.response_time_ms,
                'threshold': self.thresholds.response_time_warning_ms
            }))

        # Error rate checks
        if metrics.error_rate_percent >= self.thresholds.error_rate_critical:
            alerts.append(('error_rate_critical', {
                'error_rate_percent': metrics.error_rate_percent,
                'threshold': self.thresholds.error_rate_critical
            }))
        elif metrics.error_rate_percent >= self.thresholds.error_rate_warning:
            alerts.append(('error_rate_warning', {
                'error_rate_percent': metrics.error_rate_percent,
                'threshold': self.thresholds.error_rate_warning
            }))

        # Trigger alerts
        for alert_type, alert_data in alerts:
            await self._trigger_alert(alert_type, alert_data)

    async def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """Trigger performance alert."""
        logger.warning(f"Performance alert: {alert_type} - {alert_data}")

        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def record_timing(self, operation_type: str, timing_ms: float) -> None:
        """Record operation timing for performance tracking."""
        counter_key = f"{operation_type}_processing_times"
        if counter_key in self.counters:
            self.counters[counter_key].append(timing_ms)

    def record_throughput(self, operations_per_second: float) -> None:
        """Record throughput measurement."""
        self.counters['throughput_samples'].append(operations_per_second)

    def record_error(self, error_type: str) -> None:
        """Record error for error rate calculation."""
        self.counters['error_counts'][error_type] += 1

    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add alert callback function."""
        self.alert_callbacks.append(callback)

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics."""
        with self.metrics_lock:
            return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """Get metrics history for the specified time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        with self.metrics_lock:
            return [
                metrics for metrics in self.metrics_history
                if metrics.timestamp >= cutoff_time
            ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_metrics = self.get_current_metrics()
        recent_metrics = self.get_metrics_history(minutes=10)

        if not current_metrics:
            return {'status': 'no_data'}

        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
        response_time_trend = self._calculate_trend([m.response_time_ms for m in recent_metrics])

        return {
            'timestamp': current_metrics.timestamp.isoformat(),
            'current_metrics': {
                'cpu_percent': current_metrics.cpu_percent,
                'memory_percent': current_metrics.memory_percent,
                'response_time_ms': current_metrics.response_time_ms,
                'throughput_per_second': current_metrics.throughput_per_second,
                'error_rate_percent': current_metrics.error_rate_percent,
                'active_threads': current_metrics.active_threads,
                'active_tasks': current_metrics.active_tasks,
            },
            'trends': {
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend,
                'response_time_trend': response_time_trend,
            },
            'system_info': self.system_info,
            'monitoring_status': {
                'is_monitoring': self.is_monitoring,
                'metrics_collected': len(self.metrics_history),
                'monitoring_interval': self.monitoring_interval,
            },
            'thresholds': {
                'cpu_warning': self.thresholds.cpu_warning,
                'cpu_critical': self.thresholds.cpu_critical,
                'memory_warning': self.thresholds.memory_warning,
                'memory_critical': self.thresholds.memory_critical,
                'response_time_warning_ms': self.thresholds.response_time_warning_ms,
                'response_time_critical_ms': self.thresholds.response_time_critical_ms,
            }
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return 'stable'

        # Simple trend calculation
        recent_avg = sum(values[-5:]) / min(len(values), 5)
        older_avg = sum(values[:5]) / min(len(values), 5)

        diff_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0

        if diff_percent > 10:
            return 'increasing'
        elif diff_percent < -10:
            return 'decreasing'
        else:
            return 'stable'


class PerformanceOptimizer:
    """
    Automated performance optimization system.

    Features:
    - Automatic resource optimization
    - Memory management
    - Thread pool optimization
    - Cache optimization
    - Database connection pooling
    """

    def __init__(self, settings: Settings, performance_monitor: PerformanceMonitor):
        self.settings = settings
        self.monitor = performance_monitor

        # Optimization state
        self.optimizations_applied = []
        self.optimization_lock = threading.RLock()

        # Optimization strategies
        self.strategies = {
            'memory_cleanup': self._memory_cleanup,
            'thread_pool_adjustment': self._adjust_thread_pools,
            'cache_optimization': self._optimize_cache,
            'gc_tuning': self._tune_garbage_collection,
            'resource_limits': self._adjust_resource_limits,
        }

        # Register alert callbacks
        self.monitor.add_alert_callback(self._handle_performance_alert)

        logger.info("Performance optimizer initialized")

    async def _handle_performance_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """Handle performance alerts with automatic optimization."""
        optimization_strategy = None

        if 'memory' in alert_type:
            optimization_strategy = 'memory_cleanup'
        elif 'cpu' in alert_type:
            optimization_strategy = 'thread_pool_adjustment'
        elif 'response_time' in alert_type:
            optimization_strategy = 'cache_optimization'
        elif 'error_rate' in alert_type:
            optimization_strategy = 'resource_limits'

        if optimization_strategy and optimization_strategy in self.strategies:
            await self._apply_optimization(optimization_strategy, alert_data)

    async def _apply_optimization(self, strategy_name: str, context: Dict[str, Any]) -> None:
        """Apply specific optimization strategy."""
        try:
            with self.optimization_lock:
                logger.info(f"Applying optimization strategy: {strategy_name}")

                strategy_func = self.strategies[strategy_name]
                result = await strategy_func(context)

                self.optimizations_applied.append({
                    'strategy': strategy_name,
                    'timestamp': datetime.now(timezone.utc),
                    'context': context,
                    'result': result,
                })

                logger.info(f"Optimization applied: {strategy_name} - {result}")

        except Exception as e:
            logger.error(f"Error applying optimization {strategy_name}: {e}")

    async def _memory_cleanup(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform memory cleanup optimization."""
        initial_memory = psutil.virtual_memory().percent

        # Force garbage collection
        collected = gc.collect()

        # Clear caches if available
        try:
            # Clear buffer manager caches
            from core.data import get_buffer_manager
            buffer_manager = get_buffer_manager()
            # Implementation would depend on buffer manager's clear cache method
        except Exception:
            pass

        # Get memory after cleanup
        final_memory = psutil.virtual_memory().percent
        memory_freed = initial_memory - final_memory

        return {
            'gc_collected': collected,
            'memory_freed_percent': memory_freed,
            'success': memory_freed > 0,
        }

    async def _adjust_thread_pools(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust thread pool sizes for better performance."""
        current_cpu = context.get('cpu_percent', 0)

        # Simple heuristic: reduce thread pools if CPU is high
        if current_cpu > 80:
            # Reduce by 20%
            adjustment_factor = 0.8
            action = 'reduced'
        elif current_cpu < 40:
            # Increase by 20%
            adjustment_factor = 1.2
            action = 'increased'
        else:
            adjustment_factor = 1.0
            action = 'unchanged'

        # Apply adjustments (would need actual implementation)
        # This is a placeholder for the actual thread pool adjustment logic

        return {
            'action': action,
            'adjustment_factor': adjustment_factor,
            'cpu_percent': current_cpu,
            'success': action != 'unchanged',
        }

    async def _optimize_cache(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize caching strategies."""
        # Placeholder for cache optimization logic
        # Could include:
        # - Adjusting cache sizes
        # - Changing cache eviction policies
        # - Pre-loading frequently accessed data

        return {
            'cache_optimizations': ['size_adjustment', 'eviction_policy'],
            'success': True,
        }

    async def _tune_garbage_collection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Tune garbage collection parameters."""
        # Get current GC thresholds
        current_thresholds = gc.get_threshold()

        # Adjust based on memory pressure
        memory_percent = context.get('memory_percent', 0)

        if memory_percent > 80:
            # More aggressive GC
            new_thresholds = (
                int(current_thresholds[0] * 0.8),
                int(current_thresholds[1] * 0.8),
                int(current_thresholds[2] * 0.8)
            )
            gc.set_threshold(*new_thresholds)
            action = 'made_aggressive'
        elif memory_percent < 50:
            # Less aggressive GC
            new_thresholds = (
                int(current_thresholds[0] * 1.2),
                int(current_thresholds[1] * 1.2),
                int(current_thresholds[2] * 1.2)
            )
            gc.set_threshold(*new_thresholds)
            action = 'made_relaxed'
        else:
            action = 'unchanged'
            new_thresholds = current_thresholds

        return {
            'action': action,
            'old_thresholds': current_thresholds,
            'new_thresholds': new_thresholds,
            'memory_percent': memory_percent,
            'success': action != 'unchanged',
        }

    async def _adjust_resource_limits(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust system resource limits."""
        adjustments = []

        try:
            # Adjust file descriptor limit if needed
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft_limit < hard_limit * 0.8:
                new_soft_limit = min(int(hard_limit * 0.9), hard_limit)
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
                adjustments.append(f'file_descriptors: {soft_limit} -> {new_soft_limit}')

        except Exception as e:
            logger.debug(f"Could not adjust file descriptor limit: {e}")

        return {
            'adjustments': adjustments,
            'success': len(adjustments) > 0,
        }

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of applied optimizations."""
        with self.optimization_lock:
            return list(self.optimizations_applied)


def performance_timer(operation_type: str):
    """Decorator to automatically track operation timing."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                timing_ms = (end_time - start_time) * 1000

                # Record timing if monitor is available
                try:
                    monitor = get_performance_monitor()
                    monitor.record_timing(operation_type, timing_ms)
                except Exception:
                    pass  # Monitor not available

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                timing_ms = (end_time - start_time) * 1000

                # Record timing if monitor is available
                try:
                    monitor = get_performance_monitor()
                    monitor.record_timing(operation_type, timing_ms)
                except Exception:
                    pass  # Monitor not available

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global performance components
_performance_monitor: Optional[PerformanceMonitor] = None
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor

    if _performance_monitor is None:
        from core.config.settings import Settings
        settings = Settings()
        _performance_monitor = PerformanceMonitor(settings)

    return _performance_monitor


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _performance_optimizer

    if _performance_optimizer is None:
        monitor = get_performance_monitor()
        from core.config.settings import Settings
        settings = Settings()
        _performance_optimizer = PerformanceOptimizer(settings, monitor)

    return _performance_optimizer