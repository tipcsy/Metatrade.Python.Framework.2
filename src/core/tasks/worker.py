"""
Task worker system for executing background tasks.

This module provides worker processes and pools for executing tasks
from queues with comprehensive monitoring, resource management, and
error handling capabilities.
"""

from __future__ import annotations

import asyncio
import gc
import psutil
import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.config import get_settings
from src.core.exceptions import TaskExecutionError, ResourceError
from src.core.logging import get_logger
from .models import Task, TaskStatus, TaskResult, TaskStats
from .queue import get_task_queue_manager, PriorityTaskQueue

logger = get_logger(__name__)
settings = get_settings()


class TaskWorker:
    """
    Individual task worker for executing tasks from a queue.

    Provides comprehensive task execution with resource monitoring,
    error handling, timeout management, and performance tracking.
    """

    def __init__(
        self,
        worker_id: str,
        queue_name: str = "default",
        max_concurrent_tasks: int = 1,
        resource_limits: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize task worker.

        Args:
            worker_id: Unique worker identifier
            queue_name: Queue to consume tasks from
            max_concurrent_tasks: Maximum concurrent tasks
            resource_limits: Resource limits configuration
        """
        self.worker_id = worker_id
        self.queue_name = queue_name
        self.max_concurrent_tasks = max_concurrent_tasks

        # Resource limits
        self.resource_limits = resource_limits or {}
        self.memory_limit_mb = self.resource_limits.get("memory_mb", 1024)
        self.cpu_limit_percent = self.resource_limits.get("cpu_percent", 80)

        # State management
        self._is_running = False
        self._is_stopping = False
        self._current_tasks: Set[str] = set()
        self._lock = threading.RLock()

        # Task execution
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self._worker_thread: Optional[threading.Thread] = None

        # Performance tracking
        self._tasks_executed = 0
        self._tasks_failed = 0
        self._start_time = time.time()
        self._last_heartbeat = time.time()

        # Resource monitoring
        self._process = psutil.Process()
        self._baseline_memory = self._process.memory_info().rss / 1024 / 1024

        # Queue reference
        self._queue_manager = get_task_queue_manager()
        self._queue: Optional[PriorityTaskQueue] = None

        logger.info(f"Initialized worker {worker_id} for queue {queue_name}")

    def start(self) -> bool:
        """Start the task worker."""
        if self._is_running:
            logger.warning(f"Worker {self.worker_id} already running")
            return True

        try:
            # Get queue reference
            self._queue = self._queue_manager.get_or_create_queue(self.queue_name)

            # Start worker thread
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{self.worker_id}"
            )
            self._worker_thread.daemon = True
            self._worker_thread.start()

            self._is_running = True
            self._start_time = time.time()

            logger.info(f"Started worker {self.worker_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start worker {self.worker_id}: {e}")
            return False

    def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the task worker gracefully.

        Args:
            timeout: Maximum wait time for graceful shutdown
        """
        if not self._is_running:
            return

        logger.info(f"Stopping worker {self.worker_id}...")

        self._is_stopping = True
        self._is_running = False

        # Wait for current tasks to complete
        start_time = time.time()
        while len(self._current_tasks) > 0 and (time.time() - start_time) < timeout:
            logger.debug(f"Worker {self.worker_id} waiting for {len(self._current_tasks)} tasks")
            time.sleep(0.5)

        # Force shutdown if tasks still running
        if len(self._current_tasks) > 0:
            logger.warning(f"Force stopping worker {self.worker_id} with {len(self._current_tasks)} running tasks")
            self._executor.shutdown(wait=False)
        else:
            self._executor.shutdown(wait=True)

        # Wait for worker thread
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        logger.info(f"Stopped worker {self.worker_id}")

    def _worker_loop(self) -> None:
        """Main worker loop for processing tasks."""
        logger.debug(f"Started worker loop for {self.worker_id}")

        while self._is_running and not self._is_stopping:
            try:
                # Check resource limits
                if not self._check_resource_limits():
                    logger.warning(f"Worker {self.worker_id} resource limits exceeded, pausing")
                    time.sleep(5.0)
                    continue

                # Check if we can accept more tasks
                if len(self._current_tasks) >= self.max_concurrent_tasks:
                    time.sleep(0.1)
                    continue

                # Get task from queue
                task = self._queue.get(timeout=1.0)
                if task is None:
                    continue

                # Execute task asynchronously
                self._execute_task_async(task)

                # Update heartbeat
                self._last_heartbeat = time.time()

            except Exception as e:
                logger.error(f"Error in worker loop for {self.worker_id}: {e}")
                time.sleep(1.0)

        logger.debug(f"Exited worker loop for {self.worker_id}")

    def _execute_task_async(self, task: Task) -> None:
        """Execute task asynchronously in thread pool."""
        with self._lock:
            self._current_tasks.add(task.task_id)

        # Submit to thread pool
        future = self._executor.submit(self._execute_task, task)

        # Add completion callback
        def task_completed(fut):
            with self._lock:
                self._current_tasks.discard(task.task_id)

        future.add_done_callback(task_completed)

    def _execute_task(self, task: Task) -> TaskResult:
        """
        Execute a single task with comprehensive monitoring.

        Args:
            task: Task to execute

        Returns:
            TaskResult: Execution result
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Create initial result
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            attempt_number=task.context.get("retry_count", 0) + 1
        )

        try:
            logger.info(f"Executing task {task.task_id} ({task.name}) on worker {self.worker_id}")

            # Update task status
            task.status = TaskStatus.RUNNING
            task.worker_id = self.worker_id

            # Execute task function
            if task.function is None:
                raise TaskExecutionError(f"Task {task.task_id} has no function to execute")

            # Set up timeout
            timeout = task.config.timeout_seconds

            if timeout:
                # Use signal-based timeout (Unix-like systems only)
                if hasattr(signal, 'SIGALRM'):
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Task {task.task_id} timed out after {timeout} seconds")

                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout)

                    try:
                        task_result = task.function(*task.args, **task.kwargs)
                        signal.alarm(0)  # Cancel alarm
                        signal.signal(signal.SIGALRM, old_handler)
                    except TimeoutError as e:
                        signal.signal(signal.SIGALRM, old_handler)
                        raise e
                else:
                    # Fallback for systems without signal support
                    task_result = task.function(*task.args, **task.kwargs)
            else:
                task_result = task.function(*task.args, **task.kwargs)

            # Calculate execution metrics
            end_time = time.time()
            duration = end_time - start_time
            end_memory = self._get_memory_usage()
            memory_used = max(0, end_memory - start_memory)

            # Update result with success
            result.status = TaskStatus.COMPLETED
            result.result = task_result
            result.completed_at = datetime.now(timezone.utc)
            result.duration_seconds = duration
            result.memory_peak_mb = memory_used

            # Update task
            task.status = TaskStatus.COMPLETED
            task.result = result

            # Update worker stats
            self._tasks_executed += 1

            logger.info(
                f"Completed task {task.task_id} in {duration:.2f}s "
                f"(memory: {memory_used:.1f}MB)"
            )

        except TimeoutError as e:
            # Handle timeout
            result.status = TaskStatus.TIMEOUT
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)
            result.duration_seconds = time.time() - start_time

            task.status = TaskStatus.TIMEOUT
            task.result = result

            self._tasks_failed += 1

            logger.error(f"Task {task.task_id} timed out: {e}")

        except Exception as e:
            # Handle general errors
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.traceback = traceback.format_exc()
            result.completed_at = datetime.now(timezone.utc)
            result.duration_seconds = time.time() - start_time

            # Check if task should be retried
            if task.should_retry():
                result.status = TaskStatus.RETRY
                result.retry_count = task.result.retry_count + 1 if task.result else 1
                result.next_retry_at = task.calculate_next_retry_time()

                # Create retry task
                retry_task = task.create_retry_task()
                self._queue.put(retry_task)

                logger.info(
                    f"Task {task.task_id} failed, scheduled for retry {result.retry_count}"
                )
            else:
                task.status = TaskStatus.FAILED

            task.result = result
            self._tasks_failed += 1

            logger.error(f"Task {task.task_id} failed: {e}")

        finally:
            # Cleanup and resource management
            self._cleanup_after_task(task)

        return result

    def _check_resource_limits(self) -> bool:
        """Check if worker is within resource limits."""
        try:
            # Check memory usage
            current_memory = self._get_memory_usage()
            if current_memory > self.memory_limit_mb:
                logger.warning(
                    f"Worker {self.worker_id} memory limit exceeded: "
                    f"{current_memory:.1f}MB > {self.memory_limit_mb}MB"
                )
                return False

            # Check CPU usage
            cpu_percent = self._process.cpu_percent()
            if cpu_percent > self.cpu_limit_percent:
                logger.warning(
                    f"Worker {self.worker_id} CPU limit exceeded: "
                    f"{cpu_percent:.1f}% > {self.cpu_limit_percent}%"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking resource limits for worker {self.worker_id}: {e}")
            return True  # Allow execution if we can't check limits

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self._process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0

    def _cleanup_after_task(self, task: Task) -> None:
        """Cleanup resources after task execution."""
        try:
            # Force garbage collection
            gc.collect()

            # Log resource usage
            current_memory = self._get_memory_usage()
            memory_growth = current_memory - self._baseline_memory

            if memory_growth > 100:  # More than 100MB growth
                logger.warning(
                    f"Worker {self.worker_id} memory growth: {memory_growth:.1f}MB"
                )

        except Exception as e:
            logger.error(f"Error during cleanup for worker {self.worker_id}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get worker performance statistics."""
        current_time = time.time()
        runtime = current_time - self._start_time

        with self._lock:
            current_tasks_count = len(self._current_tasks)

        return {
            "worker_id": self.worker_id,
            "queue_name": self.queue_name,
            "is_running": self._is_running,
            "is_stopping": self._is_stopping,
            "runtime_seconds": runtime,
            "current_tasks": current_tasks_count,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "tasks_executed": self._tasks_executed,
            "tasks_failed": self._tasks_failed,
            "success_rate": (
                (self._tasks_executed - self._tasks_failed) / max(self._tasks_executed, 1) * 100
            ),
            "tasks_per_hour": (self._tasks_executed / (runtime / 3600)) if runtime > 0 else 0,
            "last_heartbeat": self._last_heartbeat,
            "memory_usage_mb": self._get_memory_usage(),
            "memory_limit_mb": self.memory_limit_mb,
            "cpu_limit_percent": self.cpu_limit_percent,
            "resource_limits": self.resource_limits
        }

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._is_running

    @property
    def is_healthy(self) -> bool:
        """Check if worker is healthy."""
        return (
            self._is_running and
            not self._is_stopping and
            (time.time() - self._last_heartbeat) < 60  # Heartbeat within last minute
        )


class TaskWorkerPool:
    """
    Pool of task workers for scalable task execution.

    Manages multiple workers with automatic scaling, load balancing,
    and health monitoring.
    """

    def __init__(
        self,
        pool_name: str = "default",
        min_workers: int = 1,
        max_workers: int = 4,
        queue_name: str = "default",
        auto_scale: bool = True
    ):
        """
        Initialize task worker pool.

        Args:
            pool_name: Pool name
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            queue_name: Queue to consume from
            auto_scale: Enable automatic scaling
        """
        self.pool_name = pool_name
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.queue_name = queue_name
        self.auto_scale = auto_scale

        # Worker management
        self._workers: Dict[str, TaskWorker] = {}
        self._lock = threading.RLock()

        # Auto-scaling
        self._scale_up_threshold = 0.8  # Scale up when 80% busy
        self._scale_down_threshold = 0.2  # Scale down when 20% busy
        self._scale_check_interval = 30  # Check every 30 seconds

        # Monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitoring_enabled = False

        logger.info(f"Initialized worker pool '{pool_name}' ({min_workers}-{max_workers} workers)")

    def start(self) -> bool:
        """Start the worker pool."""
        try:
            # Start minimum number of workers
            for i in range(self.min_workers):
                worker_id = f"{self.pool_name}-worker-{i+1}"
                if self._create_worker(worker_id):
                    logger.info(f"Started worker {worker_id}")

            # Start monitoring
            if self.auto_scale:
                self._start_monitoring()

            logger.info(f"Started worker pool '{self.pool_name}' with {len(self._workers)} workers")
            return True

        except Exception as e:
            logger.error(f"Failed to start worker pool '{self.pool_name}': {e}")
            return False

    def stop(self, timeout: float = 30.0) -> None:
        """
        Stop all workers in the pool.

        Args:
            timeout: Maximum wait time for graceful shutdown
        """
        logger.info(f"Stopping worker pool '{self.pool_name}'...")

        # Stop monitoring
        self._stop_monitoring()

        # Stop all workers
        with self._lock:
            workers = list(self._workers.values())

        for worker in workers:
            try:
                worker.stop(timeout=timeout)
            except Exception as e:
                logger.error(f"Error stopping worker {worker.worker_id}: {e}")

        # Clear worker registry
        with self._lock:
            self._workers.clear()

        logger.info(f"Stopped worker pool '{self.pool_name}'")

    def _create_worker(self, worker_id: str) -> bool:
        """Create and start a new worker."""
        try:
            with self._lock:
                if worker_id in self._workers:
                    return False

                worker = TaskWorker(
                    worker_id=worker_id,
                    queue_name=self.queue_name,
                    max_concurrent_tasks=1
                )

                if worker.start():
                    self._workers[worker_id] = worker
                    return True

                return False

        except Exception as e:
            logger.error(f"Failed to create worker {worker_id}: {e}")
            return False

    def _remove_worker(self, worker_id: str) -> bool:
        """Remove and stop a worker."""
        try:
            with self._lock:
                if worker_id not in self._workers:
                    return False

                worker = self._workers[worker_id]
                worker.stop()
                del self._workers[worker_id]

                logger.info(f"Removed worker {worker_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to remove worker {worker_id}: {e}")
            return False

    def scale_up(self, count: int = 1) -> int:
        """
        Scale up the worker pool.

        Args:
            count: Number of workers to add

        Returns:
            int: Number of workers actually added
        """
        with self._lock:
            current_count = len(self._workers)
            target_count = min(current_count + count, self.max_workers)
            workers_to_add = target_count - current_count

            added = 0
            for i in range(workers_to_add):
                worker_id = f"{self.pool_name}-worker-{current_count + i + 1}"
                if self._create_worker(worker_id):
                    added += 1

            if added > 0:
                logger.info(f"Scaled up pool '{self.pool_name}' by {added} workers")

            return added

    def scale_down(self, count: int = 1) -> int:
        """
        Scale down the worker pool.

        Args:
            count: Number of workers to remove

        Returns:
            int: Number of workers actually removed
        """
        with self._lock:
            current_count = len(self._workers)
            target_count = max(current_count - count, self.min_workers)
            workers_to_remove = current_count - target_count

            if workers_to_remove <= 0:
                return 0

            # Remove workers with lowest utilization
            worker_items = list(self._workers.items())
            worker_items.sort(key=lambda x: x[1]._tasks_executed)

            removed = 0
            for worker_id, _ in worker_items[:workers_to_remove]:
                if self._remove_worker(worker_id):
                    removed += 1

            if removed > 0:
                logger.info(f"Scaled down pool '{self.pool_name}' by {removed} workers")

            return removed

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        with self._lock:
            workers = list(self._workers.values())

        # Aggregate worker stats
        total_tasks = sum(w._tasks_executed for w in workers)
        total_failed = sum(w._tasks_failed for w in workers)
        healthy_workers = sum(1 for w in workers if w.is_healthy)
        busy_workers = sum(1 for w in workers if len(w._current_tasks) > 0)

        # Calculate utilization
        utilization = (busy_workers / len(workers) * 100) if workers else 0

        return {
            "pool_name": self.pool_name,
            "queue_name": self.queue_name,
            "worker_count": len(workers),
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "healthy_workers": healthy_workers,
            "busy_workers": busy_workers,
            "utilization_percent": utilization,
            "total_tasks_executed": total_tasks,
            "total_tasks_failed": total_failed,
            "success_rate": ((total_tasks - total_failed) / max(total_tasks, 1)) * 100,
            "auto_scale": self.auto_scale,
            "monitoring_enabled": self._monitoring_enabled,
            "workers": [w.get_stats() for w in workers]
        }

    def _start_monitoring(self) -> None:
        """Start pool monitoring and auto-scaling."""
        if self._monitoring_enabled:
            return

        async def monitor():
            while self._monitoring_enabled:
                try:
                    await asyncio.sleep(self._scale_check_interval)
                    self._check_scaling()
                except Exception as e:
                    logger.error(f"Pool monitoring error: {e}")

        self._monitoring_enabled = True
        self._monitor_task = asyncio.create_task(monitor())
        logger.info(f"Started monitoring for pool '{self.pool_name}'")

    def _stop_monitoring(self) -> None:
        """Stop pool monitoring."""
        self._monitoring_enabled = False

        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

        logger.info(f"Stopped monitoring for pool '{self.pool_name}'")

    def _check_scaling(self) -> None:
        """Check if pool should be scaled up or down."""
        if not self.auto_scale:
            return

        stats = self.get_pool_stats()
        utilization = stats["utilization_percent"]

        with self._lock:
            current_workers = len(self._workers)

        # Scale up if utilization is high
        if utilization > self._scale_up_threshold * 100 and current_workers < self.max_workers:
            self.scale_up(1)

        # Scale down if utilization is low
        elif utilization < self._scale_down_threshold * 100 and current_workers > self.min_workers:
            self.scale_down(1)


# Global worker pool manager
_worker_pools: Dict[str, TaskWorkerPool] = {}
_pools_lock = threading.RLock()


def create_worker_pool(
    pool_name: str,
    min_workers: int = 1,
    max_workers: int = 4,
    queue_name: str = "default",
    auto_scale: bool = True
) -> TaskWorkerPool:
    """Create or get existing worker pool."""
    global _worker_pools

    with _pools_lock:
        if pool_name in _worker_pools:
            return _worker_pools[pool_name]

        pool = TaskWorkerPool(
            pool_name=pool_name,
            min_workers=min_workers,
            max_workers=max_workers,
            queue_name=queue_name,
            auto_scale=auto_scale
        )

        _worker_pools[pool_name] = pool
        return pool


def get_worker_pool(pool_name: str = "default") -> Optional[TaskWorkerPool]:
    """Get existing worker pool."""
    global _worker_pools

    with _pools_lock:
        return _worker_pools.get(pool_name)


def get_default_worker_pool() -> TaskWorkerPool:
    """Get or create the default worker pool."""
    return create_worker_pool("default")


def shutdown_all_pools(timeout: float = 30.0) -> None:
    """Shutdown all worker pools."""
    global _worker_pools

    with _pools_lock:
        pools = list(_worker_pools.values())

    for pool in pools:
        try:
            pool.stop(timeout=timeout)
        except Exception as e:
            logger.error(f"Error shutting down pool {pool.pool_name}: {e}")

    with _pools_lock:
        _worker_pools.clear()

    logger.info("All worker pools shut down")