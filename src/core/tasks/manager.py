"""
Centralized task management system.

This module provides the main TaskManager class that coordinates
task scheduling, execution, monitoring, and lifecycle management.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

from src.core.config import get_settings
from src.core.logging import get_logger
from .models import Task, TaskStatus, TaskResult, TaskConfig, TaskPriority, TaskStats
from .queue import get_task_queue_manager, PriorityTaskQueue
from .worker import create_worker_pool, get_worker_pool, TaskWorkerPool
from .scheduler import TaskScheduler

logger = get_logger(__name__)
settings = get_settings()


class TaskManager:
    """
    Centralized task management system.

    Provides unified interface for task submission, scheduling, execution,
    and monitoring with comprehensive lifecycle management.
    """

    def __init__(self):
        """Initialize task manager."""
        # Core components
        self._queue_manager = get_task_queue_manager()
        self._scheduler = TaskScheduler()

        # Worker pools
        self._worker_pools: Dict[str, TaskWorkerPool] = {}
        self._pool_lock = threading.RLock()

        # Task tracking
        self._active_tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._task_stats = TaskStats()
        self._task_lock = threading.RLock()

        # System state
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Configuration
        self._cleanup_interval = 3600  # 1 hour
        self._max_completed_tasks = 10000  # Keep last 10k completed tasks

        logger.info("Task manager initialized")

    def start(self) -> bool:
        """Start the task management system."""
        if self._is_running:
            logger.warning("Task manager already running")
            return True

        try:
            # Start queue monitoring
            self._queue_manager.start_monitoring()

            # Start task scheduler
            self._scheduler.start()

            # Create default worker pool
            self._ensure_default_worker_pool()

            # Start system monitoring
            self._start_monitoring()

            self._is_running = True
            logger.info("Task manager started")
            return True

        except Exception as e:
            logger.error(f"Failed to start task manager: {e}")
            return False

    def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the task management system.

        Args:
            timeout: Maximum wait time for graceful shutdown
        """
        if not self._is_running:
            return

        logger.info("Stopping task manager...")

        self._is_running = False

        # Stop monitoring
        if self._monitor_task:
            self._monitor_task.cancel()

        # Stop scheduler
        self._scheduler.stop()

        # Stop queue monitoring
        self._queue_manager.stop_monitoring()

        # Stop all worker pools
        with self._pool_lock:
            pools = list(self._worker_pools.values())

        for pool in pools:
            try:
                pool.stop(timeout=timeout)
            except Exception as e:
                logger.error(f"Error stopping worker pool {pool.pool_name}: {e}")

        # Clear worker pools
        with self._pool_lock:
            self._worker_pools.clear()

        logger.info("Task manager stopped")

    def submit_task(
        self,
        function: Callable,
        *args,
        name: str = None,
        description: str = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        queue_name: str = "default",
        config: TaskConfig = None,
        scheduled_at: Optional[datetime] = None,
        tags: List[str] = None,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> str:
        """
        Submit task for execution.

        Args:
            function: Function to execute
            *args: Function arguments
            name: Task name
            description: Task description
            priority: Task priority
            queue_name: Target queue name
            config: Task configuration
            scheduled_at: Scheduled execution time
            tags: Task tags
            context: Task context
            **kwargs: Function keyword arguments

        Returns:
            str: Task ID
        """
        try:
            # Create task configuration
            if config is None:
                config = TaskConfig()
            config.priority = priority

            # Generate task name if not provided
            if name is None:
                name = getattr(function, "__name__", "anonymous_task")

            # Create task
            task = Task(
                name=name,
                description=description,
                function=function,
                args=list(args),
                kwargs=kwargs,
                config=config,
                scheduled_at=scheduled_at,
                tags=tags or [],
                context=context or {},
                queue_name=queue_name
            )

            # Track task
            with self._task_lock:
                self._active_tasks[task.task_id] = task

            # Submit to appropriate handler
            if scheduled_at:
                # Submit to scheduler
                success = self._scheduler.schedule_task(task)
            else:
                # Submit directly to queue
                queue = self._queue_manager.get_or_create_queue(queue_name)
                success = queue.put(task, priority)

                # Ensure worker pool exists
                self._ensure_worker_pool(queue_name)

            if success:
                logger.info(f"Submitted task {task.task_id} ({name}) to queue {queue_name}")
                return task.task_id
            else:
                # Remove from tracking if submission failed
                with self._task_lock:
                    self._active_tasks.pop(task.task_id, None)
                raise RuntimeError("Failed to submit task")

        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            raise

    async def submit_task_async(
        self,
        function: Callable,
        *args,
        **kwargs
    ) -> str:
        """Submit task asynchronously."""
        return self.submit_task(function, *args, **kwargs)

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.

        Args:
            task_id: Task ID to cancel

        Returns:
            bool: True if cancelled successfully
        """
        try:
            with self._task_lock:
                task = self._active_tasks.get(task_id)
                if not task:
                    logger.warning(f"Task {task_id} not found for cancellation")
                    return False

                # Try to remove from queue
                if task.status == TaskStatus.PENDING:
                    queue = self._queue_manager.get_queue(task.queue_name)
                    if queue and queue.remove(task_id):
                        task.status = TaskStatus.CANCELLED
                        self._move_task_to_completed(task, TaskStatus.CANCELLED)
                        logger.info(f"Cancelled pending task {task_id}")
                        return True

                # Try to remove from scheduler
                if self._scheduler.cancel_task(task_id):
                    task.status = TaskStatus.CANCELLED
                    self._move_task_to_completed(task, TaskStatus.CANCELLED)
                    logger.info(f"Cancelled scheduled task {task_id}")
                    return True

                # Task is running, cannot cancel
                if task.status == TaskStatus.RUNNING:
                    logger.warning(f"Cannot cancel running task {task_id}")
                    return False

                return False

        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        with self._task_lock:
            # Check active tasks
            task = self._active_tasks.get(task_id)
            if task:
                return task

            # Check completed tasks
            result = self._completed_tasks.get(task_id)
            if result:
                # Reconstruct task from result (limited info)
                return Task(
                    task_id=task_id,
                    name="completed_task",
                    status=result.status
                )

            return None

    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task execution result."""
        with self._task_lock:
            # Check active tasks first
            task = self._active_tasks.get(task_id)
            if task and task.result:
                return task.result

            # Check completed tasks
            return self._completed_tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status."""
        task = self.get_task(task_id)
        return task.status if task else None

    def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> Optional[TaskResult]:
        """
        Wait for task completion.

        Args:
            task_id: Task ID to wait for
            timeout: Maximum wait time in seconds

        Returns:
            TaskResult or None if timeout
        """
        start_time = time.time()

        while True:
            result = self.get_task_result(task_id)
            if result and result.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                return result

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None

            time.sleep(0.1)

    async def wait_for_task_async(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> Optional[TaskResult]:
        """Wait for task completion asynchronously."""
        start_time = time.time()

        while True:
            result = self.get_task_result(task_id)
            if result and result.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                return result

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None

            await asyncio.sleep(0.1)

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        queue_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Task]:
        """
        List tasks with optional filtering.

        Args:
            status: Filter by status
            queue_name: Filter by queue name
            tags: Filter by tags

        Returns:
            List of tasks matching criteria
        """
        with self._task_lock:
            tasks = list(self._active_tasks.values())

        # Apply filters
        if status:
            tasks = [t for t in tasks if t.status == status]

        if queue_name:
            tasks = [t for t in tasks if t.queue_name == queue_name]

        if tags:
            tasks = [
                t for t in tasks
                if any(tag in t.tags for tag in tags)
            ]

        return tasks

    def get_queue_stats(self, queue_name: str = None) -> Dict[str, Any]:
        """Get queue statistics."""
        if queue_name:
            queue = self._queue_manager.get_queue(queue_name)
            return queue.get_stats() if queue else {}

        return self._queue_manager.get_queue_stats()

    def get_worker_stats(self, pool_name: str = None) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self._pool_lock:
            if pool_name:
                pool = self._worker_pools.get(pool_name)
                return pool.get_pool_stats() if pool else {}

            return {
                "pools": {
                    name: pool.get_pool_stats()
                    for name, pool in self._worker_pools.items()
                }
            }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        with self._task_lock:
            active_count = len(self._active_tasks)
            completed_count = len(self._completed_tasks)

            # Count by status
            status_counts = {}
            for task in self._active_tasks.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "is_running": self._is_running,
            "active_tasks": active_count,
            "completed_tasks": completed_count,
            "task_stats": self._task_stats.to_dict(),
            "status_counts": status_counts,
            "queues": self.get_queue_stats(),
            "workers": self.get_worker_stats(),
            "scheduler": self._scheduler.get_stats()
        }

    def create_worker_pool(
        self,
        pool_name: str,
        queue_name: str = None,
        min_workers: int = 1,
        max_workers: int = 4,
        auto_scale: bool = True
    ) -> bool:
        """
        Create new worker pool.

        Args:
            pool_name: Pool name
            queue_name: Queue to consume from
            min_workers: Minimum workers
            max_workers: Maximum workers
            auto_scale: Enable auto-scaling

        Returns:
            bool: True if created successfully
        """
        try:
            if queue_name is None:
                queue_name = pool_name

            with self._pool_lock:
                if pool_name in self._worker_pools:
                    logger.warning(f"Worker pool {pool_name} already exists")
                    return False

                pool = create_worker_pool(
                    pool_name=pool_name,
                    min_workers=min_workers,
                    max_workers=max_workers,
                    queue_name=queue_name,
                    auto_scale=auto_scale
                )

                if pool.start():
                    self._worker_pools[pool_name] = pool
                    logger.info(f"Created worker pool {pool_name}")
                    return True

                return False

        except Exception as e:
            logger.error(f"Error creating worker pool {pool_name}: {e}")
            return False

    def _ensure_default_worker_pool(self) -> None:
        """Ensure default worker pool exists."""
        if not self.create_worker_pool("default", min_workers=2, max_workers=8):
            logger.error("Failed to create default worker pool")

    def _ensure_worker_pool(self, queue_name: str) -> None:
        """Ensure worker pool exists for queue."""
        with self._pool_lock:
            if queue_name not in self._worker_pools:
                self.create_worker_pool(queue_name, queue_name)

    def _move_task_to_completed(self, task: Task, status: TaskStatus) -> None:
        """Move task from active to completed with cleanup."""
        try:
            # Create result if not exists
            if not task.result:
                task.result = TaskResult(
                    task_id=task.task_id,
                    status=status,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc)
                )

            # Update statistics
            self._task_stats.update_from_result(task.result)

            with self._task_lock:
                # Remove from active tasks
                self._active_tasks.pop(task.task_id, None)

                # Add to completed tasks
                self._completed_tasks[task.task_id] = task.result

                # Cleanup old completed tasks
                if len(self._completed_tasks) > self._max_completed_tasks:
                    # Remove oldest 10%
                    oldest_tasks = sorted(
                        self._completed_tasks.items(),
                        key=lambda x: x[1].completed_at or x[1].started_at
                    )
                    to_remove = len(oldest_tasks) // 10
                    for task_id, _ in oldest_tasks[:to_remove]:
                        self._completed_tasks.pop(task_id, None)

        except Exception as e:
            logger.error(f"Error moving task {task.task_id} to completed: {e}")

    def _start_monitoring(self) -> None:
        """Start system monitoring."""
        async def monitor():
            while self._is_running:
                try:
                    await self._monitor_system()
                    await asyncio.sleep(60)  # Monitor every minute
                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    await asyncio.sleep(60)

        self._monitor_task = asyncio.create_task(monitor())
        logger.info("System monitoring started")

    async def _monitor_system(self) -> None:
        """Monitor system health and performance."""
        try:
            stats = self.get_system_stats()

            # Log system status
            logger.info(
                f"System Status: {stats['active_tasks']} active, "
                f"{stats['completed_tasks']} completed tasks"
            )

            # Check for issues
            self._check_system_health(stats)

            # Cleanup completed tasks periodically
            if int(time.time()) % self._cleanup_interval == 0:
                self._cleanup_completed_tasks()

        except Exception as e:
            logger.error(f"Error in system monitoring: {e}")

    def _check_system_health(self, stats: Dict[str, Any]) -> None:
        """Check system health and log warnings."""
        # Check for high failure rates
        task_stats = stats.get("task_stats", {})
        success_rate = task_stats.get("success_rate_percent", 100)

        if success_rate < 90:
            logger.warning(f"Low task success rate: {success_rate:.1f}%")

        # Check for queue backlogs
        queue_stats = stats.get("queues", {}).get("queues", {})
        for queue_name, queue_info in queue_stats.items():
            size = queue_info.get("size", 0)
            max_size = queue_info.get("max_size", 1000)

            if size > max_size * 0.8:  # More than 80% full
                logger.warning(f"Queue {queue_name} is {size}/{max_size} full")

    def _cleanup_completed_tasks(self) -> None:
        """Cleanup old completed tasks."""
        try:
            with self._task_lock:
                # Remove tasks older than 24 hours
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                to_remove = []

                for task_id, result in self._completed_tasks.items():
                    completion_time = result.completed_at or result.started_at
                    if completion_time < cutoff_time:
                        to_remove.append(task_id)

                for task_id in to_remove:
                    self._completed_tasks.pop(task_id, None)

                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} old completed tasks")

        except Exception as e:
            logger.error(f"Error cleaning up completed tasks: {e}")

    @property
    def is_running(self) -> bool:
        """Check if task manager is running."""
        return self._is_running


# Global task manager instance
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get the global task manager instance."""
    global _task_manager

    if _task_manager is None:
        _task_manager = TaskManager()

    return _task_manager