"""
Priority-based task queue system with async support.

This module provides high-performance task queuing with priority handling,
persistence, and comprehensive monitoring capabilities.
"""

from __future__ import annotations

import asyncio
import heapq
import pickle
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.core.config import get_settings
from src.core.exceptions import BaseFrameworkError
from src.core.logging import get_logger
from .models import Task, TaskPriority, TaskStatus

logger = get_logger(__name__)
settings = get_settings()


class PriorityTaskQueue:
    """
    High-performance priority task queue with persistence.

    Provides thread-safe priority-based task queuing with optional
    persistence, monitoring, and automatic cleanup.
    """

    def __init__(
        self,
        name: str = "default",
        max_size: int = 10000,
        enable_persistence: bool = False,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize priority task queue.

        Args:
            name: Queue name
            max_size: Maximum queue size
            enable_persistence: Enable queue persistence
            persistence_path: Path for persistence files
        """
        self.name = name
        self.max_size = max_size
        self.enable_persistence = enable_persistence

        # Priority queue storage (min-heap, so we negate priorities)
        self._queue: List[tuple[int, int, Task]] = []
        self._task_index = 0  # For FIFO ordering within same priority
        self._lock = threading.RLock()

        # Task tracking
        self._tasks: Dict[str, Task] = {}
        self._task_priorities: Dict[str, int] = {}

        # Async support
        self._async_lock = asyncio.Lock()
        self._put_event = asyncio.Event()

        # Performance metrics
        self._enqueued_count = 0
        self._dequeued_count = 0
        self._discarded_count = 0
        self._start_time = time.time()

        # Persistence setup
        self.persistence_path = None
        if enable_persistence:
            if persistence_path:
                self.persistence_path = Path(persistence_path)
            else:
                self.persistence_path = Path(f"data/queues/{name}.pkl")

            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

        logger.info(f"Initialized priority queue '{name}' with max_size={max_size}")

    def put(self, task: Task, priority: Optional[TaskPriority] = None) -> bool:
        """
        Add task to queue with specified priority.

        Args:
            task: Task to enqueue
            priority: Task priority (uses task.config.priority if None)

        Returns:
            bool: True if task was added successfully
        """
        try:
            with self._lock:
                # Check if queue is full
                if len(self._queue) >= self.max_size:
                    logger.warning(f"Queue '{self.name}' is full, discarding task {task.task_id}")
                    self._discarded_count += 1
                    return False

                # Check if task already exists
                if task.task_id in self._tasks:
                    logger.warning(f"Task {task.task_id} already in queue")
                    return False

                # Determine priority
                if priority is None:
                    priority = task.config.priority

                # Add to heap (negate priority for min-heap)
                heap_item = (-int(priority), self._task_index, task)
                heapq.heappush(self._queue, heap_item)

                # Track task
                self._tasks[task.task_id] = task
                self._task_priorities[task.task_id] = int(priority)
                self._task_index += 1
                self._enqueued_count += 1

                # Update task status
                task.status = TaskStatus.PENDING
                task.queue_name = self.name

                logger.debug(f"Enqueued task {task.task_id} with priority {priority}")

                # Persist if enabled
                if self.enable_persistence:
                    self._save_to_disk()

                return True

        except Exception as e:
            logger.error(f"Error adding task to queue '{self.name}': {e}")
            return False

    async def put_async(self, task: Task, priority: Optional[TaskPriority] = None) -> bool:
        """
        Add task to queue asynchronously.

        Args:
            task: Task to enqueue
            priority: Task priority

        Returns:
            bool: True if task was added successfully
        """
        try:
            async with self._async_lock:
                success = self.put(task, priority)

                # Signal waiting coroutines
                if success:
                    self._put_event.set()

                return success

        except Exception as e:
            logger.error(f"Error adding task async to queue '{self.name}': {e}")
            return False

    def get(self, timeout: Optional[float] = None) -> Optional[Task]:
        """
        Get highest priority task from queue.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Task or None if timeout/empty
        """
        start_time = time.time()

        while True:
            try:
                with self._lock:
                    if self._queue:
                        # Get highest priority task
                        _, _, task = heapq.heappop(self._queue)

                        # Remove from tracking
                        self._tasks.pop(task.task_id, None)
                        self._task_priorities.pop(task.task_id, None)
                        self._dequeued_count += 1

                        # Update task status
                        task.status = TaskStatus.RUNNING

                        logger.debug(f"Dequeued task {task.task_id}")

                        # Persist if enabled
                        if self.enable_persistence:
                            self._save_to_disk()

                        return task

                # Check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        return None

                # Sleep briefly before retry
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error getting task from queue '{self.name}': {e}")
                return None

    async def get_async(self, timeout: Optional[float] = None) -> Optional[Task]:
        """
        Get highest priority task from queue asynchronously.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Task or None if timeout/empty
        """
        start_time = time.time()

        while True:
            try:
                async with self._async_lock:
                    # Try to get task immediately
                    task = self.get(timeout=0)
                    if task:
                        return task

                # Wait for new tasks or timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return None

                    try:
                        await asyncio.wait_for(
                            self._put_event.wait(),
                            timeout=remaining
                        )
                        self._put_event.clear()
                    except asyncio.TimeoutError:
                        return None
                else:
                    await self._put_event.wait()
                    self._put_event.clear()

            except Exception as e:
                logger.error(f"Error getting task async from queue '{self.name}': {e}")
                return None

    def peek(self) -> Optional[Task]:
        """Get highest priority task without removing it."""
        try:
            with self._lock:
                if self._queue:
                    return self._queue[0][2]
                return None

        except Exception as e:
            logger.error(f"Error peeking queue '{self.name}': {e}")
            return None

    def remove(self, task_id: str) -> bool:
        """
        Remove specific task from queue.

        Args:
            task_id: Task ID to remove

        Returns:
            bool: True if task was removed
        """
        try:
            with self._lock:
                if task_id not in self._tasks:
                    return False

                # Find and remove from heap
                for i, (_, _, task) in enumerate(self._queue):
                    if task.task_id == task_id:
                        # Remove from heap
                        self._queue.pop(i)
                        heapq.heapify(self._queue)

                        # Remove from tracking
                        self._tasks.pop(task_id, None)
                        self._task_priorities.pop(task_id, None)

                        logger.debug(f"Removed task {task_id} from queue")

                        # Persist if enabled
                        if self.enable_persistence:
                            self._save_to_disk()

                        return True

                return False

        except Exception as e:
            logger.error(f"Error removing task {task_id} from queue '{self.name}': {e}")
            return False

    def clear(self) -> int:
        """
        Clear all tasks from queue.

        Returns:
            int: Number of tasks cleared
        """
        try:
            with self._lock:
                count = len(self._queue)

                self._queue.clear()
                self._tasks.clear()
                self._task_priorities.clear()

                logger.info(f"Cleared {count} tasks from queue '{self.name}'")

                # Persist if enabled
                if self.enable_persistence:
                    self._save_to_disk()

                return count

        except Exception as e:
            logger.error(f"Error clearing queue '{self.name}': {e}")
            return 0

    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def is_full(self) -> bool:
        """Check if queue is full."""
        with self._lock:
            return len(self._queue) >= self.max_size

    def contains(self, task_id: str) -> bool:
        """Check if queue contains specific task."""
        with self._lock:
            return task_id in self._tasks

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get specific task without removing it."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_tasks_by_priority(self, priority: TaskPriority) -> List[Task]:
        """Get all tasks with specific priority."""
        with self._lock:
            return [
                task for task_id, task in self._tasks.items()
                if self._task_priorities.get(task_id) == int(priority)
            ]

    def get_tasks_by_tag(self, tag: str) -> List[Task]:
        """Get all tasks with specific tag."""
        with self._lock:
            return [
                task for task in self._tasks.values()
                if task.has_tag(tag)
            ]

    def get_all_tasks(self) -> List[Task]:
        """Get all tasks in queue."""
        with self._lock:
            # Return tasks sorted by priority
            return [task for _, _, task in sorted(self._queue)]

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            runtime = time.time() - self._start_time

            # Count tasks by priority
            priority_counts = {}
            for priority in self._task_priorities.values():
                priority_counts[priority] = priority_counts.get(priority, 0) + 1

            # Count tasks by status
            status_counts = {}
            for task in self._tasks.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "name": self.name,
                "size": len(self._queue),
                "max_size": self.max_size,
                "is_empty": len(self._queue) == 0,
                "is_full": len(self._queue) >= self.max_size,
                "enqueued_count": self._enqueued_count,
                "dequeued_count": self._dequeued_count,
                "discarded_count": self._discarded_count,
                "throughput_per_second": self._dequeued_count / runtime if runtime > 0 else 0,
                "priority_counts": priority_counts,
                "status_counts": status_counts,
                "enable_persistence": self.enable_persistence,
                "persistence_path": str(self.persistence_path) if self.persistence_path else None
            }

    def _save_to_disk(self) -> None:
        """Save queue state to disk."""
        if not self.enable_persistence or not self.persistence_path:
            return

        try:
            # Prepare serializable data
            queue_data = {
                "tasks": [task.dict() for task in self._tasks.values()],
                "priorities": self._task_priorities,
                "metadata": {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "queue_name": self.name,
                    "queue_size": len(self._queue)
                }
            }

            # Save to disk
            with open(self.persistence_path, "wb") as f:
                pickle.dump(queue_data, f)

            logger.debug(f"Saved queue '{self.name}' to disk")

        except Exception as e:
            logger.error(f"Error saving queue '{self.name}' to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load queue state from disk."""
        if not self.enable_persistence or not self.persistence_path:
            return

        if not self.persistence_path.exists():
            logger.debug(f"No persistence file found for queue '{self.name}'")
            return

        try:
            with open(self.persistence_path, "rb") as f:
                queue_data = pickle.load(f)

            # Restore tasks
            for task_dict in queue_data.get("tasks", []):
                task = Task(**task_dict)
                priority = queue_data["priorities"].get(task.task_id, int(TaskPriority.NORMAL))

                # Add to queue
                heap_item = (-priority, self._task_index, task)
                heapq.heappush(self._queue, heap_item)

                self._tasks[task.task_id] = task
                self._task_priorities[task.task_id] = priority
                self._task_index += 1

            logger.info(f"Loaded {len(self._queue)} tasks from disk for queue '{self.name}'")

        except Exception as e:
            logger.error(f"Error loading queue '{self.name}' from disk: {e}")


class TaskQueueManager:
    """
    Centralized manager for multiple task queues.

    Provides unified access to named task queues with automatic
    creation, monitoring, and lifecycle management.
    """

    def __init__(self):
        """Initialize task queue manager."""
        self._queues: Dict[str, PriorityTaskQueue] = {}
        self._lock = threading.RLock()

        # Default queue configuration
        self._default_max_size = 10000
        self._default_persistence = False

        # Performance monitoring
        self._monitor_interval = 30  # seconds
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitoring_enabled = False

        logger.info("Task queue manager initialized")

    def create_queue(
        self,
        name: str,
        max_size: int = None,
        enable_persistence: bool = None
    ) -> PriorityTaskQueue:
        """
        Create or get existing task queue.

        Args:
            name: Queue name
            max_size: Maximum queue size
            enable_persistence: Enable persistence

        Returns:
            PriorityTaskQueue instance
        """
        with self._lock:
            if name in self._queues:
                return self._queues[name]

            # Use defaults if not specified
            if max_size is None:
                max_size = self._default_max_size
            if enable_persistence is None:
                enable_persistence = self._default_persistence

            # Create new queue
            queue = PriorityTaskQueue(
                name=name,
                max_size=max_size,
                enable_persistence=enable_persistence
            )

            self._queues[name] = queue
            logger.info(f"Created task queue '{name}'")

            return queue

    def get_queue(self, name: str = "default") -> Optional[PriorityTaskQueue]:
        """Get existing task queue."""
        with self._lock:
            return self._queues.get(name)

    def get_or_create_queue(self, name: str = "default") -> PriorityTaskQueue:
        """Get existing queue or create new one."""
        queue = self.get_queue(name)
        if queue is None:
            queue = self.create_queue(name)
        return queue

    def remove_queue(self, name: str) -> bool:
        """
        Remove task queue.

        Args:
            name: Queue name to remove

        Returns:
            bool: True if removed successfully
        """
        with self._lock:
            if name not in self._queues:
                return False

            # Clear queue before removal
            queue = self._queues[name]
            cleared_count = queue.clear()

            del self._queues[name]
            logger.info(f"Removed task queue '{name}' with {cleared_count} tasks")

            return True

    def list_queues(self) -> List[str]:
        """Get list of all queue names."""
        with self._lock:
            return list(self._queues.keys())

    def get_all_queues(self) -> Dict[str, PriorityTaskQueue]:
        """Get all task queues."""
        with self._lock:
            return dict(self._queues)

    def get_total_tasks(self) -> int:
        """Get total number of tasks across all queues."""
        with self._lock:
            return sum(queue.size() for queue in self._queues.values())

    def clear_all_queues(self) -> int:
        """
        Clear all queues.

        Returns:
            int: Total number of tasks cleared
        """
        with self._lock:
            total_cleared = 0

            for queue in self._queues.values():
                total_cleared += queue.clear()

            logger.info(f"Cleared {total_cleared} tasks from all queues")
            return total_cleared

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        with self._lock:
            stats = {
                "total_queues": len(self._queues),
                "total_tasks": self.get_total_tasks(),
                "queues": {}
            }

            for name, queue in self._queues.items():
                stats["queues"][name] = queue.get_stats()

            return stats

    def start_monitoring(self) -> None:
        """Start queue performance monitoring."""
        if self._monitoring_enabled:
            return

        async def monitor():
            while self._monitoring_enabled:
                try:
                    stats = self.get_queue_stats()
                    self._log_queue_metrics(stats)
                    await asyncio.sleep(self._monitor_interval)
                except Exception as e:
                    logger.error(f"Queue monitoring error: {e}")
                    await asyncio.sleep(self._monitor_interval)

        self._monitoring_enabled = True
        self._monitor_task = asyncio.create_task(monitor())
        logger.info("Queue monitoring started")

    def stop_monitoring(self) -> None:
        """Stop queue performance monitoring."""
        self._monitoring_enabled = False

        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

        logger.info("Queue monitoring stopped")

    def _log_queue_metrics(self, stats: Dict[str, Any]) -> None:
        """Log queue performance metrics."""
        logger.info(
            f"Queue Status: {stats['total_queues']} queues, {stats['total_tasks']} total tasks"
        )

        # Log warnings for full queues
        for name, queue_stats in stats["queues"].items():
            if queue_stats["is_full"]:
                logger.warning(f"Queue '{name}' is full ({queue_stats['size']}/{queue_stats['max_size']})")

            if queue_stats["discarded_count"] > 0:
                logger.warning(f"Queue '{name}' has discarded {queue_stats['discarded_count']} tasks")


# Global task queue manager instance
_queue_manager: Optional[TaskQueueManager] = None


def get_task_queue_manager() -> TaskQueueManager:
    """Get the global task queue manager instance."""
    global _queue_manager

    if _queue_manager is None:
        _queue_manager = TaskQueueManager()

    return _queue_manager


def get_default_queue() -> PriorityTaskQueue:
    """Get the default task queue."""
    return get_task_queue_manager().get_or_create_queue("default")