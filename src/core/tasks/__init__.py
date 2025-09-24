"""
Background task management system for the MetaTrader Python Framework.

This module provides high-performance task scheduling, execution, and monitoring
with priority queues, async support, and comprehensive error handling.
"""

from .manager import TaskManager, get_task_manager
from .scheduler import TaskScheduler, ScheduledTask, TaskTrigger
from .queue import PriorityTaskQueue, TaskPriority
from .worker import TaskWorker, TaskWorkerPool
from .models import Task, TaskStatus, TaskResult, TaskConfig
from .decorators import background_task, scheduled_task, retry_task

__all__ = [
    # Core management
    "TaskManager",
    "get_task_manager",

    # Scheduling
    "TaskScheduler",
    "ScheduledTask",
    "TaskTrigger",

    # Queue management
    "PriorityTaskQueue",
    "TaskPriority",

    # Workers
    "TaskWorker",
    "TaskWorkerPool",

    # Models
    "Task",
    "TaskStatus",
    "TaskResult",
    "TaskConfig",

    # Decorators
    "background_task",
    "scheduled_task",
    "retry_task",
]