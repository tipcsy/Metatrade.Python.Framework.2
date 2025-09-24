"""
Task models and data structures for the background task system.

This module defines the core data structures used throughout the task
management system for representing tasks, results, and configurations.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"
    TIMEOUT = "timeout"


class TaskPriority(IntEnum):
    """Task priority levels (higher number = higher priority)."""

    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20
    URGENT = 30


class TaskTrigger(str, Enum):
    """Task trigger types for scheduling."""

    MANUAL = "manual"
    INTERVAL = "interval"
    CRON = "cron"
    ONCE = "once"
    EVENT = "event"


class TaskConfig(BaseModel):
    """Task configuration settings."""

    # Execution settings
    timeout_seconds: Optional[int] = Field(
        default=300,
        ge=1,
        le=7200,
        description="Task timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Delay between retries in seconds"
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Exponential backoff factor"
    )

    # Priority and scheduling
    priority: TaskPriority = Field(
        default=TaskPriority.NORMAL,
        description="Task priority level"
    )
    allow_concurrent: bool = Field(
        default=True,
        description="Allow concurrent execution"
    )
    max_concurrent: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Maximum concurrent instances"
    )

    # Resource limits
    memory_limit_mb: Optional[int] = Field(
        default=None,
        ge=10,
        le=4096,
        description="Memory limit in MB"
    )
    cpu_limit_percent: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=100.0,
        description="CPU limit percentage"
    )

    # Notification settings
    notify_on_success: bool = Field(
        default=False,
        description="Send notification on success"
    )
    notify_on_failure: bool = Field(
        default=True,
        description="Send notification on failure"
    )
    notify_on_retry: bool = Field(
        default=False,
        description="Send notification on retry"
    )

    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Task logging level"
    )
    log_output: bool = Field(
        default=True,
        description="Log task output"
    )
    log_errors: bool = Field(
        default=True,
        description="Log task errors"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class TaskResult(BaseModel):
    """Task execution result."""

    task_id: str = Field(description="Task ID")
    status: TaskStatus = Field(description="Execution status")
    started_at: datetime = Field(description="Start timestamp")
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Completion timestamp"
    )

    # Execution results
    result: Any = Field(default=None, description="Task result data")
    error: Optional[str] = Field(default=None, description="Error message")
    traceback: Optional[str] = Field(default=None, description="Error traceback")

    # Execution metrics
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Execution duration"
    )
    memory_peak_mb: Optional[float] = Field(
        default=None,
        description="Peak memory usage"
    )
    cpu_usage_percent: Optional[float] = Field(
        default=None,
        description="CPU usage percentage"
    )

    # Retry information
    attempt_number: int = Field(default=1, description="Attempt number")
    retry_count: int = Field(default=0, description="Number of retries")
    next_retry_at: Optional[datetime] = Field(
        default=None,
        description="Next retry timestamp"
    )

    @validator("started_at", "completed_at", "next_retry_at")
    def validate_timestamps(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure timestamps are timezone-aware."""
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def calculate_duration(self) -> Optional[float]:
        """Calculate execution duration."""
        if self.completed_at is None:
            return None

        delta = self.completed_at - self.started_at
        return delta.total_seconds()

    def is_successful(self) -> bool:
        """Check if task execution was successful."""
        return self.status == TaskStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if task execution failed."""
        return self.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED]

    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.status == TaskStatus.RETRY

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
            "duration_seconds": self.duration_seconds or self.calculate_duration(),
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "attempt_number": self.attempt_number,
            "retry_count": self.retry_count,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None
        }

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class Task(BaseModel):
    """Task definition and execution context."""

    # Task identification
    task_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Unique task ID"
    )
    name: str = Field(description="Task name")
    description: Optional[str] = Field(
        default=None,
        description="Task description"
    )

    # Task function and arguments
    function: Optional[Callable] = Field(
        default=None,
        description="Task function to execute"
    )
    function_name: Optional[str] = Field(
        default=None,
        description="Function name for serialization"
    )
    args: List[Any] = Field(
        default_factory=list,
        description="Function arguments"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Function keyword arguments"
    )

    # Task configuration
    config: TaskConfig = Field(
        default_factory=TaskConfig,
        description="Task configuration"
    )

    # Scheduling information
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Task creation timestamp"
    )
    scheduled_at: Optional[datetime] = Field(
        default=None,
        description="Scheduled execution time"
    )
    trigger: TaskTrigger = Field(
        default=TaskTrigger.MANUAL,
        description="Task trigger type"
    )

    # Execution context
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task execution context"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Task tags for categorization"
    )

    # Current state
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current task status"
    )
    result: Optional[TaskResult] = Field(
        default=None,
        description="Task execution result"
    )

    # Worker assignment
    worker_id: Optional[str] = Field(
        default=None,
        description="Assigned worker ID"
    )
    queue_name: str = Field(
        default="default",
        description="Task queue name"
    )

    @validator("created_at", "scheduled_at")
    def validate_timestamps(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure timestamps are timezone-aware."""
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @validator("function", pre=True, always=True)
    def validate_function(cls, v, values):
        """Validate and set function name."""
        if v is not None:
            values["function_name"] = getattr(v, "__name__", str(v))
        return v

    def is_ready_to_execute(self) -> bool:
        """Check if task is ready for execution."""
        if self.status != TaskStatus.PENDING:
            return False

        if self.scheduled_at is None:
            return True

        return datetime.now(timezone.utc) >= self.scheduled_at

    def is_expired(self, max_age_seconds: int = 86400) -> bool:
        """Check if task has expired."""
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > max_age_seconds

    def should_retry(self) -> bool:
        """Check if task should be retried."""
        if not self.result:
            return False

        return (
            self.result.status == TaskStatus.FAILED and
            self.result.retry_count < self.config.max_retries
        )

    def calculate_next_retry_time(self) -> datetime:
        """Calculate next retry execution time."""
        if not self.result:
            raise ValueError("No result available for retry calculation")

        delay = self.config.retry_delay * (
            self.config.retry_backoff_factor ** self.result.retry_count
        )

        return datetime.now(timezone.utc).replace(
            microsecond=0
        ) + timedelta(seconds=delay)

    def create_retry_task(self) -> Task:
        """Create a new task for retry."""
        retry_task = self.copy(deep=True)
        retry_task.task_id = uuid4().hex
        retry_task.status = TaskStatus.PENDING
        retry_task.scheduled_at = self.calculate_next_retry_time()

        # Update retry information
        if self.result:
            retry_task.result = None
            retry_task.context["retry_count"] = self.result.retry_count + 1
            retry_task.context["original_task_id"] = self.task_id

        return retry_task

    def add_tag(self, tag: str) -> None:
        """Add tag to task."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove tag from task."""
        if tag in self.tags:
            self.tags.remove(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if task has specific tag."""
        return tag in self.tags

    def set_context(self, key: str, value: Any) -> None:
        """Set context value."""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value."""
        return self.context.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "function_name": self.function_name,
            "args": self.args,
            "kwargs": self.kwargs,
            "config": self.config.dict(),
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "trigger": self.trigger,
            "context": self.context,
            "tags": self.tags,
            "status": self.status,
            "result": self.result.to_dict() if self.result else None,
            "worker_id": self.worker_id,
            "queue_name": self.queue_name
        }

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        arbitrary_types_allowed = True  # Allow Callable type
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            Callable: lambda f: getattr(f, "__name__", str(f))
        }


class TaskStats(BaseModel):
    """Task execution statistics."""

    # Execution counts
    total_tasks: int = Field(default=0, description="Total tasks")
    pending_tasks: int = Field(default=0, description="Pending tasks")
    running_tasks: int = Field(default=0, description="Running tasks")
    completed_tasks: int = Field(default=0, description="Completed tasks")
    failed_tasks: int = Field(default=0, description="Failed tasks")
    cancelled_tasks: int = Field(default=0, description="Cancelled tasks")

    # Performance metrics
    average_duration_seconds: float = Field(default=0.0, description="Average duration")
    min_duration_seconds: Optional[float] = Field(default=None, description="Minimum duration")
    max_duration_seconds: Optional[float] = Field(default=None, description="Maximum duration")

    # Throughput metrics
    tasks_per_hour: float = Field(default=0.0, description="Tasks per hour")
    success_rate_percent: float = Field(default=0.0, description="Success rate percentage")

    # Resource usage
    average_memory_mb: float = Field(default=0.0, description="Average memory usage")
    peak_memory_mb: Optional[float] = Field(default=None, description="Peak memory usage")
    average_cpu_percent: float = Field(default=0.0, description="Average CPU usage")

    # Time tracking
    first_task_at: Optional[datetime] = Field(default=None, description="First task timestamp")
    last_task_at: Optional[datetime] = Field(default=None, description="Last task timestamp")

    @validator("first_task_at", "last_task_at")
    def validate_timestamps(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure timestamps are timezone-aware."""
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def calculate_success_rate(self) -> float:
        """Calculate success rate percentage."""
        total_finished = self.completed_tasks + self.failed_tasks + self.cancelled_tasks
        if total_finished == 0:
            return 0.0
        return (self.completed_tasks / total_finished) * 100

    def calculate_throughput(self) -> float:
        """Calculate tasks per hour."""
        if not self.first_task_at or not self.last_task_at:
            return 0.0

        duration_hours = (self.last_task_at - self.first_task_at).total_seconds() / 3600
        if duration_hours == 0:
            return 0.0

        return self.total_tasks / duration_hours

    def update_from_result(self, result: TaskResult) -> None:
        """Update statistics from task result."""
        self.total_tasks += 1

        # Update status counts
        if result.status == TaskStatus.COMPLETED:
            self.completed_tasks += 1
        elif result.status == TaskStatus.FAILED:
            self.failed_tasks += 1
        elif result.status == TaskStatus.CANCELLED:
            self.cancelled_tasks += 1

        # Update duration metrics
        if result.duration_seconds:
            if self.min_duration_seconds is None or result.duration_seconds < self.min_duration_seconds:
                self.min_duration_seconds = result.duration_seconds

            if self.max_duration_seconds is None or result.duration_seconds > self.max_duration_seconds:
                self.max_duration_seconds = result.duration_seconds

            # Calculate new average duration
            total_completed = max(self.completed_tasks + self.failed_tasks, 1)
            self.average_duration_seconds = (
                (self.average_duration_seconds * (total_completed - 1) + result.duration_seconds) /
                total_completed
            )

        # Update resource metrics
        if result.memory_peak_mb:
            if self.peak_memory_mb is None or result.memory_peak_mb > self.peak_memory_mb:
                self.peak_memory_mb = result.memory_peak_mb

            # Calculate average memory
            self.average_memory_mb = (
                (self.average_memory_mb * (self.total_tasks - 1) + result.memory_peak_mb) /
                self.total_tasks
            )

        if result.cpu_usage_percent:
            # Calculate average CPU
            self.average_cpu_percent = (
                (self.average_cpu_percent * (self.total_tasks - 1) + result.cpu_usage_percent) /
                self.total_tasks
            )

        # Update time tracking
        if self.first_task_at is None or result.started_at < self.first_task_at:
            self.first_task_at = result.started_at

        if result.completed_at:
            if self.last_task_at is None or result.completed_at > self.last_task_at:
                self.last_task_at = result.completed_at

        # Recalculate derived metrics
        self.success_rate_percent = self.calculate_success_rate()
        self.tasks_per_hour = self.calculate_throughput()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_tasks": self.total_tasks,
            "pending_tasks": self.pending_tasks,
            "running_tasks": self.running_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "cancelled_tasks": self.cancelled_tasks,
            "average_duration_seconds": self.average_duration_seconds,
            "min_duration_seconds": self.min_duration_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "tasks_per_hour": self.tasks_per_hour,
            "success_rate_percent": self.success_rate_percent,
            "average_memory_mb": self.average_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "average_cpu_percent": self.average_cpu_percent,
            "first_task_at": self.first_task_at.isoformat() if self.first_task_at else None,
            "last_task_at": self.last_task_at.isoformat() if self.last_task_at else None
        }

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }