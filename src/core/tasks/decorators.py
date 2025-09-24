"""
Task decorators for easy task creation and scheduling.

This module provides convenient decorators for converting functions
into background tasks with various scheduling options.
"""

from __future__ import annotations

import functools
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .manager import get_task_manager
from .models import TaskPriority, TaskConfig


def background_task(
    name: str = None,
    description: str = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    queue_name: str = "default",
    timeout_seconds: int = 300,
    max_retries: int = 3,
    tags: List[str] = None,
    context: Dict[str, Any] = None
) -> Callable:
    """
    Decorator to convert function into a background task.

    Args:
        name: Task name (uses function name if None)
        description: Task description
        priority: Task priority
        queue_name: Target queue name
        timeout_seconds: Execution timeout
        max_retries: Maximum retry attempts
        tags: Task tags
        context: Task context

    Returns:
        Decorated function that submits task when called
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> str:
            task_manager = get_task_manager()

            # Create task configuration
            config = TaskConfig(
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                priority=priority
            )

            # Submit task
            return task_manager.submit_task(
                function=func,
                *args,
                name=name or func.__name__,
                description=description,
                priority=priority,
                queue_name=queue_name,
                config=config,
                tags=tags or [],
                context=context or {},
                **kwargs
            )

        # Add task execution method
        def execute_now(*args, **kwargs):
            """Execute function directly (not as background task)."""
            return func(*args, **kwargs)

        wrapper.execute_now = execute_now
        wrapper.original_function = func

        return wrapper

    return decorator


def scheduled_task(
    interval_seconds: int = None,
    cron_expression: str = None,
    name: str = None,
    description: str = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    queue_name: str = "default",
    timeout_seconds: int = 300,
    max_retries: int = 3,
    max_runs: int = None,
    end_time: datetime = None,
    tags: List[str] = None,
    context: Dict[str, Any] = None,
    auto_start: bool = True
) -> Callable:
    """
    Decorator to create scheduled task.

    Args:
        interval_seconds: Execution interval (for interval scheduling)
        cron_expression: Cron expression (for cron scheduling)
        name: Task name
        description: Task description
        priority: Task priority
        queue_name: Target queue name
        timeout_seconds: Execution timeout
        max_retries: Maximum retry attempts
        max_runs: Maximum number of runs
        end_time: When to stop scheduling
        tags: Task tags
        context: Task context
        auto_start: Automatically start scheduling

    Returns:
        Decorated function with scheduling methods
    """
    if interval_seconds is None and cron_expression is None:
        raise ValueError("Must specify either interval_seconds or cron_expression")

    if interval_seconds is not None and cron_expression is not None:
        raise ValueError("Cannot specify both interval_seconds and cron_expression")

    def decorator(func: Callable) -> Callable:
        task_manager = get_task_manager()

        # Create task configuration
        config = TaskConfig(
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            priority=priority
        )

        # Create base task
        from .models import Task, TaskTrigger
        task = Task(
            name=name or func.__name__,
            description=description,
            function=func,
            config=config,
            tags=tags or [],
            context=context or {},
            queue_name=queue_name
        )

        # Schedule task
        if auto_start:
            if interval_seconds is not None:
                task_manager._scheduler.schedule_interval(
                    task=task,
                    interval_seconds=interval_seconds,
                    max_runs=max_runs,
                    end_time=end_time
                )
            else:
                task_manager._scheduler.schedule_cron(
                    task=task,
                    cron_expression=cron_expression,
                    max_runs=max_runs,
                    end_time=end_time
                )

        # Create wrapper with scheduling methods
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Execute function directly."""
            return func(*args, **kwargs)

        def start_scheduling():
            """Start task scheduling."""
            if interval_seconds is not None:
                return task_manager._scheduler.schedule_interval(
                    task=task,
                    interval_seconds=interval_seconds,
                    max_runs=max_runs,
                    end_time=end_time
                )
            else:
                return task_manager._scheduler.schedule_cron(
                    task=task,
                    cron_expression=cron_expression,
                    max_runs=max_runs,
                    end_time=end_time
                )

        def stop_scheduling():
            """Stop task scheduling."""
            return task_manager._scheduler.cancel_task(task.task_id)

        def trigger_now():
            """Trigger task execution immediately."""
            return task_manager._scheduler.trigger_task(task.task_id)

        def get_schedule_info():
            """Get scheduling information."""
            return task_manager._scheduler.get_scheduled_task(task.task_id)

        # Add methods to wrapper
        wrapper.start_scheduling = start_scheduling
        wrapper.stop_scheduling = stop_scheduling
        wrapper.trigger_now = trigger_now
        wrapper.get_schedule_info = get_schedule_info
        wrapper.task_id = task.task_id
        wrapper.original_function = func

        return wrapper

    return decorator


def retry_task(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_on_exceptions: List[Exception] = None
) -> Callable:
    """
    Decorator to add retry logic to task functions.

    Args:
        max_retries: Maximum retry attempts
        retry_delay: Initial retry delay
        retry_backoff_factor: Backoff multiplier
        retry_on_exceptions: Exception types to retry on

    Returns:
        Decorated function with retry logic
    """
    if retry_on_exceptions is None:
        retry_on_exceptions = [Exception]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    should_retry = any(
                        isinstance(e, exc_type)
                        for exc_type in retry_on_exceptions
                    )

                    if not should_retry or attempt >= max_retries:
                        raise e

                    # Calculate delay
                    delay = retry_delay * (retry_backoff_factor ** attempt)

                    # Log retry attempt
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Task {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay}s: {e}"
                    )

                    # Sleep before retry
                    import time
                    time.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception

        wrapper.original_function = func
        return wrapper

    return decorator


def periodic_task(minutes: int = None, hours: int = None, days: int = None) -> Callable:
    """
    Convenient decorator for periodic tasks.

    Args:
        minutes: Execute every N minutes
        hours: Execute every N hours
        days: Execute every N days

    Returns:
        Decorated function scheduled for periodic execution
    """
    if not any([minutes, hours, days]):
        raise ValueError("Must specify at least one time unit")

    # Calculate total seconds
    total_seconds = 0
    if minutes:
        total_seconds += minutes * 60
    if hours:
        total_seconds += hours * 3600
    if days:
        total_seconds += days * 86400

    return scheduled_task(interval_seconds=total_seconds)


def daily_task(hour: int = 0, minute: int = 0) -> Callable:
    """
    Decorator for daily scheduled tasks.

    Args:
        hour: Hour to execute (0-23)
        minute: Minute to execute (0-59)

    Returns:
        Decorated function scheduled for daily execution
    """
    cron_expr = f"{minute} {hour} * * *"
    return scheduled_task(cron_expression=cron_expr)


def weekly_task(day_of_week: int = 0, hour: int = 0, minute: int = 0) -> Callable:
    """
    Decorator for weekly scheduled tasks.

    Args:
        day_of_week: Day of week (0=Sunday, 6=Saturday)
        hour: Hour to execute (0-23)
        minute: Minute to execute (0-59)

    Returns:
        Decorated function scheduled for weekly execution
    """
    cron_expr = f"{minute} {hour} * * {day_of_week}"
    return scheduled_task(cron_expression=cron_expr)


def high_priority_task(**kwargs) -> Callable:
    """Decorator for high priority background tasks."""
    kwargs.setdefault("priority", TaskPriority.HIGH)
    return background_task(**kwargs)


def critical_task(**kwargs) -> Callable:
    """Decorator for critical priority background tasks."""
    kwargs.setdefault("priority", TaskPriority.CRITICAL)
    kwargs.setdefault("max_retries", 5)
    kwargs.setdefault("timeout_seconds", 600)
    return background_task(**kwargs)