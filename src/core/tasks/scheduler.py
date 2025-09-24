"""
Task scheduling system with cron-like functionality.

This module provides advanced task scheduling capabilities with
cron expressions, interval scheduling, and event-driven triggers.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import croniter

from src.core.config import get_settings
from src.core.logging import get_logger
from .models import Task, TaskTrigger, TaskStatus
from .queue import get_task_queue_manager

logger = get_logger(__name__)
settings = get_settings()


class ScheduledTask:
    """
    Scheduled task wrapper with trigger information.

    Represents a task with scheduling information and trigger logic.
    """

    def __init__(
        self,
        task: Task,
        trigger: TaskTrigger,
        trigger_config: Dict[str, Any]
    ):
        """
        Initialize scheduled task.

        Args:
            task: Base task to schedule
            trigger: Trigger type
            trigger_config: Trigger configuration
        """
        self.task = task
        self.trigger = trigger
        self.trigger_config = trigger_config

        # Scheduling state
        self.next_run_time: Optional[datetime] = None
        self.last_run_time: Optional[datetime] = None
        self.run_count = 0
        self.is_enabled = True

        # Initialize next run time
        self._calculate_next_run_time()

        logger.debug(f"Created scheduled task {task.task_id} with trigger {trigger}")

    def _calculate_next_run_time(self) -> None:
        """Calculate the next execution time based on trigger."""
        now = datetime.now(timezone.utc)

        try:
            if self.trigger == TaskTrigger.ONCE:
                # One-time execution
                if self.run_count == 0:
                    scheduled_at = self.trigger_config.get("scheduled_at")
                    if scheduled_at:
                        self.next_run_time = scheduled_at
                    else:
                        self.next_run_time = now
                else:
                    self.next_run_time = None  # Already executed

            elif self.trigger == TaskTrigger.INTERVAL:
                # Interval-based execution
                interval_seconds = self.trigger_config.get("interval_seconds", 60)

                if self.last_run_time:
                    self.next_run_time = self.last_run_time + timedelta(seconds=interval_seconds)
                else:
                    # First run
                    delay = self.trigger_config.get("initial_delay_seconds", 0)
                    self.next_run_time = now + timedelta(seconds=delay)

            elif self.trigger == TaskTrigger.CRON:
                # Cron expression
                cron_expr = self.trigger_config.get("cron_expression")
                if not cron_expr:
                    raise ValueError("Cron expression required for cron trigger")

                cron = croniter.croniter(cron_expr, now)
                self.next_run_time = cron.get_next(datetime)

            elif self.trigger == TaskTrigger.EVENT:
                # Event-driven (external trigger)
                self.next_run_time = None

            else:
                # Manual trigger
                self.next_run_time = None

        except Exception as e:
            logger.error(f"Error calculating next run time for task {self.task.task_id}: {e}")
            self.next_run_time = None

    def is_due(self, current_time: datetime = None) -> bool:
        """Check if task is due for execution."""
        if not self.is_enabled or not self.next_run_time:
            return False

        if current_time is None:
            current_time = datetime.now(timezone.utc)

        return current_time >= self.next_run_time

    def should_continue(self) -> bool:
        """Check if task should continue being scheduled."""
        if not self.is_enabled:
            return False

        # Check max runs limit
        max_runs = self.trigger_config.get("max_runs")
        if max_runs is not None and self.run_count >= max_runs:
            return False

        # Check end time
        end_time = self.trigger_config.get("end_time")
        if end_time and datetime.now(timezone.utc) >= end_time:
            return False

        return True

    def create_execution_task(self) -> Task:
        """Create a new task instance for execution."""
        # Create a copy of the base task
        execution_task = self.task.copy(deep=True)
        execution_task.task_id = f"{self.task.task_id}_run_{self.run_count + 1}"
        execution_task.status = TaskStatus.PENDING
        execution_task.scheduled_at = self.next_run_time

        # Add scheduling context
        execution_task.context.update({
            "scheduled_task_id": self.task.task_id,
            "run_count": self.run_count + 1,
            "trigger_type": self.trigger.value,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None
        })

        return execution_task

    def mark_executed(self) -> None:
        """Mark task as executed and update scheduling."""
        self.last_run_time = datetime.now(timezone.utc)
        self.run_count += 1

        # Calculate next run time
        self._calculate_next_run_time()

        logger.debug(
            f"Scheduled task {self.task.task_id} executed (run {self.run_count}), "
            f"next run: {self.next_run_time}"
        )

    def get_info(self) -> Dict[str, Any]:
        """Get scheduling information."""
        return {
            "task_id": self.task.task_id,
            "task_name": self.task.name,
            "trigger": self.trigger.value,
            "trigger_config": self.trigger_config,
            "next_run_time": self.next_run_time.isoformat() if self.next_run_time else None,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "run_count": self.run_count,
            "is_enabled": self.is_enabled,
            "should_continue": self.should_continue()
        }


class TaskScheduler:
    """
    Advanced task scheduler with cron-like functionality.

    Provides comprehensive task scheduling with multiple trigger types,
    timezone support, and robust error handling.
    """

    def __init__(self, check_interval: float = 1.0):
        """
        Initialize task scheduler.

        Args:
            check_interval: How often to check for due tasks (seconds)
        """
        self.check_interval = check_interval

        # Scheduled tasks storage
        self._scheduled_tasks: Dict[str, ScheduledTask] = {}
        self._lock = threading.RLock()

        # Scheduler state
        self._is_running = False
        self._scheduler_thread: Optional[threading.Thread] = None

        # Statistics
        self._tasks_scheduled = 0
        self._tasks_executed = 0
        self._tasks_failed = 0
        self._start_time = time.time()

        # Dependencies
        self._queue_manager = get_task_queue_manager()

        logger.info("Task scheduler initialized")

    def start(self) -> bool:
        """Start the task scheduler."""
        if self._is_running:
            logger.warning("Task scheduler already running")
            return True

        try:
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="TaskScheduler"
            )
            self._scheduler_thread.daemon = True
            self._scheduler_thread.start()

            self._is_running = True
            self._start_time = time.time()

            logger.info("Task scheduler started")
            return True

        except Exception as e:
            logger.error(f"Failed to start task scheduler: {e}")
            return False

    def stop(self) -> None:
        """Stop the task scheduler."""
        if not self._is_running:
            return

        logger.info("Stopping task scheduler...")

        self._is_running = False

        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)

        logger.info("Task scheduler stopped")

    def schedule_once(
        self,
        task: Task,
        scheduled_at: Optional[datetime] = None
    ) -> bool:
        """
        Schedule task for one-time execution.

        Args:
            task: Task to schedule
            scheduled_at: When to execute (now if None)

        Returns:
            bool: True if scheduled successfully
        """
        trigger_config = {
            "scheduled_at": scheduled_at or datetime.now(timezone.utc)
        }

        return self._schedule_task(task, TaskTrigger.ONCE, trigger_config)

    def schedule_interval(
        self,
        task: Task,
        interval_seconds: int,
        initial_delay_seconds: int = 0,
        max_runs: Optional[int] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """
        Schedule task for interval-based execution.

        Args:
            task: Task to schedule
            interval_seconds: Execution interval in seconds
            initial_delay_seconds: Initial delay before first execution
            max_runs: Maximum number of runs
            end_time: When to stop scheduling

        Returns:
            bool: True if scheduled successfully
        """
        trigger_config = {
            "interval_seconds": interval_seconds,
            "initial_delay_seconds": initial_delay_seconds,
            "max_runs": max_runs,
            "end_time": end_time
        }

        return self._schedule_task(task, TaskTrigger.INTERVAL, trigger_config)

    def schedule_cron(
        self,
        task: Task,
        cron_expression: str,
        max_runs: Optional[int] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """
        Schedule task using cron expression.

        Args:
            task: Task to schedule
            cron_expression: Cron expression (e.g., "0 */6 * * *")
            max_runs: Maximum number of runs
            end_time: When to stop scheduling

        Returns:
            bool: True if scheduled successfully
        """
        # Validate cron expression
        try:
            croniter.croniter(cron_expression)
        except Exception as e:
            logger.error(f"Invalid cron expression '{cron_expression}': {e}")
            return False

        trigger_config = {
            "cron_expression": cron_expression,
            "max_runs": max_runs,
            "end_time": end_time
        }

        return self._schedule_task(task, TaskTrigger.CRON, trigger_config)

    def schedule_task(self, task: Task) -> bool:
        """
        Schedule task based on its trigger configuration.

        Args:
            task: Task to schedule (must have trigger info)

        Returns:
            bool: True if scheduled successfully
        """
        if task.trigger == TaskTrigger.MANUAL:
            logger.warning(f"Cannot schedule manual trigger task {task.task_id}")
            return False

        # Extract trigger config from task context
        trigger_config = task.context.get("trigger_config", {})

        return self._schedule_task(task, task.trigger, trigger_config)

    def _schedule_task(
        self,
        task: Task,
        trigger: TaskTrigger,
        trigger_config: Dict[str, Any]
    ) -> bool:
        """Internal method to schedule a task."""
        try:
            with self._lock:
                if task.task_id in self._scheduled_tasks:
                    logger.warning(f"Task {task.task_id} already scheduled")
                    return False

                # Create scheduled task
                scheduled_task = ScheduledTask(task, trigger, trigger_config)

                # Store scheduled task
                self._scheduled_tasks[task.task_id] = scheduled_task
                self._tasks_scheduled += 1

                logger.info(
                    f"Scheduled task {task.task_id} with trigger {trigger}, "
                    f"next run: {scheduled_task.next_run_time}"
                )

                return True

        except Exception as e:
            logger.error(f"Error scheduling task {task.task_id}: {e}")
            return False

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel scheduled task.

        Args:
            task_id: Task ID to cancel

        Returns:
            bool: True if cancelled successfully
        """
        try:
            with self._lock:
                if task_id not in self._scheduled_tasks:
                    logger.warning(f"Scheduled task {task_id} not found")
                    return False

                del self._scheduled_tasks[task_id]

            logger.info(f"Cancelled scheduled task {task_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling scheduled task {task_id}: {e}")
            return False

    def enable_task(self, task_id: str) -> bool:
        """Enable scheduled task."""
        with self._lock:
            scheduled_task = self._scheduled_tasks.get(task_id)
            if scheduled_task:
                scheduled_task.is_enabled = True
                return True
            return False

    def disable_task(self, task_id: str) -> bool:
        """Disable scheduled task."""
        with self._lock:
            scheduled_task = self._scheduled_tasks.get(task_id)
            if scheduled_task:
                scheduled_task.is_enabled = False
                return True
            return False

    def trigger_task(self, task_id: str) -> bool:
        """
        Manually trigger scheduled task execution.

        Args:
            task_id: Scheduled task ID to trigger

        Returns:
            bool: True if triggered successfully
        """
        try:
            with self._lock:
                scheduled_task = self._scheduled_tasks.get(task_id)
                if not scheduled_task:
                    logger.warning(f"Scheduled task {task_id} not found")
                    return False

            # Execute immediately
            return self._execute_scheduled_task(scheduled_task)

        except Exception as e:
            logger.error(f"Error triggering task {task_id}: {e}")
            return False

    def list_scheduled_tasks(self, include_disabled: bool = True) -> List[Dict[str, Any]]:
        """
        List all scheduled tasks.

        Args:
            include_disabled: Include disabled tasks

        Returns:
            List of scheduled task information
        """
        with self._lock:
            tasks = []
            for scheduled_task in self._scheduled_tasks.values():
                if include_disabled or scheduled_task.is_enabled:
                    tasks.append(scheduled_task.get_info())

            return sorted(tasks, key=lambda x: x.get("next_run_time") or "")

    def get_scheduled_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get information about specific scheduled task."""
        with self._lock:
            scheduled_task = self._scheduled_tasks.get(task_id)
            return scheduled_task.get_info() if scheduled_task else None

    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.debug("Started scheduler loop")

        while self._is_running:
            try:
                current_time = datetime.now(timezone.utc)
                due_tasks = []

                # Find due tasks
                with self._lock:
                    for scheduled_task in self._scheduled_tasks.values():
                        if scheduled_task.is_due(current_time):
                            due_tasks.append(scheduled_task)

                # Execute due tasks
                for scheduled_task in due_tasks:
                    try:
                        self._execute_scheduled_task(scheduled_task)
                    except Exception as e:
                        logger.error(
                            f"Error executing scheduled task {scheduled_task.task.task_id}: {e}"
                        )
                        self._tasks_failed += 1

                # Cleanup completed tasks
                self._cleanup_completed_tasks()

                # Sleep until next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(self.check_interval)

        logger.debug("Exited scheduler loop")

    def _execute_scheduled_task(self, scheduled_task: ScheduledTask) -> bool:
        """Execute a scheduled task."""
        try:
            # Create execution task
            execution_task = scheduled_task.create_execution_task()

            # Submit to queue
            queue = self._queue_manager.get_or_create_queue(execution_task.queue_name)
            success = queue.put(execution_task, execution_task.config.priority)

            if success:
                # Mark as executed
                scheduled_task.mark_executed()
                self._tasks_executed += 1

                logger.debug(f"Executed scheduled task {scheduled_task.task.task_id}")

                # Check if task should continue
                if not scheduled_task.should_continue():
                    with self._lock:
                        self._scheduled_tasks.pop(scheduled_task.task.task_id, None)
                    logger.info(f"Scheduled task {scheduled_task.task.task_id} completed")

                return True
            else:
                logger.error(f"Failed to queue execution task for {scheduled_task.task.task_id}")
                return False

        except Exception as e:
            logger.error(f"Error executing scheduled task {scheduled_task.task.task_id}: {e}")
            return False

    def _cleanup_completed_tasks(self) -> None:
        """Remove completed scheduled tasks."""
        try:
            with self._lock:
                to_remove = []

                for task_id, scheduled_task in self._scheduled_tasks.items():
                    if not scheduled_task.should_continue():
                        to_remove.append(task_id)

                for task_id in to_remove:
                    self._scheduled_tasks.pop(task_id, None)

                if to_remove:
                    logger.debug(f"Cleaned up {len(to_remove)} completed scheduled tasks")

        except Exception as e:
            logger.error(f"Error during scheduled task cleanup: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        runtime = time.time() - self._start_time

        with self._lock:
            scheduled_count = len(self._scheduled_tasks)
            enabled_count = sum(1 for st in self._scheduled_tasks.values() if st.is_enabled)

        return {
            "is_running": self._is_running,
            "runtime_seconds": runtime,
            "scheduled_tasks": scheduled_count,
            "enabled_tasks": enabled_count,
            "tasks_scheduled": self._tasks_scheduled,
            "tasks_executed": self._tasks_executed,
            "tasks_failed": self._tasks_failed,
            "execution_rate": self._tasks_executed / runtime if runtime > 0 else 0,
            "success_rate": (
                (self._tasks_executed - self._tasks_failed) / max(self._tasks_executed, 1) * 100
            )
        }

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running