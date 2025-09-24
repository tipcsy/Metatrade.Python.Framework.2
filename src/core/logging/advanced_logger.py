"""
Advanced Logging System.

This module provides a comprehensive logging system with structured logging,
performance metrics, distributed tracing, log aggregation, and advanced
filtering capabilities for the MetaTrader Python Framework.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Callable, TextIO
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import json
import gzip
import threading
from contextlib import contextmanager
import traceback
import sys
import os

import structlog
from structlog import configure, get_logger as struct_get_logger
from structlog.processors import JSONRenderer, KeyValueRenderer, CallsiteParameterAdder
from structlog.stdlib import LoggerFactory, add_log_level, PositionalArgumentsFormatter
from structlog.dev import ConsoleRenderer

from src.core.config.settings import LoggingSettings
from src.core.exceptions import LoggingError

# Standard library imports for logging
import logging
import logging.handlers
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Log output formats."""
    JSON = "json"
    KEY_VALUE = "key_value"
    CONSOLE = "console"
    PLAIN = "plain"


@dataclass
class LogMetrics:
    """Logging system metrics."""
    total_logs: int = 0
    logs_by_level: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    performance_logs: List[Dict[str, Any]] = field(default_factory=list)
    last_error_time: Optional[float] = None
    system_errors: int = 0


@dataclass
class LogContext:
    """Logging context for distributed tracing."""
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None


class PerformanceTimer:
    """Context manager for performance timing."""

    def __init__(self, logger: structlog.BoundLogger, operation: str, **kwargs) -> None:
        """Initialize performance timer.

        Args:
            logger: Logger instance
            operation: Operation being timed
            **kwargs: Additional context
        """
        self.logger = logger
        self.operation = operation
        self.context = kwargs
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> PerformanceTimer:
        """Start timing."""
        self.start_time = time.time()
        self.logger.debug(
            "Operation started",
            operation=self.operation,
            **self.context
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End timing and log results."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type is None:
            self.logger.info(
                "Operation completed",
                operation=self.operation,
                duration=duration,
                **self.context
            )
        else:
            self.logger.error(
                "Operation failed",
                operation=self.operation,
                duration=duration,
                error=str(exc_val),
                **self.context
            )


class AdvancedLogger:
    """
    Advanced logging system with structured logging and performance monitoring.

    Features:
    - Structured logging with JSON output
    - Performance timing and metrics
    - Distributed tracing support
    - Log aggregation and filtering
    - Automatic log rotation and compression
    - Error rate monitoring
    - Context-aware logging
    """

    def __init__(self, settings: LoggingSettings) -> None:
        """Initialize advanced logging system.

        Args:
            settings: Logging configuration settings
        """
        self.settings = settings
        self._is_initialized = False

        # Logging context
        self._context: LogContext = LogContext()
        self._context_stack: List[LogContext] = []
        self._context_lock = threading.RLock()

        # Metrics
        self._metrics = LogMetrics()
        self._metrics_lock = threading.RLock()

        # Log filters and processors
        self._filters: List[Callable[[Dict[str, Any]], bool]] = []
        self._processors: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []

        # Background tasks
        self._metrics_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Log handlers
        self._handlers: List[logging.Handler] = []
        self._root_logger: Optional[logging.Logger] = None

    async def initialize(self) -> None:
        """Initialize the logging system."""
        if self._is_initialized:
            return

        try:
            # Configure structured logging
            await self._configure_structlog()

            # Set up log handlers
            await self._setup_handlers()

            # Configure root logger
            self._configure_root_logger()

            # Start background tasks
            if self.settings.enable_metrics:
                self._metrics_task = asyncio.create_task(self._metrics_loop())

            if self.settings.enable_cleanup:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._is_initialized = True

            # Log successful initialization
            logger = self.get_logger(__name__)
            logger.info(
                "Advanced logging system initialized",
                level=self.settings.level,
                format=self.settings.format,
                handlers_count=len(self._handlers),
            )

        except Exception as e:
            print(f"Failed to initialize logging system: {e}", file=sys.stderr)
            raise LoggingError(f"Logging initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """Shutdown the logging system."""
        if not self._is_initialized:
            return

        try:
            # Signal stop
            self._stop_event.set()

            # Wait for background tasks
            if self._metrics_task:
                self._metrics_task.cancel()
                try:
                    await self._metrics_task
                except asyncio.CancelledError:
                    pass

            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Close handlers
            for handler in self._handlers:
                handler.close()

            self._is_initialized = False

        except Exception as e:
            print(f"Error shutting down logging system: {e}", file=sys.stderr)

    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a structured logger instance.

        Args:
            name: Logger name

        Returns:
            Structured logger instance
        """
        if not self._is_initialized:
            # Return a basic logger if not initialized
            return struct_get_logger(name)

        # Get logger with current context
        logger = struct_get_logger(name)

        with self._context_lock:
            if self._context.request_id:
                logger = logger.bind(request_id=self._context.request_id)
            if self._context.session_id:
                logger = logger.bind(session_id=self._context.session_id)
            if self._context.user_id:
                logger = logger.bind(user_id=self._context.user_id)
            if self._context.component:
                logger = logger.bind(component=self._context.component)
            if self._context.trace_id:
                logger = logger.bind(trace_id=self._context.trace_id)
            if self._context.span_id:
                logger = logger.bind(span_id=self._context.span_id)

        return logger

    @contextmanager
    def context(self, **kwargs):
        """Context manager for setting logging context.

        Args:
            **kwargs: Context variables to set
        """
        with self._context_lock:
            # Push current context
            self._context_stack.append(LogContext(**self._context.__dict__))

            # Update context
            for key, value in kwargs.items():
                if hasattr(self._context, key):
                    setattr(self._context, key, value)

        try:
            yield
        finally:
            with self._context_lock:
                # Restore previous context
                if self._context_stack:
                    self._context = self._context_stack.pop()

    def performance_timer(self, operation: str, **kwargs) -> PerformanceTimer:
        """Create a performance timer context manager.

        Args:
            operation: Operation being timed
            **kwargs: Additional context

        Returns:
            Performance timer context manager
        """
        logger = self.get_logger(__name__)
        return PerformanceTimer(logger, operation, **kwargs)

    def add_filter(self, filter_func: Callable[[Dict[str, Any]], bool]) -> None:
        """Add a log filter function.

        Args:
            filter_func: Function that returns True if log should be processed
        """
        self._filters.append(filter_func)

    def add_processor(self, processor_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Add a log processor function.

        Args:
            processor_func: Function to process log entries
        """
        self._processors.append(processor_func)

    def get_metrics(self) -> Dict[str, Any]:
        """Get logging metrics.

        Returns:
            Dictionary containing logging metrics
        """
        with self._metrics_lock:
            current_time = time.time()

            # Calculate error rate (errors per minute)
            minute_ago = current_time - 60
            recent_errors = sum(1 for error_time in self._metrics.errors_per_minute if error_time > minute_ago)

            return {
                "total_logs": self._metrics.total_logs,
                "logs_by_level": dict(self._metrics.logs_by_level),
                "errors_per_minute": recent_errors,
                "last_error_time": self._metrics.last_error_time,
                "system_errors": self._metrics.system_errors,
                "performance_logs_count": len(self._metrics.performance_logs),
            }

    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """Log performance data.

        Args:
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional performance data
        """
        with self._metrics_lock:
            self._metrics.performance_logs.append({
                "timestamp": time.time(),
                "operation": operation,
                "duration": duration,
                **kwargs
            })

            # Keep only recent performance logs
            if len(self._metrics.performance_logs) > self.settings.max_performance_logs:
                self._metrics.performance_logs.pop(0)

    async def _configure_structlog(self) -> None:
        """Configure structured logging."""
        # Determine processors based on format
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.CallsiteParameterAdder(parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]),
            structlog.processors.TimeStamper(fmt="ISO"),
        ]

        # Add custom processors
        for processor in self._processors:
            processors.append(processor)

        # Add metrics processor
        processors.append(self._metrics_processor)

        # Add format-specific processor
        if self.settings.format == LogFormat.JSON.value:
            processors.append(JSONRenderer())
        elif self.settings.format == LogFormat.KEY_VALUE.value:
            processors.append(KeyValueRenderer())
        elif self.settings.format == LogFormat.CONSOLE.value:
            processors.append(ConsoleRenderer())
        else:
            processors.append(structlog.processors.KeyValueRenderer())

        # Configure structlog
        configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    async def _setup_handlers(self) -> None:
        """Set up log handlers."""
        self._handlers = []

        # Console handler
        if self.settings.console.enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.settings.console.level))

            if self.settings.format == LogFormat.CONSOLE.value:
                formatter = logging.Formatter('%(message)s')
            else:
                formatter = logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                )
            console_handler.setFormatter(formatter)
            self._handlers.append(console_handler)

        # File handler
        if self.settings.file.enabled:
            # Ensure log directory exists
            log_dir = Path(self.settings.file.path).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            if self.settings.file.rotation == "size":
                file_handler = RotatingFileHandler(
                    filename=self.settings.file.path,
                    maxBytes=self.settings.file.max_size,
                    backupCount=self.settings.file.backup_count
                )
            elif self.settings.file.rotation == "time":
                file_handler = TimedRotatingFileHandler(
                    filename=self.settings.file.path,
                    when=self.settings.file.rotation_interval,
                    interval=1,
                    backupCount=self.settings.file.backup_count
                )
            else:
                file_handler = logging.FileHandler(self.settings.file.path)

            file_handler.setLevel(getattr(logging, self.settings.file.level))

            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
            file_handler.setFormatter(formatter)
            self._handlers.append(file_handler)

        # Error file handler
        if self.settings.error_file.enabled:
            # Ensure error log directory exists
            error_log_dir = Path(self.settings.error_file.path).parent
            error_log_dir.mkdir(parents=True, exist_ok=True)

            error_handler = RotatingFileHandler(
                filename=self.settings.error_file.path,
                maxBytes=self.settings.error_file.max_size,
                backupCount=self.settings.error_file.backup_count
            )
            error_handler.setLevel(logging.ERROR)

            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s\n%(pathname)s:%(lineno)d'
            )
            error_handler.setFormatter(formatter)
            self._handlers.append(error_handler)

    def _configure_root_logger(self) -> None:
        """Configure the root logger."""
        self._root_logger = logging.getLogger()
        self._root_logger.setLevel(getattr(logging, self.settings.level))

        # Remove existing handlers
        for handler in self._root_logger.handlers[:]:
            self._root_logger.removeHandler(handler)

        # Add our handlers
        for handler in self._handlers:
            self._root_logger.addHandler(handler)

    def _metrics_processor(self, logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process log entry for metrics collection."""
        with self._metrics_lock:
            self._metrics.total_logs += 1

            # Count by level
            level = event_dict.get('level', 'unknown').upper()
            self._metrics.logs_by_level[level] += 1

            # Track errors
            if level in ['ERROR', 'CRITICAL']:
                current_time = time.time()
                self._metrics.errors_per_minute.append(current_time)
                self._metrics.last_error_time = current_time

                # Check for system errors
                if 'exc_info' in event_dict or 'error' in event_dict:
                    self._metrics.system_errors += 1

        # Apply custom filters
        for filter_func in self._filters:
            try:
                if not filter_func(event_dict):
                    return None  # Skip this log entry
            except Exception as e:
                # Filter error - log it but don't block logging
                print(f"Log filter error: {e}", file=sys.stderr)

        return event_dict

    async def _metrics_loop(self) -> None:
        """Background metrics collection loop."""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=60.0)
                break
            except asyncio.TimeoutError:
                # Perform metrics cleanup
                await self._cleanup_metrics()

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=3600.0)  # 1 hour
                break
            except asyncio.TimeoutError:
                # Perform log file cleanup
                await self._cleanup_log_files()

    async def _cleanup_metrics(self) -> None:
        """Clean up old metrics data."""
        try:
            with self._metrics_lock:
                current_time = time.time()

                # Clean up old error timestamps
                minute_ago = current_time - 60
                while (self._metrics.errors_per_minute and
                       self._metrics.errors_per_minute[0] < minute_ago):
                    self._metrics.errors_per_minute.popleft()

                # Clean up old performance logs
                if len(self._metrics.performance_logs) > self.settings.max_performance_logs:
                    # Keep only the most recent logs
                    self._metrics.performance_logs = self._metrics.performance_logs[-self.settings.max_performance_logs:]

        except Exception as e:
            print(f"Error cleaning up metrics: {e}", file=sys.stderr)

    async def _cleanup_log_files(self) -> None:
        """Clean up old log files."""
        try:
            if not self.settings.enable_cleanup:
                return

            # Clean up old log files based on retention policy
            retention_days = getattr(self.settings, 'retention_days', 30)
            cutoff_time = time.time() - (retention_days * 24 * 3600)

            # Check file log directory
            if self.settings.file.enabled:
                log_dir = Path(self.settings.file.path).parent
                self._cleanup_directory(log_dir, cutoff_time)

            # Check error log directory
            if self.settings.error_file.enabled:
                error_log_dir = Path(self.settings.error_file.path).parent
                self._cleanup_directory(error_log_dir, cutoff_time)

        except Exception as e:
            print(f"Error cleaning up log files: {e}", file=sys.stderr)

    def _cleanup_directory(self, directory: Path, cutoff_time: float) -> None:
        """Clean up old files in a directory."""
        try:
            if not directory.exists():
                return

            for file_path in directory.glob("*.log.*"):
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                    except OSError as e:
                        print(f"Failed to delete old log file {file_path}: {e}", file=sys.stderr)

        except Exception as e:
            print(f"Error cleaning up directory {directory}: {e}", file=sys.stderr)


# Global logger instance
_logger_instance: Optional[AdvancedLogger] = None


async def initialize_logging(settings: LoggingSettings) -> None:
    """Initialize the global logging system.

    Args:
        settings: Logging configuration settings
    """
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = AdvancedLogger(settings)
        await _logger_instance.initialize()


async def shutdown_logging() -> None:
    """Shutdown the global logging system."""
    global _logger_instance

    if _logger_instance:
        await _logger_instance.shutdown()
        _logger_instance = None


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    global _logger_instance

    if _logger_instance:
        return _logger_instance.get_logger(name)
    else:
        # Return basic structlog logger if advanced logging not initialized
        return struct_get_logger(name)


def performance_timer(operation: str, **kwargs) -> PerformanceTimer:
    """Create a performance timer context manager.

    Args:
        operation: Operation being timed
        **kwargs: Additional context

    Returns:
        Performance timer context manager
    """
    global _logger_instance

    if _logger_instance:
        return _logger_instance.performance_timer(operation, **kwargs)
    else:
        # Return a no-op timer if advanced logging not initialized
        return PerformanceTimer(get_logger(__name__), operation, **kwargs)


def logging_context(**kwargs):
    """Context manager for setting logging context.

    Args:
        **kwargs: Context variables to set
    """
    global _logger_instance

    if _logger_instance:
        return _logger_instance.context(**kwargs)
    else:
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()


def get_logging_metrics() -> Dict[str, Any]:
    """Get logging system metrics.

    Returns:
        Dictionary containing logging metrics
    """
    global _logger_instance

    if _logger_instance:
        return _logger_instance.get_metrics()
    else:
        return {"error": "Logging system not initialized"}