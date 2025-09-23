"""
Custom logging handlers for MetaTrader Python Framework.

This module provides specialized logging handlers for different output targets
including enhanced console output, file rotation, and remote logging.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

try:
    from rich.console import Console
    from rich.logging import RichHandler as BaseRichHandler
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class RichHandler(logging.Handler):
    """Enhanced Rich handler for beautiful console output."""

    def __init__(
        self,
        level: Union[str, int] = logging.NOTSET,
        console: Optional[Console] = None,
        show_time: bool = True,
        show_level: bool = True,
        show_path: bool = True,
        markup: bool = False,
        rich_tracebacks: bool = True,
        tracebacks_width: Optional[int] = None,
        tracebacks_extra_lines: int = 3,
        tracebacks_theme: Optional[str] = None,
        tracebacks_word_wrap: bool = True,
        tracebacks_show_locals: bool = False,
        locals_max_length: int = 10,
        locals_max_string: int = 80,
        keywords: Optional[List[str]] = None,
        log_time_format: Union[str, callable] = "[%x %X]",
    ) -> None:
        """
        Initialize Rich handler.

        Args:
            level: Logging level
            console: Rich console instance
            show_time: Show timestamp
            show_level: Show log level
            show_path: Show file path
            markup: Enable Rich markup
            rich_tracebacks: Use Rich for tracebacks
            tracebacks_width: Traceback width
            tracebacks_extra_lines: Extra lines in tracebacks
            tracebacks_theme: Traceback syntax theme
            tracebacks_word_wrap: Word wrap in tracebacks
            tracebacks_show_locals: Show local variables
            locals_max_length: Max length for locals
            locals_max_string: Max string length for locals
            keywords: Keywords to highlight
            log_time_format: Time format
        """
        super().__init__(level)

        if not RICH_AVAILABLE:
            raise ImportError("Rich is required for RichHandler")

        self.console = console or Console(stderr=True)
        self._rich_handler = BaseRichHandler(
            level=level,
            console=self.console,
            show_time=show_time,
            show_level=show_level,
            show_path=show_path,
            markup=markup,
            rich_tracebacks=rich_tracebacks,
            tracebacks_width=tracebacks_width,
            tracebacks_extra_lines=tracebacks_extra_lines,
            tracebacks_theme=tracebacks_theme,
            tracebacks_word_wrap=tracebacks_word_wrap,
            tracebacks_show_locals=tracebacks_show_locals,
            locals_max_length=locals_max_length,
            locals_max_string=locals_max_string,
            keywords=keywords,
            log_time_format=log_time_format,
        )

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record.

        Args:
            record: Log record to emit
        """
        try:
            self._rich_handler.emit(record)
        except Exception:
            self.handleError(record)


class AsyncFileHandler(logging.Handler):
    """Asynchronous file handler for high-performance logging."""

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = "a",
        encoding: Optional[str] = None,
        delay: bool = False,
        queue_size: int = 1000,
    ) -> None:
        """
        Initialize async file handler.

        Args:
            filename: Log file path
            mode: File open mode
            encoding: File encoding
            delay: Delay file opening
            queue_size: Maximum queue size
        """
        super().__init__()
        self.filename = Path(filename)
        self.mode = mode
        self.encoding = encoding or "utf-8"
        self.delay = delay

        # Create parent directory if it doesn't exist
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        # Initialize queue and worker thread
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._file: Optional[TextIO] = None

        if not delay:
            self._open_file()

        self._start_worker()

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record asynchronously.

        Args:
            record: Log record to emit
        """
        try:
            message = self.format(record)
            if not self._queue.full():
                self._queue.put_nowait(message)
            else:
                # Drop the message if queue is full
                pass
        except Exception:
            self.handleError(record)

    def _start_worker(self) -> None:
        """Start the worker thread."""
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self) -> None:
        """Worker thread for writing log messages."""
        while not self._stop_event.is_set():
            try:
                # Get message with timeout
                try:
                    message = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Write message to file
                if self._file is None:
                    self._open_file()

                if self._file:
                    self._file.write(message + "\n")
                    self._file.flush()

                self._queue.task_done()

            except Exception:
                # Log errors to stderr
                import traceback
                traceback.print_exc()

    def _open_file(self) -> None:
        """Open the log file."""
        try:
            self._file = open(self.filename, self.mode, encoding=self.encoding)
        except Exception:
            self._file = None

    def close(self) -> None:
        """Close the handler and clean up resources."""
        # Stop worker thread
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        # Close file
        if self._file:
            self._file.close()
            self._file = None

        super().close()


class RotatingAsyncFileHandler(AsyncFileHandler):
    """Asynchronous rotating file handler."""

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = "a",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: Optional[str] = None,
        delay: bool = False,
        queue_size: int = 1000,
    ) -> None:
        """
        Initialize rotating async file handler.

        Args:
            filename: Log file path
            mode: File open mode
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening
            queue_size: Maximum queue size
        """
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        super().__init__(filename, mode, encoding, delay, queue_size)

    def _worker(self) -> None:
        """Worker thread with file rotation support."""
        while not self._stop_event.is_set():
            try:
                # Get message with timeout
                try:
                    message = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Check if rotation is needed
                self._check_rotation()

                # Write message to file
                if self._file is None:
                    self._open_file()

                if self._file:
                    self._file.write(message + "\n")
                    self._file.flush()

                self._queue.task_done()

            except Exception:
                # Log errors to stderr
                import traceback
                traceback.print_exc()

    def _check_rotation(self) -> None:
        """Check if file rotation is needed."""
        if self._file and self.filename.exists():
            if self.filename.stat().st_size >= self.max_bytes:
                self._rotate_file()

    def _rotate_file(self) -> None:
        """Rotate the log file."""
        if self._file:
            self._file.close()
            self._file = None

        # Rotate backup files
        for i in range(self.backup_count - 1, 0, -1):
            src = Path(f"{self.filename}.{i}")
            dst = Path(f"{self.filename}.{i + 1}")

            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)

        # Move current file to .1
        if self.filename.exists():
            backup_path = Path(f"{self.filename}.1")
            if backup_path.exists():
                backup_path.unlink()
            self.filename.rename(backup_path)

        # Reopen file
        self._open_file()


class BufferedHandler(logging.Handler):
    """Buffered handler that flushes based on buffer size or time."""

    def __init__(
        self,
        target_handler: logging.Handler,
        buffer_size: int = 100,
        flush_interval: float = 30.0,
    ) -> None:
        """
        Initialize buffered handler.

        Args:
            target_handler: Target handler to forward buffered records
            buffer_size: Maximum buffer size before flush
            flush_interval: Maximum time between flushes (seconds)
        """
        super().__init__()
        self.target_handler = target_handler
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self._buffer: List[logging.LogRecord] = []
        self._last_flush = time.time()
        self._lock = threading.Lock()

        # Start flush timer
        self._start_flush_timer()

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to buffer.

        Args:
            record: Log record to emit
        """
        with self._lock:
            self._buffer.append(record)

            # Check if we need to flush
            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush the buffer to target handler."""
        if not self._buffer:
            return

        for record in self._buffer:
            try:
                self.target_handler.emit(record)
            except Exception:
                self.handleError(record)

        self._buffer.clear()
        self._last_flush = time.time()

    def _start_flush_timer(self) -> None:
        """Start the flush timer."""
        def flush_timer():
            while True:
                time.sleep(1.0)
                with self._lock:
                    if (
                        self._buffer
                        and time.time() - self._last_flush >= self.flush_interval
                    ):
                        self._flush_buffer()

        timer_thread = threading.Thread(target=flush_timer, daemon=True)
        timer_thread.start()

    def flush(self) -> None:
        """Force flush the buffer."""
        with self._lock:
            self._flush_buffer()
        self.target_handler.flush()

    def close(self) -> None:
        """Close the handler."""
        self.flush()
        self.target_handler.close()
        super().close()


class FilteredHandler(logging.Handler):
    """Handler that applies filters before forwarding to target handler."""

    def __init__(
        self,
        target_handler: logging.Handler,
        filters: Optional[List[logging.Filter]] = None,
    ) -> None:
        """
        Initialize filtered handler.

        Args:
            target_handler: Target handler to forward filtered records
            filters: List of filters to apply
        """
        super().__init__()
        self.target_handler = target_handler

        if filters:
            for filter_obj in filters:
                self.addFilter(filter_obj)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record after applying filters.

        Args:
            record: Log record to emit
        """
        if self.filter(record):
            self.target_handler.emit(record)

    def flush(self) -> None:
        """Flush the target handler."""
        self.target_handler.flush()

    def close(self) -> None:
        """Close the target handler."""
        self.target_handler.close()
        super().close()


class TradingHandler(logging.Handler):
    """Specialized handler for trading-related logs."""

    def __init__(
        self,
        target_handler: logging.Handler,
        include_context: bool = True,
        separate_trades: bool = True,
    ) -> None:
        """
        Initialize trading handler.

        Args:
            target_handler: Target handler for forwarding records
            include_context: Whether to include trading context
            separate_trades: Whether to separate trade logs
        """
        super().__init__()
        self.target_handler = target_handler
        self.include_context = include_context
        self.separate_trades = separate_trades

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a trading log record.

        Args:
            record: Log record to emit
        """
        # Enhance record with trading context
        if self.include_context:
            self._add_trading_context(record)

        self.target_handler.emit(record)

    def _add_trading_context(self, record: logging.LogRecord) -> None:
        """
        Add trading context to log record.

        Args:
            record: Log record to enhance
        """
        # Add timestamp for trade correlation
        if not hasattr(record, "trade_time"):
            record.trade_time = time.time()

        # Add trading session information
        if not hasattr(record, "session"):
            record.session = "main"


def create_handler(
    handler_type: str,
    **kwargs: Any,
) -> logging.Handler:
    """
    Create a handler instance based on type.

    Args:
        handler_type: Type of handler
        **kwargs: Additional arguments for handler

    Returns:
        Handler instance
    """
    handlers = {
        "rich": RichHandler,
        "async_file": AsyncFileHandler,
        "rotating_async_file": RotatingAsyncFileHandler,
        "buffered": BufferedHandler,
        "filtered": FilteredHandler,
        "trading": TradingHandler,
    }

    handler_class = handlers.get(handler_type.lower())
    if not handler_class:
        raise ValueError(f"Unknown handler type: {handler_type}")

    return handler_class(**kwargs)