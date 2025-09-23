"""
Core logging module for MetaTrader Python Framework.

This module provides a comprehensive logging framework with support for
structured logging, multiple output formats, and specialized handlers.
"""

from .formatters import (
    ColoredFormatter,
    FilteredFormatter,
    JsonFormatter,
    StructuredFormatter,
    TradingFormatter,
    create_formatter,
)
from .handlers import (
    AsyncFileHandler,
    BufferedHandler,
    FilteredHandler,
    RotatingAsyncFileHandler,
    RichHandler,
    TradingHandler,
    create_handler,
)
from .logger_factory import (
    LoggerFactory,
    create_console_handler,
    create_file_handler,
    get_logger,
    setup_logging,
)

__all__ = [
    # Logger factory
    "LoggerFactory",
    "get_logger",
    "setup_logging",
    "create_console_handler",
    "create_file_handler",
    # Formatters
    "JsonFormatter",
    "StructuredFormatter",
    "ColoredFormatter",
    "TradingFormatter",
    "FilteredFormatter",
    "create_formatter",
    # Handlers
    "RichHandler",
    "AsyncFileHandler",
    "RotatingAsyncFileHandler",
    "BufferedHandler",
    "FilteredHandler",
    "TradingHandler",
    "create_handler",
]