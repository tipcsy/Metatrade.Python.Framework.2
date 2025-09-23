"""
Logger factory module for MetaTrader Python Framework.

This module provides a centralized logger factory with support for different
output formats, automatic configuration, and structured logging.
"""

from __future__ import annotations

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog
from rich.console import Console
from rich.logging import RichHandler

from ..config import LogFormat, LogLevel, Settings


class LoggerFactory:
    """Factory for creating and configuring loggers."""

    _initialized = False
    _settings: Optional[Settings] = None
    _console: Optional[Console] = None

    @classmethod
    def initialize(cls, settings: Settings) -> None:
        """
        Initialize the logging system with the provided settings.

        Args:
            settings: Application settings containing logging configuration
        """
        cls._settings = settings
        cls._console = Console(stderr=True)

        # Configure Python logging
        cls._configure_python_logging()

        # Configure structlog
        cls._configure_structlog()

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> structlog.stdlib.BoundLogger:
        """
        Get a configured logger instance.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            raise RuntimeError("LoggerFactory not initialized. Call initialize() first.")

        return structlog.get_logger(name)

    @classmethod
    def get_stdlib_logger(cls, name: str) -> logging.Logger:
        """
        Get a standard library logger instance.

        Args:
            name: Logger name

        Returns:
            Standard library logger instance
        """
        return logging.getLogger(name)

    @classmethod
    def _configure_python_logging(cls) -> None:
        """Configure Python standard library logging."""
        if not cls._settings:
            return

        logging_config = cls._get_logging_config()
        logging.config.dictConfig(logging_config)

    @classmethod
    def _configure_structlog(cls) -> None:
        """Configure structlog for structured logging."""
        if not cls._settings:
            return

        log_format = cls._settings.logging.format

        # Choose processor chain based on format
        if log_format == LogFormat.JSON:
            processors = cls._get_json_processors()
        elif log_format == LogFormat.STRUCTURED:
            processors = cls._get_structured_processors()
        else:  # Simple format
            processors = cls._get_simple_processors()

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    @classmethod
    def _get_logging_config(cls) -> Dict[str, Any]:
        """Get Python logging configuration dictionary."""
        if not cls._settings:
            raise RuntimeError("Settings not initialized")

        log_level = cls._settings.logging.level.value
        log_file = cls._settings.logging.file_path
        max_bytes = cls._parse_size(cls._settings.logging.max_size)
        backup_count = cls._settings.logging.backup_count

        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "detailed": {
                    "format": "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "json": {
                    "()": "src.core.logging.formatters.JsonFormatter",
                },
                "structured": {
                    "()": "src.core.logging.formatters.StructuredFormatter",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "standard",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": str(log_file),
                    "maxBytes": max_bytes,
                    "backupCount": backup_count,
                    "encoding": "utf-8",
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": str(log_file.parent / "error.log"),
                    "maxBytes": max_bytes,
                    "backupCount": backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "src": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "MetaTrader5": {
                    "level": "INFO",
                    "handlers": ["file"],
                    "propagate": False,
                },
                "sqlalchemy": {
                    "level": "WARNING",
                    "handlers": ["file"],
                    "propagate": False,
                },
                "urllib3": {
                    "level": "WARNING",
                    "handlers": ["file"],
                    "propagate": False,
                },
            },
            "root": {
                "level": log_level,
                "handlers": ["console", "file"],
            },
        }

        # Add rich handler for development
        if cls._settings.is_development():
            config["handlers"]["rich"] = {
                "class": "src.core.logging.handlers.RichHandler",
                "level": log_level,
                "show_time": True,
                "show_level": True,
                "show_path": True,
                "markup": True,
                "rich_tracebacks": True,
            }
            config["loggers"]["src"]["handlers"] = ["rich", "file"]
            config["root"]["handlers"] = ["rich", "file"]

        return config

    @classmethod
    def _get_json_processors(cls) -> list:
        """Get processors for JSON output format."""
        return [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]

    @classmethod
    def _get_structured_processors(cls) -> list:
        """Get processors for structured output format."""
        return [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=cls._settings.is_development() if cls._settings else False),
        ]

    @classmethod
    def _get_simple_processors(cls) -> list:
        """Get processors for simple output format."""
        return [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.KeyValueRenderer(),
        ]

    @classmethod
    def _parse_size(cls, size_str: str) -> int:
        """
        Parse size string to bytes.

        Args:
            size_str: Size string (e.g., "10MB", "1GB")

        Returns:
            Size in bytes
        """
        size_str = size_str.upper().strip()

        if size_str.endswith("B"):
            size_str = size_str[:-1]

        multipliers = {
            "K": 1024,
            "M": 1024 ** 2,
            "G": 1024 ** 3,
            "T": 1024 ** 4,
        }

        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                return int(float(size_str[:-1]) * multiplier)

        return int(size_str)

    @classmethod
    def reset(cls) -> None:
        """Reset the logger factory (mainly for testing)."""
        cls._initialized = False
        cls._settings = None
        cls._console = None


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance (convenience function).

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return LoggerFactory.get_logger(name)


def setup_logging(settings: Settings) -> None:
    """
    Setup logging with the provided settings (convenience function).

    Args:
        settings: Application settings
    """
    LoggerFactory.initialize(settings)


def create_file_handler(
    file_path: Union[str, Path],
    level: Union[str, int] = logging.INFO,
    format_string: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Handler:
    """
    Create a rotating file handler.

    Args:
        file_path: Path to log file
        level: Logging level
        format_string: Log format string
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured file handler
    """
    from logging.handlers import RotatingFileHandler

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        filename=str(path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)

    if format_string:
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

    return handler


def create_console_handler(
    level: Union[str, int] = logging.INFO,
    format_string: Optional[str] = None,
    use_colors: bool = True,
) -> logging.Handler:
    """
    Create a console handler.

    Args:
        level: Logging level
        format_string: Log format string
        use_colors: Whether to use colored output

    Returns:
        Configured console handler
    """
    if use_colors and sys.stderr.isatty():
        from .handlers import RichHandler
        handler = RichHandler(
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
        )
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setLevel(level)

    if format_string and not use_colors:
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

    return handler