"""
Custom logging formatters for MetaTrader Python Framework.

This module provides specialized formatters for different logging output formats
including JSON, structured, and enhanced console formatting.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import structlog


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging output."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
    ) -> None:
        """
        Initialize JSON formatter.

        Args:
            fmt: Format string (ignored for JSON)
            datefmt: Date format string
            style: Format style (ignored for JSON)
        """
        super().__init__(fmt, datefmt, style)
        self.datefmt = datefmt or "%Y-%m-%dT%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        log_data = self._build_log_data(record)
        return json.dumps(log_data, default=self._json_serializer, ensure_ascii=False)

    def _build_log_data(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Build log data dictionary from log record.

        Args:
            record: Log record

        Returns:
            Log data dictionary
        """
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).strftime(self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add stack information if present
        if record.stack_info:
            log_data["stack_info"] = record.stack_info

        # Add custom fields from record
        custom_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "exc_info", "exc_text", "stack_info",
                "getMessage",
            }:
                custom_fields[key] = value

        if custom_fields:
            log_data["custom"] = custom_fields

        return log_data

    def _json_serializer(self, obj: Any) -> str:
        """
        JSON serializer for non-standard types.

        Args:
            obj: Object to serialize

        Returns:
            String representation of object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)


class StructuredFormatter(logging.Formatter):
    """Structured formatter for human-readable structured output."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        use_colors: bool = True,
    ) -> None:
        """
        Initialize structured formatter.

        Args:
            fmt: Format string (ignored)
            datefmt: Date format string
            style: Format style (ignored)
            use_colors: Whether to use ANSI color codes
        """
        super().__init__(fmt, datefmt, style)
        self.datefmt = datefmt or "%H:%M:%S"
        self.use_colors = use_colors and sys.stderr.isatty()

        # Color mappings
        self.level_colors = {
            "DEBUG": "\033[36m",    # Cyan
            "INFO": "\033[32m",     # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",    # Red
            "CRITICAL": "\033[35m", # Magenta
        }
        self.reset_color = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record in structured format.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime(self.datefmt)

        # Format level with color
        level = record.levelname
        if self.use_colors:
            color = self.level_colors.get(level, "")
            level = f"{color}{level:8}{self.reset_color}"
        else:
            level = f"{level:8}"

        # Format logger name
        logger_name = record.name
        if len(logger_name) > 20:
            logger_name = "..." + logger_name[-17:]

        # Build base message
        message_parts = [
            timestamp,
            level,
            f"{logger_name:20}",
            f"{record.funcName}:{record.lineno}",
            record.getMessage(),
        ]

        base_message = " | ".join(message_parts)

        # Add exception information if present
        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            base_message += f"\n{exception_text}"

        # Add stack information if present
        if record.stack_info:
            base_message += f"\n{record.stack_info}"

        return base_message


class ColoredFormatter(logging.Formatter):
    """Colored console formatter with enhanced readability."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
    ) -> None:
        """
        Initialize colored formatter.

        Args:
            fmt: Format string
            datefmt: Date format string
            style: Format style
        """
        super().__init__(fmt, datefmt, style)

        # ANSI color codes
        self.colors = {
            "DEBUG": "\033[36m",     # Cyan
            "INFO": "\033[32m",      # Green
            "WARNING": "\033[33m",   # Yellow
            "ERROR": "\033[31m",     # Red
            "CRITICAL": "\033[35m",  # Magenta
            "RESET": "\033[0m",      # Reset
            "BOLD": "\033[1m",       # Bold
            "DIM": "\033[2m",        # Dim
        }

        self.use_colors = sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.

        Args:
            record: Log record to format

        Returns:
            Colored formatted log string
        """
        if not self.use_colors:
            return super().format(record)

        # Save original format
        original_format = self._style._fmt

        # Apply colors based on level
        level_color = self.colors.get(record.levelname, "")
        reset = self.colors["RESET"]

        # Modify format string to include colors
        if level_color:
            colored_format = original_format.replace(
                "%(levelname)s",
                f"{level_color}%(levelname)s{reset}"
            )
            self._style._fmt = colored_format

        # Format the record
        formatted = super().format(record)

        # Restore original format
        self._style._fmt = original_format

        return formatted


class TradingFormatter(logging.Formatter):
    """Specialized formatter for trading-related logs."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        include_context: bool = True,
    ) -> None:
        """
        Initialize trading formatter.

        Args:
            fmt: Format string
            datefmt: Date format string
            style: Format style
            include_context: Whether to include trading context
        """
        super().__init__(fmt, datefmt, style)
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with trading context.

        Args:
            record: Log record to format

        Returns:
            Formatted log string with trading context
        """
        # Format base message
        formatted = super().format(record)

        if not self.include_context:
            return formatted

        # Add trading context if available
        context_parts = []

        # Check for common trading attributes
        trading_attrs = {
            "symbol": "Symbol",
            "order_id": "Order",
            "position_id": "Position",
            "strategy": "Strategy",
            "signal": "Signal",
            "price": "Price",
            "volume": "Volume",
            "profit": "P/L",
        }

        for attr, label in trading_attrs.items():
            if hasattr(record, attr):
                value = getattr(record, attr)
                context_parts.append(f"{label}={value}")

        if context_parts:
            context = " [" + " | ".join(context_parts) + "]"
            formatted += context

        return formatted


class FilteredFormatter(logging.Formatter):
    """Formatter that filters sensitive information."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        sensitive_fields: Optional[list] = None,
    ) -> None:
        """
        Initialize filtered formatter.

        Args:
            fmt: Format string
            datefmt: Date format string
            style: Format style
            sensitive_fields: List of sensitive field names to filter
        """
        super().__init__(fmt, datefmt, style)
        self.sensitive_fields = sensitive_fields or [
            "password", "token", "key", "secret", "credential",
            "login", "auth", "api_key", "session",
        ]

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with sensitive information filtered.

        Args:
            record: Log record to format

        Returns:
            Formatted log string with sensitive data masked
        """
        # Create a copy of the record to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)

        # Filter sensitive information from message
        if record_copy.msg:
            record_copy.msg = self._filter_sensitive_data(str(record_copy.msg))

        # Filter sensitive information from args
        if record_copy.args:
            filtered_args = []
            for arg in record_copy.args:
                if isinstance(arg, (dict, str)):
                    filtered_args.append(self._filter_sensitive_data(str(arg)))
                else:
                    filtered_args.append(arg)
            record_copy.args = tuple(filtered_args)

        return super().format(record_copy)

    def _filter_sensitive_data(self, text: str) -> str:
        """
        Filter sensitive data from text.

        Args:
            text: Text to filter

        Returns:
            Text with sensitive data masked
        """
        import re

        for field in self.sensitive_fields:
            # Pattern to match field=value or field: value
            pattern = rf"({field}[=:\s]+)[^\s,}}\]]+?"
            text = re.sub(pattern, r"\1***", text, flags=re.IGNORECASE)

        return text


def create_formatter(
    format_type: str,
    **kwargs: Any,
) -> logging.Formatter:
    """
    Create a formatter instance based on type.

    Args:
        format_type: Type of formatter (json, structured, colored, trading, filtered)
        **kwargs: Additional arguments for formatter

    Returns:
        Formatter instance
    """
    formatters = {
        "json": JsonFormatter,
        "structured": StructuredFormatter,
        "colored": ColoredFormatter,
        "trading": TradingFormatter,
        "filtered": FilteredFormatter,
    }

    formatter_class = formatters.get(format_type.lower())
    if not formatter_class:
        raise ValueError(f"Unknown formatter type: {format_type}")

    return formatter_class(**kwargs)