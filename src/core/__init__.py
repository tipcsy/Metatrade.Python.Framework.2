"""
Core module for MetaTrader Python Framework.

This module provides the foundational components including configuration,
logging, exception handling, and utilities.
"""

from .config import (
    ConfigLoader,
    Environment,
    Settings,
    load_settings,
    validate_settings,
)
from .exceptions import (
    BaseFrameworkError,
    ConfigurationError,
    ConnectionError,
    TradingError,
    ValidationError,
)
from .logging import (
    LoggerFactory,
    get_logger,
    setup_logging,
)
from .utils import (
    cache,
    retry,
    timeout,
    TypeValidator,
    RangeValidator,
    ValidationResult,
)

__version__ = "2.0.0"

__all__ = [
    # Version
    "__version__",
    # Configuration
    "Settings",
    "Environment",
    "ConfigLoader",
    "load_settings",
    "validate_settings",
    # Logging
    "LoggerFactory",
    "get_logger",
    "setup_logging",
    # Exceptions
    "BaseFrameworkError",
    "ConfigurationError",
    "ValidationError",
    "ConnectionError",
    "TradingError",
    # Utilities
    "cache",
    "retry",
    "timeout",
    "TypeValidator",
    "RangeValidator",
    "ValidationResult",
]