"""
MetaTrader Python Framework.

A comprehensive Python framework for MetaTrader 5 integration,
algorithmic trading, technical analysis, and strategy development.
"""

from .core import (
    __version__,
    Environment,
    Settings,
    load_settings,
    get_logger,
    setup_logging,
    BaseFrameworkError,
    TradingError,
    ValidationError,
)

# Framework metadata
__title__ = "MetaTrader Python Framework"
__description__ = "Advanced Python framework for MetaTrader 5 connections and algorithmic trading"
__author__ = "MetaTrader Framework Team"
__license__ = "MIT"
__url__ = "https://github.com/tipcsy/metatrader-python-framework"

__all__ = [
    # Metadata
    "__version__",
    "__title__",
    "__description__",
    "__author__",
    "__license__",
    "__url__",
    # Core exports
    "Environment",
    "Settings",
    "load_settings",
    "get_logger",
    "setup_logging",
    "BaseFrameworkError",
    "TradingError",
    "ValidationError",
]