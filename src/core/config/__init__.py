"""
Core configuration module for MetaTrader Python Framework.

This module provides configuration management using Pydantic Settings
with environment-specific YAML file support and validation.
"""

from .config_loader import ConfigLoader, ConfigurationError, load_settings
from .settings import (
    ApiSettings,
    BackupSettings,
    DatabaseSettings,
    Environment,
    GuiSettings,
    LoggingSettings,
    MarketDataSettings,
    Mt5Settings,
    NotificationSettings,
    PerformanceSettings,
    SecuritySettings,
    Settings,
    TradingSettings,
)
from .validation import ConfigValidator, ValidationError, check_configuration, validate_settings

__all__ = [
    # Settings classes
    "Settings",
    "Environment",
    "Mt5Settings",
    "DatabaseSettings",
    "LoggingSettings",
    "TradingSettings",
    "MarketDataSettings",
    "GuiSettings",
    "ApiSettings",
    "NotificationSettings",
    "PerformanceSettings",
    "SecuritySettings",
    "BackupSettings",
    # Configuration loading
    "ConfigLoader",
    "ConfigurationError",
    "load_settings",
    # Validation
    "ConfigValidator",
    "ValidationError",
    "validate_settings",
    "check_configuration",
]