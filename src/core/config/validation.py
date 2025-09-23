"""
Configuration validation module.

This module provides comprehensive validation for application configuration,
including security checks, dependency validation, and environment-specific rules.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .settings import Settings, Environment


class ValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigValidator:
    """Configuration validator with comprehensive checks."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the configuration validator.

        Args:
            settings: Settings instance to validate
        """
        self.settings = settings
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> Tuple[List[str], List[str]]:
        """
        Perform comprehensive validation of all configuration.

        Returns:
            Tuple of (errors, warnings)
        """
        self.errors.clear()
        self.warnings.clear()

        # Core validations
        self._validate_environment()
        self._validate_security()
        self._validate_database()
        self._validate_mt5()
        self._validate_trading()
        self._validate_paths()
        self._validate_network()
        self._validate_logging()
        self._validate_performance()
        self._validate_dependencies()

        return self.errors.copy(), self.warnings.copy()

    def validate_and_raise(self) -> None:
        """
        Validate configuration and raise ValidationError if any errors are found.

        Raises:
            ValidationError: If validation errors are found
        """
        errors, warnings = self.validate_all()

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ValidationError(error_msg)

        if warnings:
            # Log warnings but don't raise
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")

    def _validate_environment(self) -> None:
        """Validate environment-specific settings."""
        if self.settings.is_production():
            # Production-specific validations
            if self.settings.debug:
                self.errors.append("Debug mode must be disabled in production")

            if self.settings.security.encryption_key == "change-me-in-production":
                self.errors.append("Encryption key must be changed in production")

            if self.settings.security.session_secret == "change-me-in-production":
                self.errors.append("Session secret must be changed in production")

            if self.settings.api.enabled and self.settings.api.secret_key == "change-me-in-production":
                self.errors.append("API secret key must be changed in production")

            if self.settings.logging.level.value == "DEBUG":
                self.warnings.append("Debug logging level in production may impact performance")

        elif self.settings.is_development():
            # Development-specific validations
            if not self.settings.debug:
                self.warnings.append("Debug mode is typically enabled in development")

    def _validate_security(self) -> None:
        """Validate security-related settings."""
        security = self.settings.security

        # Check encryption key strength
        if len(security.encryption_key) < 32:
            self.errors.append("Encryption key must be at least 32 characters long")

        if len(security.session_secret) < 32:
            self.errors.append("Session secret must be at least 32 characters long")

        # Check for weak keys
        weak_patterns = [
            r"^password\d*$",
            r"^secret\d*$",
            r"^key\d*$",
            r"^(123|test|demo)",
        ]

        for pattern in weak_patterns:
            if re.match(pattern, security.encryption_key, re.IGNORECASE):
                self.warnings.append("Encryption key appears to be weak or predictable")
                break

        # Validate rate limiting
        if security.rate_limit_enabled:
            if security.rate_limit_requests <= 0:
                self.errors.append("Rate limit requests must be greater than 0")

            if security.rate_limit_window <= 0:
                self.errors.append("Rate limit window must be greater than 0")

    def _validate_database(self) -> None:
        """Validate database configuration."""
        database = self.settings.database

        # Validate database URL format
        if not database.url:
            self.errors.append("Database URL cannot be empty")
        elif not self._is_valid_database_url(database.url):
            self.errors.append(f"Invalid database URL format: {database.url}")

        # Check SQLite-specific settings
        if database.url.startswith("sqlite:"):
            db_path = database.url.replace("sqlite:///", "")
            if db_path != ":memory:":
                db_file = Path(db_path)
                db_file.parent.mkdir(parents=True, exist_ok=True)

        # Validate pool settings
        if database.pool_size <= 0:
            self.errors.append("Database pool size must be greater than 0")

        if database.max_overflow < 0:
            self.errors.append("Database max overflow cannot be negative")

    def _validate_mt5(self) -> None:
        """Validate MetaTrader 5 configuration."""
        mt5 = self.settings.mt5

        if mt5.enabled:
            # Check required credentials in production
            if self.settings.is_production():
                if not mt5.login:
                    self.errors.append("MT5 login is required in production")
                if not mt5.password:
                    self.errors.append("MT5 password is required in production")
                if not mt5.server:
                    self.errors.append("MT5 server is required in production")

            # Validate MT5 path if provided
            if mt5.path and not mt5.path.exists():
                self.errors.append(f"MT5 executable path does not exist: {mt5.path}")

            # Validate timeout
            if mt5.timeout < 1000:
                self.warnings.append("MT5 timeout less than 1 second may cause connection issues")
            elif mt5.timeout > 300000:  # 5 minutes
                self.warnings.append("MT5 timeout greater than 5 minutes may be excessive")

    def _validate_trading(self) -> None:
        """Validate trading configuration."""
        trading = self.settings.trading

        if trading.enabled:
            # Production trading safety checks
            if self.settings.is_production():
                if not trading.risk_management_enabled:
                    self.errors.append("Risk management must be enabled for live trading")

                if trading.max_risk_per_trade > 0.1:  # 10%
                    self.warnings.append("Risk per trade exceeds 10% - this is very high")

                if trading.max_daily_risk > 0.5:  # 50%
                    self.warnings.append("Daily risk exceeds 50% - this is extremely high")

            # Validate risk parameters
            if trading.max_risk_per_trade <= 0:
                self.errors.append("Max risk per trade must be greater than 0")

            if trading.max_risk_per_trade >= 1:
                self.errors.append("Max risk per trade cannot be 100% or more")

            if trading.max_daily_risk <= 0:
                self.errors.append("Max daily risk must be greater than 0")

            if trading.max_open_positions <= 0:
                self.errors.append("Max open positions must be greater than 0")

            # Warn about high risk settings
            if trading.max_risk_per_trade > 0.05:  # 5%
                self.warnings.append("Risk per trade exceeds 5% - consider lowering for safety")

    def _validate_paths(self) -> None:
        """Validate file and directory paths."""
        paths_to_check = [
            ("Strategy config", self.settings.strategy_config_path),
            ("Indicators config", self.settings.indicators_config_path),
            ("Patterns config", self.settings.patterns_config_path),
            ("Log file", self.settings.logging.file_path),
            ("Backup path", self.settings.backup.path),
        ]

        for name, path in paths_to_check:
            try:
                if path.suffix in {".yaml", ".yml", ".log"}:
                    # Create parent directory for files
                    path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    # Create directory
                    path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                self.errors.append(f"Cannot create {name} path {path}: {e}")

    def _validate_network(self) -> None:
        """Validate network-related settings."""
        api = self.settings.api

        if api.enabled:
            # Validate port range
            if not (1024 <= api.port <= 65535):
                self.errors.append(f"API port {api.port} is outside valid range (1024-65535)")

            # Check if port is commonly used
            common_ports = {22, 23, 25, 53, 80, 110, 143, 443, 993, 995}
            if api.port in common_ports:
                self.warnings.append(f"API port {api.port} is commonly used by other services")

            # Validate host
            if not self._is_valid_host(api.host):
                self.errors.append(f"Invalid API host: {api.host}")

    def _validate_logging(self) -> None:
        """Validate logging configuration."""
        logging = self.settings.logging

        # Check log file path permissions
        try:
            logging.file_path.parent.mkdir(parents=True, exist_ok=True)
            # Try to create a test file to check write permissions
            test_file = logging.file_path.parent / ".write_test"
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError):
            self.errors.append(f"Cannot write to log directory: {logging.file_path.parent}")

        # Validate backup count
        if logging.backup_count <= 0:
            self.warnings.append("Log backup count is 0 - logs will not be rotated")

        # Validate max size format
        if not re.match(r"^\d+[KMGT]?B$", logging.max_size.upper()):
            self.errors.append(f"Invalid log max size format: {logging.max_size}")

    def _validate_performance(self) -> None:
        """Validate performance settings."""
        performance = self.settings.performance

        # Check worker counts against CPU cores
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()

            if performance.async_workers > cpu_count * 2:
                self.warnings.append(
                    f"Async workers ({performance.async_workers}) exceed 2x CPU cores ({cpu_count})"
                )

            if performance.thread_pool_size > cpu_count * 4:
                self.warnings.append(
                    f"Thread pool size ({performance.thread_pool_size}) exceed 4x CPU cores ({cpu_count})"
                )
        except (ImportError, NotImplementedError):
            pass  # Cannot determine CPU count on this platform

        # Validate memory limit format
        if not re.match(r"^\d+[KMGT]?B$", performance.memory_limit.upper()):
            self.errors.append(f"Invalid memory limit format: {performance.memory_limit}")

    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are available."""
        required_packages = {
            "MetaTrader5": "mt5.enabled",
            "PyQt6": "gui.enabled",
            "prometheus_client": "api.enabled",
        }

        for package, condition in required_packages.items():
            if self._evaluate_condition(condition):
                try:
                    __import__(package)
                except ImportError:
                    self.warnings.append(f"Optional dependency '{package}' not found but may be required")

    def _is_valid_database_url(self, url: str) -> bool:
        """Check if database URL is valid."""
        valid_schemes = {"sqlite", "postgresql", "mysql", "mariadb", "oracle", "mssql"}
        try:
            scheme = url.split("://")[0].lower()
            return scheme in valid_schemes
        except (IndexError, AttributeError):
            return False

    def _is_valid_host(self, host: str) -> bool:
        """Check if host is valid."""
        if not host:
            return False

        # Allow localhost variations
        if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
            return True

        # Basic IPv4 check
        ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
        if re.match(ipv4_pattern, host):
            return all(0 <= int(part) <= 255 for part in host.split("."))

        # Basic hostname check
        hostname_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$"
        return bool(re.match(hostname_pattern, host))

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a simple condition string against settings."""
        try:
            parts = condition.split(".")
            value = self.settings
            for part in parts:
                value = getattr(value, part)
            return bool(value)
        except (AttributeError, TypeError):
            return False


def validate_settings(settings: Settings) -> None:
    """
    Validate settings and raise ValidationError if any errors are found.

    Args:
        settings: Settings instance to validate

    Raises:
        ValidationError: If validation errors are found
    """
    validator = ConfigValidator(settings)
    validator.validate_and_raise()


def check_configuration(settings: Settings) -> Tuple[List[str], List[str]]:
    """
    Check configuration and return errors and warnings.

    Args:
        settings: Settings instance to check

    Returns:
        Tuple of (errors, warnings)
    """
    validator = ConfigValidator(settings)
    return validator.validate_all()