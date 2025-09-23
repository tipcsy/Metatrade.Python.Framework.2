"""
Configuration loader module for YAML files.

This module provides functionality to load and merge configuration from YAML files
with environment-specific overrides and validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

from .settings import Settings, Environment


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


class ConfigLoader:
    """Configuration loader for YAML files with environment-specific overrides."""

    def __init__(self, config_dir: Union[str, Path] = "config") -> None:
        """
        Initialize the configuration loader.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a YAML file and return its contents.

        Args:
            file_path: Path to the YAML file

        Returns:
            Dictionary containing the YAML contents

        Raises:
            ConfigurationError: If the file cannot be loaded or parsed
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return {}

            with path.open("r", encoding="utf-8") as file:
                content = yaml.safe_load(file)
                return content if content is not None else {}

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML file {file_path}: {e}") from e
        except OSError as e:
            raise ConfigurationError(f"Error reading file {file_path}: {e}") from e

    def save_yaml(self, data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        Save data to a YAML file.

        Args:
            data: Dictionary to save
            file_path: Path to save the YAML file

        Raises:
            ConfigurationError: If the file cannot be saved
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(
                    data,
                    file,
                    default_flow_style=False,
                    indent=2,
                    sort_keys=False,
                    allow_unicode=True,
                )

        except OSError as e:
            raise ConfigurationError(f"Error writing file {file_path}: {e}") from e

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.

        Later configurations override earlier ones. Nested dictionaries are merged recursively.

        Args:
            *configs: Configuration dictionaries to merge

        Returns:
            Merged configuration dictionary
        """
        result: Dict[str, Any] = {}

        for config in configs:
            if not isinstance(config, dict):
                continue

            for key, value in config.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = self.merge_configs(result[key], value)
                else:
                    result[key] = value

        return result

    def load_environment_config(self, environment: Union[str, Environment]) -> Dict[str, Any]:
        """
        Load configuration for a specific environment.

        Args:
            environment: Environment name or Environment enum

        Returns:
            Environment-specific configuration dictionary
        """
        if isinstance(environment, Environment):
            env_name = environment.value
        else:
            env_name = str(environment).lower()

        # Load base configuration
        base_config = self.load_yaml(self.config_dir / "base.yaml")

        # Load environment-specific configuration
        env_config = self.load_yaml(self.config_dir / f"{env_name}.yaml")

        # Load local overrides (not tracked in git)
        local_config = self.load_yaml(self.config_dir / "local.yaml")

        # Merge configurations (later ones override earlier)
        return self.merge_configs(base_config, env_config, local_config)

    def load_strategy_config(self) -> Dict[str, Any]:
        """
        Load strategy configuration.

        Returns:
            Strategy configuration dictionary
        """
        return self.load_yaml(self.config_dir / "strategies.yaml")

    def load_indicators_config(self) -> Dict[str, Any]:
        """
        Load indicators configuration.

        Returns:
            Indicators configuration dictionary
        """
        return self.load_yaml(self.config_dir / "indicators.yaml")

    def load_patterns_config(self) -> Dict[str, Any]:
        """
        Load patterns configuration.

        Returns:
            Patterns configuration dictionary
        """
        return self.load_yaml(self.config_dir / "patterns.yaml")

    def load_logging_config(self) -> Dict[str, Any]:
        """
        Load logging configuration.

        Returns:
            Logging configuration dictionary
        """
        return self.load_yaml(self.config_dir / "logging.yaml")

    def save_strategy_config(self, config: Dict[str, Any]) -> None:
        """
        Save strategy configuration.

        Args:
            config: Strategy configuration to save
        """
        self.save_yaml(config, self.config_dir / "strategies.yaml")

    def save_indicators_config(self, config: Dict[str, Any]) -> None:
        """
        Save indicators configuration.

        Args:
            config: Indicators configuration to save
        """
        self.save_yaml(config, self.config_dir / "indicators.yaml")

    def save_patterns_config(self, config: Dict[str, Any]) -> None:
        """
        Save patterns configuration.

        Args:
            config: Patterns configuration to save
        """
        self.save_yaml(config, self.config_dir / "patterns.yaml")

    def create_default_configs(self) -> None:
        """Create default configuration files if they don't exist."""
        configs = {
            "base.yaml": self._get_base_config(),
            "development.yaml": self._get_development_config(),
            "testing.yaml": self._get_testing_config(),
            "production.yaml": self._get_production_config(),
            "logging.yaml": self._get_logging_config(),
            "strategies.yaml": self._get_strategies_config(),
            "indicators.yaml": self._get_indicators_config(),
            "patterns.yaml": self._get_patterns_config(),
        }

        for filename, config in configs.items():
            file_path = self.config_dir / filename
            if not file_path.exists():
                self.save_yaml(config, file_path)

    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration template."""
        return {
            "app_name": "MetaTrader Python Framework",
            "app_version": "2.0.0",
            "time_zone": "Europe/Budapest",
            "mt5": {
                "enabled": True,
                "timeout": 60000,
            },
            "database": {
                "echo": False,
                "pool_size": 5,
                "max_overflow": 10,
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/application.log",
                "max_size": "10MB",
                "backup_count": 5,
                "format": "structured",
            },
            "trading": {
                "enabled": False,
                "risk_management_enabled": True,
                "max_risk_per_trade": 0.02,
                "max_daily_risk": 0.10,
                "max_open_positions": 5,
            },
            "market_data": {
                "provider": "mt5",
                "cache_enabled": True,
                "cache_ttl": 300,
                "historical_data_days": 365,
            },
            "gui": {
                "enabled": True,
                "theme": "dark",
                "language": "hu",
                "update_interval": 1000,
            },
            "performance": {
                "async_workers": 4,
                "thread_pool_size": 8,
                "memory_limit": "1GB",
            },
            "backup": {
                "enabled": True,
                "interval": "daily",
                "retention_days": 30,
                "path": "data/backups/",
            },
        }

    def _get_development_config(self) -> Dict[str, Any]:
        """Get development configuration template."""
        return {
            "environment": "development",
            "debug": True,
            "database": {
                "url": "sqlite:///data/trading_dev.db",
                "echo": True,
            },
            "logging": {
                "level": "DEBUG",
            },
            "api": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 8000,
            },
            "security": {
                "rate_limit_enabled": False,
            },
        }

    def _get_testing_config(self) -> Dict[str, Any]:
        """Get testing configuration template."""
        return {
            "environment": "testing",
            "debug": False,
            "database": {
                "url": "sqlite:///:memory:",
                "echo": False,
            },
            "logging": {
                "level": "WARNING",
                "file_path": "logs/test.log",
            },
            "mt5": {
                "enabled": False,
            },
            "trading": {
                "enabled": False,
            },
            "notifications": {
                "enabled": False,
            },
            "backup": {
                "enabled": False,
            },
        }

    def _get_production_config(self) -> Dict[str, Any]:
        """Get production configuration template."""
        return {
            "environment": "production",
            "debug": False,
            "database": {
                "url": "sqlite:///data/trading_prod.db",
                "echo": False,
            },
            "logging": {
                "level": "INFO",
                "format": "json",
            },
            "security": {
                "rate_limit_enabled": True,
                "rate_limit_requests": 100,
                "rate_limit_window": 60,
            },
        }

    def _get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration template."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                },
                "detailed": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
                },
                "json": {
                    "()": "src.core.logging.formatters.JsonFormatter"
                },
                "structured": {
                    "()": "src.core.logging.formatters.StructuredFormatter"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/application.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": "logs/error.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
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
            },
            "root": {
                "level": "INFO",
                "handlers": ["console", "file"],
            },
        }

    def _get_strategies_config(self) -> Dict[str, Any]:
        """Get strategies configuration template."""
        return {
            "strategies": {
                "rsi_strategy": {
                    "enabled": True,
                    "parameters": {
                        "period": 14,
                        "overbought": 70,
                        "oversold": 30,
                    },
                    "symbols": ["EURUSD", "GBPUSD"],
                    "timeframe": "H1",
                },
                "macd_strategy": {
                    "enabled": False,
                    "parameters": {
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9,
                    },
                    "symbols": ["EURUSD"],
                    "timeframe": "H1",
                },
            }
        }

    def _get_indicators_config(self) -> Dict[str, Any]:
        """Get indicators configuration template."""
        return {
            "indicators": {
                "rsi": {
                    "period": 14,
                    "enabled": True,
                },
                "macd": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9,
                    "enabled": True,
                },
                "bollinger_bands": {
                    "period": 20,
                    "deviation": 2.0,
                    "enabled": True,
                },
                "stochastic": {
                    "k_period": 14,
                    "d_period": 3,
                    "slowing": 3,
                    "enabled": False,
                },
            }
        }

    def _get_patterns_config(self) -> Dict[str, Any]:
        """Get patterns configuration template."""
        return {
            "patterns": {
                "doji": {
                    "enabled": True,
                    "body_ratio_threshold": 0.1,
                },
                "hammer": {
                    "enabled": True,
                    "body_ratio_threshold": 0.3,
                    "shadow_ratio_threshold": 2.0,
                },
                "shooting_star": {
                    "enabled": True,
                    "body_ratio_threshold": 0.3,
                    "shadow_ratio_threshold": 2.0,
                },
                "engulfing": {
                    "enabled": True,
                    "min_body_ratio": 0.5,
                },
            }
        }


def load_settings(
    environment: Optional[Union[str, Environment]] = None,
    config_dir: Union[str, Path] = "config",
) -> Settings:
    """
    Load application settings with environment-specific configuration.

    Args:
        environment: Environment to load (defaults to ENVIRONMENT env var or development)
        config_dir: Configuration directory path

    Returns:
        Validated Settings instance

    Raises:
        ConfigurationError: If configuration loading or validation fails
    """
    try:
        # Determine environment
        if environment is None:
            environment = os.getenv("ENVIRONMENT", Environment.DEVELOPMENT.value)

        if isinstance(environment, str):
            environment = Environment(environment.lower())

        # Load configuration
        loader = ConfigLoader(config_dir)
        loader.create_default_configs()

        config_data = loader.load_environment_config(environment)

        # Create settings with environment override
        config_data["environment"] = environment.value

        # Load from environment variables and validate
        settings = Settings(**config_data)

        return settings

    except ValidationError as e:
        raise ConfigurationError(f"Configuration validation error: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {e}") from e