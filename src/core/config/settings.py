"""
Configuration settings module using Pydantic Settings.

This module defines the application configuration structure using Pydantic models
for type safety, validation, and environment variable loading.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    validator,
    root_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment enumeration."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log format enumeration."""

    STRUCTURED = "structured"
    JSON = "json"
    SIMPLE = "simple"


class GuiTheme(str, Enum):
    """GUI theme enumeration."""

    DARK = "dark"
    LIGHT = "light"


class Language(str, Enum):
    """Supported language enumeration."""

    HUNGARIAN = "hu"
    ENGLISH = "en"


class DataProvider(str, Enum):
    """Data provider enumeration."""

    MT5 = "mt5"
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"


class BackupInterval(str, Enum):
    """Backup interval enumeration."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


class Mt5Settings(BaseModel):
    """MetaTrader 5 connection settings."""

    enabled: bool = Field(default=True, description="Enable MT5 connection")
    login: Optional[str] = Field(default=None, description="MT5 login number")
    password: Optional[str] = Field(default=None, description="MT5 password")
    server: Optional[str] = Field(default=None, description="MT5 server name")
    path: Optional[Path] = Field(
        default=None,
        description="Path to MT5 terminal executable"
    )
    timeout: int = Field(
        default=60000,
        ge=1000,
        le=300000,
        description="Connection timeout in milliseconds"
    )

    @validator("path", pre=True)
    def validate_path(cls, v: Union[str, Path, None]) -> Optional[Path]:
        """Validate MT5 path."""
        if v is None:
            return None
        path = Path(v)
        if not path.exists():
            raise ValueError(f"MT5 path does not exist: {path}")
        return path

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

    url: str = Field(
        default="sqlite:///data/trading.db",
        description="Database connection URL"
    )
    echo: bool = Field(default=False, description="Echo SQL queries")
    pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Database connection pool size"
    )
    max_overflow: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Maximum overflow connections"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class LoggingSettings(BaseModel):
    """Logging configuration settings."""

    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    file_path: Path = Field(
        default=Path("logs/application.log"),
        description="Log file path"
    )
    max_size: str = Field(
        default="10MB",
        description="Maximum log file size"
    )
    backup_count: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of backup log files"
    )
    format: LogFormat = Field(
        default=LogFormat.STRUCTURED,
        description="Log format type"
    )

    @validator("file_path", pre=True)
    def validate_file_path(cls, v: Union[str, Path]) -> Path:
        """Validate and create log directory."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class TradingSettings(BaseModel):
    """Trading configuration settings."""

    enabled: bool = Field(
        default=False,
        description="Enable live trading (use with caution!)"
    )
    risk_management_enabled: bool = Field(
        default=True,
        description="Enable risk management"
    )
    max_risk_per_trade: float = Field(
        default=0.02,
        ge=0.001,
        le=0.5,
        description="Maximum risk per trade (as decimal)"
    )
    max_daily_risk: float = Field(
        default=0.10,
        ge=0.01,
        le=1.0,
        description="Maximum daily risk (as decimal)"
    )
    max_open_positions: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of open positions"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class MarketDataSettings(BaseModel):
    """Market data configuration settings."""

    provider: DataProvider = Field(
        default=DataProvider.MT5,
        description="Primary data provider"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable data caching"
    )
    cache_ttl: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Cache time-to-live in seconds"
    )
    historical_data_days: int = Field(
        default=365,
        ge=1,
        le=3650,
        description="Number of days of historical data to maintain"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class GuiSettings(BaseModel):
    """GUI configuration settings."""

    enabled: bool = Field(default=True, description="Enable GUI")
    theme: GuiTheme = Field(default=GuiTheme.DARK, description="GUI theme")
    language: Language = Field(
        default=Language.HUNGARIAN,
        description="GUI language"
    )
    update_interval: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="GUI update interval in milliseconds"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class ApiSettings(BaseModel):
    """API configuration settings."""

    enabled: bool = Field(default=False, description="Enable API server")
    host: str = Field(default="127.0.0.1", description="API host")
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API port"
    )
    secret_key: str = Field(
        default="change-me-in-production",
        min_length=16,
        description="API secret key"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class NotificationSettings(BaseModel):
    """Notification configuration settings."""

    enabled: bool = Field(default=False, description="Enable notifications")
    email_enabled: bool = Field(default=False, description="Enable email notifications")
    telegram_enabled: bool = Field(default=False, description="Enable Telegram notifications")

    # Email settings
    email_smtp_server: Optional[str] = Field(default=None, description="SMTP server")
    email_smtp_port: Optional[int] = Field(default=587, description="SMTP port")
    email_username: Optional[str] = Field(default=None, description="Email username")
    email_password: Optional[str] = Field(default=None, description="Email password")
    email_to: Optional[str] = Field(default=None, description="Recipient email")

    # Telegram settings
    telegram_bot_token: Optional[str] = Field(default=None, description="Telegram bot token")
    telegram_chat_id: Optional[str] = Field(default=None, description="Telegram chat ID")

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class PerformanceSettings(BaseModel):
    """Performance configuration settings."""

    async_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of async workers"
    )
    thread_pool_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Thread pool size"
    )
    memory_limit: str = Field(
        default="1GB",
        description="Memory limit"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class SecuritySettings(BaseModel):
    """Security configuration settings."""

    encryption_key: str = Field(
        default="change-me-in-production",
        min_length=32,
        description="Encryption key"
    )
    session_secret: str = Field(
        default="change-me-in-production",
        min_length=32,
        description="Session secret"
    )
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Rate limit requests per window"
    )
    rate_limit_window: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Rate limit window in seconds"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class BackupSettings(BaseModel):
    """Backup configuration settings."""

    enabled: bool = Field(default=True, description="Enable backups")
    interval: BackupInterval = Field(
        default=BackupInterval.DAILY,
        description="Backup interval"
    )
    retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Backup retention in days"
    )
    path: Path = Field(
        default=Path("data/backups/"),
        description="Backup directory path"
    )

    @validator("path", pre=True)
    def validate_backup_path(cls, v: Union[str, Path]) -> Path:
        """Validate and create backup directory."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class Settings(BaseSettings):
    """Main application settings."""

    # Application settings
    app_name: str = Field(
        default="MetaTrader Python Framework",
        description="Application name"
    )
    app_version: str = Field(default="2.0.0", description="Application version")
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    debug: bool = Field(default=True, description="Debug mode")

    # Component settings
    mt5: Mt5Settings = Field(default_factory=Mt5Settings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    gui: GuiSettings = Field(default_factory=GuiSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    backup: BackupSettings = Field(default_factory=BackupSettings)

    # Configuration paths
    strategy_config_path: Path = Field(
        default=Path("config/strategies.yaml"),
        description="Strategy configuration file path"
    )
    indicators_config_path: Path = Field(
        default=Path("config/indicators.yaml"),
        description="Indicators configuration file path"
    )
    patterns_config_path: Path = Field(
        default=Path("config/patterns.yaml"),
        description="Patterns configuration file path"
    )

    # Time zone
    time_zone: str = Field(
        default="Europe/Budapest",
        description="Application time zone"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        validate_assignment=True,
        extra="forbid",
    )

    @root_validator
    def validate_production_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate production-specific settings."""
        environment = values.get("environment")

        if environment == Environment.PRODUCTION:
            # Ensure debug is disabled in production
            values["debug"] = False

            # Validate security settings
            security = values.get("security", {})
            if isinstance(security, SecuritySettings):
                if security.encryption_key == "change-me-in-production":
                    raise ValueError("Encryption key must be changed in production")
                if security.session_secret == "change-me-in-production":
                    raise ValueError("Session secret must be changed in production")

            # Validate API settings
            api = values.get("api", {})
            if isinstance(api, ApiSettings) and api.enabled:
                if api.secret_key == "change-me-in-production":
                    raise ValueError("API secret key must be changed in production")

        return values

    @validator("strategy_config_path", "indicators_config_path", "patterns_config_path", pre=True)
    def validate_config_paths(cls, v: Union[str, Path]) -> Path:
        """Validate configuration file paths."""
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    def get_database_url(self) -> str:
        """Get the database URL."""
        return self.database.url

    def get_log_level(self) -> str:
        """Get the logging level."""
        return self.logging.level.value

    def get_config_dir(self) -> Path:
        """Get the configuration directory."""
        return Path("config")

    def get_data_dir(self) -> Path:
        """Get the data directory."""
        return Path("data")

    def get_logs_dir(self) -> Path:
        """Get the logs directory."""
        return self.logging.file_path.parent

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        use_enum_values = True