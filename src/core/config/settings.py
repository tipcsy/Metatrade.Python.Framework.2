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


class Mt5ConnectionType(str, Enum):
    """MT5 connection type enumeration."""

    LIVE = "live"
    DEMO = "demo"


class Mt5OrderType(str, Enum):
    """MT5 order type enumeration."""

    BUY = "ORDER_TYPE_BUY"
    SELL = "ORDER_TYPE_SELL"
    BUY_LIMIT = "ORDER_TYPE_BUY_LIMIT"
    SELL_LIMIT = "ORDER_TYPE_SELL_LIMIT"
    BUY_STOP = "ORDER_TYPE_BUY_STOP"
    SELL_STOP = "ORDER_TYPE_SELL_STOP"
    BUY_STOP_LIMIT = "ORDER_TYPE_BUY_STOP_LIMIT"
    SELL_STOP_LIMIT = "ORDER_TYPE_SELL_STOP_LIMIT"


class Mt5TimeFrame(str, Enum):
    """MT5 timeframe enumeration."""

    M1 = "TIMEFRAME_M1"
    M2 = "TIMEFRAME_M2"
    M3 = "TIMEFRAME_M3"
    M4 = "TIMEFRAME_M4"
    M5 = "TIMEFRAME_M5"
    M6 = "TIMEFRAME_M6"
    M10 = "TIMEFRAME_M10"
    M12 = "TIMEFRAME_M12"
    M15 = "TIMEFRAME_M15"
    M20 = "TIMEFRAME_M20"
    M30 = "TIMEFRAME_M30"
    H1 = "TIMEFRAME_H1"
    H2 = "TIMEFRAME_H2"
    H3 = "TIMEFRAME_H3"
    H4 = "TIMEFRAME_H4"
    H6 = "TIMEFRAME_H6"
    H8 = "TIMEFRAME_H8"
    H12 = "TIMEFRAME_H12"
    D1 = "TIMEFRAME_D1"
    W1 = "TIMEFRAME_W1"
    MN1 = "TIMEFRAME_MN1"


class Mt5AccountSettings(BaseModel):
    """Individual MT5 account configuration."""

    name: str = Field(description="Account display name")
    login: int = Field(description="MT5 login number")
    password: str = Field(description="MT5 password")
    server: str = Field(description="MT5 server name")
    connection_type: Mt5ConnectionType = Field(default=Mt5ConnectionType.DEMO)
    enabled: bool = Field(default=True, description="Enable this account")

    # Advanced connection settings
    timeout: int = Field(
        default=30000,
        ge=5000,
        le=120000,
        description="Connection timeout in milliseconds"
    )
    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts"
    )
    retry_delay: int = Field(
        default=1000,
        ge=500,
        le=10000,
        description="Delay between retries in milliseconds"
    )

    # Performance settings
    max_symbols: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum symbols to track"
    )
    tick_buffer_size: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Tick data buffer size"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class Mt5PerformanceSettings(BaseModel):
    """MT5 performance configuration."""

    # Connection pooling
    pool_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Connection pool size"
    )
    max_overflow: int = Field(
        default=10,
        ge=0,
        le=50,
        description="Maximum overflow connections"
    )

    # Real-time data settings
    tick_processing_threads: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of tick processing threads"
    )
    quote_processing_threads: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of quote processing threads"
    )

    # Latency targets (in milliseconds)
    max_response_latency: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum API response latency (P95)"
    )
    max_tick_latency: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Maximum tick processing latency"
    )

    # Throughput targets
    max_ticks_per_second: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum ticks per second to process"
    )
    max_orders_per_second: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum orders per second"
    )

    # Memory management
    memory_limit_mb: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Memory limit in MB"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class Mt5SecuritySettings(BaseModel):
    """MT5 security configuration."""

    # Credential encryption
    encrypt_credentials: bool = Field(
        default=True,
        description="Encrypt stored credentials"
    )
    credential_rotation_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Credential rotation period in days"
    )

    # Access control
    allowed_symbols: Optional[List[str]] = Field(
        default=None,
        description="List of allowed trading symbols"
    )
    blocked_symbols: Optional[List[str]] = Field(
        default=None,
        description="List of blocked trading symbols"
    )

    # Trading restrictions
    max_lot_size: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=1000.0,
        description="Maximum lot size per trade"
    )
    max_daily_trades: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description="Maximum trades per day"
    )

    # Audit settings
    audit_all_operations: bool = Field(
        default=True,
        description="Audit all MT5 operations"
    )
    audit_retention_days: int = Field(
        default=365,
        ge=30,
        le=2555,
        description="Audit log retention in days"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class Mt5MonitoringSettings(BaseModel):
    """MT5 monitoring and alerting configuration."""

    # Health checks
    health_check_interval: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Health check interval in seconds"
    )
    connection_timeout_threshold: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="Connection timeout threshold in milliseconds"
    )

    # Performance monitoring
    performance_metrics_enabled: bool = Field(
        default=True,
        description="Enable performance metrics collection"
    )
    metrics_collection_interval: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Metrics collection interval in seconds"
    )

    # Alerting
    alerts_enabled: bool = Field(
        default=True,
        description="Enable monitoring alerts"
    )
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "latency_p95": 100.0,
            "error_rate": 0.01,
            "memory_usage": 0.8,
            "connection_failures": 5.0
        },
        description="Alert threshold configurations"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class Mt5Settings(BaseModel):
    """Enterprise MetaTrader 5 configuration settings."""

    # Basic settings
    enabled: bool = Field(default=True, description="Enable MT5 integration")
    auto_connect: bool = Field(default=True, description="Auto-connect on startup")

    # Installation path
    path: Optional[Path] = Field(
        default=None,
        description="Path to MT5 terminal executable"
    )

    # Global timeout settings
    global_timeout: int = Field(
        default=60000,
        ge=5000,
        le=300000,
        description="Global operation timeout in milliseconds"
    )
    initialization_timeout: int = Field(
        default=30000,
        ge=5000,
        le=120000,
        description="Initialization timeout in milliseconds"
    )

    # Account configurations
    accounts: List[Mt5AccountSettings] = Field(
        default_factory=list,
        description="List of MT5 account configurations"
    )
    default_account: Optional[str] = Field(
        default=None,
        description="Default account name"
    )

    # Component settings
    performance: Mt5PerformanceSettings = Field(default_factory=Mt5PerformanceSettings)
    security: Mt5SecuritySettings = Field(default_factory=Mt5SecuritySettings)
    monitoring: Mt5MonitoringSettings = Field(default_factory=Mt5MonitoringSettings)

    # Circuit breaker settings
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker for fault tolerance"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Failure threshold for circuit breaker"
    )
    circuit_breaker_timeout: int = Field(
        default=30000,
        ge=5000,
        le=300000,
        description="Circuit breaker timeout in milliseconds"
    )

    # Data settings
    default_timeframes: List[Mt5TimeFrame] = Field(
        default_factory=lambda: [Mt5TimeFrame.M1, Mt5TimeFrame.M5, Mt5TimeFrame.H1],
        description="Default timeframes to subscribe to"
    )
    historical_data_limit: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum historical data points to fetch"
    )

    # Event settings
    event_buffer_size: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Event buffer size for real-time processing"
    )
    event_processing_threads: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of event processing threads"
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

    @validator("accounts")
    def validate_accounts(cls, v: List[Mt5AccountSettings]) -> List[Mt5AccountSettings]:
        """Validate account configurations."""
        if not v:
            return v

        # Check for duplicate account names
        names = [account.name for account in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate account names found")

        # Check for duplicate login numbers
        logins = [account.login for account in v]
        if len(logins) != len(set(logins)):
            raise ValueError("Duplicate login numbers found")

        return v

    @root_validator
    def validate_default_account(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate default account exists."""
        default_account = values.get("default_account")
        accounts = values.get("accounts", [])

        if default_account and accounts:
            account_names = [account.name for account in accounts]
            if default_account not in account_names:
                raise ValueError(f"Default account '{default_account}' not found in accounts")

        return values

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