"""
Common types and data structures for the Phase 5 Trading System.

This module defines shared types, enums, and data structures used across
all trading components for consistency and type safety.
"""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


# Re-export commonly used types from component modules
from .trading_engine import (
    OrderType,
    OrderStatus,
    TimeInForce,
    AdvancedOrder,
    OrderResponse,
    ExecutionResult
)

from .risk_manager import (
    RiskLevel,
    RiskLimitType,
    RiskLimit,
    RiskMetrics,
    RiskAlert,
    RiskValidationResult
)

from .data_processor import (
    DataSource,
    DataType,
    DataQuality,
    TickDataPoint,
    OHLCDataPoint,
    MarketEvent
)

from .ml_pipeline import (
    ModelType,
    FeatureType,
    FeatureDefinition,
    FeatureVector,
    ModelConfig,
    TrainingResult,
    PredictionRequest,
    PredictionResponse
)

from .portfolio_optimizer import (
    OptimizationObjective,
    RebalancingFrequency,
    OptimizationConstraint,
    MarketView,
    PortfolioConstraints,
    OptimizationResult,
    RebalanceSignal
)

from .metrics_collector import (
    MetricType,
    AlertSeverity,
    MetricDefinition,
    MetricPoint,
    AlertRule,
    Alert,
    PerformanceSnapshot
)


# Common trading types
class Side(Enum):
    """Order/Position side."""
    BUY = "BUY"
    SELL = "SELL"
    LONG = "LONG"
    SHORT = "SHORT"


class Currency(Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    AUD = "AUD"
    CAD = "CAD"
    NZD = "NZD"


class AssetClass(Enum):
    """Asset class categories."""
    EQUITY = "equity"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    CRYPTO = "crypto"
    OPTION = "option"
    FUTURE = "future"
    ETF = "etf"


@dataclass
class Symbol:
    """Trading symbol information."""
    symbol: str
    name: str
    asset_class: AssetClass
    currency: Currency
    exchange: str

    # Trading specifications
    tick_size: Decimal
    min_size: Decimal
    max_size: Decimal

    # Market hours (simplified)
    market_open: str  # "09:30"
    market_close: str  # "16:00"
    timezone: str = "US/Eastern"

    # Additional metadata
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[Decimal] = None
    is_active: bool = True


@dataclass
class Account:
    """Trading account information."""
    account_id: str
    account_name: str
    account_type: str  # "LIVE", "DEMO", "PAPER"

    # Balance information
    balance: Decimal
    equity: Decimal
    margin_available: Decimal
    margin_used: Decimal

    # Account limits
    max_position_size: Optional[Decimal] = None
    max_daily_loss: Optional[Decimal] = None

    # Account metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


@dataclass
class Position:
    """Trading position information."""
    position_id: str
    account_id: str
    symbol: str
    side: Side

    # Position sizes
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal

    # P&L information
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal('0')

    # Position timing
    opened_at: datetime
    closed_at: Optional[datetime] = None

    # Risk management
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    # Position metadata
    strategy_id: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class Trade:
    """Completed trade information."""
    trade_id: str
    account_id: str
    symbol: str
    side: Side

    # Trade execution
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal

    # Trade timing
    open_time: datetime
    close_time: datetime

    # Trade results
    gross_pnl: Decimal
    net_pnl: Decimal
    commission: Decimal

    # Trade metadata
    strategy_id: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class MarketData:
    """Market data snapshot."""
    symbol: str
    timestamp: datetime

    # Price data
    bid: Decimal
    ask: Decimal
    last: Decimal

    # Volume data
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None
    volume: Optional[Decimal] = None

    # Derived data
    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    strategy_id: str
    strategy_name: str
    strategy_type: str

    # Strategy parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Risk parameters
    max_position_size: Optional[Decimal] = None
    max_loss_per_trade: Optional[Decimal] = None

    # Execution parameters
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)

    # Strategy metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


@dataclass
class PerformanceMetrics:
    """Strategy/Portfolio performance metrics."""
    # Return metrics
    total_return: Decimal
    annual_return: Decimal
    monthly_return: Decimal
    daily_return: Decimal

    # Risk metrics
    volatility: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal

    # Other metrics
    profit_factor: Decimal
    avg_trade_duration: Optional[timedelta] = None

    # Time period
    start_date: datetime
    end_date: datetime


# Type aliases for common combinations
PriceData = Union[TickDataPoint, OHLCDataPoint, MarketData]
OrderData = Union[AdvancedOrder, OrderResponse]
RiskData = Union[RiskMetrics, RiskAlert, RiskValidationResult]
PerformanceData = Union[PerformanceMetrics, PerformanceSnapshot]


# Constants
DEFAULT_PRECISION = 8
DEFAULT_CURRENCY = Currency.USD
DEFAULT_ASSET_CLASS = AssetClass.EQUITY

# Performance thresholds (in microseconds)
LATENCY_THRESHOLDS = {
    'order_submission': 50_000,    # 50μs
    'order_modification': 30_000,  # 30μs
    'order_cancellation': 20_000,  # 20μs
    'risk_validation': 10_000,     # 10μs
    'price_update': 5_000,         # 5μs
    'ml_prediction': 1_000_000,    # 1ms
}

# Risk limits (default values)
DEFAULT_RISK_LIMITS = {
    'max_order_size': Decimal('1000000'),      # $1M
    'max_position_size': Decimal('5000000'),   # $5M
    'max_portfolio_exposure': Decimal('50000000'),  # $50M
    'max_daily_loss': Decimal('1000000'),      # $1M
    'max_drawdown': Decimal('0.20'),           # 20%
    'max_leverage': Decimal('5'),              # 5:1
}


__all__ = [
    # Enums
    'Side', 'Currency', 'AssetClass',
    'OrderType', 'OrderStatus', 'TimeInForce',
    'RiskLevel', 'RiskLimitType',
    'DataSource', 'DataType', 'DataQuality',
    'ModelType', 'FeatureType',
    'OptimizationObjective', 'RebalancingFrequency',
    'MetricType', 'AlertSeverity',

    # Data classes
    'Symbol', 'Account', 'Position', 'Trade', 'MarketData',
    'StrategyConfig', 'PerformanceMetrics',
    'AdvancedOrder', 'OrderResponse', 'ExecutionResult',
    'RiskLimit', 'RiskMetrics', 'RiskAlert', 'RiskValidationResult',
    'TickDataPoint', 'OHLCDataPoint', 'MarketEvent',
    'FeatureDefinition', 'FeatureVector', 'ModelConfig',
    'TrainingResult', 'PredictionRequest', 'PredictionResponse',
    'OptimizationConstraint', 'MarketView', 'PortfolioConstraints',
    'OptimizationResult', 'RebalanceSignal',
    'MetricDefinition', 'MetricPoint', 'AlertRule', 'Alert',
    'PerformanceSnapshot',

    # Type aliases
    'PriceData', 'OrderData', 'RiskData', 'PerformanceData',

    # Constants
    'DEFAULT_PRECISION', 'DEFAULT_CURRENCY', 'DEFAULT_ASSET_CLASS',
    'LATENCY_THRESHOLDS', 'DEFAULT_RISK_LIMITS'
]