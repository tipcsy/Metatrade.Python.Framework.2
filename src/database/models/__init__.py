"""
Database models package for the MetaTrader Python Framework.

This package contains all database model definitions organized by domain:
- base: Base model classes and mixins
- symbols: Symbol and instrument models
- accounts: Account and user management models
- market_data: Market data and time series models
- trading: Trading orders, positions, and trades
- strategies: Strategy and performance tracking models
- system: System configuration and audit models
"""

from __future__ import annotations

# Import base classes first
from .base import Base, BaseModel
from .mixins import (
    TimeSeriesMixin,
    FinancialMixin,
    OHLCVMixin,
    TradingMixin,
    PerformanceMixin,
    ConfigurationMixin,
)

# Import all model classes
from .symbols import Symbol, SymbolGroup, SymbolSession
from .accounts import User, Account, Transaction, AccountSettings
from .market_data import (
    MarketData,
    TickData,
    MarketDepth,
    MarketSession,
    DataQuality,
)
from .trading import Order, OrderFill, Position, Trade
from .strategies import (
    Strategy,
    StrategySession,
    BacktestResult,
    StrategyParameter,
)
from .system import (
    SystemConfiguration,
    AuditLog,
    SystemMonitoring,
    FeatureFlag,
    SystemAlert,
)

# Export all classes for external use
__all__ = [
    # Base classes
    "Base",
    "BaseModel",

    # Mixins
    "TimeSeriesMixin",
    "FinancialMixin",
    "OHLCVMixin",
    "TradingMixin",
    "PerformanceMixin",
    "ConfigurationMixin",

    # Symbol models
    "Symbol",
    "SymbolGroup",
    "SymbolSession",

    # Account models
    "User",
    "Account",
    "Transaction",
    "AccountSettings",

    # Market data models
    "MarketData",
    "TickData",
    "MarketDepth",
    "MarketSession",
    "DataQuality",

    # Trading models
    "Order",
    "OrderFill",
    "Position",
    "Trade",

    # Strategy models
    "Strategy",
    "StrategySession",
    "BacktestResult",
    "StrategyParameter",

    # System models
    "SystemConfiguration",
    "AuditLog",
    "SystemMonitoring",
    "FeatureFlag",
    "SystemAlert",
]