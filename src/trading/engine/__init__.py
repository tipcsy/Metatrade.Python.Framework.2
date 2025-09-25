"""
Trading Engine Module.

This module provides the core trading engine components including:
- Multi-strategy trading system
- Trade execution engine
- Strategy management
- Risk management integration
"""

from .trading_engine import (
    TradingEngine,
    TradingEngineConfig,
    EngineState,
    EngineMetrics,
)
from .strategy_manager import (
    StrategyManager,
    StrategyConfig,
    StrategyInstance,
    StrategyStatus,
)
from .execution_engine import (
    ExecutionEngine,
    ExecutionConfig,
    ExecutionResult,
    ExecutionMetrics,
)

__all__ = [
    "TradingEngine",
    "TradingEngineConfig",
    "EngineState",
    "EngineMetrics",
    "StrategyManager",
    "StrategyConfig",
    "StrategyInstance",
    "StrategyStatus",
    "ExecutionEngine",
    "ExecutionConfig",
    "ExecutionResult",
    "ExecutionMetrics",
]