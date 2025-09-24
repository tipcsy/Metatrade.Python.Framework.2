"""
Symbol management system for the MetaTrader Python Framework.

This module provides comprehensive symbol management with MT5 integration,
persistence, monitoring, and real-time updates.
"""

from .manager import SymbolManager, get_symbol_manager
from .models import SymbolInfo, SymbolStatus, SymbolGroup, MarketSession
from .subscriber import SymbolSubscriber, SubscriptionType
from .monitor import SymbolMonitor, SymbolHealthCheck

__all__ = [
    # Core management
    "SymbolManager",
    "get_symbol_manager",

    # Models
    "SymbolInfo",
    "SymbolStatus",
    "SymbolGroup",
    "MarketSession",

    # Subscription
    "SymbolSubscriber",
    "SubscriptionType",

    # Monitoring
    "SymbolMonitor",
    "SymbolHealthCheck",
]