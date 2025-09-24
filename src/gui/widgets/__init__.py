"""
GUI widgets package for the MetaTrader Python Framework.

This package contains all custom PyQt6 widgets used in the application,
including symbol management, market data display, and utility widgets.
"""

from __future__ import annotations

from .symbol_management import SymbolManagementWidget
from .market_data_table import MarketDataTableWidget
from .status_bar import StatusBarWidget

__all__ = [
    "SymbolManagementWidget",
    "MarketDataTableWidget",
    "StatusBarWidget",
]