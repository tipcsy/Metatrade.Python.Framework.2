"""
GUI package for the MetaTrader Python Framework.

This package contains all GUI components built with PyQt6, including:
- Main application window and framework
- Symbol management widgets
- Market data display components
- Real-time data visualization
- Theme and localization systems
"""

from __future__ import annotations

# Import main application components
from .app import MetaTraderApp
from .main_window import MainWindow

# Import widget components
from .widgets import (
    SymbolManagementWidget,
    MarketDataTableWidget,
    StatusBarWidget,
)

# Import utility components
from .models import (
    SymbolTableModel,
    MarketDataModel,
)

from .themes import ThemeManager
from .localization import LocalizationManager

# Export all GUI components
__all__ = [
    # Main application
    "MetaTraderApp",
    "MainWindow",

    # Widgets
    "SymbolManagementWidget",
    "MarketDataTableWidget",
    "StatusBarWidget",

    # Models
    "SymbolTableModel",
    "MarketDataModel",

    # Utilities
    "ThemeManager",
    "LocalizationManager",
]