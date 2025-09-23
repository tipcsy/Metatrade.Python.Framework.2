"""
Services package for the MetaTrader Python Framework.

This package contains business logic services that provide high-level
operations for different domain areas including symbols, accounts,
trading, and system management.
"""

from __future__ import annotations

# Import base service classes
from .base import BaseService, CachedService

# Import domain services
from .symbols import SymbolGroupService, SymbolService, SymbolSessionService
from .accounts import UserService, AccountService, TransactionService, AccountSettingsService

# Export all services
__all__ = [
    # Base services
    "BaseService",
    "CachedService",

    # Symbol services
    "SymbolGroupService",
    "SymbolService",
    "SymbolSessionService",

    # Account services
    "UserService",
    "AccountService",
    "TransactionService",
    "AccountSettingsService",
]