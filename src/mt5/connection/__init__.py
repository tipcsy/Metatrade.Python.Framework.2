"""
MT5 Connection Management Module.

This module provides enterprise-grade connection management for MetaTrader 5,
including connection pooling, session lifecycle management, and fault tolerance.
"""

from .manager import Mt5ConnectionManager, get_mt5_session_manager
from .pool import Mt5ConnectionPool
from .session import Mt5Session
from .circuit_breaker import Mt5CircuitBreaker

__all__ = [
    "Mt5ConnectionManager",
    "Mt5ConnectionPool",
    "Mt5Session",
    "Mt5CircuitBreaker",
    "get_mt5_session_manager",
]