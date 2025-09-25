"""
Phase 5 Advanced Trading Components for MetaTrader Python Framework.

This module provides institutional-grade trading capabilities including:
- Advanced Order Management System (OMS)
- Real-time Risk Management Engine
- ML-powered Strategy Optimization
- High-performance Portfolio Management
- Real-time Data Processing Pipeline
- Comprehensive Performance Monitoring

Architecture Features:
- Microsecond-level order execution latency
- Real-time risk monitoring and controls
- ML model integration for strategy optimization
- Multi-asset portfolio management
- Event-driven architecture for scalability
- Comprehensive audit trails and compliance
"""

from .trading_engine import TradingEngine
from .order_manager import OrderManager
from .risk_manager import RiskManager
from .portfolio_optimizer import PortfolioOptimizer
from .ml_pipeline import MLPipeline
from .data_processor import DataProcessor
from .metrics_collector import MetricsCollector

__all__ = [
    'TradingEngine',
    'OrderManager',
    'RiskManager',
    'PortfolioOptimizer',
    'MLPipeline',
    'DataProcessor',
    'MetricsCollector'
]

__version__ = '5.0.0'