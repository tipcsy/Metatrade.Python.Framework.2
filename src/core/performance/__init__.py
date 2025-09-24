"""
Performance optimization components for the MetaTrader Python Framework.

This module provides comprehensive performance optimization capabilities
including monitoring, profiling, caching, and resource optimization.
"""

from .monitor import PerformanceMonitor, MetricType, PerformanceMetric, SystemMetrics
from .profiler import CodeProfiler, ProfileResult, ProfilingContext
from .cache import CacheManager, CacheStrategy, CacheEntry
from .optimizer import (
    PerformanceOptimizer as Phase2Optimizer,
    OptimizationRule,
    OptimizationResult,
)
from .circuit_breaker import CircuitBreaker, CircuitState, BreakerConfig

# Phase 3 components
from .optimizer import (
    PerformanceMonitor as Phase3Monitor,
    PerformanceOptimizer as Phase3Optimizer,
    PerformanceMetrics,
    PerformanceThresholds,
    performance_timer,
    get_performance_monitor,
    get_performance_optimizer,
)

__all__ = [
    # Monitoring
    "PerformanceMonitor",
    "MetricType",
    "PerformanceMetric",
    "SystemMetrics",

    # Profiling
    "CodeProfiler",
    "ProfileResult",
    "ProfilingContext",

    # Caching
    "CacheManager",
    "CacheStrategy",
    "CacheEntry",

    # Optimization
    "Phase2Optimizer",
    "OptimizationRule",
    "OptimizationResult",

    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "BreakerConfig",

    # Phase 3 Components
    "Phase3Monitor",
    "Phase3Optimizer",
    "PerformanceMetrics",
    "PerformanceThresholds",
    "performance_timer",
    "get_performance_monitor",
    "get_performance_optimizer",
]