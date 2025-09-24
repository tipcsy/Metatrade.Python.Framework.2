"""
Code profiling system for performance analysis.

This module provides comprehensive profiling capabilities to identify
performance bottlenecks and optimize code execution.
"""

from __future__ import annotations

import cProfile
import functools
import io
import pstats
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ProfileResult(BaseModel):
    """Code profiling result."""

    profile_name: str = Field(description="Profile name")
    start_time: datetime = Field(description="Profiling start time")
    end_time: datetime = Field(description="Profiling end time")
    duration_seconds: float = Field(description="Total profiling duration")

    # Function statistics
    total_calls: int = Field(description="Total function calls")
    primitive_calls: int = Field(description="Primitive calls (non-recursive)")
    total_time: float = Field(description="Total time spent")

    # Top functions by various metrics
    top_by_cumulative: List[Dict[str, Any]] = Field(default_factory=list, description="Top functions by cumulative time")
    top_by_total_time: List[Dict[str, Any]] = Field(default_factory=list, description="Top functions by total time")
    top_by_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Top functions by call count")

    # Memory information (if available)
    peak_memory_mb: Optional[float] = Field(default=None, description="Peak memory usage in MB")

    # Raw stats for detailed analysis
    raw_stats: Optional[str] = Field(default=None, description="Raw profiling statistics")

    def get_summary(self) -> str:
        """Get human-readable summary."""
        summary = [
            f"Profile: {self.profile_name}",
            f"Duration: {self.duration_seconds:.3f} seconds",
            f"Total calls: {self.total_calls}",
            f"Total time: {self.total_time:.3f} seconds",
            ""
        ]

        if self.top_by_cumulative:
            summary.append("Top functions by cumulative time:")
            for i, func in enumerate(self.top_by_cumulative[:10], 1):
                summary.append(f"  {i}. {func['function']} - {func['cumulative']:.3f}s ({func['calls']} calls)")
            summary.append("")

        if self.peak_memory_mb:
            summary.append(f"Peak memory: {self.peak_memory_mb:.1f} MB")

        return "\n".join(summary)


class ProfilingContext:
    """Context manager for code profiling."""

    def __init__(self, profiler: 'CodeProfiler', name: str):
        """Initialize profiling context."""
        self.profiler = profiler
        self.name = name
        self._profile = None
        self._start_time = None

    def __enter__(self):
        """Start profiling."""
        self._profile = cProfile.Profile()
        self._start_time = time.time()
        self._profile.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and collect results."""
        self._profile.disable()
        end_time = time.time()
        duration = end_time - self._start_time

        # Process profiling results
        result = self.profiler._process_profile_data(
            self.name, self._profile, self._start_time, end_time, duration
        )

        # Store result
        self.profiler._add_result(result)


class CodeProfiler:
    """
    Comprehensive code profiling system.

    Provides function-level profiling, performance analysis,
    and bottleneck identification capabilities.
    """

    def __init__(self):
        """Initialize code profiler."""
        self._results: Dict[str, List[ProfileResult]] = {}
        self._lock = threading.RLock()
        self._active_profiles: Dict[str, ProfilingContext] = {}

        # Configuration
        self._max_results_per_profile = 100
        self._enable_memory_profiling = True

        # Statistics
        self._profiler_stats = {
            "total_profiles": 0,
            "active_profiles": 0,
            "total_duration": 0.0
        }

        logger.info("Code profiler initialized")

    def profile(self, name: str) -> ProfilingContext:
        """
        Create profiling context.

        Args:
            name: Profile name

        Returns:
            Profiling context manager
        """
        return ProfilingContext(self, name)

    def profile_function(self, name: str = None):
        """
        Decorator for profiling functions.

        Args:
            name: Profile name (uses function name if None)

        Example:
            @profiler.profile_function("critical_calculation")
            def calculate_indicators(data):
                # Function implementation
                pass
        """
        def decorator(func):
            profile_name = name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile(profile_name):
                    return func(*args, **kwargs)

            return wrapper
        return decorator

    def start_profile(self, name: str) -> bool:
        """
        Start named profile.

        Args:
            name: Profile name

        Returns:
            True if started successfully
        """
        with self._lock:
            if name in self._active_profiles:
                logger.warning(f"Profile '{name}' already active")
                return False

            context = ProfilingContext(self, name)
            context.__enter__()
            self._active_profiles[name] = context
            self._profiler_stats["active_profiles"] += 1

            logger.info(f"Started profile: {name}")
            return True

    def stop_profile(self, name: str) -> Optional[ProfileResult]:
        """
        Stop named profile and return results.

        Args:
            name: Profile name

        Returns:
            Profile result or None if not found
        """
        with self._lock:
            if name not in self._active_profiles:
                logger.warning(f"Profile '{name}' not active")
                return None

            context = self._active_profiles.pop(name)
            self._profiler_stats["active_profiles"] -= 1

            # Stop profiling
            context.__exit__(None, None, None)

            # Return latest result
            if name in self._results and self._results[name]:
                result = self._results[name][-1]
                logger.info(f"Stopped profile: {name} (duration: {result.duration_seconds:.3f}s)")
                return result

            return None

    def get_results(self, name: str, limit: int = 10) -> List[ProfileResult]:
        """
        Get profiling results for named profile.

        Args:
            name: Profile name
            limit: Maximum number of results

        Returns:
            List of profile results
        """
        with self._lock:
            if name not in self._results:
                return []

            return self._results[name][-limit:]

    def get_all_results(self) -> Dict[str, List[ProfileResult]]:
        """Get all profiling results."""
        with self._lock:
            return {name: results.copy() for name, results in self._results.items()}

    def get_profile_names(self) -> List[str]:
        """Get all profile names."""
        with self._lock:
            return list(self._results.keys())

    def get_active_profiles(self) -> List[str]:
        """Get active profile names."""
        with self._lock:
            return list(self._active_profiles.keys())

    def clear_results(self, name: str = None) -> None:
        """
        Clear profiling results.

        Args:
            name: Specific profile name (all if None)
        """
        with self._lock:
            if name:
                self._results.pop(name, None)
                logger.info(f"Cleared results for profile: {name}")
            else:
                self._results.clear()
                logger.info("Cleared all profiling results")

    def get_summary_report(self, name: str = None) -> str:
        """
        Get summary report for profiles.

        Args:
            name: Specific profile name (all if None)

        Returns:
            Human-readable summary report
        """
        with self._lock:
            if name:
                if name not in self._results or not self._results[name]:
                    return f"No results found for profile: {name}"

                latest_result = self._results[name][-1]
                return latest_result.get_summary()

            else:
                # Generate summary for all profiles
                summary = ["Profiling Summary", "=" * 50, ""]

                for profile_name, results in self._results.items():
                    if not results:
                        continue

                    latest = results[-1]
                    summary.extend([
                        f"Profile: {profile_name}",
                        f"  Runs: {len(results)}",
                        f"  Latest duration: {latest.duration_seconds:.3f}s",
                        f"  Total calls: {latest.total_calls}",
                        f"  Total time: {latest.total_time:.3f}s",
                        ""
                    ])

                return "\n".join(summary)

    def analyze_performance_trends(self, name: str, min_runs: int = 5) -> Dict[str, Any]:
        """
        Analyze performance trends for a profile.

        Args:
            name: Profile name
            min_runs: Minimum number of runs required for analysis

        Returns:
            Performance trend analysis
        """
        with self._lock:
            if name not in self._results or len(self._results[name]) < min_runs:
                return {"error": f"Insufficient data for trend analysis (need at least {min_runs} runs)"}

            results = self._results[name]
            durations = [r.duration_seconds for r in results]
            total_times = [r.total_time for r in results]
            call_counts = [r.total_calls for r in results]

            # Calculate trends
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)

            avg_total_time = sum(total_times) / len(total_times)
            avg_calls = sum(call_counts) / len(call_counts)

            # Recent vs historical comparison
            recent_count = max(3, len(results) // 4)
            recent_durations = durations[-recent_count:]
            historical_durations = durations[:-recent_count] if len(durations) > recent_count else durations

            recent_avg = sum(recent_durations) / len(recent_durations)
            historical_avg = sum(historical_durations) / len(historical_durations)

            trend_direction = "stable"
            if recent_avg > historical_avg * 1.1:
                trend_direction = "degrading"
            elif recent_avg < historical_avg * 0.9:
                trend_direction = "improving"

            return {
                "profile_name": name,
                "total_runs": len(results),
                "duration_stats": {
                    "average": avg_duration,
                    "minimum": min_duration,
                    "maximum": max_duration,
                    "recent_average": recent_avg,
                    "historical_average": historical_avg
                },
                "performance_trend": trend_direction,
                "average_total_time": avg_total_time,
                "average_calls": avg_calls,
                "latest_result": results[-1].get_summary()
            }

    def get_profiler_stats(self) -> Dict[str, Any]:
        """Get profiler statistics."""
        with self._lock:
            stats = self._profiler_stats.copy()
            stats["total_profile_names"] = len(self._results)
            stats["total_results"] = sum(len(results) for results in self._results.values())

            return stats

    def _process_profile_data(
        self,
        name: str,
        profile: cProfile.Profile,
        start_time: float,
        end_time: float,
        duration: float
    ) -> ProfileResult:
        """Process raw profile data into structured result."""

        # Capture profile statistics
        stats_buffer = io.StringIO()
        ps = pstats.Stats(profile, stream=stats_buffer)
        ps.sort_stats('cumulative')

        # Extract key statistics
        total_calls = ps.total_calls
        primitive_calls = ps.prim_calls
        total_time = ps.total_tt

        # Get top functions by different metrics
        top_by_cumulative = self._extract_top_functions(ps, 'cumulative', 20)

        ps.sort_stats('tottime')
        top_by_total_time = self._extract_top_functions(ps, 'tottime', 20)

        ps.sort_stats('ncalls')
        top_by_calls = self._extract_top_functions(ps, 'ncalls', 20)

        # Get raw stats string
        ps.print_stats()
        raw_stats = stats_buffer.getvalue()

        # Memory profiling (if enabled)
        peak_memory_mb = None
        if self._enable_memory_profiling:
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                peak_memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            except ImportError:
                pass

        return ProfileResult(
            profile_name=name,
            start_time=datetime.fromtimestamp(start_time, timezone.utc),
            end_time=datetime.fromtimestamp(end_time, timezone.utc),
            duration_seconds=duration,
            total_calls=total_calls,
            primitive_calls=primitive_calls,
            total_time=total_time,
            top_by_cumulative=top_by_cumulative,
            top_by_total_time=top_by_total_time,
            top_by_calls=top_by_calls,
            peak_memory_mb=peak_memory_mb,
            raw_stats=raw_stats
        )

    def _extract_top_functions(self, ps: pstats.Stats, sort_key: str, limit: int) -> List[Dict[str, Any]]:
        """Extract top functions from profile statistics."""
        functions = []

        try:
            # Get sorted function data
            for func_key, (cc, nc, tt, ct, callers) in ps.stats.items():
                filename, line_number, function_name = func_key

                functions.append({
                    'function': f"{filename}:{function_name}:{line_number}",
                    'calls': cc,
                    'primitive_calls': nc,
                    'total_time': tt,
                    'cumulative': ct,
                    'per_call': tt / cc if cc > 0 else 0,
                    'cumulative_per_call': ct / cc if cc > 0 else 0
                })

            # Sort by requested metric
            if sort_key == 'cumulative':
                functions.sort(key=lambda x: x['cumulative'], reverse=True)
            elif sort_key == 'tottime':
                functions.sort(key=lambda x: x['total_time'], reverse=True)
            elif sort_key == 'ncalls':
                functions.sort(key=lambda x: x['calls'], reverse=True)

            return functions[:limit]

        except Exception as e:
            logger.error(f"Error extracting function data: {e}")
            return []

    def _add_result(self, result: ProfileResult) -> None:
        """Add profiling result to storage."""
        with self._lock:
            if result.profile_name not in self._results:
                self._results[result.profile_name] = []

            self._results[result.profile_name].append(result)

            # Limit result history
            if len(self._results[result.profile_name]) > self._max_results_per_profile:
                self._results[result.profile_name] = self._results[result.profile_name][-self._max_results_per_profile:]

            # Update statistics
            self._profiler_stats["total_profiles"] += 1
            self._profiler_stats["total_duration"] += result.duration_seconds


# Global profiler instance
_code_profiler: Optional[CodeProfiler] = None


def get_code_profiler() -> CodeProfiler:
    """Get the global code profiler instance."""
    global _code_profiler

    if _code_profiler is None:
        _code_profiler = CodeProfiler()

    return _code_profiler


# Convenience decorators and context managers
def profile_function(name: str = None):
    """Decorator for profiling functions."""
    return get_code_profiler().profile_function(name)


@contextmanager
def profile_code(name: str):
    """Context manager for profiling code blocks."""
    profiler = get_code_profiler()
    with profiler.profile(name):
        yield