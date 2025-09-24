"""
Circuit breaker pattern implementation for fault tolerance.

This module provides circuit breaker functionality to prevent cascade failures
and improve system resilience under high load or error conditions.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class BreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    failure_threshold: int = Field(default=5, description="Number of failures to open circuit")
    success_threshold: int = Field(default=3, description="Number of successes to close circuit from half-open")
    timeout_seconds: int = Field(default=60, description="Timeout before trying half-open from open")
    max_failures: int = Field(default=10, description="Maximum failures to track")
    slow_call_threshold: float = Field(default=5.0, description="Slow call threshold in seconds")
    slow_call_rate_threshold: float = Field(default=0.5, description="Slow call rate threshold (0-1)")
    minimum_calls: int = Field(default=10, description="Minimum calls before evaluating circuit")

    # Advanced configuration
    exponential_backoff: bool = Field(default=False, description="Use exponential backoff for timeout")
    max_timeout_seconds: int = Field(default=300, description="Maximum timeout for exponential backoff")
    jitter: bool = Field(default=True, description="Add jitter to timeout")


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    def __init__(self, message: str, circuit_name: str):
        super().__init__(message)
        self.circuit_name = circuit_name


class CallResult(BaseModel):
    """Result of a protected call."""

    success: bool = Field(description="Whether call was successful")
    duration: float = Field(description="Call duration in seconds")
    timestamp: datetime = Field(description="Call timestamp")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    Provides automatic failure detection, fail-fast behavior,
    and automatic recovery testing to improve system resilience.
    """

    def __init__(self, name: str, config: BreakerConfig = None):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or BreakerConfig()

        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()

        # Failure tracking
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None

        # Call history for rate calculations
        self._call_history: List[CallResult] = []
        self._state_change_time = datetime.now(timezone.utc)

        # Statistics
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
            "slow_calls": 0,
            "state_changes": 0,
            "last_state_change": self._state_change_time
        }

        # Callbacks
        self._state_change_callbacks: List[Callable[[str, CircuitState, CircuitState], None]] = []

        logger.info(f"Circuit breaker '{name}' initialized in {self._state.value} state")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            self._check_and_update_state()

            if self._state == CircuitState.OPEN:
                self._stats["rejected_calls"] += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open",
                    self.name
                )

        # Execute the call
        start_time = time.time()
        call_success = False
        error_message = None

        try:
            result = func(*args, **kwargs)
            call_success = True
            return result

        except Exception as e:
            error_message = str(e)
            raise

        finally:
            duration = time.time() - start_time
            self._record_call_result(call_success, duration, error_message)

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            self._check_and_update_state()

            if self._state == CircuitState.OPEN:
                self._stats["rejected_calls"] += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open",
                    self.name
                )

        # Execute the async call
        start_time = time.time()
        call_success = False
        error_message = None

        try:
            result = await func(*args, **kwargs)
            call_success = True
            return result

        except Exception as e:
            error_message = str(e)
            raise

        finally:
            duration = time.time() - start_time
            self._record_call_result(call_success, duration, error_message)

    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                self._change_state(CircuitState.OPEN)
                logger.warning(f"Circuit breaker '{self.name}' forced open")

    def force_close(self) -> None:
        """Force circuit breaker to closed state."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                self._reset_counters()
                self._change_state(CircuitState.CLOSED)
                logger.info(f"Circuit breaker '{self.name}' forced closed")

    def force_half_open(self) -> None:
        """Force circuit breaker to half-open state."""
        with self._lock:
            if self._state != CircuitState.HALF_OPEN:
                self._change_state(CircuitState.HALF_OPEN)
                logger.info(f"Circuit breaker '{self.name}' forced half-open")

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._reset_counters()
            self._change_state(CircuitState.CLOSED)
            logger.info(f"Circuit breaker '{self.name}' reset")

    def add_state_change_callback(self, callback: Callable[[str, CircuitState, CircuitState], None]) -> None:
        """Add state change callback."""
        self._state_change_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "config": self.config.model_dump(),
                "failure_rate": self._calculate_failure_rate(),
                "slow_call_rate": self._calculate_slow_call_rate(),
                "state_duration_seconds": (datetime.now(timezone.utc) - self._state_change_time).total_seconds()
            })

            if self._last_failure_time:
                stats["last_failure"] = self._last_failure_time.isoformat()
            if self._last_success_time:
                stats["last_success"] = self._last_success_time.isoformat()

            return stats

    def _check_and_update_state(self) -> None:
        """Check and update circuit breaker state based on current conditions."""
        current_time = datetime.now(timezone.utc)

        if self._state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if self._should_open_circuit():
                self._change_state(CircuitState.OPEN)

        elif self._state == CircuitState.OPEN:
            # Check if we should try half-open
            timeout_duration = self._calculate_timeout_duration()
            if current_time >= self._state_change_time + timeout_duration:
                self._change_state(CircuitState.HALF_OPEN)

        elif self._state == CircuitState.HALF_OPEN:
            # Check if we should close or re-open
            if self._success_count >= self.config.success_threshold:
                self._reset_counters()
                self._change_state(CircuitState.CLOSED)
            elif self._failure_count >= 1:  # Any failure in half-open goes back to open
                self._change_state(CircuitState.OPEN)

    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened."""
        # Check minimum calls requirement
        if len(self._call_history) < self.config.minimum_calls:
            return False

        # Check failure threshold
        if self._failure_count >= self.config.failure_threshold:
            return True

        # Check failure rate
        failure_rate = self._calculate_failure_rate()
        if failure_rate > 0.5 and len(self._call_history) >= self.config.minimum_calls:
            return True

        # Check slow call rate
        slow_call_rate = self._calculate_slow_call_rate()
        if slow_call_rate > self.config.slow_call_rate_threshold:
            return True

        return False

    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate."""
        if not self._call_history:
            return 0.0

        recent_calls = self._get_recent_calls()
        if not recent_calls:
            return 0.0

        failed_calls = sum(1 for call in recent_calls if not call.success)
        return failed_calls / len(recent_calls)

    def _calculate_slow_call_rate(self) -> float:
        """Calculate current slow call rate."""
        if not self._call_history:
            return 0.0

        recent_calls = self._get_recent_calls()
        if not recent_calls:
            return 0.0

        slow_calls = sum(1 for call in recent_calls if call.duration > self.config.slow_call_threshold)
        return slow_calls / len(recent_calls)

    def _get_recent_calls(self, window_minutes: int = 5) -> List[CallResult]:
        """Get recent calls within time window."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        return [call for call in self._call_history if call.timestamp >= cutoff_time]

    def _calculate_timeout_duration(self) -> timedelta:
        """Calculate timeout duration for open state."""
        base_timeout = self.config.timeout_seconds

        if self.config.exponential_backoff:
            # Exponential backoff based on consecutive failures
            multiplier = min(2 ** (self._stats["state_changes"] // 2),
                           self.config.max_timeout_seconds // base_timeout)
            timeout = base_timeout * multiplier
        else:
            timeout = base_timeout

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            import random
            jitter = random.uniform(0.8, 1.2)
            timeout = int(timeout * jitter)

        return timedelta(seconds=min(timeout, self.config.max_timeout_seconds))

    def _record_call_result(self, success: bool, duration: float, error: Optional[str] = None) -> None:
        """Record the result of a protected call."""
        with self._lock:
            current_time = datetime.now(timezone.utc)

            # Create call result
            call_result = CallResult(
                success=success,
                duration=duration,
                timestamp=current_time,
                error=error
            )

            # Add to history
            self._call_history.append(call_result)

            # Limit history size
            if len(self._call_history) > self.config.max_failures * 2:
                self._call_history = self._call_history[-self.config.max_failures * 2:]

            # Update statistics
            self._stats["total_calls"] += 1

            if success:
                self._success_count += 1
                self._stats["successful_calls"] += 1
                self._last_success_time = current_time

                # Reset failure count on success in closed state
                if self._state == CircuitState.CLOSED:
                    self._failure_count = 0

            else:
                self._failure_count += 1
                self._stats["failed_calls"] += 1
                self._last_failure_time = current_time

                # Reset success count on failure
                if self._state == CircuitState.HALF_OPEN:
                    self._success_count = 0

            # Track slow calls
            if duration > self.config.slow_call_threshold:
                self._stats["slow_calls"] += 1

    def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit breaker state."""
        old_state = self._state
        self._state = new_state
        self._state_change_time = datetime.now(timezone.utc)
        self._stats["state_changes"] += 1
        self._stats["last_state_change"] = self._state_change_time

        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback failed: {e}")

        logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")

    def _reset_counters(self) -> None:
        """Reset failure and success counters."""
        self._failure_count = 0
        self._success_count = 0


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""

    def __init__(self):
        """Initialize circuit breaker manager."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

        logger.info("Circuit breaker manager initialized")

    def get_breaker(self, name: str, config: BreakerConfig = None) -> CircuitBreaker:
        """
        Get or create circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration (uses default if None)

        Returns:
            Circuit breaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)

            return self._breakers[name]

    def remove_breaker(self, name: str) -> bool:
        """Remove circuit breaker."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.info(f"Removed circuit breaker: {name}")
                return True
            return False

    def get_all_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers."""
        with self._lock:
            return self._breakers.copy()

    def get_breaker_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return [breaker.get_stats() for breaker in self._breakers.values()]

    def reset_all_breakers(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("Reset all circuit breakers")


# Global circuit breaker manager
_circuit_breaker_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager instance."""
    global _circuit_breaker_manager

    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()

    return _circuit_breaker_manager


def circuit_breaker(name: str, config: BreakerConfig = None):
    """
    Decorator for circuit breaker protection.

    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration

    Example:
        @circuit_breaker("external_api")
        def call_external_api():
            # API call code
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            breaker = get_circuit_breaker_manager().get_breaker(name, config)
            return breaker.call(func, *args, **kwargs)

        async def async_wrapper(*args, **kwargs):
            breaker = get_circuit_breaker_manager().get_breaker(name, config)
            return await breaker.call_async(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator