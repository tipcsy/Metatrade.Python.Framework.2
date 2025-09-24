"""
MT5 Circuit Breaker Implementation.

This module provides a circuit breaker pattern implementation for MT5 connections
to handle failures gracefully and prevent cascading failures in the system.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from src.core.exceptions import Mt5CircuitBreakerError, Mt5CircuitBreakerOpenError
from src.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


class Mt5CircuitBreaker:
    """
    Circuit breaker for MT5 operations with configurable thresholds and timeouts.

    Implements the circuit breaker pattern to prevent cascading failures
    when MT5 operations are failing consistently.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: float = 30.0,
        expected_exception: type = Exception,
        fallback_function: Optional[Callable] = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Circuit breaker name for logging
            failure_threshold: Number of failures before opening
            timeout_seconds: Time to wait before attempting half-open
            expected_exception: Exception type that triggers the circuit breaker
            fallback_function: Optional fallback function when circuit is open
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        self.fallback_function = fallback_function

        # State tracking
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._opened_time: Optional[float] = None

        # Statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._circuit_opened_count = 0

        # Thread safety
        self._lock = asyncio.Lock()

        logger.info(
            "Circuit breaker initialized",
            extra={
                "name": self.name,
                "failure_threshold": self.failure_threshold,
                "timeout_seconds": self.timeout_seconds,
            }
        )

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed (normal operation)."""
        return self._state == CircuitBreakerState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open (failing fast)."""
        return self._state == CircuitBreakerState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open (testing)."""
        return self._state == CircuitBreakerState.HALF_OPEN

    async def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Mt5CircuitBreakerOpenError: If circuit is open
            Exception: Original exception if function fails
        """
        async with self._lock:
            self._total_requests += 1

            # Check circuit state
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                    logger.info(
                        "Circuit breaker transitioning to half-open",
                        extra={"name": self.name}
                    )
                else:
                    # Circuit is open, fail fast
                    self._failed_requests += 1

                    if self.fallback_function:
                        logger.debug(
                            "Circuit breaker open, using fallback",
                            extra={"name": self.name}
                        )
                        return await self._call_fallback(*args, **kwargs)

                    raise Mt5CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is open",
                        failure_count=self._failure_count,
                        threshold=self.failure_threshold
                    )

        # Execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            await self._record_success()
            return result

        except self.expected_exception as e:
            # Record failure
            await self._record_failure(e)
            raise

        except Exception as e:
            # Unexpected exception, don't trigger circuit breaker
            logger.warning(
                "Unexpected exception in circuit breaker",
                extra={
                    "name": self.name,
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                }
            )
            raise

    async def _record_success(self) -> None:
        """Record successful operation."""
        async with self._lock:
            self._successful_requests += 1
            self._last_success_time = time.time()

            if self._state == CircuitBreakerState.HALF_OPEN:
                # Reset to closed state
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                logger.info(
                    "Circuit breaker closed after successful test",
                    extra={"name": self.name}
                )

    async def _record_failure(self, exception: Exception) -> None:
        """Record failed operation.

        Args:
            exception: Exception that occurred
        """
        async with self._lock:
            self._failed_requests += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            logger.debug(
                "Circuit breaker recorded failure",
                extra={
                    "name": self.name,
                    "failure_count": self._failure_count,
                    "exception": str(exception),
                }
            )

            # Check if we should open the circuit
            if (
                self._failure_count >= self.failure_threshold and
                self._state != CircuitBreakerState.OPEN
            ):
                self._state = CircuitBreakerState.OPEN
                self._opened_time = time.time()
                self._circuit_opened_count += 1

                logger.warning(
                    "Circuit breaker opened due to failures",
                    extra={
                        "name": self.name,
                        "failure_count": self._failure_count,
                        "threshold": self.failure_threshold,
                        "last_exception": str(exception),
                    }
                )

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker.

        Returns:
            True if we should attempt reset, False otherwise
        """
        if self._opened_time is None:
            return False

        return time.time() - self._opened_time >= self.timeout_seconds

    async def _call_fallback(self, *args: Any, **kwargs: Any) -> Any:
        """Call fallback function.

        Args:
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Fallback function result
        """
        if self.fallback_function is None:
            return None

        try:
            if asyncio.iscoroutinefunction(self.fallback_function):
                return await self.fallback_function(*args, **kwargs)
            else:
                return self.fallback_function(*args, **kwargs)
        except Exception as e:
            logger.error(
                "Fallback function failed",
                extra={
                    "name": self.name,
                    "error": str(e),
                },
                exc_info=True
            )
            return None

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._opened_time = None

            logger.info(
                "Circuit breaker manually reset",
                extra={"name": self.name}
            )

    def get_statistics(self) -> Dict[str, Union[int, float, str]]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary containing statistics
        """
        current_time = time.time()

        return {
            "name": self.name,
            "state": self._state.value,
            "failure_threshold": self.failure_threshold,
            "timeout_seconds": self.timeout_seconds,
            "failure_count": self._failure_count,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                self._successful_requests / self._total_requests
                if self._total_requests > 0 else 0.0
            ),
            "circuit_opened_count": self._circuit_opened_count,
            "last_failure_time": self._last_failure_time,
            "last_success_time": self._last_success_time,
            "opened_time": self._opened_time,
            "time_since_opened": (
                current_time - self._opened_time
                if self._opened_time else None
            ),
            "should_attempt_reset": (
                self._should_attempt_reset()
                if self._state == CircuitBreakerState.OPEN else False
            ),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of circuit breaker.

        Returns:
            Dictionary containing health status
        """
        stats = self.get_statistics()

        # Determine health based on state and statistics
        if self._state == CircuitBreakerState.CLOSED:
            health = "healthy"
        elif self._state == CircuitBreakerState.HALF_OPEN:
            health = "testing"
        else:  # OPEN
            health = "unhealthy"

        return {
            "health": health,
            "state": self._state.value,
            "is_operational": self._state != CircuitBreakerState.OPEN,
            "failure_count": self._failure_count,
            "success_rate": stats["success_rate"],
            "recommendations": self._get_health_recommendations(),
        }

    def _get_health_recommendations(self) -> list[str]:
        """Get health recommendations based on current state.

        Returns:
            List of recommendations
        """
        recommendations = []

        if self._state == CircuitBreakerState.OPEN:
            recommendations.append("Service is currently unavailable")
            recommendations.append("Check MT5 connection and terminal status")
            recommendations.append("Verify network connectivity")

        elif self._failure_count > self.failure_threshold * 0.5:
            recommendations.append("High failure rate detected")
            recommendations.append("Monitor service closely")

        elif self._total_requests > 0 and self._successful_requests / self._total_requests < 0.9:
            recommendations.append("Success rate below optimal threshold")
            recommendations.append("Consider checking service configuration")

        return recommendations

    def __str__(self) -> str:
        """String representation of circuit breaker."""
        return (
            f"Mt5CircuitBreaker(name='{self.name}', "
            f"state={self._state.value}, "
            f"failures={self._failure_count}/{self.failure_threshold})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of circuit breaker."""
        return (
            f"Mt5CircuitBreaker("
            f"name='{self.name}', "
            f"state={self._state.value}, "
            f"failure_count={self._failure_count}, "
            f"failure_threshold={self.failure_threshold}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"total_requests={self._total_requests}"
            f")"
        )