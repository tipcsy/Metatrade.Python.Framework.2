"""
Utility decorators for MetaTrader Python Framework.

This module provides various decorators for common functionality like
retry logic, timeout handling, caching, performance monitoring, and validation.
"""

from __future__ import annotations

import asyncio
import functools
import time
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from ..exceptions import RetryableError, TimeoutError

T = TypeVar("T", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[T], T]:
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each attempt
        exceptions: Tuple of exceptions to catch and retry on
        on_retry: Callback function called on each retry

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Don't retry on the last attempt
                    if attempt == max_attempts - 1:
                        break

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt + 1, e)

                    # Wait before retrying
                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            # Raise RetryableError with context
            raise RetryableError(
                f"Failed after {max_attempts} attempts",
                cause=last_exception,
                max_retries=max_attempts,
                retry_count=max_attempts,
            )

        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[T], T]:
    """
    Async retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each attempt
        exceptions: Tuple of exceptions to catch and retry on
        on_retry: Callback function called on each retry

    Returns:
        Decorated async function with retry logic
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Don't retry on the last attempt
                    if attempt == max_attempts - 1:
                        break

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt + 1, e)

                    # Wait before retrying
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

            # Raise RetryableError with context
            raise RetryableError(
                f"Failed after {max_attempts} attempts",
                cause=last_exception,
                max_retries=max_attempts,
                retry_count=max_attempts,
            )

        return wrapper
    return decorator


def timeout(seconds: float) -> Callable[[T], T]:
    """
    Timeout decorator for synchronous functions.

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorated function with timeout logic
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except FutureTimeoutError:
                    raise TimeoutError(
                        f"Function {func.__name__} timed out after {seconds} seconds",
                        timeout=seconds,
                        operation=func.__name__,
                    )

        return wrapper
    return decorator


def async_timeout(seconds: float) -> Callable[[T], T]:
    """
    Timeout decorator for async functions.

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorated async function with timeout logic
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {seconds} seconds",
                    timeout=seconds,
                    operation=func.__name__,
                )

        return wrapper
    return decorator


def cache(
    maxsize: int = 128,
    ttl: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable[[T], T]:
    """
    Cache decorator with optional TTL.

    Args:
        maxsize: Maximum cache size
        ttl: Time to live in seconds (None for no expiration)
        key_func: Function to generate cache keys

    Returns:
        Decorated function with caching
    """
    def decorator(func: T) -> T:
        cache_data: Dict[str, tuple] = {}
        cache_lock = threading.RLock()

        def default_key_func(*args: Any, **kwargs: Any) -> str:
            """Default cache key function."""
            return str(args) + str(sorted(kwargs.items()))

        key_generator = key_func or default_key_func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = key_generator(*args, **kwargs)
            current_time = time.time()

            with cache_lock:
                # Check if key exists and is not expired
                if key in cache_data:
                    value, timestamp = cache_data[key]
                    if ttl is None or (current_time - timestamp) < ttl:
                        return value
                    else:
                        # Remove expired entry
                        del cache_data[key]

                # Compute new value
                result = func(*args, **kwargs)

                # Store in cache
                cache_data[key] = (result, current_time)

                # Enforce maxsize
                if len(cache_data) > maxsize:
                    # Remove oldest entry
                    oldest_key = min(cache_data.keys(), key=lambda k: cache_data[k][1])
                    del cache_data[oldest_key]

                return result

        def cache_info() -> Dict[str, Any]:
            """Get cache information."""
            with cache_lock:
                return {
                    "size": len(cache_data),
                    "maxsize": maxsize,
                    "ttl": ttl,
                }

        def cache_clear() -> None:
            """Clear the cache."""
            with cache_lock:
                cache_data.clear()

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear

        return wrapper
    return decorator


def rate_limit(
    calls: int,
    period: float,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable[[T], T]:
    """
    Rate limiting decorator.

    Args:
        calls: Number of calls allowed
        period: Time period in seconds
        key_func: Function to generate rate limit keys

    Returns:
        Decorated function with rate limiting
    """
    def decorator(func: T) -> T:
        call_times: Dict[str, list] = {}
        rate_lock = threading.RLock()

        def default_key_func(*args: Any, **kwargs: Any) -> str:
            """Default rate limit key function."""
            return "default"

        key_generator = key_func or default_key_func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = key_generator(*args, **kwargs)
            current_time = time.time()

            with rate_lock:
                if key not in call_times:
                    call_times[key] = []

                # Remove old calls outside the period
                call_times[key] = [
                    t for t in call_times[key]
                    if current_time - t < period
                ]

                # Check rate limit
                if len(call_times[key]) >= calls:
                    from ..exceptions import RateLimitError
                    oldest_call = min(call_times[key])
                    retry_after = period - (current_time - oldest_call)
                    raise RateLimitError(
                        f"Rate limit exceeded: {calls} calls per {period} seconds",
                        limit=calls,
                        window=int(period),
                        retry_after=int(retry_after) + 1,
                    )

                # Record this call
                call_times[key].append(current_time)

            return func(*args, **kwargs)

        return wrapper
    return decorator


def measure_time(
    logger: Optional[Any] = None,
    level: str = "INFO",
    message: Optional[str] = None,
) -> Callable[[T], T]:
    """
    Measure and log function execution time.

    Args:
        logger: Logger to use (None for print)
        level: Log level
        message: Custom message template

    Returns:
        Decorated function with time measurement
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                log_message = message or f"Function {func.__name__} executed in {execution_time:.4f} seconds"

                if logger:
                    log_func = getattr(logger, level.lower(), logger.info)
                    log_func(log_message, execution_time=execution_time, function=func.__name__)
                else:
                    print(log_message)

        return wrapper
    return decorator


def deprecated(
    reason: str = "",
    version: Optional[str] = None,
    removal_version: Optional[str] = None,
) -> Callable[[T], T]:
    """
    Mark a function as deprecated.

    Args:
        reason: Reason for deprecation
        version: Version when deprecated
        removal_version: Version when it will be removed

    Returns:
        Decorated function with deprecation warning
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            message = f"Function {func.__name__} is deprecated"
            if version:
                message += f" since version {version}"
            if reason:
                message += f": {reason}"
            if removal_version:
                message += f". It will be removed in version {removal_version}"

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper
    return decorator


def validate_types(**type_checks: Type) -> Callable[[T], T]:
    """
    Validate function argument types.

    Args:
        **type_checks: Keyword arguments mapping parameter names to expected types

    Returns:
        Decorated function with type validation
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate types
            for param_name, expected_type in type_checks.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        from ..exceptions import ValidationError
                        raise ValidationError(
                            f"Parameter {param_name} must be of type {expected_type.__name__}, got {type(value).__name__}",
                            field=param_name,
                            value=value,
                            expected_type=expected_type,
                        )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def singleton(cls: Type) -> Type:
    """
    Singleton decorator for classes.

    Args:
        cls: Class to make singleton

    Returns:
        Singleton class
    """
    instances = {}
    lock = threading.Lock()

    def get_instance(*args: Any, **kwargs: Any) -> Any:
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def thread_safe(func: T) -> T:
    """
    Make a function thread-safe using a lock.

    Args:
        func: Function to make thread-safe

    Returns:
        Thread-safe function
    """
    lock = threading.RLock()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with lock:
            return func(*args, **kwargs)

    return wrapper


def property_cached(func: T) -> T:
    """
    Cache property value after first access.

    Args:
        func: Property function

    Returns:
        Cached property
    """
    attr_name = f"_cached_{func.__name__}"

    @functools.wraps(func)
    def wrapper(self: Any) -> Any:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return property(wrapper)