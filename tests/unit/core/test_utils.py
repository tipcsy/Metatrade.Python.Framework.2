"""
Unit tests for core utilities.

Tests decorators, validators, and helper functions.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import RetryableError, TimeoutError, ValidationError
from src.core.utils import (
    TypeValidator,
    RangeValidator,
    ValidationResult,
    retry,
    timeout,
    cache,
    calculate_percentage,
    generate_uuid,
    is_number,
    format_datetime,
    utc_now,
)


class TestValidators:
    """Test validator classes."""

    def test_type_validator_success(self):
        """Test successful type validation."""
        validator = TypeValidator(str)
        result = validator.validate("test string")

        assert result.is_valid
        assert len(result.errors) == 0

    def test_type_validator_failure(self):
        """Test failed type validation."""
        validator = TypeValidator(str)
        result = validator.validate(123)

        assert not result.is_valid
        assert len(result.errors) == 1
        assert "must be of type str" in result.errors[0]

    def test_type_validator_with_none(self):
        """Test type validator with None values."""
        # Disallow None
        validator = TypeValidator(str, allow_none=False)
        result = validator.validate(None)
        assert not result.is_valid

        # Allow None
        validator = TypeValidator(str, allow_none=True)
        result = validator.validate(None)
        assert result.is_valid

    def test_range_validator_success(self):
        """Test successful range validation."""
        validator = RangeValidator(min_value=0, max_value=100)

        # Test valid values
        assert validator.validate(50).is_valid
        assert validator.validate(0).is_valid
        assert validator.validate(100).is_valid

    def test_range_validator_failure(self):
        """Test failed range validation."""
        validator = RangeValidator(min_value=0, max_value=100)

        # Test invalid values
        assert not validator.validate(-1).is_valid
        assert not validator.validate(101).is_valid

    def test_validation_result_merge(self):
        """Test merging validation results."""
        result1 = ValidationResult(True)
        result1.add_warning("Warning 1")

        result2 = ValidationResult(False)
        result2.add_error("Error 1")

        result1.merge(result2)

        assert not result1.is_valid
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1


class TestDecorators:
    """Test decorator functions."""

    def test_retry_decorator_success(self):
        """Test retry decorator with successful execution."""
        call_count = 0

        @retry(max_attempts=3)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_decorator_failure_then_success(self):
        """Test retry decorator with initial failures."""
        call_count = 0

        @retry(max_attempts=3, delay=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Test error")
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count == 3

    def test_retry_decorator_max_attempts_exceeded(self):
        """Test retry decorator when max attempts exceeded."""
        call_count = 0

        @retry(max_attempts=2, delay=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(RetryableError):
            test_function()

        assert call_count == 2

    def test_timeout_decorator_success(self):
        """Test timeout decorator with successful execution."""
        @timeout(1.0)
        def quick_function():
            return "success"

        result = quick_function()
        assert result == "success"

    def test_timeout_decorator_timeout(self):
        """Test timeout decorator with timeout."""
        @timeout(0.5)
        def slow_function():
            time.sleep(1.0)
            return "should not reach here"

        with pytest.raises(TimeoutError):
            slow_function()

    def test_cache_decorator(self):
        """Test cache decorator functionality."""
        call_count = 0

        @cache(maxsize=2)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same argument (should use cache)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # No additional call

        # Call with different argument
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

        # Check cache info
        cache_info = expensive_function.cache_info()
        assert cache_info["size"] == 2
        assert cache_info["maxsize"] == 2


class TestHelpers:
    """Test helper functions."""

    def test_calculate_percentage(self):
        """Test percentage calculation."""
        assert calculate_percentage(25, 100) == 25.0
        assert calculate_percentage(50, 200) == 25.0
        assert calculate_percentage(10, 0) == 0.0  # Division by zero

    def test_is_number(self):
        """Test number detection."""
        assert is_number(42)
        assert is_number(3.14)
        assert is_number("123")
        assert is_number("45.67")
        assert not is_number("not a number")
        assert not is_number([1, 2, 3])

    def test_generate_uuid(self):
        """Test UUID generation."""
        uuid1 = generate_uuid()
        uuid2 = generate_uuid()

        assert isinstance(uuid1, str)
        assert isinstance(uuid2, str)
        assert uuid1 != uuid2
        assert len(uuid1) == 36  # Standard UUID length

    def test_format_datetime(self):
        """Test datetime formatting."""
        dt = datetime(2023, 12, 25, 15, 30, 45)
        formatted = format_datetime(dt, "%Y-%m-%d %H:%M:%S")
        assert formatted == "2023-12-25 15:30:45"

    def test_utc_now(self):
        """Test UTC now function."""
        now = utc_now()
        assert isinstance(now, datetime)
        assert now.tzinfo is not None


class TestAsyncDecorators:
    """Test async decorators."""

    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        """Test async retry decorator with success."""
        from src.core.utils.decorators import async_retry

        call_count = 0

        @async_retry(max_attempts=3, delay=0.1)
        async def async_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await async_function()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_failure_then_success(self):
        """Test async retry decorator with initial failures."""
        from src.core.utils.decorators import async_retry

        call_count = 0

        @async_retry(max_attempts=3, delay=0.1)
        async def async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Test error")
            return "success"

        result = await async_function()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_timeout_success(self):
        """Test async timeout decorator with success."""
        from src.core.utils.decorators import async_timeout

        @async_timeout(1.0)
        async def quick_async_function():
            await asyncio.sleep(0.1)
            return "success"

        result = await quick_async_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_timeout_failure(self):
        """Test async timeout decorator with timeout."""
        from src.core.utils.decorators import async_timeout

        @async_timeout(0.2)
        async def slow_async_function():
            await asyncio.sleep(0.5)
            return "should not reach here"

        with pytest.raises(TimeoutError):
            await slow_async_function()


class TestTradingValidators:
    """Test trading-specific validators."""

    def test_trading_symbol_validator(self):
        """Test trading symbol validation."""
        from src.core.utils.validators import TradingSymbolValidator

        validator = TradingSymbolValidator()

        # Valid symbols
        assert validator.validate("EURUSD").is_valid
        assert validator.validate("GBPJPY").is_valid
        assert validator.validate("US30").is_valid

        # Invalid symbols
        assert not validator.validate("eur/usd").is_valid  # Invalid characters
        assert not validator.validate("").is_valid         # Empty
        assert not validator.validate(123).is_valid        # Not string

    def test_price_validator(self):
        """Test price validation."""
        from src.core.utils.validators import PriceValidator

        validator = PriceValidator(min_price=0.0001, max_price=10000.0)

        # Valid prices
        assert validator.validate(1.2345).is_valid
        assert validator.validate(100.0).is_valid

        # Invalid prices
        assert not validator.validate(0.0).is_valid        # Too low
        assert not validator.validate(20000.0).is_valid    # Too high
        assert not validator.validate("not a price").is_valid  # Not numeric

    def test_volume_validator(self):
        """Test volume validation."""
        from src.core.utils.validators import VolumeValidator

        validator = VolumeValidator(min_volume=0.01, max_volume=100.0, volume_step=0.01)

        # Valid volumes
        assert validator.validate(0.01).is_valid
        assert validator.validate(1.0).is_valid
        assert validator.validate(0.50).is_valid

        # Invalid volumes
        assert not validator.validate(0.005).is_valid      # Too low
        assert not validator.validate(200.0).is_valid      # Too high
        assert not validator.validate(0.015).is_valid      # Not multiple of step


@pytest.mark.performance
class TestPerformance:
    """Test performance characteristics."""

    def test_cache_performance(self):
        """Test cache performance improvement."""
        call_count = 0

        @cache(maxsize=100)
        def fibonacci(n):
            nonlocal call_count
            call_count += 1
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)

        # Calculate fibonacci(20) - should be much faster with cache
        start_time = time.perf_counter()
        result = fibonacci(20)
        end_time = time.perf_counter()

        assert result == 6765  # Correct fibonacci(20)
        assert end_time - start_time < 0.1  # Should be very fast with cache
        # Without cache, this would require 21891 calls
        # With cache, it should be much fewer
        assert call_count < 100

    def test_retry_with_backoff_timing(self):
        """Test retry decorator backoff timing."""
        call_times = []

        @retry(max_attempts=3, delay=0.1, backoff_factor=2.0)
        def failing_function():
            call_times.append(time.perf_counter())
            raise ValueError("Always fails")

        start_time = time.perf_counter()

        with pytest.raises(RetryableError):
            failing_function()

        # Check that delays increase with backoff factor
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            assert delay1 >= 0.1  # First delay

        if len(call_times) >= 3:
            delay2 = call_times[2] - call_times[1]
            assert delay2 >= 0.2  # Second delay (backoff factor applied)