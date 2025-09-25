"""
Base classes for technical indicators.

This module provides the foundation for all technical indicators
with standardized interfaces, configuration, and result handling.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from src.core.logging import get_logger

logger = get_logger(__name__)


class IndicatorConfig(BaseModel):
    """Base configuration for indicators."""

    # Time periods
    period: int = Field(
        default=14,
        ge=1,
        le=1000,
        description="Main period for calculation"
    )

    # Data source
    price_type: str = Field(
        default="close",
        description="Price type to use (open, high, low, close, typical, weighted)"
    )

    # Calculation settings
    use_weights: bool = Field(
        default=False,
        description="Use weighted calculations"
    )
    warmup_period: int = Field(
        default=0,
        ge=0,
        description="Warmup period before valid results"
    )

    # Output settings
    precision: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Decimal precision for results"
    )

    # Custom parameters
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional indicator-specific parameters"
    )

    # Pydantic v2 configuration moved to model_config

class IndicatorResult(BaseModel):
    """Standard result structure for indicators."""

    # Identification
    indicator_name: str = Field(description="Indicator name")
    symbol: str = Field(description="Symbol")
    timeframe: str = Field(description="Timeframe")

    # Timing
    timestamp: datetime = Field(description="Calculation timestamp")
    bar_timestamp: datetime = Field(description="Source bar timestamp")

    # Results
    values: Dict[str, float] = Field(
        default_factory=dict,
        description="Calculated indicator values"
    )
    signals: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generated signals"
    )

    # Metadata
    is_valid: bool = Field(default=True, description="Result validity")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in result (0-1)"
    )
    data_quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Input data quality (0-1)"
    )

    # Context
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context information"
    )

    @field_validator("timestamp", "bar_timestamp")
    def validate_timestamps(cls, v: datetime) -> datetime:
        """Ensure timestamps are timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def get_value(self, key: str, default: float = 0.0) -> float:
        """Get specific indicator value."""
        return self.values.get(key, default)

    def get_signal(self, key: str, default: Any = None) -> Any:
        """Get specific signal value."""
        return self.signals.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "indicator_name": self.indicator_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "bar_timestamp": self.bar_timestamp.isoformat(),
            "values": self.values,
            "signals": self.signals,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "data_quality": self.data_quality,
            "context": self.context
        }

    # Pydantic v2 configuration moved to model_config

class BaseIndicator(ABC):
    """
    Base class for all technical indicators.

    Provides standardized interface and common functionality
    for indicator calculation and management.
    """

    def __init__(
        self,
        name: str,
        config: IndicatorConfig = None
    ):
        """
        Initialize base indicator.

        Args:
            name: Indicator name
            config: Indicator configuration
        """
        self.name = name
        self.config = config or IndicatorConfig()

        # State management
        self._is_initialized = False
        self._data_buffer: List[Dict[str, float]] = []
        self._results_cache: List[IndicatorResult] = []

        # Performance tracking
        self._calculations_count = 0
        self._calculation_times: List[float] = []
        self._errors_count = 0

        # Cache configuration
        self._max_buffer_size = 1000
        self._max_cache_size = 500

        logger.debug(f"Initialized indicator {self.name}")

    @property
    @abstractmethod
    def required_periods(self) -> int:
        """Get minimum periods required for calculation."""
        pass

    @abstractmethod
    def _calculate(self, data: List[Dict[str, float]]) -> IndicatorResult:
        """
        Perform indicator calculation.

        Args:
            data: Price data for calculation

        Returns:
            IndicatorResult with calculated values
        """
        pass

    def calculate(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, float]],
        bar_timestamp: datetime = None
    ) -> Optional[IndicatorResult]:
        """
        Calculate indicator with error handling and performance tracking.

        Args:
            symbol: Symbol name
            timeframe: Timeframe
            data: Price data
            bar_timestamp: Source bar timestamp

        Returns:
            IndicatorResult or None if calculation failed
        """
        start_time = time.time()

        try:
            # Validate input data
            if not self._validate_data(data):
                logger.warning(f"Invalid data for {self.name} calculation")
                return None

            # Check if we have enough data
            if len(data) < self.required_periods:
                logger.debug(
                    f"Insufficient data for {self.name}: "
                    f"need {self.required_periods}, got {len(data)}"
                )
                return None

            # Update data buffer
            self._update_buffer(data)

            # Perform calculation
            result = self._calculate(data)

            # Set result metadata
            result.indicator_name = self.name
            result.symbol = symbol
            result.timeframe = timeframe
            result.timestamp = datetime.now(timezone.utc)

            if bar_timestamp:
                result.bar_timestamp = bar_timestamp
            elif data:
                # Use timestamp from last data point
                last_data = data[-1]
                if 'timestamp' in last_data:
                    result.bar_timestamp = datetime.fromtimestamp(
                        last_data['timestamp'], timezone.utc
                    )
                else:
                    result.bar_timestamp = result.timestamp

            # Assess data quality
            result.data_quality = self._assess_data_quality(data)

            # Cache result
            self._cache_result(result)

            # Update performance metrics
            self._calculations_count += 1
            calculation_time = time.time() - start_time
            self._calculation_times.append(calculation_time)

            # Limit calculation time history
            if len(self._calculation_times) > 100:
                self._calculation_times = self._calculation_times[-100:]

            logger.debug(
                f"Calculated {self.name} for {symbol} in {calculation_time*1000:.2f}ms"
            )

            return result

        except Exception as e:
            self._errors_count += 1
            logger.error(f"Error calculating {self.name} for {symbol}: {e}")
            return None

    def add_data_point(
        self,
        data_point: Dict[str, float]
    ) -> None:
        """
        Add single data point to buffer.

        Args:
            data_point: OHLC data point
        """
        if self._validate_data_point(data_point):
            self._data_buffer.append(data_point)

            # Limit buffer size
            if len(self._data_buffer) > self._max_buffer_size:
                self._data_buffer = self._data_buffer[-self._max_buffer_size:]

    def get_last_result(self) -> Optional[IndicatorResult]:
        """Get the most recent calculation result."""
        return self._results_cache[-1] if self._results_cache else None

    def get_results_history(self, count: int = 10) -> List[IndicatorResult]:
        """
        Get recent calculation results.

        Args:
            count: Number of results to return

        Returns:
            List of recent results
        """
        return self._results_cache[-count:] if self._results_cache else []

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get indicator performance statistics."""
        avg_calc_time = (
            sum(self._calculation_times) / len(self._calculation_times)
            if self._calculation_times else 0
        )

        return {
            "indicator_name": self.name,
            "calculations_count": self._calculations_count,
            "errors_count": self._errors_count,
            "success_rate": (
                (self._calculations_count - self._errors_count) /
                max(self._calculations_count, 1) * 100
            ),
            "avg_calculation_time_ms": avg_calc_time * 1000,
            "buffer_size": len(self._data_buffer),
            "cache_size": len(self._results_cache),
            "is_initialized": self._is_initialized
        }

    def reset(self) -> None:
        """Reset indicator state."""
        self._data_buffer.clear()
        self._results_cache.clear()
        self._calculations_count = 0
        self._calculation_times.clear()
        self._errors_count = 0
        self._is_initialized = False

        logger.debug(f"Reset indicator {self.name}")

    def _validate_data(self, data: List[Dict[str, float]]) -> bool:
        """
        Validate input data for calculation.

        Args:
            data: Price data to validate

        Returns:
            bool: True if data is valid
        """
        if not data:
            return False

        # Check required fields
        required_fields = ['open', 'high', 'low', 'close']

        for data_point in data:
            if not all(field in data_point for field in required_fields):
                return False

            # Check for valid prices
            if any(data_point[field] <= 0 for field in required_fields):
                return False

            # Check high >= low
            if data_point['high'] < data_point['low']:
                return False

        return True

    def _validate_data_point(self, data_point: Dict[str, float]) -> bool:
        """
        Validate single data point.

        Args:
            data_point: Single OHLC data point

        Returns:
            bool: True if valid
        """
        required_fields = ['open', 'high', 'low', 'close']

        if not all(field in data_point for field in required_fields):
            return False

        if any(data_point[field] <= 0 for field in required_fields):
            return False

        if data_point['high'] < data_point['low']:
            return False

        return True

    def _assess_data_quality(self, data: List[Dict[str, float]]) -> float:
        """
        Assess quality of input data.

        Args:
            data: Price data to assess

        Returns:
            float: Quality score (0-1)
        """
        if not data:
            return 0.0

        quality_score = 1.0

        # Check for gaps in timestamps
        if len(data) > 1:
            timestamps = [d.get('timestamp', 0) for d in data if 'timestamp' in d]
            if len(timestamps) == len(data):
                # Calculate average interval
                intervals = [
                    timestamps[i] - timestamps[i-1]
                    for i in range(1, len(timestamps))
                ]

                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    # Penalize large gaps
                    large_gaps = sum(1 for interval in intervals if interval > avg_interval * 2)
                    quality_score -= (large_gaps / len(intervals)) * 0.2

        # Check for suspicious price movements
        for i in range(1, len(data)):
            prev_close = data[i-1]['close']
            curr_open = data[i]['open']

            # Large gaps between close and next open
            if abs(curr_open - prev_close) / prev_close > 0.1:  # 10% gap
                quality_score -= 0.05

        return max(0.0, quality_score)

    def _update_buffer(self, data: List[Dict[str, float]]) -> None:
        """Update internal data buffer."""
        # For now, just keep the latest data
        # More sophisticated buffering can be implemented by subclasses
        self._data_buffer = data[-self._max_buffer_size:]

    def _cache_result(self, result: IndicatorResult) -> None:
        """Cache calculation result."""
        self._results_cache.append(result)

        # Limit cache size
        if len(self._results_cache) > self._max_cache_size:
            self._results_cache = self._results_cache[-self._max_cache_size:]

    def _get_price_series(
        self,
        data: List[Dict[str, float]],
        price_type: str = None
    ) -> List[float]:
        """
        Extract price series from OHLC data.

        Args:
            data: OHLC data
            price_type: Type of price to extract

        Returns:
            List of prices
        """
        if price_type is None:
            price_type = self.config.price_type

        if price_type == "typical":
            return [(d['high'] + d['low'] + d['close']) / 3 for d in data]
        elif price_type == "weighted":
            return [(d['open'] + d['high'] + d['low'] + d['close']) / 4 for d in data]
        else:
            return [d.get(price_type, d['close']) for d in data]

    @property
    def is_initialized(self) -> bool:
        """Check if indicator is initialized."""
        return self._is_initialized