"""
MACD (Moving Average Convergence Divergence) indicator implementation.

This module provides high-performance MACD calculation with signal generation,
trend analysis, and multi-timeframe support for comprehensive market analysis.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.core.logging import get_logger
from .base import BaseIndicator, IndicatorConfig, IndicatorResult

logger = get_logger(__name__)


class MACDSignal(str, Enum):
    """MACD signal types."""

    BUY = "buy"
    SELL = "sell"
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    ZERO_CROSS_UP = "zero_cross_up"
    ZERO_CROSS_DOWN = "zero_cross_down"
    SIGNAL_CROSS_UP = "signal_cross_up"
    SIGNAL_CROSS_DOWN = "signal_cross_down"
    NONE = "none"


class MACDConfig(IndicatorConfig):
    """MACD-specific configuration."""

    fast_period: int = Field(
        default=12,
        ge=1,
        le=100,
        description="Fast EMA period"
    )
    slow_period: int = Field(
        default=26,
        ge=1,
        le=200,
        description="Slow EMA period"
    )
    signal_period: int = Field(
        default=9,
        ge=1,
        le=50,
        description="Signal line EMA period"
    )

    # Signal generation thresholds
    zero_cross_threshold: float = Field(
        default=0.0001,
        ge=0.0,
        description="Minimum value for zero line cross detection"
    )
    signal_cross_threshold: float = Field(
        default=0.0001,
        ge=0.0,
        description="Minimum difference for signal line cross detection"
    )

    # Divergence detection
    enable_divergence_detection: bool = Field(
        default=True,
        description="Enable divergence detection"
    )
    divergence_lookback_periods: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Periods to look back for divergence detection"
    )

    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class MACDAnalysis(BaseModel):
    """Comprehensive MACD analysis result."""

    # Current values
    macd_line: float = Field(description="MACD line value")
    signal_line: float = Field(description="Signal line value")
    histogram: float = Field(description="MACD histogram value")

    # Trend analysis
    trend_direction: str = Field(description="Current trend direction")
    trend_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Trend strength (0-1)"
    )

    # Signals
    primary_signal: MACDSignal = Field(description="Primary trading signal")
    secondary_signals: List[MACDSignal] = Field(
        default_factory=list,
        description="Additional signals"
    )

    # Momentum analysis
    momentum_increasing: bool = Field(description="Is momentum increasing")
    momentum_rate: float = Field(description="Rate of momentum change")

    # Divergence information
    has_divergence: bool = Field(default=False, description="Divergence detected")
    divergence_type: Optional[str] = Field(default=None, description="Type of divergence")

    # Confidence metrics
    signal_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in primary signal"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "macd_line": self.macd_line,
            "signal_line": self.signal_line,
            "histogram": self.histogram,
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "primary_signal": self.primary_signal.value,
            "secondary_signals": [s.value for s in self.secondary_signals],
            "momentum_increasing": self.momentum_increasing,
            "momentum_rate": self.momentum_rate,
            "has_divergence": self.has_divergence,
            "divergence_type": self.divergence_type,
            "signal_confidence": self.signal_confidence
        }

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class MACDIndicator(BaseIndicator):
    """
    High-performance MACD indicator with advanced signal generation.

    Provides comprehensive MACD calculation with trend analysis,
    divergence detection, and multi-timeframe support.
    """

    def __init__(self, config: MACDConfig = None):
        """
        Initialize MACD indicator.

        Args:
            config: MACD configuration
        """
        super().__init__("MACD", config or MACDConfig())
        self.macd_config = self.config if isinstance(self.config, MACDConfig) else MACDConfig()

        # EMA calculation state
        self._fast_ema: Optional[float] = None
        self._slow_ema: Optional[float] = None
        self._signal_ema: Optional[float] = None

        # Historical values for analysis
        self._macd_history: List[float] = []
        self._signal_history: List[float] = []
        self._histogram_history: List[float] = []
        self._price_history: List[float] = []

        # Multipliers for EMA calculation
        self._fast_multiplier = 2.0 / (self.macd_config.fast_period + 1)
        self._slow_multiplier = 2.0 / (self.macd_config.slow_period + 1)
        self._signal_multiplier = 2.0 / (self.macd_config.signal_period + 1)

        logger.debug(
            f"Initialized MACD indicator: "
            f"fast={self.macd_config.fast_period}, "
            f"slow={self.macd_config.slow_period}, "
            f"signal={self.macd_config.signal_period}"
        )

    @property
    def required_periods(self) -> int:
        """Get minimum periods required for calculation."""
        return max(
            self.macd_config.slow_period + self.macd_config.signal_period,
            50  # Minimum for reliable signals
        )

    def _calculate(self, data: List[Dict[str, float]]) -> IndicatorResult:
        """
        Calculate MACD values and generate signals.

        Args:
            data: OHLC price data

        Returns:
            IndicatorResult with MACD analysis
        """
        prices = self._get_price_series(data, "close")

        # Calculate EMAs
        fast_ema, slow_ema = self._calculate_emas(prices)
        macd_line = fast_ema - slow_ema

        # Calculate signal line
        signal_line = self._calculate_signal_line(macd_line)
        histogram = macd_line - signal_line

        # Update history
        self._update_history(macd_line, signal_line, histogram, prices[-1])

        # Generate comprehensive analysis
        analysis = self._analyze_macd(macd_line, signal_line, histogram, prices)

        # Create result
        result = IndicatorResult(
            indicator_name=self.name,
            symbol="",  # Will be set by caller
            timeframe="",  # Will be set by caller
            timestamp=datetime.now(timezone.utc),
            bar_timestamp=datetime.now(timezone.utc)
        )

        # Set calculated values
        result.values = {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
            "fast_ema": fast_ema,
            "slow_ema": slow_ema
        }

        # Set signals and analysis
        result.signals = analysis.to_dict()

        # Set confidence based on data quality and signal strength
        result.confidence = self._calculate_confidence(analysis, len(data))

        return result

    def _calculate_emas(self, prices: List[float]) -> Tuple[float, float]:
        """
        Calculate fast and slow EMAs.

        Args:
            prices: Price series

        Returns:
            Tuple of (fast_ema, slow_ema)
        """
        if not self._fast_ema or not self._slow_ema:
            # Initialize with SMA if first calculation
            if len(prices) >= self.macd_config.slow_period:
                self._fast_ema = sum(prices[-self.macd_config.fast_period:]) / self.macd_config.fast_period
                self._slow_ema = sum(prices[-self.macd_config.slow_period:]) / self.macd_config.slow_period
            else:
                # Not enough data for proper initialization
                self._fast_ema = prices[-1]
                self._slow_ema = prices[-1]

        # Update EMAs with latest price
        current_price = prices[-1]
        self._fast_ema = (current_price * self._fast_multiplier) + (self._fast_ema * (1 - self._fast_multiplier))
        self._slow_ema = (current_price * self._slow_multiplier) + (self._slow_ema * (1 - self._slow_multiplier))

        return self._fast_ema, self._slow_ema

    def _calculate_signal_line(self, macd_value: float) -> float:
        """
        Calculate signal line EMA.

        Args:
            macd_value: Current MACD value

        Returns:
            Signal line value
        """
        if not self._signal_ema:
            self._signal_ema = macd_value
        else:
            self._signal_ema = (macd_value * self._signal_multiplier) + (self._signal_ema * (1 - self._signal_multiplier))

        return self._signal_ema

    def _update_history(
        self,
        macd: float,
        signal: float,
        histogram: float,
        price: float
    ) -> None:
        """Update historical values for analysis."""
        max_history = max(100, self.macd_config.divergence_lookback_periods * 2)

        self._macd_history.append(macd)
        self._signal_history.append(signal)
        self._histogram_history.append(histogram)
        self._price_history.append(price)

        # Limit history size
        if len(self._macd_history) > max_history:
            self._macd_history = self._macd_history[-max_history:]
            self._signal_history = self._signal_history[-max_history:]
            self._histogram_history = self._histogram_history[-max_history:]
            self._price_history = self._price_history[-max_history:]

    def _analyze_macd(
        self,
        macd: float,
        signal: float,
        histogram: float,
        prices: List[float]
    ) -> MACDAnalysis:
        """
        Perform comprehensive MACD analysis.

        Args:
            macd: Current MACD value
            signal: Current signal line value
            histogram: Current histogram value
            prices: Price series

        Returns:
            MACDAnalysis with comprehensive analysis
        """
        # Generate signals
        primary_signal, secondary_signals = self._generate_signals(macd, signal, histogram)

        # Analyze trend
        trend_direction, trend_strength = self._analyze_trend(macd, signal, histogram)

        # Analyze momentum
        momentum_increasing, momentum_rate = self._analyze_momentum()

        # Detect divergences
        has_divergence, divergence_type = self._detect_divergence(prices)

        # Calculate signal confidence
        signal_confidence = self._calculate_signal_confidence(
            primary_signal, trend_strength, len(self._macd_history)
        )

        return MACDAnalysis(
            macd_line=macd,
            signal_line=signal,
            histogram=histogram,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            primary_signal=primary_signal,
            secondary_signals=secondary_signals,
            momentum_increasing=momentum_increasing,
            momentum_rate=momentum_rate,
            has_divergence=has_divergence,
            divergence_type=divergence_type,
            signal_confidence=signal_confidence
        )

    def _generate_signals(
        self,
        macd: float,
        signal: float,
        histogram: float
    ) -> Tuple[MACDSignal, List[MACDSignal]]:
        """
        Generate trading signals from MACD values.

        Args:
            macd: Current MACD value
            signal: Current signal line value
            histogram: Current histogram value

        Returns:
            Tuple of (primary_signal, secondary_signals)
        """
        primary_signal = MACDSignal.NONE
        secondary_signals = []

        if len(self._macd_history) < 2:
            return primary_signal, secondary_signals

        prev_macd = self._macd_history[-2]
        prev_signal = self._signal_history[-2]
        prev_histogram = self._histogram_history[-2]

        # Zero line crossovers
        if prev_macd <= 0 and macd > self.macd_config.zero_cross_threshold:
            secondary_signals.append(MACDSignal.ZERO_CROSS_UP)
            if abs(macd) > self.macd_config.zero_cross_threshold * 2:
                primary_signal = MACDSignal.BUY

        elif prev_macd >= 0 and macd < -self.macd_config.zero_cross_threshold:
            secondary_signals.append(MACDSignal.ZERO_CROSS_DOWN)
            if abs(macd) > self.macd_config.zero_cross_threshold * 2:
                primary_signal = MACDSignal.SELL

        # Signal line crossovers
        if prev_macd <= prev_signal and macd > signal + self.macd_config.signal_cross_threshold:
            secondary_signals.append(MACDSignal.SIGNAL_CROSS_UP)
            if primary_signal == MACDSignal.NONE and macd > 0:
                primary_signal = MACDSignal.BUY

        elif prev_macd >= prev_signal and macd < signal - self.macd_config.signal_cross_threshold:
            secondary_signals.append(MACDSignal.SIGNAL_CROSS_DOWN)
            if primary_signal == MACDSignal.NONE and macd < 0:
                primary_signal = MACDSignal.SELL

        return primary_signal, secondary_signals

    def _analyze_trend(
        self,
        macd: float,
        signal: float,
        histogram: float
    ) -> Tuple[str, float]:
        """
        Analyze trend direction and strength.

        Args:
            macd: Current MACD value
            signal: Current signal line value
            histogram: Current histogram value

        Returns:
            Tuple of (trend_direction, trend_strength)
        """
        if len(self._macd_history) < 10:
            return "unknown", 0.5

        # Determine trend direction
        recent_macd = self._macd_history[-10:]
        macd_slope = self._calculate_slope(recent_macd)

        if macd_slope > 0.001:
            trend_direction = "bullish"
        elif macd_slope < -0.001:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"

        # Calculate trend strength
        # Based on MACD distance from zero, histogram direction, and consistency
        zero_distance = abs(macd) / (max(self._macd_history) - min(self._macd_history) + 0.0001)
        histogram_strength = abs(histogram) / (max(self._histogram_history) - min(self._histogram_history) + 0.0001)

        # Consistency factor
        recent_slopes = []
        for i in range(max(1, len(self._macd_history) - 5), len(self._macd_history)):
            if i >= 3:
                period_data = self._macd_history[i-3:i]
                recent_slopes.append(self._calculate_slope(period_data))

        consistency = 0.5
        if recent_slopes:
            # Check if slopes have same sign (consistent direction)
            positive_slopes = sum(1 for s in recent_slopes if s > 0)
            consistency = max(positive_slopes, len(recent_slopes) - positive_slopes) / len(recent_slopes)

        trend_strength = min(1.0, (zero_distance + histogram_strength + consistency) / 3)

        return trend_direction, trend_strength

    def _analyze_momentum(self) -> Tuple[bool, float]:
        """
        Analyze momentum changes.

        Returns:
            Tuple of (momentum_increasing, momentum_rate)
        """
        if len(self._histogram_history) < 5:
            return False, 0.0

        recent_histogram = self._histogram_history[-5:]
        histogram_slope = self._calculate_slope(recent_histogram)

        momentum_increasing = histogram_slope > 0
        momentum_rate = abs(histogram_slope)

        return momentum_increasing, momentum_rate

    def _detect_divergence(self, prices: List[float]) -> Tuple[bool, Optional[str]]:
        """
        Detect bullish and bearish divergences.

        Args:
            prices: Price series

        Returns:
            Tuple of (has_divergence, divergence_type)
        """
        if not self.macd_config.enable_divergence_detection:
            return False, None

        lookback = self.macd_config.divergence_lookback_periods
        if len(self._price_history) < lookback or len(self._macd_history) < lookback:
            return False, None

        try:
            recent_prices = self._price_history[-lookback:]
            recent_macd = self._macd_history[-lookback:]

            # Find local peaks and troughs
            price_peaks = self._find_peaks(recent_prices)
            price_troughs = self._find_troughs(recent_prices)
            macd_peaks = self._find_peaks(recent_macd)
            macd_troughs = self._find_troughs(recent_macd)

            # Check for bullish divergence (price makes lower low, MACD makes higher low)
            if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
                last_price_trough = price_troughs[-1]
                prev_price_trough = price_troughs[-2]
                last_macd_trough = macd_troughs[-1]
                prev_macd_trough = macd_troughs[-2]

                if (recent_prices[last_price_trough] < recent_prices[prev_price_trough] and
                    recent_macd[last_macd_trough] > recent_macd[prev_macd_trough]):
                    return True, "bullish"

            # Check for bearish divergence (price makes higher high, MACD makes lower high)
            if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
                last_price_peak = price_peaks[-1]
                prev_price_peak = price_peaks[-2]
                last_macd_peak = macd_peaks[-1]
                prev_macd_peak = macd_peaks[-2]

                if (recent_prices[last_price_peak] > recent_prices[prev_price_peak] and
                    recent_macd[last_macd_peak] < recent_macd[prev_macd_peak]):
                    return True, "bearish"

        except Exception as e:
            logger.debug(f"Error in divergence detection: {e}")

        return False, None

    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate slope of value series using linear regression."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))

        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0

        slope = (n * xy_sum - x_sum * y_sum) / denominator
        return slope

    def _find_peaks(self, values: List[float]) -> List[int]:
        """Find local peaks in value series."""
        peaks = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks.append(i)
        return peaks

    def _find_troughs(self, values: List[float]) -> List[int]:
        """Find local troughs in value series."""
        troughs = []
        for i in range(1, len(values) - 1):
            if values[i] < values[i-1] and values[i] < values[i+1]:
                troughs.append(i)
        return troughs

    def _calculate_signal_confidence(
        self,
        signal: MACDSignal,
        trend_strength: float,
        history_length: int
    ) -> float:
        """
        Calculate confidence in the generated signal.

        Args:
            signal: Generated signal
            trend_strength: Calculated trend strength
            history_length: Length of historical data

        Returns:
            Confidence score (0-1)
        """
        base_confidence = 0.5

        # Boost confidence for strong signals
        if signal in [MACDSignal.BUY, MACDSignal.SELL]:
            base_confidence += 0.3

        # Adjust based on trend strength
        base_confidence += trend_strength * 0.2

        # Adjust based on data sufficiency
        if history_length >= self.required_periods:
            data_factor = min(1.0, history_length / (self.required_periods * 2))
            base_confidence += data_factor * 0.1

        return min(1.0, base_confidence)

    def _calculate_confidence(self, analysis: MACDAnalysis, data_length: int) -> float:
        """Calculate overall result confidence."""
        confidence = 0.7  # Base confidence

        # Adjust based on trend strength
        confidence += analysis.trend_strength * 0.2

        # Adjust based on signal quality
        if analysis.primary_signal != MACDSignal.NONE:
            confidence += 0.1

        # Adjust based on data sufficiency
        if data_length >= self.required_periods * 2:
            confidence += 0.1

        return min(1.0, confidence)

    def reset(self) -> None:
        """Reset MACD indicator state."""
        super().reset()

        self._fast_ema = None
        self._slow_ema = None
        self._signal_ema = None

        self._macd_history.clear()
        self._signal_history.clear()
        self._histogram_history.clear()
        self._price_history.clear()