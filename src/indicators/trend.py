"""
Advanced trend analysis system with multi-timeframe support.

This module provides comprehensive trend analysis using MACD and other
indicators across multiple timeframes for robust trend identification.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger
from .base import BaseIndicator, IndicatorConfig, IndicatorResult
from .macd import MACDIndicator, MACDConfig, MACDSignal

logger = get_logger(__name__)
settings = get_settings()


class TrendDirection(str, Enum):
    """Trend direction enumeration."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class TrendSignal(str, Enum):
    """Trend analysis signals."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TimeframeWeight(BaseModel):
    """Weight configuration for different timeframes."""

    timeframe: str = Field(description="Timeframe identifier")
    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Weight factor for this timeframe"
    )
    required: bool = Field(
        default=False,
        description="Whether this timeframe is required for analysis"
    )


class TrendConfig(IndicatorConfig):
    """Trend analyzer configuration."""

    # Timeframe weights
    timeframe_weights: List[TimeframeWeight] = Field(
        default_factory=lambda: [
            TimeframeWeight(timeframe="M1", weight=0.1),
            TimeframeWeight(timeframe="M3", weight=0.15),
            TimeframeWeight(timeframe="M5", weight=0.2, required=True),
            TimeframeWeight(timeframe="M15", weight=0.25, required=True),
            TimeframeWeight(timeframe="H1", weight=0.3, required=True),
        ],
        description="Timeframe weight configuration"
    )

    # MACD configurations for different timeframes
    macd_configs: Dict[str, MACDConfig] = Field(
        default_factory=lambda: {
            "M1": MACDConfig(fast_period=12, slow_period=26, signal_period=9),
            "M5": MACDConfig(fast_period=12, slow_period=26, signal_period=9),
            "M15": MACDConfig(fast_period=8, slow_period=21, signal_period=5),
            "H1": MACDConfig(fast_period=6, slow_period=14, signal_period=4),
        },
        description="MACD configurations by timeframe"
    )

    # Trend strength thresholds
    strong_trend_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for strong trend classification"
    )
    weak_trend_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Threshold for weak trend classification"
    )

    # Signal generation
    min_agreement_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of agreeing timeframes for signal"
    )

    # Conflict resolution
    enable_conflict_resolution: bool = Field(
        default=True,
        description="Enable automatic conflict resolution"
    )
    higher_timeframe_priority: bool = Field(
        default=True,
        description="Give higher priority to longer timeframes"
    )

    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class TimeframeTrend(BaseModel):
    """Trend analysis for a single timeframe."""

    timeframe: str = Field(description="Timeframe identifier")
    direction: TrendDirection = Field(description="Trend direction")
    strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Trend strength"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Analysis confidence"
    )
    macd_signal: MACDSignal = Field(description="MACD signal for this timeframe")
    weight: float = Field(description="Weight of this timeframe")

    # Additional context
    macd_values: Dict[str, float] = Field(
        default_factory=dict,
        description="MACD indicator values"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "macd_signal": self.macd_signal.value,
            "weight": self.weight,
            "macd_values": self.macd_values,
            "last_updated": self.last_updated.isoformat()
        }

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class TrendAnalysis(BaseModel):
    """Comprehensive multi-timeframe trend analysis."""

    symbol: str = Field(description="Symbol analyzed")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Overall trend assessment
    overall_direction: TrendDirection = Field(description="Overall trend direction")
    overall_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall trend strength"
    )
    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall analysis confidence"
    )

    # Signal generation
    trend_signal: TrendSignal = Field(description="Generated trend signal")
    signal_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Signal strength"
    )

    # Timeframe analysis
    timeframe_trends: List[TimeframeTrend] = Field(
        default_factory=list,
        description="Individual timeframe analyses"
    )

    # Agreement metrics
    bullish_agreement: float = Field(
        ge=0.0,
        le=1.0,
        description="Ratio of timeframes showing bullish trend"
    )
    bearish_agreement: float = Field(
        ge=0.0,
        le=1.0,
        description="Ratio of timeframes showing bearish trend"
    )

    # Conflict information
    has_conflicts: bool = Field(
        default=False,
        description="Whether timeframes show conflicting trends"
    )
    conflict_details: List[str] = Field(
        default_factory=list,
        description="Details of trend conflicts"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "overall_direction": self.overall_direction.value,
            "overall_strength": self.overall_strength,
            "overall_confidence": self.overall_confidence,
            "trend_signal": self.trend_signal.value,
            "signal_strength": self.signal_strength,
            "timeframe_trends": [tf.to_dict() for tf in self.timeframe_trends],
            "bullish_agreement": self.bullish_agreement,
            "bearish_agreement": self.bearish_agreement,
            "has_conflicts": self.has_conflicts,
            "conflict_details": self.conflict_details
        }

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class TrendAnalyzer(BaseIndicator):
    """
    Multi-timeframe trend analyzer using MACD and other indicators.

    Provides comprehensive trend analysis across multiple timeframes
    with conflict resolution and weighted signal generation.
    """

    def __init__(self, config: TrendConfig = None):
        """
        Initialize trend analyzer.

        Args:
            config: Trend analyzer configuration
        """
        super().__init__("TrendAnalyzer", config or TrendConfig())
        self.trend_config = self.config if isinstance(self.config, TrendConfig) else TrendConfig()

        # Initialize MACD indicators for each timeframe
        self._macd_indicators: Dict[str, MACDIndicator] = {}
        for timeframe, macd_config in self.trend_config.macd_configs.items():
            self._macd_indicators[timeframe] = MACDIndicator(macd_config)

        # Historical trend analyses
        self._trend_history: List[TrendAnalysis] = []
        self._max_history_size = 1000

        logger.info(f"Initialized trend analyzer with {len(self._macd_indicators)} timeframes")

    @property
    def required_periods(self) -> int:
        """Get minimum periods required for analysis."""
        # Return maximum required periods across all MACD indicators
        if not self._macd_indicators:
            return 100

        return max(indicator.required_periods for indicator in self._macd_indicators.values())

    def _calculate(self, data: List[Dict[str, float]]) -> IndicatorResult:
        """
        Perform multi-timeframe trend analysis.

        Args:
            data: OHLC price data (assumed to be for highest timeframe)

        Returns:
            IndicatorResult with trend analysis
        """
        # For this implementation, we'll analyze the provided data as the base timeframe
        # In a real implementation, you would have separate data for each timeframe
        symbol = ""  # Will be set by caller
        base_timeframe = "H1"  # Assume hourly data as base

        # Perform single timeframe analysis for demonstration
        trend_analysis = self._analyze_single_timeframe(symbol, base_timeframe, data)

        # Create result
        result = IndicatorResult(
            indicator_name=self.name,
            symbol=symbol,
            timeframe="MULTI",
            timestamp=datetime.now(timezone.utc),
            bar_timestamp=datetime.now(timezone.utc)
        )

        # Set values and signals
        result.values = {
            "overall_direction": trend_analysis.overall_direction.value,
            "overall_strength": trend_analysis.overall_strength,
            "overall_confidence": trend_analysis.overall_confidence,
            "bullish_agreement": trend_analysis.bullish_agreement,
            "bearish_agreement": trend_analysis.bearish_agreement
        }

        result.signals = trend_analysis.to_dict()

        # Set confidence
        result.confidence = trend_analysis.overall_confidence

        # Store in history
        self._store_analysis(trend_analysis)

        return result

    def analyze_multi_timeframe(
        self,
        symbol: str,
        timeframe_data: Dict[str, List[Dict[str, float]]]
    ) -> TrendAnalysis:
        """
        Perform comprehensive multi-timeframe trend analysis.

        Args:
            symbol: Symbol to analyze
            timeframe_data: Data for each timeframe

        Returns:
            TrendAnalysis with comprehensive results
        """
        timeframe_trends = []

        # Analyze each timeframe
        for timeframe, data in timeframe_data.items():
            if timeframe in self._macd_indicators and len(data) >= self.required_periods:
                try:
                    trend = self._analyze_single_timeframe(symbol, timeframe, data)
                    if trend.timeframe_trends:
                        timeframe_trends.append(trend.timeframe_trends[0])
                except Exception as e:
                    logger.error(f"Error analyzing {timeframe} for {symbol}: {e}")

        if not timeframe_trends:
            # Return neutral analysis if no timeframes could be analyzed
            return TrendAnalysis(
                symbol=symbol,
                overall_direction=TrendDirection.UNKNOWN,
                overall_strength=0.0,
                overall_confidence=0.0,
                trend_signal=TrendSignal.NEUTRAL,
                signal_strength=0.0
            )

        # Combine timeframe analyses
        combined_analysis = self._combine_timeframe_analyses(symbol, timeframe_trends)

        # Store in history
        self._store_analysis(combined_analysis)

        return combined_analysis

    def _analyze_single_timeframe(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, float]]
    ) -> TrendAnalysis:
        """
        Analyze trend for a single timeframe.

        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe identifier
            data: OHLC data for the timeframe

        Returns:
            TrendAnalysis for the timeframe
        """
        if timeframe not in self._macd_indicators:
            logger.warning(f"No MACD indicator configured for timeframe {timeframe}")
            return TrendAnalysis(
                symbol=symbol,
                overall_direction=TrendDirection.UNKNOWN,
                overall_strength=0.0,
                overall_confidence=0.0,
                trend_signal=TrendSignal.NEUTRAL,
                signal_strength=0.0
            )

        # Get MACD analysis
        macd_indicator = self._macd_indicators[timeframe]
        macd_result = macd_indicator.calculate(symbol, timeframe, data)

        if not macd_result or not macd_result.is_valid:
            logger.warning(f"Invalid MACD result for {symbol} {timeframe}")
            return TrendAnalysis(
                symbol=symbol,
                overall_direction=TrendDirection.UNKNOWN,
                overall_strength=0.0,
                overall_confidence=0.0,
                trend_signal=TrendSignal.NEUTRAL,
                signal_strength=0.0
            )

        # Extract MACD analysis
        macd_analysis = macd_result.signals

        # Convert MACD trend to TrendDirection
        trend_direction = self._macd_trend_to_trend_direction(
            macd_analysis.get("trend_direction", "neutral")
        )

        # Get timeframe weight
        weight = self._get_timeframe_weight(timeframe)

        # Create timeframe trend
        timeframe_trend = TimeframeTrend(
            timeframe=timeframe,
            direction=trend_direction,
            strength=macd_analysis.get("trend_strength", 0.0),
            confidence=macd_result.confidence,
            macd_signal=MACDSignal(macd_analysis.get("primary_signal", "none")),
            weight=weight,
            macd_values={
                "macd": macd_result.values.get("macd", 0.0),
                "signal": macd_result.values.get("signal", 0.0),
                "histogram": macd_result.values.get("histogram", 0.0)
            }
        )

        # Create single-timeframe analysis
        return TrendAnalysis(
            symbol=symbol,
            overall_direction=trend_direction,
            overall_strength=timeframe_trend.strength,
            overall_confidence=timeframe_trend.confidence,
            trend_signal=self._generate_trend_signal(
                trend_direction,
                timeframe_trend.strength,
                timeframe_trend.confidence
            ),
            signal_strength=timeframe_trend.strength * timeframe_trend.confidence,
            timeframe_trends=[timeframe_trend],
            bullish_agreement=1.0 if trend_direction == TrendDirection.BULLISH else 0.0,
            bearish_agreement=1.0 if trend_direction == TrendDirection.BEARISH else 0.0
        )

    def _combine_timeframe_analyses(
        self,
        symbol: str,
        timeframe_trends: List[TimeframeTrend]
    ) -> TrendAnalysis:
        """
        Combine multiple timeframe analyses into overall trend.

        Args:
            symbol: Symbol being analyzed
            timeframe_trends: Individual timeframe trend analyses

        Returns:
            Combined TrendAnalysis
        """
        if not timeframe_trends:
            return TrendAnalysis(
                symbol=symbol,
                overall_direction=TrendDirection.UNKNOWN,
                overall_strength=0.0,
                overall_confidence=0.0,
                trend_signal=TrendSignal.NEUTRAL,
                signal_strength=0.0
            )

        # Calculate weighted averages
        total_weight = sum(tf.weight for tf in timeframe_trends)
        if total_weight == 0:
            total_weight = len(timeframe_trends)
            for tf in timeframe_trends:
                tf.weight = 1.0 / len(timeframe_trends)

        # Weighted trend strength and confidence
        weighted_strength = sum(
            tf.strength * tf.weight for tf in timeframe_trends
        ) / total_weight

        weighted_confidence = sum(
            tf.confidence * tf.weight for tf in timeframe_trends
        ) / total_weight

        # Agreement analysis
        bullish_weight = sum(
            tf.weight for tf in timeframe_trends
            if tf.direction == TrendDirection.BULLISH
        ) / total_weight

        bearish_weight = sum(
            tf.weight for tf in timeframe_trends
            if tf.direction == TrendDirection.BEARISH
        ) / total_weight

        neutral_weight = 1.0 - bullish_weight - bearish_weight

        # Determine overall direction
        if bullish_weight >= self.trend_config.min_agreement_ratio:
            overall_direction = TrendDirection.BULLISH
        elif bearish_weight >= self.trend_config.min_agreement_ratio:
            overall_direction = TrendDirection.BEARISH
        elif neutral_weight >= 0.5:
            overall_direction = TrendDirection.NEUTRAL
        else:
            overall_direction = TrendDirection.UNKNOWN

        # Detect conflicts
        conflicts = []
        has_conflicts = False

        if bullish_weight > 0.3 and bearish_weight > 0.3:
            has_conflicts = True
            conflicts.append(f"Mixed signals: {bullish_weight:.1%} bullish, {bearish_weight:.1%} bearish")

        # Generate overall signal
        trend_signal = self._generate_trend_signal(
            overall_direction,
            weighted_strength,
            weighted_confidence
        )

        signal_strength = weighted_strength * weighted_confidence

        # Adjust confidence based on agreement
        agreement_factor = max(bullish_weight, bearish_weight, neutral_weight)
        adjusted_confidence = weighted_confidence * agreement_factor

        return TrendAnalysis(
            symbol=symbol,
            overall_direction=overall_direction,
            overall_strength=weighted_strength,
            overall_confidence=adjusted_confidence,
            trend_signal=trend_signal,
            signal_strength=signal_strength,
            timeframe_trends=timeframe_trends,
            bullish_agreement=bullish_weight,
            bearish_agreement=bearish_weight,
            has_conflicts=has_conflicts,
            conflict_details=conflicts
        )

    def _macd_trend_to_trend_direction(self, macd_trend: str) -> TrendDirection:
        """Convert MACD trend string to TrendDirection enum."""
        trend_map = {
            "bullish": TrendDirection.BULLISH,
            "bearish": TrendDirection.BEARISH,
            "neutral": TrendDirection.NEUTRAL
        }
        return trend_map.get(macd_trend.lower(), TrendDirection.UNKNOWN)

    def _get_timeframe_weight(self, timeframe: str) -> float:
        """Get weight for specific timeframe."""
        for weight_config in self.trend_config.timeframe_weights:
            if weight_config.timeframe == timeframe:
                return weight_config.weight
        return 0.1  # Default weight

    def _generate_trend_signal(
        self,
        direction: TrendDirection,
        strength: float,
        confidence: float
    ) -> TrendSignal:
        """
        Generate trend signal based on direction, strength, and confidence.

        Args:
            direction: Trend direction
            strength: Trend strength (0-1)
            confidence: Analysis confidence (0-1)

        Returns:
            TrendSignal
        """
        combined_score = strength * confidence

        if direction == TrendDirection.BULLISH:
            if combined_score >= self.trend_config.strong_trend_threshold:
                return TrendSignal.STRONG_BUY
            elif combined_score >= 0.5:
                return TrendSignal.BUY
            elif combined_score >= self.trend_config.weak_trend_threshold:
                return TrendSignal.WEAK_BUY
            else:
                return TrendSignal.NEUTRAL

        elif direction == TrendDirection.BEARISH:
            if combined_score >= self.trend_config.strong_trend_threshold:
                return TrendSignal.STRONG_SELL
            elif combined_score >= 0.5:
                return TrendSignal.SELL
            elif combined_score >= self.trend_config.weak_trend_threshold:
                return TrendSignal.WEAK_SELL
            else:
                return TrendSignal.NEUTRAL

        else:
            return TrendSignal.NEUTRAL

    def _store_analysis(self, analysis: TrendAnalysis) -> None:
        """Store trend analysis in history."""
        self._trend_history.append(analysis)

        # Limit history size
        if len(self._trend_history) > self._max_history_size:
            self._trend_history = self._trend_history[-self._max_history_size:]

    def get_trend_history(self, symbol: str, count: int = 10) -> List[TrendAnalysis]:
        """
        Get recent trend analyses for symbol.

        Args:
            symbol: Symbol to get history for
            count: Number of analyses to return

        Returns:
            List of recent trend analyses
        """
        symbol_history = [
            analysis for analysis in self._trend_history
            if analysis.symbol == symbol
        ]

        return symbol_history[-count:] if symbol_history else []

    def get_current_trend(self, symbol: str) -> Optional[TrendAnalysis]:
        """Get the most recent trend analysis for symbol."""
        history = self.get_trend_history(symbol, 1)
        return history[0] if history else None

    def reset(self) -> None:
        """Reset trend analyzer state."""
        super().reset()

        # Reset all MACD indicators
        for indicator in self._macd_indicators.values():
            indicator.reset()

        self._trend_history.clear()

        logger.debug("Reset trend analyzer")