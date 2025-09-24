"""
Data enrichment components for the market data processing pipeline.

This module provides data enrichment capabilities to add derived metrics,
technical indicators, and additional context to market data.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.data.models import TickData, OHLCData, MarketEvent
from src.indicators.manager import get_indicator_manager

logger = get_logger(__name__)
settings = get_settings()


class EnrichmentRule(BaseModel):
    """Base class for data enrichment rules."""

    name: str = Field(description="Rule name")
    description: str = Field(description="Rule description")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    priority: int = Field(default=100, description="Rule priority (lower = higher priority)")

    def enrich(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enrich data with additional information.

        Args:
            data: Data to enrich
            context: Additional context information

        Returns:
            Dictionary with enrichment data
        """
        raise NotImplementedError("Subclasses must implement enrich method")


class SpreadEnrichmentRule(EnrichmentRule):
    """Enriches tick data with spread calculations."""

    def enrich(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate spread metrics for tick data."""
        enrichment = {}

        if isinstance(data, TickData) or (isinstance(data, dict) and "bid" in data and "ask" in data):
            # Extract bid/ask prices
            if isinstance(data, TickData):
                bid, ask = data.bid, data.ask
            else:
                bid = Decimal(str(data["bid"]))
                ask = Decimal(str(data["ask"]))

            # Calculate spread metrics
            spread = ask - bid
            mid_price = (bid + ask) / 2
            spread_percent = (spread / mid_price) * 100 if mid_price > 0 else 0
            spread_pips = spread * 10000  # Assuming 4-decimal currencies

            enrichment.update({
                "spread": float(spread),
                "mid_price": float(mid_price),
                "spread_percent": float(spread_percent),
                "spread_pips": float(spread_pips)
            })

        return enrichment


class VolumeEnrichmentRule(EnrichmentRule):
    """Enriches data with volume-based metrics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._volume_history: Dict[str, List[int]] = {}
        self._max_history_size = 100

    def enrich(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate volume-based enrichments."""
        enrichment = {}

        # Extract symbol and volume
        symbol = None
        volume = None

        if isinstance(data, (TickData, OHLCData)):
            symbol = data.symbol
            volume = data.volume
        elif isinstance(data, dict):
            symbol = data.get("symbol")
            volume = data.get("volume")

        if not symbol or volume is None:
            return enrichment

        # Initialize history for symbol
        if symbol not in self._volume_history:
            self._volume_history[symbol] = []

        # Add current volume to history
        self._volume_history[symbol].append(volume)

        # Limit history size
        if len(self._volume_history[symbol]) > self._max_history_size:
            self._volume_history[symbol] = self._volume_history[symbol][-self._max_history_size:]

        # Calculate volume metrics
        history = self._volume_history[symbol]
        if len(history) > 1:
            avg_volume = sum(history) / len(history)
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

            # Volume percentile (approximate)
            sorted_history = sorted(history)
            volume_percentile = (sorted_history.index(volume) + 1) / len(sorted_history) * 100

            enrichment.update({
                "volume_avg": avg_volume,
                "volume_ratio": volume_ratio,
                "volume_percentile": volume_percentile,
                "is_high_volume": volume_ratio > 1.5,
                "is_low_volume": volume_ratio < 0.5
            })

        return enrichment


class OHLCEnrichmentRule(EnrichmentRule):
    """Enriches OHLC data with technical metrics."""

    def enrich(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate OHLC-based enrichments."""
        enrichment = {}

        if not isinstance(data, OHLCData) and not (isinstance(data, dict) and all(
            key in data for key in ["open", "high", "low", "close"]
        )):
            return enrichment

        # Extract OHLC values
        if isinstance(data, OHLCData):
            open_price, high, low, close = data.open, data.high, data.low, data.close
        else:
            try:
                open_price = Decimal(str(data["open"]))
                high = Decimal(str(data["high"]))
                low = Decimal(str(data["low"]))
                close = Decimal(str(data["close"]))
            except (ValueError, TypeError):
                return enrichment

        # Basic metrics
        body_size = abs(close - open_price)
        total_range = high - low
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low

        # Percentages
        body_percent = (body_size / total_range * 100) if total_range > 0 else 0
        upper_shadow_percent = (upper_shadow / total_range * 100) if total_range > 0 else 0
        lower_shadow_percent = (lower_shadow / total_range * 100) if total_range > 0 else 0

        # Price change
        price_change = close - open_price
        price_change_percent = (price_change / open_price * 100) if open_price > 0 else 0

        # Candle patterns
        is_bullish = close > open_price
        is_bearish = close < open_price
        is_doji = abs(price_change_percent) < 0.1
        is_hammer = (lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5)
        is_shooting_star = (upper_shadow > body_size * 2 and lower_shadow < body_size * 0.5)

        enrichment.update({
            "body_size": float(body_size),
            "total_range": float(total_range),
            "upper_shadow": float(upper_shadow),
            "lower_shadow": float(lower_shadow),
            "body_percent": float(body_percent),
            "upper_shadow_percent": float(upper_shadow_percent),
            "lower_shadow_percent": float(lower_shadow_percent),
            "price_change": float(price_change),
            "price_change_percent": float(price_change_percent),
            "is_bullish": is_bullish,
            "is_bearish": is_bearish,
            "is_doji": is_doji,
            "is_hammer": is_hammer,
            "is_shooting_star": is_shooting_star
        })

        return enrichment


class TechnicalIndicatorEnrichmentRule(EnrichmentRule):
    """Enriches data with technical indicator values."""

    def __init__(self, indicators: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.indicators = indicators or ["MACD"]
        self.indicator_manager = get_indicator_manager()
        self._data_cache: Dict[str, List[Dict]] = {}
        self._cache_size = 50

    def enrich(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add technical indicator values."""
        enrichment = {}

        # Extract symbol and timeframe
        symbol = None
        timeframe = "M1"  # Default timeframe

        if isinstance(data, (TickData, OHLCData, MarketEvent)):
            symbol = data.symbol
        elif isinstance(data, dict):
            symbol = data.get("symbol")
            timeframe = data.get("timeframe", timeframe)

        if not symbol:
            return enrichment

        # Convert data to OHLC format for indicators
        ohlc_data = self._convert_to_ohlc(data)
        if not ohlc_data:
            return enrichment

        # Update data cache
        cache_key = f"{symbol}_{timeframe}"
        if cache_key not in self._data_cache:
            self._data_cache[cache_key] = []

        self._data_cache[cache_key].append(ohlc_data)
        if len(self._data_cache[cache_key]) > self._cache_size:
            self._data_cache[cache_key] = self._data_cache[cache_key][-self._cache_size:]

        # Calculate indicators
        try:
            if "MACD" in self.indicators:
                macd_result = self.indicator_manager.calculate_macd(
                    symbol, timeframe, self._data_cache[cache_key]
                )
                if macd_result:
                    enrichment.update({
                        f"macd_{key}": value
                        for key, value in macd_result.values.items()
                    })
                    enrichment["macd_signal"] = macd_result.signals

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")

        return enrichment

    def _convert_to_ohlc(self, data: Any) -> Optional[Dict[str, float]]:
        """Convert various data formats to OHLC dictionary."""
        if isinstance(data, OHLCData):
            return {
                "open": float(data.open),
                "high": float(data.high),
                "low": float(data.low),
                "close": float(data.close),
                "volume": data.volume
            }
        elif isinstance(data, TickData):
            # Convert tick to OHLC (single point)
            mid_price = float((data.bid + data.ask) / 2)
            return {
                "open": mid_price,
                "high": mid_price,
                "low": mid_price,
                "close": mid_price,
                "volume": data.volume
            }
        elif isinstance(data, dict) and all(key in data for key in ["open", "high", "low", "close"]):
            return {
                "open": float(data["open"]),
                "high": float(data["high"]),
                "low": float(data["low"]),
                "close": float(data["close"]),
                "volume": data.get("volume", 0)
            }

        return None


class MarketConditionEnrichmentRule(EnrichmentRule):
    """Enriches data with market condition assessments."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._price_history: Dict[str, List[float]] = {}
        self._volatility_window = 20
        self._trend_window = 10

    def enrich(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess market conditions and volatility."""
        enrichment = {}

        # Extract symbol and price
        symbol = None
        price = None

        if isinstance(data, TickData):
            symbol = data.symbol
            price = float((data.bid + data.ask) / 2)
        elif isinstance(data, OHLCData):
            symbol = data.symbol
            price = float(data.close)
        elif isinstance(data, dict):
            symbol = data.get("symbol")
            if "close" in data:
                price = float(data["close"])
            elif "bid" in data and "ask" in data:
                price = (float(data["bid"]) + float(data["ask"])) / 2

        if not symbol or price is None:
            return enrichment

        # Initialize price history
        if symbol not in self._price_history:
            self._price_history[symbol] = []

        # Add current price
        self._price_history[symbol].append(price)

        # Limit history size
        max_history = max(self._volatility_window, self._trend_window)
        if len(self._price_history[symbol]) > max_history:
            self._price_history[symbol] = self._price_history[symbol][-max_history:]

        prices = self._price_history[symbol]

        # Calculate volatility
        if len(prices) >= self._volatility_window:
            recent_prices = prices[-self._volatility_window:]
            returns = [
                (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                for i in range(1, len(recent_prices))
            ]

            if returns:
                volatility = math.sqrt(sum(r*r for r in returns) / len(returns)) * 100
                avg_return = sum(returns) / len(returns) * 100

                enrichment.update({
                    "volatility_percent": volatility,
                    "average_return_percent": avg_return,
                    "is_high_volatility": volatility > 2.0,
                    "is_low_volatility": volatility < 0.5
                })

        # Calculate trend
        if len(prices) >= self._trend_window:
            recent_prices = prices[-self._trend_window:]
            first_price = recent_prices[0]
            last_price = recent_prices[-1]

            trend_percent = (last_price - first_price) / first_price * 100

            # Simple linear regression for trend strength
            n = len(recent_prices)
            sum_x = sum(range(n))
            sum_y = sum(recent_prices)
            sum_xy = sum(i * p for i, p in enumerate(recent_prices))
            sum_x2 = sum(i * i for i in range(n))

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            trend_strength = abs(slope) / (sum_y / n) * 100  # Normalize by average price

            enrichment.update({
                "trend_percent": trend_percent,
                "trend_strength": trend_strength,
                "is_uptrend": trend_percent > 0.5,
                "is_downtrend": trend_percent < -0.5,
                "is_strong_trend": trend_strength > 1.0
            })

        return enrichment


class TimingEnrichmentRule(EnrichmentRule):
    """Enriches data with timing and session information."""

    def enrich(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add timing-based enrichments."""
        enrichment = {}

        # Extract timestamp
        timestamp = None
        if isinstance(data, (TickData, OHLCData, MarketEvent)):
            timestamp = data.timestamp
        elif isinstance(data, dict):
            timestamp_val = data.get("timestamp")
            if isinstance(timestamp_val, datetime):
                timestamp = timestamp_val

        if not timestamp:
            return enrichment

        # Ensure timezone awareness
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Market session detection
        utc_hour = timestamp.hour
        london_session = 8 <= utc_hour <= 16
        new_york_session = 13 <= utc_hour <= 21
        tokyo_session = 23 <= utc_hour or utc_hour <= 7
        overlap_london_ny = 13 <= utc_hour <= 16
        overlap_tokyo_london = 8 <= utc_hour <= 9

        # Time-based metrics
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
        is_weekend = day_of_week >= 5
        minute_of_hour = timestamp.minute
        is_hour_open = minute_of_hour == 0
        is_round_time = minute_of_hour in [0, 15, 30, 45]

        enrichment.update({
            "utc_hour": utc_hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "london_session": london_session,
            "new_york_session": new_york_session,
            "tokyo_session": tokyo_session,
            "overlap_london_ny": overlap_london_ny,
            "overlap_tokyo_london": overlap_tokyo_london,
            "is_hour_open": is_hour_open,
            "is_round_time": is_round_time,
            "minute_of_hour": minute_of_hour
        })

        return enrichment


class DataEnricher:
    """
    Main data enricher that applies multiple enrichment rules.

    Coordinates enrichment across multiple rules to add comprehensive
    derived metrics and contextual information to market data.
    """

    def __init__(self):
        """Initialize data enricher."""
        self._rules: List[EnrichmentRule] = []
        self._enrichment_stats = {
            "total_enrichments": 0,
            "total_errors": 0,
            "rules_applied": 0,
            "fields_added": 0
        }

        # Register default rules
        self._register_default_rules()

        logger.info("Data enricher initialized")

    def add_rule(self, rule: EnrichmentRule) -> None:
        """Add enrichment rule."""
        self._rules.append(rule)
        # Sort by priority
        self._rules.sort(key=lambda r: r.priority)
        logger.debug(f"Added enrichment rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove enrichment rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                del self._rules[i]
                logger.debug(f"Removed enrichment rule: {rule_name}")
                return True
        return False

    def enrich(
        self,
        data: Any,
        context: Dict[str, Any] = None,
        rule_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Enrich data with additional information.

        Args:
            data: Data to enrich
            context: Additional context information
            rule_names: Specific rules to apply (all if None)

        Returns:
            Dictionary with all enrichment data
        """
        enrichment = {}
        rules_applied = 0
        context = context or {}

        try:
            # Apply enrichment rules
            for rule in self._rules:
                if not rule.enabled:
                    continue

                if rule_names and rule.name not in rule_names:
                    continue

                try:
                    rule_enrichment = rule.enrich(data, context)
                    if rule_enrichment:
                        enrichment.update(rule_enrichment)
                        rules_applied += 1

                except Exception as e:
                    error_msg = f"Enrichment rule '{rule.name}' failed: {e}"
                    logger.error(error_msg)
                    self._enrichment_stats["total_errors"] += 1

            # Update statistics
            self._enrichment_stats["total_enrichments"] += 1
            self._enrichment_stats["rules_applied"] += rules_applied
            self._enrichment_stats["fields_added"] += len(enrichment)

            if enrichment:
                logger.debug(f"Enriched data with {len(enrichment)} fields using {rules_applied} rules")

            return enrichment

        except Exception as e:
            error_msg = f"Enrichment process failed: {e}"
            logger.error(error_msg)
            self._enrichment_stats["total_errors"] += 1
            return {}

    def get_rules(self) -> List[EnrichmentRule]:
        """Get all enrichment rules."""
        return self._rules.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get enrichment statistics."""
        stats = self._enrichment_stats.copy()
        stats["total_rules"] = len(self._rules)
        stats["enabled_rules"] = sum(1 for rule in self._rules if rule.enabled)

        if stats["total_enrichments"] > 0:
            stats["average_fields_per_enrichment"] = stats["fields_added"] / stats["total_enrichments"]
            stats["average_rules_per_enrichment"] = stats["rules_applied"] / stats["total_enrichments"]

        return stats

    def reset_stats(self) -> None:
        """Reset enrichment statistics."""
        self._enrichment_stats = {
            "total_enrichments": 0,
            "total_errors": 0,
            "rules_applied": 0,
            "fields_added": 0
        }
        logger.info("Enrichment statistics reset")

    def _register_default_rules(self) -> None:
        """Register default enrichment rules."""
        # Basic enrichments
        self.add_rule(SpreadEnrichmentRule(
            name="spread_metrics",
            description="Calculate spread and mid-price metrics",
            priority=10
        ))

        self.add_rule(VolumeEnrichmentRule(
            name="volume_metrics",
            description="Calculate volume-based metrics",
            priority=20
        ))

        self.add_rule(OHLCEnrichmentRule(
            name="ohlc_metrics",
            description="Calculate OHLC-based technical metrics",
            priority=30
        ))

        # Advanced enrichments
        self.add_rule(TechnicalIndicatorEnrichmentRule(
            name="technical_indicators",
            description="Add technical indicator values",
            priority=40
        ))

        self.add_rule(MarketConditionEnrichmentRule(
            name="market_conditions",
            description="Assess market conditions and volatility",
            priority=50
        ))

        self.add_rule(TimingEnrichmentRule(
            name="timing_info",
            description="Add timing and session information",
            priority=60
        ))

        logger.info(f"Registered {len(self._rules)} default enrichment rules")