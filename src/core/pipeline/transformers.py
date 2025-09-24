"""
Data transformation components for the market data processing pipeline.

This module provides transformation capabilities to convert, aggregate,
and restructure market data for various downstream uses.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Callable

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.data.models import TickData, OHLCData, MarketEvent

logger = get_logger(__name__)
settings = get_settings()


class TransformationRule(BaseModel):
    """Base class for data transformation rules."""

    name: str = Field(description="Rule name")
    description: str = Field(description="Rule description")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    priority: int = Field(default=100, description="Rule priority (lower = higher priority)")

    def transform(self, data: Any, context: Dict[str, Any] = None) -> Any:
        """
        Transform data.

        Args:
            data: Data to transform
            context: Additional context information

        Returns:
            Transformed data
        """
        raise NotImplementedError("Subclasses must implement transform method")


class NormalizationRule(TransformationRule):
    """Normalizes data formats and structures."""

    def __init__(self, target_format: str = "dict", **kwargs):
        super().__init__(**kwargs)
        self.target_format = target_format

    def transform(self, data: Any, context: Dict[str, Any] = None) -> Any:
        """Normalize data to consistent format."""
        if self.target_format == "dict":
            return self._to_dict(data)
        elif self.target_format == "json":
            return self._to_json(data)
        elif self.target_format == "pydantic":
            return self._to_pydantic(data, context)
        else:
            return data

    def _to_dict(self, data: Any) -> Dict[str, Any]:
        """Convert data to dictionary format."""
        if isinstance(data, (TickData, OHLCData, MarketEvent)):
            result = data.model_dump()
            # Convert Decimal to float for JSON serialization
            for key, value in result.items():
                if isinstance(value, Decimal):
                    result[key] = float(value)
                elif isinstance(value, datetime):
                    result[key] = value.isoformat()
            return result
        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if isinstance(value, Decimal):
                    result[key] = float(value)
                elif isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
            return result
        else:
            return {"value": data}

    def _to_json(self, data: Any) -> str:
        """Convert data to JSON string."""
        dict_data = self._to_dict(data)
        return json.dumps(dict_data, default=str, separators=(',', ':'))

    def _to_pydantic(self, data: Any, context: Dict[str, Any] = None) -> Any:
        """Convert data to appropriate Pydantic model."""
        if isinstance(data, (TickData, OHLCData, MarketEvent)):
            return data
        elif isinstance(data, dict):
            # Try to determine the appropriate model
            if "bid" in data and "ask" in data:
                return TickData(**data)
            elif all(key in data for key in ["open", "high", "low", "close"]):
                return OHLCData(**data)
            elif "event_type" in data:
                return MarketEvent(**data)

        return data


class AggregationRule(TransformationRule):
    """Aggregates data over time periods or groups."""

    def __init__(self,
                 aggregation_type: str = "time_based",
                 time_window: timedelta = timedelta(minutes=1),
                 group_by_field: str = "symbol",
                 **kwargs):
        super().__init__(**kwargs)
        self.aggregation_type = aggregation_type
        self.time_window = time_window
        self.group_by_field = group_by_field
        self._aggregation_buffer: Dict[str, List[Any]] = {}
        self._last_aggregation: Dict[str, datetime] = {}

    def transform(self, data: Any, context: Dict[str, Any] = None) -> Optional[Any]:
        """Aggregate data based on configured rules."""
        if self.aggregation_type == "time_based":
            return self._time_based_aggregation(data, context)
        elif self.aggregation_type == "tick_based":
            return self._tick_based_aggregation(data, context)
        else:
            return data

    def _time_based_aggregation(self, data: Any, context: Dict[str, Any] = None) -> Optional[Any]:
        """Aggregate data over time windows."""
        # Extract grouping key and timestamp
        group_key = self._get_group_key(data)
        timestamp = self._get_timestamp(data)

        if not group_key or not timestamp:
            return None

        # Initialize buffer for this group
        if group_key not in self._aggregation_buffer:
            self._aggregation_buffer[group_key] = []
            self._last_aggregation[group_key] = timestamp

        # Add data to buffer
        self._aggregation_buffer[group_key].append(data)

        # Check if aggregation window is complete
        time_since_last = timestamp - self._last_aggregation[group_key]
        if time_since_last >= self.time_window:
            # Perform aggregation
            aggregated_data = self._aggregate_buffer(group_key)

            # Clear buffer and update timestamp
            self._aggregation_buffer[group_key] = []
            self._last_aggregation[group_key] = timestamp

            return aggregated_data

        return None  # Not ready for aggregation yet

    def _tick_based_aggregation(self, data: Any, context: Dict[str, Any] = None) -> Optional[Any]:
        """Aggregate data after collecting specified number of ticks."""
        group_key = self._get_group_key(data)
        if not group_key:
            return None

        # Initialize buffer
        if group_key not in self._aggregation_buffer:
            self._aggregation_buffer[group_key] = []

        # Add data to buffer
        self._aggregation_buffer[group_key].append(data)

        # Check if we have enough ticks (default: 10)
        if len(self._aggregation_buffer[group_key]) >= 10:
            aggregated_data = self._aggregate_buffer(group_key)
            self._aggregation_buffer[group_key] = []
            return aggregated_data

        return None

    def _get_group_key(self, data: Any) -> Optional[str]:
        """Extract grouping key from data."""
        if isinstance(data, (TickData, OHLCData, MarketEvent)):
            return getattr(data, self.group_by_field, None)
        elif isinstance(data, dict):
            return data.get(self.group_by_field)
        return None

    def _get_timestamp(self, data: Any) -> Optional[datetime]:
        """Extract timestamp from data."""
        if isinstance(data, (TickData, OHLCData, MarketEvent)):
            return data.timestamp
        elif isinstance(data, dict):
            timestamp_val = data.get("timestamp")
            if isinstance(timestamp_val, datetime):
                return timestamp_val
            elif isinstance(timestamp_val, str):
                try:
                    return datetime.fromisoformat(timestamp_val)
                except ValueError:
                    pass
        return None

    def _aggregate_buffer(self, group_key: str) -> Dict[str, Any]:
        """Aggregate buffered data."""
        buffer = self._aggregation_buffer[group_key]
        if not buffer:
            return {}

        # Determine data type for aggregation
        first_item = buffer[0]

        if isinstance(first_item, TickData) or (isinstance(first_item, dict) and "bid" in first_item):
            return self._aggregate_tick_data(buffer, group_key)
        elif isinstance(first_item, OHLCData) or (isinstance(first_item, dict) and "open" in first_item):
            return self._aggregate_ohlc_data(buffer, group_key)
        else:
            return self._aggregate_generic_data(buffer, group_key)

    def _aggregate_tick_data(self, ticks: List[Any], symbol: str) -> Dict[str, Any]:
        """Aggregate tick data into OHLC."""
        if not ticks:
            return {}

        # Extract mid prices
        mid_prices = []
        volumes = []
        timestamps = []

        for tick in ticks:
            if isinstance(tick, TickData):
                mid_price = (tick.bid + tick.ask) / 2
                mid_prices.append(float(mid_price))
                volumes.append(tick.volume)
                timestamps.append(tick.timestamp)
            elif isinstance(tick, dict):
                bid = Decimal(str(tick.get("bid", 0)))
                ask = Decimal(str(tick.get("ask", 0)))
                mid_price = (bid + ask) / 2
                mid_prices.append(float(mid_price))
                volumes.append(tick.get("volume", 0))
                timestamps.append(tick.get("timestamp"))

        if not mid_prices:
            return {}

        # Create OHLC from mid prices
        open_price = mid_prices[0]
        close_price = mid_prices[-1]
        high_price = max(mid_prices)
        low_price = min(mid_prices)
        total_volume = sum(volumes)

        return {
            "symbol": symbol,
            "timeframe": "AGG",
            "timestamp": timestamps[-1] if timestamps[-1] else datetime.now(timezone.utc),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": total_volume,
            "tick_count": len(ticks)
        }

    def _aggregate_ohlc_data(self, ohlc_list: List[Any], symbol: str) -> Dict[str, Any]:
        """Aggregate OHLC data into higher timeframe OHLC."""
        if not ohlc_list:
            return {}

        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        timestamps = []

        for ohlc in ohlc_list:
            if isinstance(ohlc, OHLCData):
                opens.append(float(ohlc.open))
                highs.append(float(ohlc.high))
                lows.append(float(ohlc.low))
                closes.append(float(ohlc.close))
                volumes.append(ohlc.volume)
                timestamps.append(ohlc.timestamp)
            elif isinstance(ohlc, dict):
                opens.append(float(ohlc.get("open", 0)))
                highs.append(float(ohlc.get("high", 0)))
                lows.append(float(ohlc.get("low", 0)))
                closes.append(float(ohlc.get("close", 0)))
                volumes.append(ohlc.get("volume", 0))
                timestamps.append(ohlc.get("timestamp"))

        if not opens:
            return {}

        return {
            "symbol": symbol,
            "timeframe": "AGG",
            "timestamp": timestamps[-1] if timestamps[-1] else datetime.now(timezone.utc),
            "open": opens[0],
            "high": max(highs),
            "low": min(lows),
            "close": closes[-1],
            "volume": sum(volumes),
            "bar_count": len(ohlc_list)
        }

    def _aggregate_generic_data(self, data_list: List[Any], group_key: str) -> Dict[str, Any]:
        """Aggregate generic data."""
        return {
            "group_key": group_key,
            "count": len(data_list),
            "timestamp": datetime.now(timezone.utc),
            "items": data_list
        }


class FilteringRule(TransformationRule):
    """Filters data based on conditions."""

    def __init__(self,
                 filter_conditions: List[Callable[[Any], bool]] = None,
                 include_fields: List[str] = None,
                 exclude_fields: List[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.filter_conditions = filter_conditions or []
        self.include_fields = include_fields
        self.exclude_fields = exclude_fields or []

    def transform(self, data: Any, context: Dict[str, Any] = None) -> Optional[Any]:
        """Filter data based on conditions and field selection."""
        # Apply filter conditions
        for condition in self.filter_conditions:
            try:
                if not condition(data):
                    return None  # Data filtered out
            except Exception as e:
                logger.warning(f"Filter condition failed: {e}")
                return None

        # Apply field filtering
        if isinstance(data, dict):
            filtered_data = {}

            # Include specific fields
            if self.include_fields:
                for field in self.include_fields:
                    if field in data:
                        filtered_data[field] = data[field]
            else:
                filtered_data = data.copy()

            # Exclude specific fields
            for field in self.exclude_fields:
                filtered_data.pop(field, None)

            return filtered_data
        elif isinstance(data, (TickData, OHLCData, MarketEvent)):
            # Convert to dict, apply filtering, then back to model
            dict_data = data.model_dump()
            filtered_dict = self.transform(dict_data, context)

            if filtered_dict is None:
                return None

            try:
                return data.__class__(**filtered_dict)
            except Exception as e:
                logger.warning(f"Failed to reconstruct model after filtering: {e}")
                return filtered_dict

        return data


class EnrichmentIntegrationRule(TransformationRule):
    """Integrates enrichment data into the main data structure."""

    def transform(self, data: Any, context: Dict[str, Any] = None) -> Any:
        """Integrate enrichment data."""
        if not context or "enrichment" not in context:
            return data

        enrichment = context["enrichment"]
        if not enrichment:
            return data

        if isinstance(data, dict):
            # Merge enrichment into dict
            result = data.copy()
            result.update(enrichment)
            return result
        elif isinstance(data, (TickData, OHLCData, MarketEvent)):
            # Convert to dict, add enrichment, keep as dict (or convert back if needed)
            result = data.model_dump()
            result.update(enrichment)
            return result

        return data


class TypeConversionRule(TransformationRule):
    """Converts data types for consistency."""

    def __init__(self,
                 decimal_fields: List[str] = None,
                 float_fields: List[str] = None,
                 int_fields: List[str] = None,
                 datetime_fields: List[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.decimal_fields = decimal_fields or ["bid", "ask", "open", "high", "low", "close"]
        self.float_fields = float_fields or []
        self.int_fields = int_fields or ["volume"]
        self.datetime_fields = datetime_fields or ["timestamp"]

    def transform(self, data: Any, context: Dict[str, Any] = None) -> Any:
        """Convert field types."""
        if not isinstance(data, dict):
            return data

        result = {}

        for key, value in data.items():
            if value is None:
                result[key] = value
                continue

            try:
                if key in self.decimal_fields:
                    result[key] = Decimal(str(value))
                elif key in self.float_fields:
                    result[key] = float(value)
                elif key in self.int_fields:
                    result[key] = int(float(value))  # Handle string numbers
                elif key in self.datetime_fields:
                    if isinstance(value, str):
                        result[key] = datetime.fromisoformat(value)
                    elif isinstance(value, datetime):
                        result[key] = value
                    else:
                        result[key] = value
                else:
                    result[key] = value

            except (ValueError, TypeError) as e:
                logger.warning(f"Type conversion failed for {key}={value}: {e}")
                result[key] = value

        return result


class DataTransformer:
    """
    Main data transformer that applies multiple transformation rules.

    Coordinates transformation across multiple rules to convert,
    aggregate, filter, and restructure market data.
    """

    def __init__(self):
        """Initialize data transformer."""
        self._rules: List[TransformationRule] = []
        self._transformation_stats = {
            "total_transformations": 0,
            "total_errors": 0,
            "rules_applied": 0,
            "data_filtered": 0,
            "data_aggregated": 0
        }

        # Register default rules
        self._register_default_rules()

        logger.info("Data transformer initialized")

    def add_rule(self, rule: TransformationRule) -> None:
        """Add transformation rule."""
        self._rules.append(rule)
        # Sort by priority
        self._rules.sort(key=lambda r: r.priority)
        logger.debug(f"Added transformation rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove transformation rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                del self._rules[i]
                logger.debug(f"Removed transformation rule: {rule_name}")
                return True
        return False

    def transform(
        self,
        data: Any,
        context: Dict[str, Any] = None,
        rule_names: Optional[List[str]] = None
    ) -> Optional[Any]:
        """
        Transform data using configured rules.

        Args:
            data: Data to transform
            context: Additional context information
            rule_names: Specific rules to apply (all if None)

        Returns:
            Transformed data or None if filtered out
        """
        if data is None:
            return None

        transformed_data = data
        rules_applied = 0
        context = context or {}

        try:
            # Apply transformation rules in priority order
            for rule in self._rules:
                if not rule.enabled:
                    continue

                if rule_names and rule.name not in rule_names:
                    continue

                try:
                    transformed_data = rule.transform(transformed_data, context)
                    rules_applied += 1

                    # If data was filtered out, stop processing
                    if transformed_data is None:
                        self._transformation_stats["data_filtered"] += 1
                        break

                except Exception as e:
                    error_msg = f"Transformation rule '{rule.name}' failed: {e}"
                    logger.error(error_msg)
                    self._transformation_stats["total_errors"] += 1

            # Update statistics
            self._transformation_stats["total_transformations"] += 1
            self._transformation_stats["rules_applied"] += rules_applied

            if transformed_data is not None:
                logger.debug(f"Transformed data using {rules_applied} rules")
            else:
                logger.debug(f"Data filtered out after {rules_applied} rules")

            return transformed_data

        except Exception as e:
            error_msg = f"Transformation process failed: {e}"
            logger.error(error_msg)
            self._transformation_stats["total_errors"] += 1
            return None

    def get_rules(self) -> List[TransformationRule]:
        """Get all transformation rules."""
        return self._rules.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get transformation statistics."""
        stats = self._transformation_stats.copy()
        stats["total_rules"] = len(self._rules)
        stats["enabled_rules"] = sum(1 for rule in self._rules if rule.enabled)

        if stats["total_transformations"] > 0:
            stats["average_rules_per_transformation"] = stats["rules_applied"] / stats["total_transformations"]
            stats["filter_rate"] = stats["data_filtered"] / stats["total_transformations"]

        return stats

    def reset_stats(self) -> None:
        """Reset transformation statistics."""
        self._transformation_stats = {
            "total_transformations": 0,
            "total_errors": 0,
            "rules_applied": 0,
            "data_filtered": 0,
            "data_aggregated": 0
        }
        logger.info("Transformation statistics reset")

    def _register_default_rules(self) -> None:
        """Register default transformation rules."""
        # Type conversion (highest priority)
        self.add_rule(TypeConversionRule(
            name="type_conversion",
            description="Convert data types for consistency",
            priority=10
        ))

        # Filtering
        self.add_rule(FilteringRule(
            name="basic_filtering",
            description="Basic data filtering",
            priority=20
        ))

        # Enrichment integration
        self.add_rule(EnrichmentIntegrationRule(
            name="enrichment_integration",
            description="Integrate enrichment data",
            priority=30
        ))

        # Normalization
        self.add_rule(NormalizationRule(
            name="format_normalization",
            description="Normalize data formats",
            priority=40,
            target_format="dict"
        ))

        # Aggregation (lowest priority - applied last)
        self.add_rule(AggregationRule(
            name="time_aggregation",
            description="Time-based data aggregation",
            priority=50,
            aggregation_type="time_based",
            time_window=timedelta(minutes=1)
        ))

        logger.info(f"Registered {len(self._rules)} default transformation rules")