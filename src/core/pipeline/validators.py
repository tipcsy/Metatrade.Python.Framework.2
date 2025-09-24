"""
Data validation components for the market data processing pipeline.

This module provides comprehensive validation rules and validators
for ensuring data quality and consistency in the processing pipeline.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.data.models import TickData, OHLCData, MarketEvent

logger = get_logger(__name__)
settings = get_settings()


class ValidationResult(BaseModel):
    """Result of data validation."""

    is_valid: bool = Field(description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation error messages")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional validation metadata")

    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)

    def merge(self, other: ValidationResult) -> None:
        """Merge another validation result."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.metadata.update(other.metadata)


class ValidationRule(BaseModel):
    """Base class for validation rules."""

    name: str = Field(description="Rule name")
    description: str = Field(description="Rule description")
    severity: str = Field(default="error", description="Rule severity: error, warning")
    enabled: bool = Field(default=True, description="Whether rule is enabled")

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate data against this rule.

        Args:
            data: Data to validate

        Returns:
            ValidationResult with validation outcome
        """
        raise NotImplementedError("Subclasses must implement validate method")


class SymbolValidationRule(ValidationRule):
    """Validates trading symbol format and existence."""

    allowed_symbols: Optional[Set[str]] = Field(default=None, description="Allowed symbols")
    symbol_pattern: str = Field(default=r"^[A-Z]{6}$", description="Symbol regex pattern")

    def validate(self, data: Any) -> ValidationResult:
        """Validate symbol format and existence."""
        result = ValidationResult()

        # Extract symbol from different data types
        symbol = None
        if isinstance(data, (TickData, OHLCData, MarketEvent)):
            symbol = data.symbol
        elif isinstance(data, dict):
            symbol = data.get("symbol")
        elif isinstance(data, str):
            symbol = data

        if not symbol:
            result.add_error("Missing symbol in data")
            return result

        # Check symbol format
        if not re.match(self.symbol_pattern, symbol):
            if self.severity == "error":
                result.add_error(f"Invalid symbol format: {symbol}")
            else:
                result.add_warning(f"Invalid symbol format: {symbol}")

        # Check allowed symbols
        if self.allowed_symbols and symbol not in self.allowed_symbols:
            if self.severity == "error":
                result.add_error(f"Symbol not allowed: {symbol}")
            else:
                result.add_warning(f"Symbol not in allowed list: {symbol}")

        result.metadata["symbol"] = symbol
        return result


class PriceValidationRule(ValidationRule):
    """Validates price data for reasonableness."""

    min_price: Optional[Decimal] = Field(default=None, description="Minimum valid price")
    max_price: Optional[Decimal] = Field(default=None, description="Maximum valid price")
    max_spread_percent: float = Field(default=5.0, description="Maximum spread percentage")

    def validate(self, data: Any) -> ValidationResult:
        """Validate price data."""
        result = ValidationResult()

        prices = self._extract_prices(data)
        if not prices:
            result.add_error("No price data found")
            return result

        # Check price ranges
        for price_type, price in prices.items():
            if price <= 0:
                result.add_error(f"Invalid {price_type}: {price} (must be positive)")
                continue

            if self.min_price and price < self.min_price:
                if self.severity == "error":
                    result.add_error(f"{price_type} below minimum: {price}")
                else:
                    result.add_warning(f"{price_type} below minimum: {price}")

            if self.max_price and price > self.max_price:
                if self.severity == "error":
                    result.add_error(f"{price_type} above maximum: {price}")
                else:
                    result.add_warning(f"{price_type} above maximum: {price}")

        # Check spread for tick data
        if "bid" in prices and "ask" in prices:
            bid, ask = prices["bid"], prices["ask"]
            if ask <= bid:
                result.add_error(f"Ask ({ask}) must be greater than bid ({bid})")
            else:
                spread_percent = ((ask - bid) / bid) * 100
                if spread_percent > self.max_spread_percent:
                    if self.severity == "error":
                        result.add_error(f"Spread too wide: {spread_percent:.2f}%")
                    else:
                        result.add_warning(f"Spread is wide: {spread_percent:.2f}%")

                result.metadata["spread_percent"] = float(spread_percent)

        return result

    def _extract_prices(self, data: Any) -> Dict[str, Decimal]:
        """Extract price data from various data formats."""
        prices = {}

        if isinstance(data, TickData):
            prices["bid"] = data.bid
            prices["ask"] = data.ask
        elif isinstance(data, OHLCData):
            prices["open"] = data.open
            prices["high"] = data.high
            prices["low"] = data.low
            prices["close"] = data.close
        elif isinstance(data, dict):
            for key in ["bid", "ask", "open", "high", "low", "close"]:
                if key in data and data[key] is not None:
                    try:
                        prices[key] = Decimal(str(data[key]))
                    except (ValueError, TypeError):
                        pass

        return prices


class TimestampValidationRule(ValidationRule):
    """Validates timestamp data for consistency and reasonableness."""

    max_age_hours: float = Field(default=24.0, description="Maximum data age in hours")
    max_future_minutes: float = Field(default=5.0, description="Maximum future timestamp in minutes")
    require_timezone: bool = Field(default=True, description="Whether timezone is required")

    def validate(self, data: Any) -> ValidationResult:
        """Validate timestamp data."""
        result = ValidationResult()

        timestamp = self._extract_timestamp(data)
        if not timestamp:
            result.add_error("Missing timestamp in data")
            return result

        # Check timezone
        if self.require_timezone and timestamp.tzinfo is None:
            if self.severity == "error":
                result.add_error("Timestamp missing timezone information")
            else:
                result.add_warning("Timestamp missing timezone information")

        current_time = datetime.now(timezone.utc)

        # Check if timestamp is too old
        age = current_time - timestamp
        max_age = timedelta(hours=self.max_age_hours)
        if age > max_age:
            if self.severity == "error":
                result.add_error(f"Data too old: {age.total_seconds()/3600:.1f} hours")
            else:
                result.add_warning(f"Data is old: {age.total_seconds()/3600:.1f} hours")

        # Check if timestamp is too far in the future
        future_delta = timestamp - current_time
        max_future = timedelta(minutes=self.max_future_minutes)
        if future_delta > max_future:
            if self.severity == "error":
                result.add_error(f"Timestamp too far in future: {future_delta.total_seconds()/60:.1f} minutes")
            else:
                result.add_warning(f"Timestamp in future: {future_delta.total_seconds()/60:.1f} minutes")

        result.metadata["timestamp"] = timestamp
        result.metadata["age_seconds"] = age.total_seconds()
        return result

    def _extract_timestamp(self, data: Any) -> Optional[datetime]:
        """Extract timestamp from various data formats."""
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


class VolumeValidationRule(ValidationRule):
    """Validates volume data."""

    min_volume: int = Field(default=0, description="Minimum valid volume")
    max_volume: Optional[int] = Field(default=None, description="Maximum valid volume")

    def validate(self, data: Any) -> ValidationResult:
        """Validate volume data."""
        result = ValidationResult()

        volume = self._extract_volume(data)
        if volume is None:
            # Volume might be optional for some data types
            return result

        if volume < self.min_volume:
            if self.severity == "error":
                result.add_error(f"Volume below minimum: {volume}")
            else:
                result.add_warning(f"Volume below minimum: {volume}")

        if self.max_volume and volume > self.max_volume:
            if self.severity == "error":
                result.add_error(f"Volume above maximum: {volume}")
            else:
                result.add_warning(f"Volume above maximum: {volume}")

        result.metadata["volume"] = volume
        return result

    def _extract_volume(self, data: Any) -> Optional[int]:
        """Extract volume from various data formats."""
        if isinstance(data, (TickData, OHLCData)):
            return data.volume
        elif isinstance(data, dict):
            volume_val = data.get("volume")
            if isinstance(volume_val, (int, float)):
                return int(volume_val)

        return None


class OHLCConsistencyRule(ValidationRule):
    """Validates OHLC data consistency (High >= Low, etc.)."""

    def validate(self, data: Any) -> ValidationResult:
        """Validate OHLC data consistency."""
        result = ValidationResult()

        if not isinstance(data, OHLCData) and not (isinstance(data, dict) and all(
            key in data for key in ["open", "high", "low", "close"]
        )):
            return result  # Not OHLC data, skip

        # Extract OHLC values
        if isinstance(data, OHLCData):
            open_price, high, low, close = data.open, data.high, data.low, data.close
        else:
            try:
                open_price = Decimal(str(data["open"]))
                high = Decimal(str(data["high"]))
                low = Decimal(str(data["low"]))
                close = Decimal(str(data["close"]))
            except (ValueError, TypeError, KeyError) as e:
                result.add_error(f"Invalid OHLC data format: {e}")
                return result

        # Validate OHLC relationships
        if high < low:
            result.add_error(f"High ({high}) cannot be less than low ({low})")

        if high < open_price:
            result.add_error(f"High ({high}) cannot be less than open ({open_price})")

        if high < close:
            result.add_error(f"High ({high}) cannot be less than close ({close})")

        if low > open_price:
            result.add_error(f"Low ({low}) cannot be greater than open ({open_price})")

        if low > close:
            result.add_error(f"Low ({low}) cannot be greater than close ({close})")

        # Calculate additional metrics
        if high > low:
            range_percent = ((high - low) / low) * 100
            result.metadata["range_percent"] = float(range_percent)

        body_size = abs(close - open_price)
        total_range = high - low
        if total_range > 0:
            body_percent = (body_size / total_range) * 100
            result.metadata["body_percent"] = float(body_percent)

        return result


class DataValidator:
    """
    Main validator class that applies multiple validation rules.

    Coordinates validation across multiple rules and provides
    comprehensive data quality assessment.
    """

    def __init__(self):
        """Initialize data validator."""
        self._rules: List[ValidationRule] = []
        self._validation_stats = {
            "total_validations": 0,
            "total_errors": 0,
            "total_warnings": 0,
            "rules_applied": 0
        }

        # Register default rules
        self._register_default_rules()

        logger.info("Data validator initialized")

    def add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule."""
        self._rules.append(rule)
        logger.debug(f"Added validation rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove validation rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                del self._rules[i]
                logger.debug(f"Removed validation rule: {rule_name}")
                return True
        return False

    def validate(self, data: Any, rule_names: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate data against all or specified rules.

        Args:
            data: Data to validate
            rule_names: Specific rules to apply (all if None)

        Returns:
            Combined validation result
        """
        combined_result = ValidationResult()
        rules_applied = 0

        try:
            # Apply validation rules
            for rule in self._rules:
                if not rule.enabled:
                    continue

                if rule_names and rule.name not in rule_names:
                    continue

                try:
                    rule_result = rule.validate(data)
                    combined_result.merge(rule_result)
                    rules_applied += 1

                except Exception as e:
                    error_msg = f"Rule '{rule.name}' failed: {e}"
                    combined_result.add_error(error_msg)
                    logger.error(error_msg)

            # Update statistics
            self._validation_stats["total_validations"] += 1
            self._validation_stats["total_errors"] += len(combined_result.errors)
            self._validation_stats["total_warnings"] += len(combined_result.warnings)
            self._validation_stats["rules_applied"] += rules_applied

            combined_result.metadata["rules_applied"] = rules_applied

            if combined_result.errors:
                logger.debug(f"Validation failed with {len(combined_result.errors)} errors")
            elif combined_result.warnings:
                logger.debug(f"Validation passed with {len(combined_result.warnings)} warnings")

            return combined_result

        except Exception as e:
            error_msg = f"Validation process failed: {e}"
            logger.error(error_msg)
            combined_result.add_error(error_msg)
            return combined_result

    def get_rules(self) -> List[ValidationRule]:
        """Get all validation rules."""
        return self._rules.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self._validation_stats.copy()
        stats["total_rules"] = len(self._rules)
        stats["enabled_rules"] = sum(1 for rule in self._rules if rule.enabled)

        if stats["total_validations"] > 0:
            stats["average_errors_per_validation"] = stats["total_errors"] / stats["total_validations"]
            stats["average_warnings_per_validation"] = stats["total_warnings"] / stats["total_validations"]

        return stats

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self._validation_stats = {
            "total_validations": 0,
            "total_errors": 0,
            "total_warnings": 0,
            "rules_applied": 0
        }
        logger.info("Validation statistics reset")

    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        # Symbol validation
        self.add_rule(SymbolValidationRule(
            name="symbol_format",
            description="Validate trading symbol format"
        ))

        # Price validation
        self.add_rule(PriceValidationRule(
            name="price_range",
            description="Validate price ranges and spreads",
            min_price=Decimal("0.00001"),
            max_price=Decimal("100000")
        ))

        # Timestamp validation
        self.add_rule(TimestampValidationRule(
            name="timestamp_consistency",
            description="Validate timestamp consistency and age"
        ))

        # Volume validation
        self.add_rule(VolumeValidationRule(
            name="volume_range",
            description="Validate volume ranges"
        ))

        # OHLC consistency
        self.add_rule(OHLCConsistencyRule(
            name="ohlc_consistency",
            description="Validate OHLC data consistency"
        ))

        logger.info(f"Registered {len(self._rules)} default validation rules")