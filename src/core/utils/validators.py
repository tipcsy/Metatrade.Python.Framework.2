"""
Data validation utilities for MetaTrader Python Framework.

This module provides various validators for data integrity, type checking,
range validation, and business rule enforcement.
"""

from __future__ import annotations

import re
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Type, Union

from ..exceptions import ValidationError


class ValidationResult:
    """Result of a validation operation."""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize validation result.

        Args:
            is_valid: Whether validation passed
            errors: List of error messages
            warnings: List of warning messages
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def merge(self, other: ValidationResult) -> None:
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False

    def __bool__(self) -> bool:
        """Return validation status."""
        return self.is_valid

    def __str__(self) -> str:
        """Return string representation."""
        if self.is_valid:
            status = "Valid"
        else:
            status = "Invalid"

        if self.errors:
            status += f" (Errors: {len(self.errors)})"
        if self.warnings:
            status += f" (Warnings: {len(self.warnings)})"

        return status


class BaseValidator:
    """Base class for all validators."""

    def __init__(self, error_message: Optional[str] = None) -> None:
        """
        Initialize base validator.

        Args:
            error_message: Custom error message
        """
        self.error_message = error_message

    def validate(self, value: Any, field_name: str = "value") -> ValidationResult:
        """
        Validate a value.

        Args:
            value: Value to validate
            field_name: Name of the field being validated

        Returns:
            Validation result
        """
        try:
            if self._validate(value):
                return ValidationResult(True)
            else:
                error_msg = self.error_message or self._get_default_error_message(value, field_name)
                return ValidationResult(False, [error_msg])
        except Exception as e:
            error_msg = self.error_message or f"Validation error for {field_name}: {str(e)}"
            return ValidationResult(False, [error_msg])

    def _validate(self, value: Any) -> bool:
        """
        Perform the actual validation.

        Args:
            value: Value to validate

        Returns:
            True if valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement _validate method")

    def _get_default_error_message(self, value: Any, field_name: str) -> str:
        """
        Get default error message.

        Args:
            value: Invalid value
            field_name: Field name

        Returns:
            Default error message
        """
        return f"Invalid value for {field_name}: {value}"

    def __call__(self, value: Any, field_name: str = "value") -> ValidationResult:
        """Make validator callable."""
        return self.validate(value, field_name)


class TypeValidator(BaseValidator):
    """Validate value type."""

    def __init__(
        self,
        expected_type: Union[Type, tuple],
        allow_none: bool = False,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Initialize type validator.

        Args:
            expected_type: Expected type or tuple of types
            allow_none: Whether None values are allowed
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.expected_type = expected_type
        self.allow_none = allow_none

    def _validate(self, value: Any) -> bool:
        """Validate type."""
        if value is None:
            return self.allow_none
        return isinstance(value, self.expected_type)

    def _get_default_error_message(self, value: Any, field_name: str) -> str:
        """Get default error message for type validation."""
        if isinstance(self.expected_type, tuple):
            type_names = " or ".join(t.__name__ for t in self.expected_type)
        else:
            type_names = self.expected_type.__name__

        return f"{field_name} must be of type {type_names}, got {type(value).__name__}"


class RangeValidator(BaseValidator):
    """Validate numeric range."""

    def __init__(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        inclusive: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Initialize range validator.

        Args:
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            inclusive: Whether bounds are inclusive
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def _validate(self, value: Any) -> bool:
        """Validate range."""
        if not isinstance(value, (int, float, Decimal)):
            return False

        if self.min_value is not None:
            if self.inclusive:
                if value < self.min_value:
                    return False
            else:
                if value <= self.min_value:
                    return False

        if self.max_value is not None:
            if self.inclusive:
                if value > self.max_value:
                    return False
            else:
                if value >= self.max_value:
                    return False

        return True

    def _get_default_error_message(self, value: Any, field_name: str) -> str:
        """Get default error message for range validation."""
        bounds = []
        if self.min_value is not None:
            op = ">=" if self.inclusive else ">"
            bounds.append(f"{field_name} {op} {self.min_value}")

        if self.max_value is not None:
            op = "<=" if self.inclusive else "<"
            bounds.append(f"{field_name} {op} {self.max_value}")

        constraint = " and ".join(bounds)
        return f"Value {value} violates constraint: {constraint}"


class LengthValidator(BaseValidator):
    """Validate string or collection length."""

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Initialize length validator.

        Args:
            min_length: Minimum length
            max_length: Maximum length
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.min_length = min_length
        self.max_length = max_length

    def _validate(self, value: Any) -> bool:
        """Validate length."""
        try:
            length = len(value)
        except TypeError:
            return False

        if self.min_length is not None and length < self.min_length:
            return False

        if self.max_length is not None and length > self.max_length:
            return False

        return True

    def _get_default_error_message(self, value: Any, field_name: str) -> str:
        """Get default error message for length validation."""
        try:
            length = len(value)
            constraints = []

            if self.min_length is not None:
                constraints.append(f"minimum {self.min_length}")
            if self.max_length is not None:
                constraints.append(f"maximum {self.max_length}")

            constraint_text = " and ".join(constraints)
            return f"{field_name} length {length} violates constraint: {constraint_text}"
        except TypeError:
            return f"{field_name} does not support length validation"


class RegexValidator(BaseValidator):
    """Validate against regular expression."""

    def __init__(
        self,
        pattern: Union[str, Pattern],
        flags: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Initialize regex validator.

        Args:
            pattern: Regular expression pattern
            flags: Regex flags
            error_message: Custom error message
        """
        super().__init__(error_message)
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern, flags)
        else:
            self.pattern = pattern

    def _validate(self, value: Any) -> bool:
        """Validate against regex."""
        if not isinstance(value, str):
            return False
        return bool(self.pattern.match(value))

    def _get_default_error_message(self, value: Any, field_name: str) -> str:
        """Get default error message for regex validation."""
        return f"{field_name} '{value}' does not match required pattern"


class EmailValidator(RegexValidator):
    """Validate email addresses."""

    def __init__(self, error_message: Optional[str] = None) -> None:
        """Initialize email validator."""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        super().__init__(email_pattern, error_message=error_message)

    def _get_default_error_message(self, value: Any, field_name: str) -> str:
        """Get default error message for email validation."""
        return f"{field_name} '{value}' is not a valid email address"


class UrlValidator(RegexValidator):
    """Validate URLs."""

    def __init__(self, error_message: Optional[str] = None) -> None:
        """Initialize URL validator."""
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        super().__init__(url_pattern, re.IGNORECASE, error_message)

    def _get_default_error_message(self, value: Any, field_name: str) -> str:
        """Get default error message for URL validation."""
        return f"{field_name} '{value}' is not a valid URL"


class DateValidator(BaseValidator):
    """Validate dates."""

    def __init__(
        self,
        min_date: Optional[Union[date, datetime]] = None,
        max_date: Optional[Union[date, datetime]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Initialize date validator.

        Args:
            min_date: Minimum allowed date
            max_date: Maximum allowed date
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.min_date = min_date
        self.max_date = max_date

    def _validate(self, value: Any) -> bool:
        """Validate date."""
        if not isinstance(value, (date, datetime)):
            return False

        # Extract date part for comparison
        if isinstance(value, datetime):
            value_date = value.date()
        else:
            value_date = value

        if self.min_date is not None:
            min_date_part = self.min_date.date() if isinstance(self.min_date, datetime) else self.min_date
            if value_date < min_date_part:
                return False

        if self.max_date is not None:
            max_date_part = self.max_date.date() if isinstance(self.max_date, datetime) else self.max_date
            if value_date > max_date_part:
                return False

        return True


class ChoiceValidator(BaseValidator):
    """Validate against a list of choices."""

    def __init__(
        self,
        choices: List[Any],
        case_sensitive: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Initialize choice validator.

        Args:
            choices: List of valid choices
            case_sensitive: Whether string comparison is case sensitive
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.choices = choices
        self.case_sensitive = case_sensitive

    def _validate(self, value: Any) -> bool:
        """Validate choice."""
        if not self.case_sensitive and isinstance(value, str):
            return any(
                isinstance(choice, str) and value.lower() == choice.lower()
                for choice in self.choices
            )
        return value in self.choices

    def _get_default_error_message(self, value: Any, field_name: str) -> str:
        """Get default error message for choice validation."""
        choices_str = ", ".join(str(choice) for choice in self.choices)
        return f"{field_name} '{value}' is not a valid choice. Valid choices: {choices_str}"


class PathValidator(BaseValidator):
    """Validate file paths."""

    def __init__(
        self,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Initialize path validator.

        Args:
            must_exist: Whether path must exist
            must_be_file: Whether path must be a file
            must_be_dir: Whether path must be a directory
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.must_exist = must_exist
        self.must_be_file = must_be_file
        self.must_be_dir = must_be_dir

    def _validate(self, value: Any) -> bool:
        """Validate path."""
        try:
            path = Path(value)
        except (TypeError, ValueError):
            return False

        if self.must_exist and not path.exists():
            return False

        if self.must_be_file and (not path.exists() or not path.is_file()):
            return False

        if self.must_be_dir and (not path.exists() or not path.is_dir()):
            return False

        return True


class TradingSymbolValidator(BaseValidator):
    """Validate trading symbols."""

    def __init__(
        self,
        allowed_symbols: Optional[List[str]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Initialize trading symbol validator.

        Args:
            allowed_symbols: List of allowed symbols (None for any)
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.allowed_symbols = allowed_symbols

    def _validate(self, value: Any) -> bool:
        """Validate trading symbol."""
        if not isinstance(value, str):
            return False

        # Basic format check (letters and numbers only)
        if not re.match(r"^[A-Z0-9]+$", value.upper()):
            return False

        # Check against allowed symbols if specified
        if self.allowed_symbols is not None:
            return value.upper() in [s.upper() for s in self.allowed_symbols]

        return True

    def _get_default_error_message(self, value: Any, field_name: str) -> str:
        """Get default error message for symbol validation."""
        if self.allowed_symbols:
            return f"{field_name} '{value}' is not an allowed trading symbol"
        return f"{field_name} '{value}' is not a valid trading symbol format"


class PriceValidator(BaseValidator):
    """Validate trading prices."""

    def __init__(
        self,
        min_price: float = 0.00001,
        max_price: float = 1000000.0,
        decimal_places: int = 5,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Initialize price validator.

        Args:
            min_price: Minimum allowed price
            max_price: Maximum allowed price
            decimal_places: Maximum decimal places
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.min_price = min_price
        self.max_price = max_price
        self.decimal_places = decimal_places

    def _validate(self, value: Any) -> bool:
        """Validate price."""
        try:
            price = float(value)
        except (TypeError, ValueError):
            return False

        if price < self.min_price or price > self.max_price:
            return False

        # Check decimal places
        decimal_str = str(price)
        if "." in decimal_str:
            decimal_part = decimal_str.split(".")[1]
            if len(decimal_part) > self.decimal_places:
                return False

        return True


class VolumeValidator(BaseValidator):
    """Validate trading volumes."""

    def __init__(
        self,
        min_volume: float = 0.01,
        max_volume: float = 1000.0,
        volume_step: float = 0.01,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Initialize volume validator.

        Args:
            min_volume: Minimum allowed volume
            max_volume: Maximum allowed volume
            volume_step: Volume step size
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.volume_step = volume_step

    def _validate(self, value: Any) -> bool:
        """Validate volume."""
        try:
            volume = float(value)
        except (TypeError, ValueError):
            return False

        if volume < self.min_volume or volume > self.max_volume:
            return False

        # Check if volume is a multiple of volume step
        remainder = (volume - self.min_volume) % self.volume_step
        return abs(remainder) < 1e-6  # Account for floating point precision


def validate_all(
    data: Dict[str, Any],
    validators: Dict[str, List[BaseValidator]],
) -> ValidationResult:
    """
    Validate multiple fields using multiple validators.

    Args:
        data: Dictionary of field names to values
        validators: Dictionary of field names to validator lists

    Returns:
        Combined validation result
    """
    result = ValidationResult(True)

    for field_name, value in data.items():
        if field_name in validators:
            for validator in validators[field_name]:
                field_result = validator.validate(value, field_name)
                result.merge(field_result)

    return result


def create_validator_chain(*validators: BaseValidator) -> Callable[[Any, str], ValidationResult]:
    """
    Create a validator chain that runs multiple validators.

    Args:
        *validators: Validators to chain

    Returns:
        Validator function
    """
    def validate(value: Any, field_name: str = "value") -> ValidationResult:
        result = ValidationResult(True)
        for validator in validators:
            validator_result = validator.validate(value, field_name)
            result.merge(validator_result)
            # Stop on first error for efficiency
            if not validator_result.is_valid:
                break
        return result

    return validate