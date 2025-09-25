"""
Base exception classes for MetaTrader Python Framework.

This module defines the exception hierarchy with base classes that provide
structured error handling, logging integration, and context preservation.
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union


class BaseFrameworkError(Exception):
    """
    Base exception class for all framework errors.

    This class provides common functionality for all framework exceptions
    including error codes, context preservation, and structured logging.
    """

    error_code: str = "FRAMEWORK_ERROR"
    error_category: str = "general"
    severity: str = "error"  # debug, info, warning, error, critical

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
    ) -> None:
        """
        Initialize base framework error.

        Args:
            message: Detailed error message for developers
            error_code: Specific error code for programmatic handling
            context: Additional context information
            cause: Original exception that caused this error
            user_message: User-friendly error message
            suggestion: Suggested solution or next steps
        """
        super().__init__(message)

        self.error_code = error_code or self.__class__.error_code
        self.context = context or {}
        self.cause = cause
        self.user_message = user_message or message
        self.suggestion = suggestion
        self.timestamp = datetime.utcnow()

        # Capture stack trace
        self.stack_trace = traceback.format_stack()[:-1]

        # Set the cause if provided
        if cause:
            self.__cause__ = cause

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for structured logging.

        Returns:
            Dictionary representation of the exception
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "error_category": self.error_category,
            "severity": self.severity,
            "message": str(self),
            "user_message": self.user_message,
            "suggestion": self.suggestion,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
            "cause_type": self.cause.__class__.__name__ if self.cause else None,
        }

    def add_context(self, key: str, value: Any) -> None:
        """
        Add context information to the exception.

        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value

    def with_context(self, **kwargs: Any) -> BaseFrameworkError:
        """
        Add context and return self for method chaining.

        Args:
            **kwargs: Context key-value pairs

        Returns:
            Self for method chaining
        """
        self.context.update(kwargs)
        return self

    def log_error(self, logger: Optional[Any] = None) -> None:
        """
        Log the error with appropriate level and context.

        Args:
            logger: Logger instance to use
        """
        if logger is None:
            # Import here to avoid circular imports
            from ..logging import get_logger
            logger = get_logger(__name__)

        log_data = self.to_dict()

        if self.severity == "critical":
            logger.critical(str(self), **log_data)
        elif self.severity == "error":
            logger.error(str(self), **log_data)
        elif self.severity == "warning":
            logger.warning(str(self), **log_data)
        else:
            logger.info(str(self), **log_data)

    def __str__(self) -> str:
        """Return string representation of the exception."""
        message = super().__str__()
        if self.error_code != "FRAMEWORK_ERROR":
            message = f"[{self.error_code}] {message}"
        return message

    def __repr__(self) -> str:
        """Return detailed representation of the exception."""
        return (
            f"{self.__class__.__name__}("
            f"message='{str(self)}', "
            f"error_code='{self.error_code}', "
            f"context={self.context})"
        )


class ConfigurationError(BaseFrameworkError):
    """Exception raised for configuration-related errors."""

    error_code = "CONFIG_ERROR"
    error_category = "configuration"
    severity = "error"


class ValidationError(BaseFrameworkError):
    """Exception raised for validation errors."""

    error_code = "VALIDATION_ERROR"
    error_category = "validation"
    severity = "error"

    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected_type: Optional[Type] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize validation error.

        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            expected_type: Expected type or format
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if field:
            self.add_context("field", field)
        if value is not None:
            self.add_context("value", str(value)[:100])  # Truncate long values
        if expected_type:
            self.add_context("expected_type", expected_type.__name__)


class InitializationError(BaseFrameworkError):
    """Exception raised during component initialization."""

    error_code = "INIT_ERROR"
    error_category = "initialization"
    severity = "critical"


class DependencyError(BaseFrameworkError):
    """Exception raised for missing or incompatible dependencies."""

    error_code = "DEPENDENCY_ERROR"
    error_category = "dependencies"
    severity = "error"

    def __init__(
        self,
        message: str,
        *,
        dependency: Optional[str] = None,
        required_version: Optional[str] = None,
        current_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize dependency error.

        Args:
            message: Error message
            dependency: Name of the missing or incompatible dependency
            required_version: Required version
            current_version: Current version (if any)
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if dependency:
            self.add_context("dependency", dependency)
        if required_version:
            self.add_context("required_version", required_version)
        if current_version:
            self.add_context("current_version", current_version)


class DatabaseError(BaseFrameworkError):
    """Exception raised for database-related errors."""

    error_code = "DATABASE_ERROR"
    error_category = "database"
    severity = "error"

    def __init__(
        self,
        message: str,
        *,
        database_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize database error.

        Args:
            message: Error message
            database_name: Name of the database
            operation: Database operation that failed
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if database_name:
            self.add_context("database_name", database_name)
        if operation:
            self.add_context("operation", operation)


class NotFoundError(BaseFrameworkError):
    """Exception raised when a requested resource is not found."""

    error_code = "NOT_FOUND_ERROR"
    error_category = "not_found"
    severity = "error"

    def __init__(
        self,
        message: str,
        *,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize not found error.

        Args:
            message: Error message
            resource_type: Type of resource that was not found
            resource_id: ID of the resource that was not found
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if resource_type:
            self.add_context("resource_type", resource_type)
        if resource_id:
            self.add_context("resource_id", resource_id)


class BusinessLogicError(BaseFrameworkError):
    """Exception raised for business logic violations."""

    error_code = "BUSINESS_LOGIC_ERROR"
    error_category = "business_logic"
    severity = "error"

    def __init__(
        self,
        message: str,
        *,
        operation: Optional[str] = None,
        violation_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize business logic error.

        Args:
            message: Error message
            operation: Business operation that failed
            violation_type: Type of business rule violation
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if operation:
            self.add_context("operation", operation)
        if violation_type:
            self.add_context("violation_type", violation_type)


class SecurityError(BaseFrameworkError):
    """Exception raised for security-related issues."""

    error_code = "SECURITY_ERROR"
    error_category = "security"
    severity = "critical"


class RateLimitError(BaseFrameworkError):
    """Exception raised when rate limits are exceeded."""

    error_code = "RATE_LIMIT_ERROR"
    error_category = "rate_limit"
    severity = "warning"

    def __init__(
        self,
        message: str,
        *,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        retry_after: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize rate limit error.

        Args:
            message: Error message
            limit: Rate limit threshold
            window: Time window in seconds
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if limit:
            self.add_context("limit", limit)
        if window:
            self.add_context("window", window)
        if retry_after:
            self.add_context("retry_after", retry_after)


class TimeoutError(BaseFrameworkError):
    """Exception raised when operations timeout."""

    error_code = "TIMEOUT_ERROR"
    error_category = "timeout"
    severity = "warning"

    def __init__(
        self,
        message: str,
        *,
        timeout: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize timeout error.

        Args:
            message: Error message
            timeout: Timeout value in seconds
            operation: Operation that timed out
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if timeout:
            self.add_context("timeout", timeout)
        if operation:
            self.add_context("operation", operation)


class RetryableError(BaseFrameworkError):
    """Exception for errors that can be retried."""

    error_code = "RETRYABLE_ERROR"
    error_category = "retryable"
    severity = "warning"

    def __init__(
        self,
        message: str,
        *,
        max_retries: Optional[int] = None,
        retry_count: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize retryable error.

        Args:
            message: Error message
            max_retries: Maximum number of retries
            retry_count: Current retry count
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)

        if max_retries:
            self.add_context("max_retries", max_retries)
        self.add_context("retry_count", retry_count)

    def can_retry(self, max_retries: int) -> bool:
        """
        Check if the operation can be retried.

        Args:
            max_retries: Maximum number of retries allowed

        Returns:
            True if can retry, False otherwise
        """
        return self.context.get("retry_count", 0) < max_retries


class FrameworkWarning(UserWarning):
    """Base warning class for framework warnings."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize framework warning.

        Args:
            message: Warning message
            category: Warning category
            context: Additional context
        """
        super().__init__(message)
        self.category = category
        self.context = context or {}


def handle_exception(
    exception: Exception,
    logger: Optional[Any] = None,
    reraise: bool = True,
) -> Optional[BaseFrameworkError]:
    """
    Handle any exception and convert to framework exception if needed.

    Args:
        exception: Exception to handle
        logger: Logger to use for logging
        reraise: Whether to reraise the exception

    Returns:
        Framework exception if not reraised
    """
    if isinstance(exception, BaseFrameworkError):
        framework_error = exception
    else:
        framework_error = BaseFrameworkError(
            message=str(exception),
            cause=exception,
        )

    # Log the error
    if logger:
        framework_error.log_error(logger)

    if reraise:
        raise framework_error from exception

    return framework_error


def create_error_context() -> Dict[str, Any]:
    """
    Create error context with system information.

    Returns:
        Dictionary with system context information
    """
    frame = sys._getframe(1)
    return {
        "python_version": sys.version,
        "platform": sys.platform,
        "file": frame.f_code.co_filename,
        "function": frame.f_code.co_name,
        "line": frame.f_lineno,
        "local_vars": {
            k: str(v)[:100] for k, v in frame.f_locals.items()
            if not k.startswith("_") and not callable(v)
        },
    }