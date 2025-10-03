# Shared Utilities
# Common utility functions used across all services

import logging
from datetime import datetime


def setup_logging(service_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration for a service

    Args:
        service_name: Name of the service
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(service_name)
    return logger


def get_timestamp_ms() -> int:
    """
    Get current timestamp in milliseconds

    Returns:
        Current timestamp in milliseconds
    """
    return int(datetime.now().timestamp() * 1000)


def format_response(success: bool, data=None, error: str = None, message: str = None):
    """
    Format standard API response

    Args:
        success: Whether the operation was successful
        data: Response data
        error: Error message if any
        message: Success message if any

    Returns:
        Formatted response dictionary
    """
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat()
    }

    if data is not None:
        response["data"] = data

    if message:
        response["message"] = message

    if error:
        response["error"] = error

    return response
