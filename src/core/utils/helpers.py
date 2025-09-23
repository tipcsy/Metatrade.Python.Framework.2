"""
General helper utilities for MetaTrader Python Framework.

This module provides various utility functions for common operations like
data conversion, file operations, string manipulation, and mathematical calculations.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import secrets
import string
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, TypeVar, Union

T = TypeVar("T")


# Date and Time Utilities

def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """
    Convert timestamp to datetime.

    Args:
        timestamp: Unix timestamp

    Returns:
        Datetime object in UTC
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> float:
    """
    Convert datetime to timestamp.

    Args:
        dt: Datetime object

    Returns:
        Unix timestamp
    """
    return dt.timestamp()


def format_datetime(dt: datetime, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime as string.

    Args:
        dt: Datetime object
        format_string: Format string

    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_string)


def parse_datetime(date_string: str, format_string: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    Parse datetime from string.

    Args:
        date_string: Date string
        format_string: Format string

    Returns:
        Parsed datetime object
    """
    return datetime.strptime(date_string, format_string)


# String Utilities

def generate_random_string(length: int = 32, use_uppercase: bool = True, use_lowercase: bool = True, use_digits: bool = True) -> str:
    """
    Generate a random string.

    Args:
        length: Length of the string
        use_uppercase: Include uppercase letters
        use_lowercase: Include lowercase letters
        use_digits: Include digits

    Returns:
        Random string
    """
    chars = ""
    if use_uppercase:
        chars += string.ascii_uppercase
    if use_lowercase:
        chars += string.ascii_lowercase
    if use_digits:
        chars += string.digits

    if not chars:
        raise ValueError("At least one character type must be enabled")

    return "".join(secrets.choice(chars) for _ in range(length))


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Sanitize filename by replacing invalid characters.

    Args:
        filename: Original filename
        replacement: Character to replace invalid chars with

    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, replacement)
    return filename


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text

    truncate_length = max_length - len(suffix)
    if truncate_length <= 0:
        return suffix[:max_length]

    return text[:truncate_length] + suffix


def camel_to_snake(text: str) -> str:
    """
    Convert camelCase to snake_case.

    Args:
        text: CamelCase string

    Returns:
        snake_case string
    """
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(text: str) -> str:
    """
    Convert snake_case to camelCase.

    Args:
        text: snake_case string

    Returns:
        camelCase string
    """
    components = text.split('_')
    return components[0] + ''.join(word.capitalize() for word in components[1:])


# Numeric Utilities

def round_decimal(value: Union[float, Decimal], decimal_places: int = 2) -> Decimal:
    """
    Round decimal to specified places.

    Args:
        value: Value to round
        decimal_places: Number of decimal places

    Returns:
        Rounded decimal
    """
    if not isinstance(value, Decimal):
        value = Decimal(str(value))

    places = Decimal('0.1') ** decimal_places
    return value.quantize(places, rounding=ROUND_HALF_UP)


def calculate_percentage(part: Union[int, float], total: Union[int, float]) -> float:
    """
    Calculate percentage.

    Args:
        part: Part value
        total: Total value

    Returns:
        Percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (part / total) * 100


def clamp(value: Union[int, float], min_value: Union[int, float], max_value: Union[int, float]) -> Union[int, float]:
    """
    Clamp value between min and max.

    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value

    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def is_number(value: Any) -> bool:
    """
    Check if value is a number.

    Args:
        value: Value to check

    Returns:
        True if number, False otherwise
    """
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


# Collection Utilities

def chunk_list(lst: List[T], chunk_size: int) -> Generator[List[T], None, None]:
    """
    Split list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Yields:
        List chunks
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def flatten_list(nested_list: List[Union[T, List]]) -> List[T]:
    """
    Flatten nested list.

    Args:
        nested_list: Nested list

    Returns:
        Flattened list
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def remove_duplicates(lst: List[T], preserve_order: bool = True) -> List[T]:
    """
    Remove duplicates from list.

    Args:
        lst: List with potential duplicates
        preserve_order: Whether to preserve order

    Returns:
        List without duplicates
    """
    if preserve_order:
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        return list(set(lst))


def group_by(lst: List[T], key_func: callable) -> Dict[Any, List[T]]:
    """
    Group list items by key function.

    Args:
        lst: List to group
        key_func: Function to extract grouping key

    Returns:
        Dictionary of grouped items
    """
    groups = {}
    for item in lst:
        key = key_func(item)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups


# File and Path Utilities

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def file_exists(path: Union[str, Path]) -> bool:
    """
    Check if file exists.

    Args:
        path: File path

    Returns:
        True if exists, False otherwise
    """
    return Path(path).is_file()


def directory_exists(path: Union[str, Path]) -> bool:
    """
    Check if directory exists.

    Args:
        path: Directory path

    Returns:
        True if exists, False otherwise
    """
    return Path(path).is_dir()


def get_file_size(path: Union[str, Path]) -> int:
    """
    Get file size in bytes.

    Args:
        path: File path

    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size


def get_file_modification_time(path: Union[str, Path]) -> datetime:
    """
    Get file modification time.

    Args:
        path: File path

    Returns:
        Modification time as datetime
    """
    timestamp = Path(path).stat().st_mtime
    return timestamp_to_datetime(timestamp)


def read_text_file(path: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    Read text file contents.

    Args:
        path: File path
        encoding: File encoding

    Returns:
        File contents
    """
    return Path(path).read_text(encoding=encoding)


def write_text_file(path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
    """
    Write text to file.

    Args:
        path: File path
        content: Content to write
        encoding: File encoding
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(content, encoding=encoding)


def backup_file(path: Union[str, Path], backup_suffix: str = ".bak") -> Path:
    """
    Create backup of file.

    Args:
        path: File path
        backup_suffix: Backup file suffix

    Returns:
        Backup file path
    """
    path_obj = Path(path)
    backup_path = path_obj.with_suffix(path_obj.suffix + backup_suffix)

    if path_obj.exists():
        import shutil
        shutil.copy2(path_obj, backup_path)

    return backup_path


# JSON Utilities

def load_json(path: Union[str, Path], default: Any = None) -> Any:
    """
    Load JSON from file.

    Args:
        path: File path
        default: Default value if file doesn't exist

    Returns:
        Parsed JSON data
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return default

    try:
        with path_obj.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data as JSON file.

    Args:
        data: Data to save
        path: File path
        indent: JSON indentation
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with path_obj.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def json_serialize(obj: Any) -> str:
    """
    Serialize object to JSON string.

    Args:
        obj: Object to serialize

    Returns:
        JSON string
    """
    def default_serializer(o):
        """Handle special types."""
        if isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, Decimal):
            return float(o)
        elif hasattr(o, '__dict__'):
            return o.__dict__
        else:
            return str(o)

    return json.dumps(obj, default=default_serializer, ensure_ascii=False)


# Pickle Utilities

def save_pickle(data: Any, path: Union[str, Path]) -> None:
    """
    Save data using pickle.

    Args:
        data: Data to save
        path: File path
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with path_obj.open('wb') as f:
        pickle.dump(data, f)


def load_pickle(path: Union[str, Path], default: Any = None) -> Any:
    """
    Load data from pickle file.

    Args:
        path: File path
        default: Default value if file doesn't exist

    Returns:
        Unpickled data
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return default

    try:
        with path_obj.open('rb') as f:
            return pickle.load(f)
    except (pickle.PickleError, OSError):
        return default


# Hash Utilities

def calculate_file_hash(path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Calculate file hash.

    Args:
        path: File path
        algorithm: Hash algorithm

    Returns:
        File hash as hex string
    """
    hash_obj = hashlib.new(algorithm)

    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def calculate_string_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Calculate string hash.

    Args:
        text: Text to hash
        algorithm: Hash algorithm

    Returns:
        Hash as hex string
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()


# System Utilities

def get_python_version() -> str:
    """Get Python version string."""
    return sys.version


def get_platform_info() -> Dict[str, str]:
    """
    Get platform information.

    Returns:
        Dictionary with platform info
    """
    import platform
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }


def get_memory_usage() -> Dict[str, int]:
    """
    Get current memory usage.

    Returns:
        Dictionary with memory info in bytes
    """
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "rss": memory_info.rss,
        "vms": memory_info.vms,
        "percent": process.memory_percent(),
    }


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes as human readable string.

    Args:
        bytes_value: Bytes value

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


# Environment Utilities

def get_env_var(name: str, default: Optional[str] = None, cast_type: type = str) -> Any:
    """
    Get environment variable with type casting.

    Args:
        name: Environment variable name
        default: Default value
        cast_type: Type to cast to

    Returns:
        Environment variable value
    """
    value = os.getenv(name, default)

    if value is None:
        return None

    if cast_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    elif cast_type in (int, float):
        try:
            return cast_type(value)
        except ValueError:
            return default
    else:
        return cast_type(value)


def set_env_var(name: str, value: Any) -> None:
    """
    Set environment variable.

    Args:
        name: Environment variable name
        value: Value to set
    """
    os.environ[name] = str(value)


# Trading Utilities

def normalize_symbol(symbol: str) -> str:
    """
    Normalize trading symbol format.

    Args:
        symbol: Trading symbol

    Returns:
        Normalized symbol
    """
    return symbol.upper().strip()


def calculate_pip_value(symbol: str, lot_size: float = 1.0) -> float:
    """
    Calculate pip value for a trading symbol.

    Args:
        symbol: Trading symbol
        lot_size: Lot size

    Returns:
        Pip value
    """
    # This is a simplified calculation
    # In practice, this would depend on the symbol specifications
    if "JPY" in symbol:
        return 0.01 * lot_size
    else:
        return 0.0001 * lot_size


def format_price(price: float, decimal_places: int = 5) -> str:
    """
    Format price for display.

    Args:
        price: Price value
        decimal_places: Number of decimal places

    Returns:
        Formatted price string
    """
    return f"{price:.{decimal_places}f}"


def safe_divide(dividend: Union[int, float], divisor: Union[int, float], default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safe division that handles division by zero.

    Args:
        dividend: Dividend
        divisor: Divisor
        default: Default value for division by zero

    Returns:
        Division result or default
    """
    if divisor == 0:
        return default
    return dividend / divisor