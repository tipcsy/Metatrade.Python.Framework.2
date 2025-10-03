# Shared Configuration Utilities
# Configuration loading and management

import json
import os
from typing import Dict, Any


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file

    Args:
        config_path: Path to config.json file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def get_service_config(service_name: str, config_path: str = "config.json") -> Dict[str, Any]:
    """
    Get configuration for a specific service

    Args:
        service_name: Name of the service
        config_path: Path to config.json file

    Returns:
        Service-specific configuration
    """
    config = load_config(config_path)

    if "services" not in config or service_name not in config["services"]:
        raise ValueError(f"Service '{service_name}' not found in configuration")

    return config["services"][service_name]
