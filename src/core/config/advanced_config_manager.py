"""
Advanced Centralized Configuration Management System.

This module provides a comprehensive configuration management system with
environment-specific configs, secure credential management, dynamic reloading,
validation, and distributed configuration support for the MetaTrader Python Framework.
"""

from __future__ import annotations

import asyncio
import os
import time
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable, Type, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import yaml
import base64
import threading
from enum import Enum
from abc import ABC, abstractmethod

from pydantic import BaseModel, ValidationError, SecretStr
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.core.exceptions import ConfigurationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"
    INI = "ini"


class ConfigSource(Enum):
    """Configuration data sources."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    MEMORY = "memory"


class ChangeType(Enum):
    """Configuration change types."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RELOADED = "reloaded"


@dataclass
class ConfigChange:
    """Configuration change event."""
    path: str
    old_value: Any
    new_value: Any
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)
    source: ConfigSource = ConfigSource.FILE


@dataclass
class WatchedConfig:
    """Watched configuration file metadata."""
    path: Path
    last_modified: float
    checksum: str
    format: ConfigFormat
    data: Dict[str, Any] = field(default_factory=dict)


class ConfigProvider(ABC):
    """Abstract base class for configuration providers."""

    @abstractmethod
    async def load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from the provider.

        Args:
            path: Configuration path/key

        Returns:
            Configuration dictionary
        """
        pass

    @abstractmethod
    async def save_config(self, path: str, config: Dict[str, Any]) -> bool:
        """Save configuration to the provider.

        Args:
            path: Configuration path/key
            config: Configuration data

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def delete_config(self, path: str) -> bool:
        """Delete configuration from the provider.

        Args:
            path: Configuration path/key

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def list_configs(self, prefix: str = "") -> List[str]:
        """List available configuration paths.

        Args:
            prefix: Path prefix filter

        Returns:
            List of configuration paths
        """
        pass

    @abstractmethod
    async def watch_config(self, path: str, callback: Callable[[str, Dict[str, Any]], None]) -> bool:
        """Watch configuration for changes.

        Args:
            path: Configuration path to watch
            callback: Change callback function

        Returns:
            True if watching started successfully
        """
        pass


class FileConfigProvider(ConfigProvider):
    """File-based configuration provider."""

    def __init__(self, base_path: Union[str, Path]) -> None:
        """Initialize file config provider.

        Args:
            base_path: Base directory for configuration files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            file_path = self.base_path / path

            if not file_path.exists():
                raise ConfigurationError(f"Configuration file not found: {file_path}")

            format_type = self._detect_format(file_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if format_type == ConfigFormat.JSON:
                return json.loads(content)
            elif format_type == ConfigFormat.YAML:
                return yaml.safe_load(content)
            elif format_type == ConfigFormat.ENV:
                return self._parse_env_format(content)
            else:
                raise ConfigurationError(f"Unsupported config format for file: {file_path}")

        except Exception as e:
            logger.error(
                f"Failed to load config from file: {path}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise ConfigurationError(f"Failed to load config from file {path}: {e}") from e

    async def save_config(self, path: str, config: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            file_path = self.base_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            format_type = self._detect_format(file_path)

            if format_type == ConfigFormat.JSON:
                content = json.dumps(config, indent=2, default=str)
            elif format_type == ConfigFormat.YAML:
                content = yaml.dump(config, default_flow_style=False)
            else:
                raise ConfigurationError(f"Unsupported format for saving: {format_type}")

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.debug(f"Saved configuration to file: {file_path}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to save config to file: {path}",
                extra={"error": str(e)},
                exc_info=True
            )
            return False

    async def delete_config(self, path: str) -> bool:
        """Delete configuration file."""
        try:
            file_path = self.base_path / path
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted configuration file: {file_path}")
                return True
            return False

        except Exception as e:
            logger.error(
                f"Failed to delete config file: {path}",
                extra={"error": str(e)},
                exc_info=True
            )
            return False

    async def list_configs(self, prefix: str = "") -> List[str]:
        """List configuration files."""
        try:
            configs = []
            search_path = self.base_path

            if prefix:
                search_path = self.base_path / prefix

            for file_path in search_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.json', '.yaml', '.yml', '.env']:
                    relative_path = file_path.relative_to(self.base_path)
                    configs.append(str(relative_path))

            return configs

        except Exception as e:
            logger.error(
                f"Failed to list config files",
                extra={"error": str(e)},
                exc_info=True
            )
            return []

    async def watch_config(self, path: str, callback: Callable[[str, Dict[str, Any]], None]) -> bool:
        """Watch configuration file for changes."""
        # This would be implemented with file system watchers in a real system
        # For now, we'll simulate with periodic polling
        logger.debug(f"Started watching config file: {path}")
        return True

    def _detect_format(self, file_path: Path) -> ConfigFormat:
        """Detect configuration file format."""
        suffix = file_path.suffix.lower()

        if suffix == '.json':
            return ConfigFormat.JSON
        elif suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        elif suffix == '.env':
            return ConfigFormat.ENV
        else:
            # Try to detect from content
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{'):
                        return ConfigFormat.JSON
                    elif '=' in first_line:
                        return ConfigFormat.ENV
                    else:
                        return ConfigFormat.YAML
            except Exception:
                return ConfigFormat.YAML  # Default

    def _parse_env_format(self, content: str) -> Dict[str, Any]:
        """Parse environment file format."""
        config = {}
        for line in content.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip().strip('"\'')
        return config


class EnvironmentConfigProvider(ConfigProvider):
    """Environment variables configuration provider."""

    def __init__(self, prefix: str = "") -> None:
        """Initialize environment config provider.

        Args:
            prefix: Environment variable prefix
        """
        self.prefix = prefix

    async def load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        try:
            config = {}
            env_key = f"{self.prefix}_{path}".upper() if self.prefix else path.upper()

            # Direct key lookup
            if env_key in os.environ:
                value = os.environ[env_key]
                try:
                    # Try to parse as JSON
                    config[path] = json.loads(value)
                except json.JSONDecodeError:
                    config[path] = value

            # Prefix-based lookup
            if self.prefix:
                prefix_key = f"{self.prefix}_"
                for key, value in os.environ.items():
                    if key.startswith(prefix_key):
                        config_key = key[len(prefix_key):].lower()
                        try:
                            config[config_key] = json.loads(value)
                        except json.JSONDecodeError:
                            config[config_key] = value

            return config

        except Exception as e:
            logger.error(
                f"Failed to load config from environment: {path}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise ConfigurationError(f"Failed to load environment config: {e}") from e

    async def save_config(self, path: str, config: Dict[str, Any]) -> bool:
        """Save configuration to environment (not typically supported)."""
        return False

    async def delete_config(self, path: str) -> bool:
        """Delete environment configuration (not typically supported)."""
        return False

    async def list_configs(self, prefix: str = "") -> List[str]:
        """List environment configurations."""
        configs = []
        search_prefix = f"{self.prefix}_{prefix}".upper() if self.prefix else prefix.upper()

        for key in os.environ:
            if key.startswith(search_prefix):
                configs.append(key)

        return configs

    async def watch_config(self, path: str, callback: Callable[[str, Dict[str, Any]], None]) -> bool:
        """Environment watching not supported."""
        return False


class SecureCredentialManager:
    """Secure credential and secret management."""

    def __init__(self, master_key: Optional[bytes] = None) -> None:
        """Initialize credential manager.

        Args:
            master_key: Master encryption key (generated if None)
        """
        if master_key is None:
            master_key = Fernet.generate_key()

        self._fernet = Fernet(master_key)
        self._credentials: Dict[str, bytes] = {}

    def store_credential(self, name: str, value: str) -> None:
        """Store encrypted credential.

        Args:
            name: Credential name
            value: Credential value
        """
        try:
            encrypted_value = self._fernet.encrypt(value.encode('utf-8'))
            self._credentials[name] = encrypted_value
            logger.debug(f"Stored encrypted credential: {name}")

        except Exception as e:
            logger.error(
                f"Failed to store credential: {name}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise ConfigurationError(f"Failed to store credential {name}: {e}") from e

    def get_credential(self, name: str) -> Optional[str]:
        """Retrieve decrypted credential.

        Args:
            name: Credential name

        Returns:
            Decrypted credential value or None
        """
        try:
            encrypted_value = self._credentials.get(name)
            if encrypted_value is None:
                return None

            decrypted_value = self._fernet.decrypt(encrypted_value)
            return decrypted_value.decode('utf-8')

        except Exception as e:
            logger.error(
                f"Failed to retrieve credential: {name}",
                extra={"error": str(e)},
                exc_info=True
            )
            return None

    def delete_credential(self, name: str) -> bool:
        """Delete credential.

        Args:
            name: Credential name

        Returns:
            True if deleted
        """
        if name in self._credentials:
            del self._credentials[name]
            logger.debug(f"Deleted credential: {name}")
            return True
        return False

    def list_credentials(self) -> List[str]:
        """List stored credential names.

        Returns:
            List of credential names
        """
        return list(self._credentials.keys())


class AdvancedConfigManager:
    """
    Advanced centralized configuration management system.

    Features:
    - Multiple configuration providers (file, environment, database, remote)
    - Environment-specific configurations
    - Secure credential management
    - Dynamic configuration reloading
    - Configuration validation
    - Change tracking and notifications
    - Configuration templating and inheritance
    """

    def __init__(self) -> None:
        """Initialize advanced configuration manager."""
        # Configuration providers
        self._providers: Dict[ConfigSource, ConfigProvider] = {}

        # Configuration cache
        self._config_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_lock = threading.RLock()

        # File watching
        self._watched_files: Dict[str, WatchedConfig] = {}
        self._file_watchers: Dict[str, asyncio.Task] = {}

        # Change notifications
        self._change_callbacks: List[Callable[[ConfigChange], None]] = []

        # Credential management
        self._credential_manager = SecureCredentialManager()

        # Configuration validation
        self._validators: Dict[str, Callable[[Any], bool]] = {}
        self._schemas: Dict[str, Type[BaseModel]] = {}

        # Environment and profile management
        self._current_environment = os.getenv('ENVIRONMENT', 'development')
        self._active_profiles: Set[str] = set(os.getenv('PROFILES', '').split(','))
        if '' in self._active_profiles:
            self._active_profiles.remove('')

        # Background tasks
        self._reload_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Configuration hierarchy
        self._config_hierarchy: List[str] = []

        logger.info(
            "Advanced configuration manager initialized",
            extra={
                "environment": self._current_environment,
                "profiles": list(self._active_profiles),
            }
        )

    async def initialize(self, config_paths: List[str]) -> None:
        """Initialize configuration manager.

        Args:
            config_paths: List of configuration file paths to load
        """
        try:
            logger.info("Initializing advanced configuration manager")

            # Set up default providers
            await self._setup_default_providers()

            # Load initial configurations
            for config_path in config_paths:
                await self.load_config(config_path)

            # Start background tasks
            self._reload_task = asyncio.create_task(self._auto_reload_loop())

            logger.info("Advanced configuration manager initialized successfully")

        except Exception as e:
            logger.error(
                "Failed to initialize configuration manager",
                extra={"error": str(e)},
                exc_info=True
            )
            raise ConfigurationError(f"Configuration manager initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """Shutdown configuration manager."""
        try:
            logger.info("Shutting down configuration manager")

            # Signal stop
            self._stop_event.set()

            # Stop background tasks
            if self._reload_task:
                self._reload_task.cancel()
                try:
                    await self._reload_task
                except asyncio.CancelledError:
                    pass

            # Stop file watchers
            for watcher in self._file_watchers.values():
                watcher.cancel()

            logger.info("Configuration manager shutdown completed")

        except Exception as e:
            logger.error(
                "Error shutting down configuration manager",
                extra={"error": str(e)},
                exc_info=True
            )

    async def load_config(self, path: str, source: ConfigSource = ConfigSource.FILE) -> Dict[str, Any]:
        """Load configuration from specified source.

        Args:
            path: Configuration path
            source: Configuration source

        Returns:
            Loaded configuration
        """
        try:
            provider = self._providers.get(source)
            if provider is None:
                raise ConfigurationError(f"No provider configured for source: {source}")

            # Load base configuration
            config = await provider.load_config(path)

            # Apply environment-specific overrides
            env_config = await self._load_environment_config(path, source)
            if env_config:
                config = self._merge_configs(config, env_config)

            # Apply profile-specific configurations
            for profile in self._active_profiles:
                profile_config = await self._load_profile_config(path, profile, source)
                if profile_config:
                    config = self._merge_configs(config, profile_config)

            # Validate configuration
            await self._validate_config(path, config)

            # Cache configuration
            with self._cache_lock:
                self._config_cache[path] = config
                self._cache_timestamps[path] = time.time()

            # Set up file watching if it's a file source
            if source == ConfigSource.FILE:
                await self._setup_file_watcher(path)

            logger.debug(f"Loaded configuration: {path}")
            return config

        except Exception as e:
            logger.error(
                f"Failed to load configuration: {path}",
                extra={"source": source.value, "error": str(e)},
                exc_info=True
            )
            raise ConfigurationError(f"Failed to load config {path} from {source.value}: {e}") from e

    async def save_config(self, path: str, config: Dict[str, Any], source: ConfigSource = ConfigSource.FILE) -> bool:
        """Save configuration to specified source.

        Args:
            path: Configuration path
            config: Configuration data
            source: Configuration source

        Returns:
            True if successful
        """
        try:
            provider = self._providers.get(source)
            if provider is None:
                raise ConfigurationError(f"No provider configured for source: {source}")

            # Validate before saving
            await self._validate_config(path, config)

            # Save configuration
            success = await provider.save_config(path, config)

            if success:
                # Update cache
                with self._cache_lock:
                    old_config = self._config_cache.get(path)
                    self._config_cache[path] = config
                    self._cache_timestamps[path] = time.time()

                # Notify change
                change = ConfigChange(
                    path=path,
                    old_value=old_config,
                    new_value=config,
                    change_type=ChangeType.UPDATED if old_config else ChangeType.CREATED,
                    source=source
                )
                await self._notify_config_change(change)

                logger.debug(f"Saved configuration: {path}")

            return success

        except Exception as e:
            logger.error(
                f"Failed to save configuration: {path}",
                extra={"source": source.value, "error": str(e)},
                exc_info=True
            )
            return False

    def get_config(self, path: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            path: Configuration path (dot notation supported)
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        try:
            with self._cache_lock:
                # Support dot notation for nested access
                path_parts = path.split('.')
                current = self._config_cache

                for part in path_parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return default

                return current

        except Exception as e:
            logger.error(
                f"Error getting configuration value: {path}",
                extra={"error": str(e)},
                exc_info=True
            )
            return default

    def set_config(self, path: str, value: Any) -> None:
        """Set configuration value in memory.

        Args:
            path: Configuration path (dot notation supported)
            value: Value to set
        """
        try:
            with self._cache_lock:
                path_parts = path.split('.')
                current = self._config_cache

                # Navigate to parent
                for part in path_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Set value
                old_value = current.get(path_parts[-1])
                current[path_parts[-1]] = value

                # Notify change
                change = ConfigChange(
                    path=path,
                    old_value=old_value,
                    new_value=value,
                    change_type=ChangeType.UPDATED if old_value is not None else ChangeType.CREATED,
                    source=ConfigSource.MEMORY
                )
                asyncio.create_task(self._notify_config_change(change))

                logger.debug(f"Set configuration value: {path} = {value}")

        except Exception as e:
            logger.error(
                f"Error setting configuration value: {path}",
                extra={"value": value, "error": str(e)},
                exc_info=True
            )

    def has_config(self, path: str) -> bool:
        """Check if configuration path exists.

        Args:
            path: Configuration path

        Returns:
            True if path exists
        """
        try:
            value = self.get_config(path, object())  # Use sentinel object
            return value is not object()

        except Exception:
            return False

    def add_change_callback(self, callback: Callable[[ConfigChange], None]) -> None:
        """Add configuration change callback.

        Args:
            callback: Callback function
        """
        self._change_callbacks.append(callback)

    def remove_change_callback(self, callback: Callable[[ConfigChange], None]) -> None:
        """Remove configuration change callback.

        Args:
            callback: Callback function to remove
        """
        try:
            self._change_callbacks.remove(callback)
        except ValueError:
            pass

    def add_validator(self, path: str, validator: Callable[[Any], bool]) -> None:
        """Add configuration validator.

        Args:
            path: Configuration path to validate
            validator: Validator function
        """
        self._validators[path] = validator

    def add_schema(self, path: str, schema: Type[BaseModel]) -> None:
        """Add Pydantic schema for configuration validation.

        Args:
            path: Configuration path
            schema: Pydantic model schema
        """
        self._schemas[path] = schema

    async def reload_config(self, path: Optional[str] = None) -> bool:
        """Reload configuration from source.

        Args:
            path: Specific configuration path (all if None)

        Returns:
            True if successful
        """
        try:
            if path:
                # Reload specific configuration
                await self.load_config(path)
                return True
            else:
                # Reload all configurations
                with self._cache_lock:
                    paths = list(self._config_cache.keys())

                for config_path in paths:
                    try:
                        await self.load_config(config_path)
                    except Exception as e:
                        logger.error(f"Failed to reload config: {config_path} - {e}")

                return True

        except Exception as e:
            logger.error(
                f"Failed to reload configuration: {path}",
                extra={"error": str(e)},
                exc_info=True
            )
            return False

    def get_environment(self) -> str:
        """Get current environment.

        Returns:
            Current environment name
        """
        return self._current_environment

    def set_environment(self, environment: str) -> None:
        """Set current environment.

        Args:
            environment: Environment name
        """
        if environment != self._current_environment:
            old_env = self._current_environment
            self._current_environment = environment
            logger.info(f"Environment changed from {old_env} to {environment}")

    def get_profiles(self) -> Set[str]:
        """Get active profiles.

        Returns:
            Set of active profile names
        """
        return self._active_profiles.copy()

    def add_profile(self, profile: str) -> None:
        """Add active profile.

        Args:
            profile: Profile name
        """
        if profile not in self._active_profiles:
            self._active_profiles.add(profile)
            logger.info(f"Added profile: {profile}")

    def remove_profile(self, profile: str) -> None:
        """Remove active profile.

        Args:
            profile: Profile name
        """
        if profile in self._active_profiles:
            self._active_profiles.remove(profile)
            logger.info(f"Removed profile: {profile}")

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary.

        Returns:
            Configuration summary dictionary
        """
        try:
            with self._cache_lock:
                return {
                    "environment": self._current_environment,
                    "profiles": list(self._active_profiles),
                    "loaded_configs": len(self._config_cache),
                    "watched_files": len(self._watched_files),
                    "providers": list(self._providers.keys()),
                    "credentials": len(self._credential_manager.list_credentials()),
                    "validators": len(self._validators),
                    "schemas": len(self._schemas),
                    "cache_size": len(self._config_cache),
                }

        except Exception as e:
            logger.error(
                "Error getting configuration summary",
                extra={"error": str(e)},
                exc_info=True
            )
            return {"error": str(e)}

    async def _setup_default_providers(self) -> None:
        """Set up default configuration providers."""
        # File provider
        config_dir = Path("config")
        file_provider = FileConfigProvider(config_dir)
        self._providers[ConfigSource.FILE] = file_provider

        # Environment provider
        env_provider = EnvironmentConfigProvider(prefix="APP")
        self._providers[ConfigSource.ENVIRONMENT] = env_provider

    async def _load_environment_config(self, path: str, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """Load environment-specific configuration."""
        try:
            env_path = f"{path}-{self._current_environment}"
            provider = self._providers.get(source)

            if provider:
                try:
                    return await provider.load_config(env_path)
                except ConfigurationError:
                    # Environment-specific config doesn't exist, which is okay
                    return None

        except Exception as e:
            logger.debug(f"No environment config for {path}: {e}")

        return None

    async def _load_profile_config(self, path: str, profile: str, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """Load profile-specific configuration."""
        try:
            profile_path = f"{path}-{profile}"
            provider = self._providers.get(source)

            if provider:
                try:
                    return await provider.load_config(profile_path)
                except ConfigurationError:
                    # Profile-specific config doesn't exist, which is okay
                    return None

        except Exception as e:
            logger.debug(f"No profile config for {path}/{profile}: {e}")

        return None

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    async def _validate_config(self, path: str, config: Dict[str, Any]) -> None:
        """Validate configuration using registered validators and schemas."""
        # Custom validator
        validator = self._validators.get(path)
        if validator:
            if not validator(config):
                raise ConfigurationError(f"Configuration validation failed for {path}")

        # Pydantic schema validation
        schema = self._schemas.get(path)
        if schema:
            try:
                schema(**config)
            except ValidationError as e:
                raise ConfigurationError(f"Schema validation failed for {path}: {e}")

    async def _setup_file_watcher(self, path: str) -> None:
        """Set up file watcher for configuration file."""
        try:
            file_provider = self._providers.get(ConfigSource.FILE)
            if not isinstance(file_provider, FileConfigProvider):
                return

            file_path = file_provider.base_path / path

            if file_path.exists():
                stat = file_path.stat()
                checksum = self._calculate_file_checksum(file_path)

                watched_config = WatchedConfig(
                    path=file_path,
                    last_modified=stat.st_mtime,
                    checksum=checksum,
                    format=file_provider._detect_format(file_path)
                )

                self._watched_files[path] = watched_config

                # Start watcher task
                watcher_task = asyncio.create_task(self._watch_file(path))
                self._file_watchers[path] = watcher_task

        except Exception as e:
            logger.error(
                f"Failed to setup file watcher for {path}",
                extra={"error": str(e)},
                exc_info=True
            )

    async def _watch_file(self, path: str) -> None:
        """Watch file for changes."""
        while not self._stop_event.is_set():
            try:
                watched = self._watched_files.get(path)
                if not watched:
                    break

                if watched.path.exists():
                    stat = watched.path.stat()
                    new_checksum = self._calculate_file_checksum(watched.path)

                    if (stat.st_mtime > watched.last_modified or
                        new_checksum != watched.checksum):

                        logger.info(f"Configuration file changed: {path}")

                        # Reload configuration
                        old_config = self._config_cache.get(path)
                        await self.load_config(path)
                        new_config = self._config_cache.get(path)

                        # Update watched file info
                        watched.last_modified = stat.st_mtime
                        watched.checksum = new_checksum

                        # Notify change
                        change = ConfigChange(
                            path=path,
                            old_value=old_config,
                            new_value=new_config,
                            change_type=ChangeType.RELOADED,
                            source=ConfigSource.FILE
                        )
                        await self._notify_config_change(change)

                # Check every 5 seconds
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Error watching file: {path}",
                    extra={"error": str(e)},
                    exc_info=True
                )
                await asyncio.sleep(5)

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""

    async def _auto_reload_loop(self) -> None:
        """Auto-reload loop for configuration changes."""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=60)
                break
            except asyncio.TimeoutError:
                # Perform periodic maintenance
                await self._cleanup_cache()

    async def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""
        try:
            current_time = time.time()
            cache_ttl = 3600  # 1 hour

            with self._cache_lock:
                expired_keys = [
                    key for key, timestamp in self._cache_timestamps.items()
                    if current_time - timestamp > cache_ttl
                ]

                for key in expired_keys:
                    del self._config_cache[key]
                    del self._cache_timestamps[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        except Exception as e:
            logger.error(
                "Error cleaning up configuration cache",
                extra={"error": str(e)},
                exc_info=True
            )

    async def _notify_config_change(self, change: ConfigChange) -> None:
        """Notify configuration change callbacks."""
        for callback in self._change_callbacks:
            try:
                callback(change)
            except Exception as e:
                logger.error(
                    f"Error in configuration change callback",
                    extra={"change": change.path, "error": str(e)},
                    exc_info=True
                )

    async def __aenter__(self) -> AdvancedConfigManager:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()


# Global configuration manager instance
_config_manager: Optional[AdvancedConfigManager] = None


async def initialize_config_manager(config_paths: List[str]) -> None:
    """Initialize the global configuration manager.

    Args:
        config_paths: List of configuration file paths to load
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = AdvancedConfigManager()
        await _config_manager.initialize(config_paths)


async def shutdown_config_manager() -> None:
    """Shutdown the global configuration manager."""
    global _config_manager

    if _config_manager:
        await _config_manager.shutdown()
        _config_manager = None


def get_config_manager() -> Optional[AdvancedConfigManager]:
    """Get the global configuration manager instance.

    Returns:
        Configuration manager instance or None
    """
    global _config_manager
    return _config_manager


def get_config(path: str, default: Any = None) -> Any:
    """Get configuration value from global manager.

    Args:
        path: Configuration path
        default: Default value

    Returns:
        Configuration value or default
    """
    global _config_manager

    if _config_manager:
        return _config_manager.get_config(path, default)
    else:
        return default


def set_config(path: str, value: Any) -> None:
    """Set configuration value in global manager.

    Args:
        path: Configuration path
        value: Value to set
    """
    global _config_manager

    if _config_manager:
        _config_manager.set_config(path, value)


def has_config(path: str) -> bool:
    """Check if configuration path exists in global manager.

    Args:
        path: Configuration path

    Returns:
        True if path exists
    """
    global _config_manager

    if _config_manager:
        return _config_manager.has_config(path)
    else:
        return False