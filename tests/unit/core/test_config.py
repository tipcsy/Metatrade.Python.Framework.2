"""
Unit tests for configuration module.

Tests the configuration system including Settings classes, YAML loading,
validation, and environment-specific configurations.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.core.config import (
    ConfigLoader,
    ConfigurationError,
    Environment,
    Settings,
    ValidationError,
    load_settings,
    validate_settings,
)


class TestSettings:
    """Test Settings class functionality."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()

        assert settings.app_name == "MetaTrader Python Framework"
        assert settings.app_version == "2.0.0"
        assert settings.environment == Environment.DEVELOPMENT
        assert settings.debug is True

    def test_environment_methods(self):
        """Test environment checking methods."""
        # Test development
        dev_settings = Settings(environment=Environment.DEVELOPMENT)
        assert dev_settings.is_development()
        assert not dev_settings.is_testing()
        assert not dev_settings.is_production()

        # Test testing
        test_settings = Settings(environment=Environment.TESTING)
        assert not test_settings.is_development()
        assert test_settings.is_testing()
        assert not test_settings.is_production()

        # Test production
        prod_settings = Settings(environment=Environment.PRODUCTION)
        assert not prod_settings.is_development()
        assert not prod_settings.is_testing()
        assert prod_settings.is_production()

    def test_get_methods(self):
        """Test getter methods."""
        settings = Settings()

        assert isinstance(settings.get_database_url(), str)
        assert isinstance(settings.get_log_level(), str)
        assert isinstance(settings.get_config_dir(), Path)
        assert isinstance(settings.get_data_dir(), Path)
        assert isinstance(settings.get_logs_dir(), Path)

    def test_production_validation(self):
        """Test production-specific validation."""
        # Should raise error with default security settings in production
        with pytest.raises(ValidationError):
            Settings(
                environment=Environment.PRODUCTION,
                security={"encryption_key": "change-me-in-production"}
            )

    def test_nested_model_validation(self):
        """Test nested model validation."""
        # Test invalid MT5 timeout
        with pytest.raises(ValidationError):
            Settings(mt5={"timeout": 500})  # Too low

        # Test invalid database pool size
        with pytest.raises(ValidationError):
            Settings(database={"pool_size": 0})  # Too low

        # Test invalid risk parameters
        with pytest.raises(ValidationError):
            Settings(trading={"max_risk_per_trade": 1.5})  # > 100%


class TestConfigLoader:
    """Test ConfigLoader functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.loader = ConfigLoader(self.config_dir)

    def test_load_yaml_existing_file(self):
        """Test loading existing YAML file."""
        config_file = self.config_dir / "test.yaml"
        test_data = {"key": "value", "number": 42}

        with config_file.open("w") as f:
            yaml.safe_dump(test_data, f)

        result = self.loader.load_yaml(config_file)
        assert result == test_data

    def test_load_yaml_nonexistent_file(self):
        """Test loading non-existent YAML file."""
        result = self.loader.load_yaml(self.config_dir / "nonexistent.yaml")
        assert result == {}

    def test_load_yaml_invalid_yaml(self):
        """Test loading invalid YAML file."""
        config_file = self.config_dir / "invalid.yaml"
        config_file.write_text("invalid: yaml: content:")

        with pytest.raises(ConfigurationError):
            self.loader.load_yaml(config_file)

    def test_save_yaml(self):
        """Test saving YAML file."""
        config_file = self.config_dir / "save_test.yaml"
        test_data = {"key": "value", "nested": {"inner": "data"}}

        self.loader.save_yaml(test_data, config_file)

        assert config_file.exists()
        loaded_data = self.loader.load_yaml(config_file)
        assert loaded_data == test_data

    def test_merge_configs(self):
        """Test configuration merging."""
        base_config = {
            "app": {"name": "Base App", "version": "1.0"},
            "database": {"url": "base_url"},
        }

        override_config = {
            "app": {"version": "2.0"},
            "new_key": "new_value",
        }

        result = self.loader.merge_configs(base_config, override_config)

        expected = {
            "app": {"name": "Base App", "version": "2.0"},
            "database": {"url": "base_url"},
            "new_key": "new_value",
        }

        assert result == expected

    def test_load_environment_config(self):
        """Test loading environment-specific configuration."""
        # Create base config
        base_config = {"app": {"name": "Test App"}, "base_only": "value"}
        self.loader.save_yaml(base_config, self.config_dir / "base.yaml")

        # Create development config
        dev_config = {"app": {"debug": True}, "dev_only": "value"}
        self.loader.save_yaml(dev_config, self.config_dir / "development.yaml")

        result = self.loader.load_environment_config(Environment.DEVELOPMENT)

        assert result["app"]["name"] == "Test App"  # From base
        assert result["app"]["debug"] is True  # From development
        assert result["base_only"] == "value"  # From base
        assert result["dev_only"] == "value"  # From development

    def test_create_default_configs(self):
        """Test creation of default configuration files."""
        self.loader.create_default_configs()

        expected_files = [
            "base.yaml",
            "development.yaml",
            "testing.yaml",
            "production.yaml",
            "logging.yaml",
            "strategies.yaml",
            "indicators.yaml",
            "patterns.yaml",
        ]

        for filename in expected_files:
            config_file = self.config_dir / filename
            assert config_file.exists(), f"Config file {filename} not created"

            # Verify it's valid YAML
            data = self.loader.load_yaml(config_file)
            assert isinstance(data, dict), f"Config file {filename} doesn't contain valid YAML dict"


class TestConfigValidation:
    """Test configuration validation functionality."""

    def test_validate_settings_success(self):
        """Test successful settings validation."""
        settings = Settings(environment=Environment.TESTING)
        # Should not raise any exception
        validate_settings(settings)

    def test_validate_settings_failure(self):
        """Test settings validation failure."""
        # Create settings with invalid configuration
        with pytest.raises(ValidationError):
            settings = Settings(
                environment=Environment.PRODUCTION,
                security={"encryption_key": "change-me-in-production"}
            )
            validate_settings(settings)

    def test_mt5_path_validation(self):
        """Test MT5 path validation."""
        # Should work with None path
        settings = Settings(mt5={"path": None})
        validate_settings(settings)

        # Should fail with non-existent path
        with pytest.raises(ValidationError):
            Settings(mt5={"path": "/nonexistent/path/to/mt5.exe"})


class TestLoadSettings:
    """Test load_settings function."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def test_load_settings_default_environment(self):
        """Test loading settings with default environment."""
        with patch.dict(os.environ, {"ENVIRONMENT": "testing"}):
            settings = load_settings(config_dir=self.config_dir)
            assert settings.environment == Environment.TESTING

    def test_load_settings_specific_environment(self):
        """Test loading settings with specific environment."""
        settings = load_settings(
            environment=Environment.PRODUCTION,
            config_dir=self.config_dir
        )
        assert settings.environment == Environment.PRODUCTION

    def test_load_settings_with_env_vars(self):
        """Test loading settings with environment variables."""
        env_vars = {
            "APP_NAME": "Custom App Name",
            "DEBUG": "false",
            "DATABASE__URL": "sqlite:///custom.db",
            "LOGGING__LEVEL": "ERROR",
        }

        with patch.dict(os.environ, env_vars):
            settings = load_settings(config_dir=self.config_dir)

            assert settings.app_name == "Custom App Name"
            assert settings.debug is False
            assert settings.database.url == "sqlite:///custom.db"
            assert settings.logging.level.value == "ERROR"

    def test_load_settings_invalid_config(self):
        """Test loading settings with invalid configuration."""
        # Create invalid config file
        invalid_config = {"invalid": "structure"}
        config_file = self.config_dir / "testing.yaml"

        with config_file.open("w") as f:
            yaml.safe_dump(invalid_config, f)

        # Should still work as it falls back to defaults
        settings = load_settings(
            environment=Environment.TESTING,
            config_dir=self.config_dir
        )
        assert isinstance(settings, Settings)

    def test_load_settings_missing_config_dir(self):
        """Test loading settings with missing config directory."""
        missing_dir = self.temp_dir / "missing"

        # Should work and create the directory
        settings = load_settings(config_dir=missing_dir)
        assert isinstance(settings, Settings)
        assert missing_dir.exists()


class TestEnvironmentEnum:
    """Test Environment enum functionality."""

    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.PRODUCTION.value == "production"

    def test_environment_creation_from_string(self):
        """Test creating environment from string."""
        assert Environment("development") == Environment.DEVELOPMENT
        assert Environment("testing") == Environment.TESTING
        assert Environment("production") == Environment.PRODUCTION

    def test_environment_invalid_value(self):
        """Test creating environment with invalid value."""
        with pytest.raises(ValueError):
            Environment("invalid")


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration system."""

    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.temp_dir / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        # Create config loader and generate default configs
        loader = ConfigLoader(self.config_dir)
        loader.create_default_configs()

        # Load settings for each environment
        for env in [Environment.DEVELOPMENT, Environment.TESTING, Environment.PRODUCTION]:
            settings = load_settings(environment=env, config_dir=self.config_dir)

            assert isinstance(settings, Settings)
            assert settings.environment == env

            # Validate settings
            validate_settings(settings)

    def test_config_override_precedence(self):
        """Test configuration override precedence."""
        # Create base config
        base_config = {"app": {"name": "Base App"}, "test_value": "base"}
        loader = ConfigLoader(self.config_dir)
        loader.save_yaml(base_config, self.config_dir / "base.yaml")

        # Create environment config
        env_config = {"app": {"name": "Env App"}, "test_value": "env"}
        loader.save_yaml(env_config, self.config_dir / "development.yaml")

        # Create local config
        local_config = {"test_value": "local"}
        loader.save_yaml(local_config, self.config_dir / "local.yaml")

        # Load and verify precedence: local > env > base
        config_data = loader.load_environment_config(Environment.DEVELOPMENT)

        assert config_data["app"]["name"] == "Env App"  # env overrides base
        assert config_data["test_value"] == "local"  # local overrides env

    def test_config_with_environment_variables(self):
        """Test configuration with environment variable overrides."""
        env_vars = {
            "ENVIRONMENT": "testing",
            "APP_NAME": "Test Framework",
            "MT5__ENABLED": "false",
            "TRADING__ENABLED": "false",
            "DATABASE__URL": "sqlite:///test.db",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = load_settings(config_dir=self.config_dir)

            assert settings.environment == Environment.TESTING
            assert settings.app_name == "Test Framework"
            assert settings.mt5.enabled is False
            assert settings.trading.enabled is False
            assert "test.db" in settings.database.url