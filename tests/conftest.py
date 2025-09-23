"""
Pytest configuration and fixtures for MetaTrader Python Framework.

This module provides shared fixtures and configuration for all tests,
including test database setup, mock services, and utility fixtures.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import Environment, Settings, load_settings
from src.core.logging import LoggerFactory
from src.core.exceptions import BaseFrameworkError


# Test Environment Setup

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["LOG_LEVEL"] = "WARNING"
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["MT5_ENABLED"] = "false"
    os.environ["TRADING_ENABLED"] = "false"
    os.environ["API_ENABLED"] = "false"
    os.environ["GUI_ENABLED"] = "false"
    os.environ["NOTIFICATIONS_ENABLED"] = "false"
    os.environ["BACKUP_ENABLED"] = "false"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Configuration Fixtures

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_config_dir(temp_dir: Path) -> Path:
    """Create temporary config directory."""
    config_dir = temp_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def test_data_dir(temp_dir: Path) -> Path:
    """Create temporary data directory."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def test_logs_dir(temp_dir: Path) -> Path:
    """Create temporary logs directory."""
    logs_dir = temp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


@pytest.fixture
def test_settings(test_config_dir: Path, test_data_dir: Path, test_logs_dir: Path) -> Settings:
    """Create test settings."""
    # Override paths to use temporary directories
    with patch.dict(os.environ, {
        "DATABASE_URL": f"sqlite:///{test_data_dir}/test.db",
        "LOG_FILE_PATH": str(test_logs_dir / "test.log"),
    }):
        settings = load_settings(Environment.TESTING, test_config_dir)

        # Ensure test-specific overrides
        settings.environment = Environment.TESTING
        settings.debug = False
        settings.database.url = f"sqlite:///{test_data_dir}/test.db"
        settings.logging.file_path = test_logs_dir / "test.log"

        return settings


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for unit tests."""
    return Settings(
        environment=Environment.TESTING,
        debug=False,
    )


# Database Fixtures

@pytest.fixture
def test_database_url(test_data_dir: Path) -> str:
    """Get test database URL."""
    return f"sqlite:///{test_data_dir}/test.db"


@pytest.fixture
def test_database_engine(test_database_url: str):
    """Create test database engine."""
    engine = create_engine(test_database_url, echo=False)
    yield engine
    engine.dispose()


@pytest.fixture
def test_database_session(test_database_engine):
    """Create test database session."""
    SessionLocal = sessionmaker(bind=test_database_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# Logging Fixtures

@pytest.fixture
def test_logger(test_settings: Settings):
    """Create test logger."""
    LoggerFactory.reset()  # Reset any previous configuration
    LoggerFactory.initialize(test_settings)
    logger = LoggerFactory.get_logger("test")
    yield logger
    LoggerFactory.reset()


@pytest.fixture
def capture_logs(test_logger):
    """Capture log messages for testing."""
    from unittest.mock import MagicMock

    # Mock the logger to capture calls
    original_info = test_logger.info
    original_warning = test_logger.warning
    original_error = test_logger.error
    original_debug = test_logger.debug

    captured_logs = {
        "info": [],
        "warning": [],
        "error": [],
        "debug": [],
    }

    def capture_info(*args, **kwargs):
        captured_logs["info"].append((args, kwargs))
        return original_info(*args, **kwargs)

    def capture_warning(*args, **kwargs):
        captured_logs["warning"].append((args, kwargs))
        return original_warning(*args, **kwargs)

    def capture_error(*args, **kwargs):
        captured_logs["error"].append((args, kwargs))
        return original_error(*args, **kwargs)

    def capture_debug(*args, **kwargs):
        captured_logs["debug"].append((args, kwargs))
        return original_debug(*args, **kwargs)

    test_logger.info = capture_info
    test_logger.warning = capture_warning
    test_logger.error = capture_error
    test_logger.debug = capture_debug

    yield captured_logs

    # Restore original methods
    test_logger.info = original_info
    test_logger.warning = original_warning
    test_logger.error = original_error
    test_logger.debug = original_debug


# Mock Fixtures

@pytest.fixture
def mock_mt5():
    """Mock MetaTrader 5 module."""
    with patch("MetaTrader5") as mock:
        # Configure mock to simulate MT5 behavior
        mock.initialize.return_value = True
        mock.login.return_value = True
        mock.account_info.return_value = {
            "login": 123456,
            "server": "TestServer",
            "balance": 10000.0,
            "equity": 10000.0,
            "margin": 0.0,
            "free_margin": 10000.0,
        }
        mock.symbols_get.return_value = [
            {"name": "EURUSD", "digits": 5, "trade_contract_size": 100000},
            {"name": "GBPUSD", "digits": 5, "trade_contract_size": 100000},
        ]
        mock.copy_rates_from_pos.return_value = []
        mock.copy_ticks_from.return_value = []
        mock.order_send.return_value = {"retcode": 10009}  # TRADE_RETCODE_DONE
        mock.positions_get.return_value = []
        mock.orders_get.return_value = []
        mock.shutdown.return_value = None

        yield mock


@pytest.fixture
def mock_file_system(temp_dir: Path):
    """Mock file system operations to use temp directory."""
    original_cwd = Path.cwd()

    def mock_path_resolve(self):
        """Mock Path.resolve to work with temp directory."""
        if self.is_absolute():
            return self
        return temp_dir / self

    with patch.object(Path, "resolve", mock_path_resolve):
        yield temp_dir


# Test Data Fixtures

@pytest.fixture
def sample_ohlc_data() -> list:
    """Generate sample OHLC data for testing."""
    import random
    from datetime import datetime, timedelta

    data = []
    base_price = 1.2000
    current_time = datetime.now()

    for i in range(100):
        time = current_time - timedelta(minutes=i)

        # Generate realistic OHLC data
        open_price = base_price + random.uniform(-0.01, 0.01)
        high_price = open_price + random.uniform(0, 0.005)
        low_price = open_price - random.uniform(0, 0.005)
        close_price = open_price + random.uniform(-0.003, 0.003)
        volume = random.randint(100, 1000)

        data.append({
            "time": time,
            "open": round(open_price, 5),
            "high": round(high_price, 5),
            "low": round(low_price, 5),
            "close": round(close_price, 5),
            "volume": volume,
        })

        base_price = close_price  # Use close as next base

    return list(reversed(data))  # Chronological order


@pytest.fixture
def sample_tick_data() -> list:
    """Generate sample tick data for testing."""
    import random
    from datetime import datetime, timedelta

    data = []
    base_price = 1.2000
    current_time = datetime.now()

    for i in range(1000):
        time = current_time - timedelta(seconds=i)
        bid = base_price + random.uniform(-0.0005, 0.0005)
        ask = bid + random.uniform(0.0001, 0.0003)

        data.append({
            "time": time,
            "bid": round(bid, 5),
            "ask": round(ask, 5),
            "volume": random.randint(1, 10),
        })

    return list(reversed(data))  # Chronological order


@pytest.fixture
def sample_trading_symbols() -> list:
    """Get sample trading symbols for testing."""
    return [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
        "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "EURCHF"
    ]


# Exception Testing Fixtures

@pytest.fixture
def sample_exception() -> BaseFrameworkError:
    """Create sample framework exception for testing."""
    return BaseFrameworkError(
        "Test error message",
        error_code="TEST_ERROR",
        context={"test_key": "test_value"},
        user_message="User-friendly test message",
        suggestion="Try again later",
    )


# Async Testing Fixtures

@pytest.fixture
async def async_test_context():
    """Provide async test context."""
    class AsyncTestContext:
        def __init__(self):
            self.tasks = []

        async def create_task(self, coro):
            task = asyncio.create_task(coro)
            self.tasks.append(task)
            return task

        async def cleanup(self):
            for task in self.tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    context = AsyncTestContext()
    yield context
    await context.cleanup()


# Performance Testing Fixtures

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    import time
    import psutil
    import threading

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.peak_memory = None
            self.monitoring = False
            self.monitor_thread = None

        def start(self):
            self.start_time = time.perf_counter()
            process = psutil.Process()
            self.start_memory = process.memory_info().rss
            self.peak_memory = self.start_memory
            self.monitoring = True

            def monitor():
                while self.monitoring:
                    current_memory = psutil.Process().memory_info().rss
                    if current_memory > self.peak_memory:
                        self.peak_memory = current_memory
                    time.sleep(0.1)

            self.monitor_thread = threading.Thread(target=monitor, daemon=True)
            self.monitor_thread.start()

        def stop(self):
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)

            self.end_time = time.perf_counter()
            process = psutil.Process()
            self.end_memory = process.memory_info().rss

        def get_metrics(self) -> Dict[str, Any]:
            if self.start_time is None or self.end_time is None:
                return {}

            return {
                "execution_time": self.end_time - self.start_time,
                "memory_start": self.start_memory,
                "memory_end": self.end_memory,
                "memory_peak": self.peak_memory,
                "memory_delta": self.end_memory - self.start_memory,
            }

    return PerformanceMonitor()


# Pytest Marks and Configuration

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "mt5: mark test as requiring MetaTrader 5"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # Add integration marker to tests in integration directory
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Add slow marker to tests that contain "slow" in their name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# Test Utilities

def assert_no_exceptions(func, *args, **kwargs):
    """Assert that function doesn't raise any exceptions."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        pytest.fail(f"Function {func.__name__} raised {type(e).__name__}: {e}")


def assert_framework_error(func, error_code: Optional[str] = None, *args, **kwargs):
    """Assert that function raises a BaseFrameworkError with optional error code check."""
    with pytest.raises(BaseFrameworkError) as exc_info:
        func(*args, **kwargs)

    if error_code:
        assert exc_info.value.error_code == error_code

    return exc_info.value


def create_test_file(path: Path, content: str = "test content") -> Path:
    """Create a test file with content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path