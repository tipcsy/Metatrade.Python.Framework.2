"""
Main GUI entry point for the MetaTrader Python Framework.

This module provides the primary entry point for running the GUI application
with proper initialization, error handling, and graceful shutdown.
"""

from __future__ import annotations

import sys
import os
import signal
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import get_settings
from src.core.logging import setup_logging, get_logger
from src.core.exceptions import BaseFrameworkError

# Import GUI components
from src.gui.app import create_application, run_application

logger = None


def setup_environment() -> bool:
    """
    Setup application environment and dependencies.

    Returns:
        bool: True if setup successful
    """
    try:
        # Load settings first
        settings = get_settings()

        # Setup logging with settings
        setup_logging(settings)
        global logger
        logger = get_logger(__name__)

        logger.info("Setting up GUI application environment...")

        # Validate GUI settings
        if not settings.gui.enabled:
            logger.error("GUI is disabled in configuration")
            return False

        # Create necessary directories
        directories = [
            Path("data"),
            Path("logs"),
            Path("localization"),
            Path("config"),
        ]

        for directory in directories:
            directory.mkdir(exist_ok=True)

        # Verify PyQt6 availability
        try:
            import PyQt6
            logger.info(f"PyQt6 version: {PyQt6.QtCore.PYQT_VERSION_STR}")
        except ImportError as e:
            logger.error(f"PyQt6 not available: {e}")
            return False

        logger.info("Environment setup completed successfully")
        return True

    except Exception as e:
        if logger:
            logger.error(f"Environment setup failed: {e}")
        else:
            print(f"Environment setup failed: {e}")
        return False


def check_dependencies() -> bool:
    """
    Check required dependencies are available.

    Returns:
        bool: True if all dependencies are available
    """
    try:
        required_modules = [
            'PyQt6',
            'PyQt6.QtCore',
            'PyQt6.QtGui',
            'PyQt6.QtWidgets',
            'src.core.config',
            'src.core.logging',
            'src.database.database',
            'src.core.symbols.manager',
            'src.mt5.connection',
        ]

        missing_modules = []

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        if missing_modules:
            if logger:
                logger.error(f"Missing required modules: {missing_modules}")
            else:
                print(f"Missing required modules: {missing_modules}")
            return False

        logger.info("All required dependencies are available")
        return True

    except Exception as e:
        if logger:
            logger.error(f"Dependency check failed: {e}")
        else:
            print(f"Dependency check failed: {e}")
        return False


def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        if logger:
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        else:
            print(f"Received signal {signum}, shutting down...")

        # Exit gracefully
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Windows doesn't have SIGQUIT
    if hasattr(signal, 'SIGQUIT'):
        signal.signal(signal.SIGQUIT, signal_handler)


def validate_runtime_environment() -> bool:
    """
    Validate runtime environment for GUI execution.

    Returns:
        bool: True if environment is valid
    """
    try:
        # Check Python version
        if sys.version_info < (3, 9):
            logger.error(f"Python 3.9+ required, found {sys.version}")
            return False

        # Check if running in GUI environment
        if os.name == 'nt':  # Windows
            # On Windows, GUI should always work
            pass
        else:  # Unix-like systems
            if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
                logger.warning("No display environment detected")
                # Don't fail here as there might be other ways to display GUI

        # Check write permissions for data directories
        data_dir = Path("data")
        if data_dir.exists() and not os.access(data_dir, os.W_OK):
            logger.error("No write permission for data directory")
            return False

        logger.info("Runtime environment validation passed")
        return True

    except Exception as e:
        logger.error(f"Runtime environment validation failed: {e}")
        return False


def initialize_backend_services() -> bool:
    """
    Initialize required backend services.

    Returns:
        bool: True if initialization successful
    """
    try:
        logger.info("Initializing backend services...")

        # Initialize database
        from src.database.database import get_database_manager
        db_manager = get_database_manager()

        # Database is already initialized by get_database_manager()
        if not db_manager.is_initialized:
            logger.error("Failed to initialize database")
            return False

        # Initialize symbol manager
        from src.core.symbols.manager import get_symbol_manager
        symbol_manager = get_symbol_manager()

        if not symbol_manager.start():
            logger.warning("Symbol manager failed to start (may affect functionality)")

        # Initialize MT5 connection manager (optional)
        try:
            from src.mt5.connection import get_mt5_session_manager
            mt5_manager = get_mt5_session_manager()
            logger.info("MT5 connection manager initialized")
        except Exception as e:
            logger.warning(f"MT5 connection manager not available: {e}")

        logger.info("Backend services initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Backend services initialization failed: {e}")
        return False


def cleanup_on_exit() -> None:
    """Cleanup resources on application exit."""
    try:
        if logger:
            logger.info("Cleaning up application resources...")

        # Cleanup symbol manager
        try:
            from src.core.symbols.manager import get_symbol_manager
            symbol_manager = get_symbol_manager()
            symbol_manager.stop()
        except Exception as e:
            if logger:
                logger.debug(f"Error stopping symbol manager: {e}")

        # Cleanup database connections
        try:
            from src.database.database import get_database_manager
            db_manager = get_database_manager()
            db_manager.shutdown()
        except Exception as e:
            if logger:
                logger.debug(f"Error shutting down database: {e}")

        if logger:
            logger.info("Application cleanup completed")

    except Exception as e:
        if logger:
            logger.error(f"Error during cleanup: {e}")


def main() -> int:
    """
    Main entry point for the GUI application.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    exit_code = 0

    try:
        # Setup environment
        if not setup_environment():
            return 1

        # Setup signal handlers
        setup_signal_handlers()

        # Check dependencies
        if not check_dependencies():
            return 1

        # Validate runtime environment
        if not validate_runtime_environment():
            return 1

        # Initialize backend services
        if not initialize_backend_services():
            logger.error("Backend services initialization failed")
            return 1

        # Run the GUI application
        logger.info("Starting MetaTrader Python Framework GUI...")
        exit_code = run_application()

        if exit_code == 0:
            logger.info("GUI application exited normally")
        else:
            logger.warning(f"GUI application exited with code: {exit_code}")

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C)")
        exit_code = 0

    except BaseFrameworkError as e:
        logger.error(f"Framework error: {e}")
        exit_code = 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit_code = 1

    finally:
        # Cleanup resources
        cleanup_on_exit()

    return exit_code


def run_gui() -> int:
    """
    Alternative entry point for running GUI.

    Returns:
        int: Exit code
    """
    return main()


if __name__ == "__main__":
    # Set exit code
    exit_code = main()
    sys.exit(exit_code)