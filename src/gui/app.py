"""
Main PyQt6 application for the MetaTrader Python Framework.

This module defines the main QApplication subclass that coordinates all GUI
components, manages application lifecycle, and integrates with the backend services.
"""

from __future__ import annotations

import sys
import signal
from typing import Optional

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer, QThread, QObject, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon, QPalette

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.exceptions import BaseFrameworkError

from .main_window import MainWindow
from .themes import ThemeManager
from .localization import LocalizationManager

logger = None  # Will be initialized after LoggerFactory setup


def _ensure_logger():
    """Ensure logger is initialized."""
    global logger
    if logger is None:
        logger = get_logger(__name__)


class MetaTraderApp(QApplication):
    """
    Main application class for the MetaTrader Python Framework GUI.

    Manages the application lifecycle, coordinates all GUI components,
    and provides integration with backend services.
    """

    # Custom signals for application events
    shutdownRequested = pyqtSignal()
    errorOccurred = pyqtSignal(str, str)  # title, message

    def __init__(self, argv: list[str] = None):
        """
        Initialize the MetaTrader application.

        Args:
            argv: Command line arguments
        """
        if argv is None:
            argv = sys.argv

        super().__init__(argv)

        # Store application settings
        self.settings = get_settings()

        # Initialize components
        self.theme_manager: Optional[ThemeManager] = None
        self.localization_manager: Optional[LocalizationManager] = None
        self.main_window: Optional[MainWindow] = None

        # Performance monitoring
        self.startup_timer = QTimer()
        self.performance_metrics = {
            'startup_time': 0.0,
            'update_count': 0,
            'memory_usage': 0.0,
        }

        # Setup application properties
        self._setup_application()

        # Initialize GUI components
        try:
            print("About to call _initialize_components...")
            self._initialize_components()
            print("_initialize_components completed successfully")
        except Exception as e:
            print(f"ERROR in _initialize_components: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise

        # Setup signal handlers
        self._setup_signal_handlers()

        _ensure_logger()
        if logger is not None:
            logger.info("MetaTrader application initialized")

    def _setup_application(self) -> None:
        """Setup basic application properties."""
        # Application metadata
        self.setApplicationName("MetaTrader Python Framework")
        self.setApplicationVersion("2.0.0")
        self.setOrganizationName("MetaTrader Framework")
        self.setOrganizationDomain("metatrader-framework.local")

        # Window icon (if available)
        try:
            icon = QIcon("resources/icons/app_icon.png")
            self.setWindowIcon(icon)
        except Exception:
            _ensure_logger()
            if logger is not None:
                logger.debug("Application icon not found")

        # Font configuration for high DPI displays
        font = QFont("Segoe UI", 9)
        font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
        self.setFont(font)

        # High DPI support is automatic in PyQt6
        # AA_EnableHighDpiScaling and AA_UseHighDpiPixmaps are deprecated and not needed

        # Application behavior
        self.setQuitOnLastWindowClosed(True)

        _ensure_logger()
        if logger is not None:
            logger.debug("Application properties configured")

    def _initialize_components(self) -> None:
        """Initialize GUI component managers."""
        try:
            print("Starting GUI components initialization...")

            # Temporarily disable complex components to isolate the issue
            # Initialize theme manager
            print("Initializing ThemeManager...")
            try:
                self.theme_manager = ThemeManager(self.settings.gui.theme)
                print("✅ ThemeManager initialized successfully")
            except Exception as e:
                print(f"❌ ThemeManager failed: {e}")
                self.theme_manager = None

            # Initialize localization manager
            print("Initializing LocalizationManager...")
            try:
                self.localization_manager = LocalizationManager(self.settings.gui.language)
                print("✅ LocalizationManager initialized successfully")
            except Exception as e:
                print(f"❌ LocalizationManager failed: {e}")
                self.localization_manager = None

            # Apply theme (only if theme manager exists)
            if self.theme_manager is not None:
                print("Applying theme...")
                try:
                    self.theme_manager.apply_theme(self)
                    print("✅ Theme applied successfully")
                except Exception as e:
                    print(f"❌ Theme application failed: {e}")

            # Apply translations (only if localization manager exists)
            if self.localization_manager is not None:
                print("Installing translator...")
                try:
                    self.installTranslator(self.localization_manager.get_translator())
                    print("✅ Translator installed successfully")
                except Exception as e:
                    print(f"❌ Translator installation failed: {e}")

            # Initialize main window
            print("Initializing MainWindow...")
            try:
                self.main_window = MainWindow(
                    theme_manager=self.theme_manager,
                    localization_manager=self.localization_manager
                )
                print("✅ MainWindow initialized successfully")
            except Exception as e:
                print(f"❌ MainWindow initialization failed: {e}")
                self.main_window = None

            _ensure_logger()
            if logger is not None:
                logger.info("GUI components initialized (with fallbacks)")

        except Exception as e:
            _ensure_logger()
            if logger is not None:
                logger.error(f"Failed to initialize GUI components: {e}")
            self._show_error_message("Initialization Error", str(e))
            sys.exit(1)

    def _setup_signal_handlers(self) -> None:
        """Setup system and custom signal handlers."""
        # System signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Custom application signals
        self.shutdownRequested.connect(self._handle_shutdown)
        self.errorOccurred.connect(self._handle_error)

        # Application signals
        self.aboutToQuit.connect(self._cleanup_before_exit)

        _ensure_logger()
        if logger is not None:
            logger.debug("Signal handlers configured")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle system signals gracefully."""
        _ensure_logger()
        if logger is not None:
            logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdownRequested.emit()

    def _handle_shutdown(self) -> None:
        """Handle application shutdown request."""
        _ensure_logger()
        if logger is not None:
            logger.info("Shutdown requested")

        if self.main_window:
            self.main_window.close()

        self.quit()

    def _handle_error(self, title: str, message: str) -> None:
        """Handle application errors with user notification."""
        _ensure_logger()
        if logger is not None:
            logger.error(f"Application error: {title} - {message}")
        self._show_error_message(title, message)

    def _show_error_message(self, title: str, message: str) -> None:
        """Show error message to user."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)

        if self.theme_manager:
            self.theme_manager.apply_theme_to_widget(msg)

        msg.exec()

    def _cleanup_before_exit(self) -> None:
        """Cleanup resources before application exit."""
        _ensure_logger()
        logger.info("Cleaning up before exit...")

        try:
            # Save window state and settings
            if self.main_window:
                self.main_window.save_settings()

            # Save theme preferences
            if self.theme_manager:
                self.theme_manager.save_preferences()

            # Save localization preferences
            if self.localization_manager:
                self.localization_manager.save_preferences()

            _ensure_logger()
            if logger is not None:
                logger.info("Application cleanup completed")

        except Exception as e:
            _ensure_logger()
            if logger is not None:
                logger.error(f"Error during cleanup: {e}")

    def show_main_window(self) -> None:
        """Show the main application window."""
        if self.main_window:
            self.main_window.show()
            self.main_window.raise_()
            self.main_window.activateWindow()
        else:
            _ensure_logger()
            if logger is not None:
                logger.error("Main window not initialized")

    def get_theme_manager(self) -> Optional[ThemeManager]:
        """Get the theme manager instance."""
        return self.theme_manager

    def get_localization_manager(self) -> Optional[LocalizationManager]:
        """Get the localization manager instance."""
        return self.localization_manager

    def get_main_window(self) -> Optional[MainWindow]:
        """Get the main window instance."""
        return self.main_window

    def switch_theme(self, theme_name: str) -> None:
        """Switch application theme."""
        if self.theme_manager:
            self.theme_manager.set_theme(theme_name)
            self.theme_manager.apply_theme(self)

            if self.main_window:
                self.theme_manager.apply_theme_to_widget(self.main_window)

    def switch_language(self, language_code: str) -> None:
        """Switch application language."""
        if self.localization_manager:
            self.localization_manager.set_language(language_code)
            translator = self.localization_manager.get_translator()
            self.installTranslator(translator)

            if self.main_window:
                self.main_window.retranslate_ui()

    def report_error(self, title: str, message: str) -> None:
        """Report application error."""
        self.errorOccurred.emit(title, message)

    def get_performance_metrics(self) -> dict:
        """Get application performance metrics."""
        return self.performance_metrics.copy()


def create_application(argv: list[str] = None) -> MetaTraderApp:
    """
    Create and configure the MetaTrader application.

    Args:
        argv: Command line arguments

    Returns:
        Configured MetaTrader application instance
    """
    try:
        app = MetaTraderApp(argv)
        _ensure_logger()
        if logger is not None:
            logger.info("MetaTrader application created successfully")
        return app

    except Exception as e:
        _ensure_logger()
        if logger is not None:
            logger.error(f"Failed to create application: {e}")
        raise BaseFrameworkError(f"Application creation failed: {e}")


def run_application() -> int:
    """
    Run the MetaTrader GUI application.

    Returns:
        Application exit code
    """
    try:
        # Create application
        app = create_application()

        # Show main window
        app.show_main_window()

        _ensure_logger()
        if logger is not None:
            logger.info("Starting GUI application...")

        # Run application
        return app.exec()

    except KeyboardInterrupt:
        _ensure_logger()
        if logger is not None:
            logger.info("Application interrupted by user")
        return 0

    except Exception as e:
        _ensure_logger()
        if logger is not None:
            logger.error(f"Application execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_application())