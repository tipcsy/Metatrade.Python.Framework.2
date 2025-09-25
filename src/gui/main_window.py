"""
Main window for the MetaTrader Python Framework GUI.

This module defines the main application window that contains all GUI widgets
and coordinates the user interface components.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QSplitter, QMenuBar, QStatusBar, QToolBar,
    QApplication, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QCloseEvent

from src.core.config import get_settings
from src.core.logging import get_logger

from .widgets import (
    SymbolManagementWidget,
    MarketDataTableWidget,
    StatusBarWidget
)
from .themes import ThemeManager
from .localization import LocalizationManager
from .models import MarketDataModel

logger = None  # Will be initialized after LoggerFactory setup


def _ensure_logger():
    """Ensure logger is initialized."""
    global logger
    if logger is None:
        logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window for the MetaTrader Python Framework.

    Contains all GUI widgets and provides the main user interface for
    symbol management, market data display, and real-time monitoring.
    """

    # Custom signals
    windowClosed = pyqtSignal()
    settingsChanged = pyqtSignal(dict)

    def __init__(
        self,
        theme_manager: ThemeManager = None,
        localization_manager: LocalizationManager = None,
        parent: QWidget = None
    ):
        """
        Initialize the main window.

        Args:
            theme_manager: Theme management instance
            localization_manager: Localization management instance
            parent: Parent widget
        """
        super().__init__(parent)

        # Store dependencies
        self.settings = get_settings()
        self.theme_manager = theme_manager
        self.localization_manager = localization_manager

        # Window state
        self.is_closing = False

        # Initialize UI components
        self._init_window_properties()
        self._create_widgets()
        self._create_menus()
        self._create_toolbars()
        self._create_status_bar()
        self._setup_layout()
        self._setup_connections()
        self._restore_window_state()

        # Start update timers
        self._setup_timers()

        _ensure_logger()
        logger.info("Main window initialized")

    def _init_window_properties(self) -> None:
        """Initialize window properties and settings."""
        # Window title and icon
        self.setWindowTitle("MetaTrader Python Framework - Real-time Market Data")

        # Window size and position
        self.resize(1400, 900)
        self.setMinimumSize(800, 600)

        # Center window on screen
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

        # Window flags
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        logger.debug("Window properties initialized")

    def _create_widgets(self) -> None:
        """Create all child widgets."""
        try:
            # Create central widget and main layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)

            # Create main data model
            self.market_data_model = MarketDataModel()

            # Create GUI-backend integrator
            from .integration import GuiBackendIntegrator
            self.integrator = GuiBackendIntegrator(self.market_data_model)

            # Create symbol management widget
            self.symbol_widget = SymbolManagementWidget(
                theme_manager=self.theme_manager,
                localization_manager=self.localization_manager
            )

            # Create market data table widget
            self.market_data_widget = MarketDataTableWidget(
                model=self.market_data_model,
                theme_manager=self.theme_manager,
                localization_manager=self.localization_manager
            )

            # Create status bar widget
            self.status_widget = StatusBarWidget(
                theme_manager=self.theme_manager,
                localization_manager=self.localization_manager
            )

            logger.debug("Child widgets created")

        except Exception as e:
            logger.error(f"Failed to create widgets: {e}")
            raise

    def _create_menus(self) -> None:
        """Create application menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Theme submenu
        theme_menu = view_menu.addMenu("&Theme")

        light_theme_action = QAction("&Light", self)
        light_theme_action.setCheckable(True)
        light_theme_action.triggered.connect(lambda: self._switch_theme("light"))
        theme_menu.addAction(light_theme_action)

        dark_theme_action = QAction("&Dark", self)
        dark_theme_action.setCheckable(True)
        dark_theme_action.setChecked(True)  # Default
        dark_theme_action.triggered.connect(lambda: self._switch_theme("dark"))
        theme_menu.addAction(dark_theme_action)

        # Language submenu
        language_menu = view_menu.addMenu("&Language")

        english_action = QAction("&English", self)
        english_action.setCheckable(True)
        english_action.triggered.connect(lambda: self._switch_language("en"))
        language_menu.addAction(english_action)

        hungarian_action = QAction("&Magyar", self)
        hungarian_action.setCheckable(True)
        hungarian_action.setChecked(True)  # Default
        hungarian_action.triggered.connect(lambda: self._switch_language("hu"))
        language_menu.addAction(hungarian_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        refresh_action = QAction("&Refresh Data", self)
        refresh_action.setShortcut(QKeySequence.StandardKey.Refresh)
        refresh_action.setStatusTip("Refresh market data")
        refresh_action.triggered.connect(self._refresh_data)
        tools_menu.addAction(refresh_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.setStatusTip("About the application")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        logger.debug("Menus created")

    def _create_toolbars(self) -> None:
        """Create application toolbars."""
        # Main toolbar
        main_toolbar = self.addToolBar("Main")
        main_toolbar.setObjectName("main_toolbar")

        # Add symbol refresh action
        refresh_action = QAction("Refresh", self)
        refresh_action.setStatusTip("Refresh market data")
        refresh_action.triggered.connect(self._refresh_data)
        main_toolbar.addAction(refresh_action)

        main_toolbar.addSeparator()

        # Add connection status indicator
        # This would be implemented with a custom widget showing MT5 connection status

        logger.debug("Toolbars created")

    def _create_status_bar(self) -> None:
        """Create status bar."""
        status_bar = self.statusBar()
        status_bar.addWidget(self.status_widget)
        status_bar.showMessage("Ready")

        logger.debug("Status bar created")

    def _setup_layout(self) -> None:
        """Setup main window layout."""
        central_widget = self.centralWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Create main splitter for symbol management and market data
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Symbol management (300px width)
        main_splitter.addWidget(self.symbol_widget)

        # Right panel - Market data table
        main_splitter.addWidget(self.market_data_widget)

        # Set splitter proportions (symbol: 300px, market data: remaining)
        main_splitter.setSizes([300, 1100])
        main_splitter.setStretchFactor(0, 0)  # Fixed width for symbol panel
        main_splitter.setStretchFactor(1, 1)  # Expandable market data panel

        main_layout.addWidget(main_splitter)

        logger.debug("Layout configured")

    def _setup_connections(self) -> None:
        """Setup signal connections between widgets."""
        try:
            # Connect symbol widget signals to market data widget
            self.symbol_widget.symbolAdded.connect(
                self.market_data_widget.add_symbol
            )
            self.symbol_widget.symbolRemoved.connect(
                self.market_data_widget.remove_symbol
            )
            self.symbol_widget.symbolSelectionChanged.connect(
                self.market_data_widget.highlight_symbol
            )

            # Connect market data widget signals to status bar
            self.market_data_widget.dataUpdated.connect(
                self.status_widget.update_data_status
            )
            self.market_data_widget.errorOccurred.connect(
                self.status_widget.show_error
            )

            # Connect integrator signals
            self.integrator.dataIntegrated.connect(self._on_data_integrated)
            self.integrator.serviceStatusChanged.connect(self._on_service_status_changed)
            self.integrator.integrationError.connect(self.status_widget.show_error)

            # Connect symbol widget to integrator
            self.symbol_widget.symbolAdded.connect(self.integrator.add_symbol_to_integration)
            self.symbol_widget.symbolRemoved.connect(self.integrator.remove_symbol_from_integration)

            logger.debug("Widget connections established")

        except Exception as e:
            logger.error(f"Failed to setup connections: {e}")

    def _setup_timers(self) -> None:
        """Setup update timers."""
        # Main update timer for real-time data
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_data)
        self.update_timer.start(self.settings.gui.update_interval)

        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(5000)  # Update every 5 seconds

        # Start backend integration
        if self.integrator:
            QTimer.singleShot(2000, self._start_integration)  # Start after 2 seconds

        logger.debug("Update timers started")

    def _restore_window_state(self) -> None:
        """Restore window state from settings."""
        try:
            settings = QSettings()

            # Restore window geometry
            geometry = settings.value("window/geometry")
            if geometry:
                self.restoreGeometry(geometry)

            # Restore window state (toolbars, docks, etc.)
            state = settings.value("window/state")
            if state:
                self.restoreState(state)

            logger.debug("Window state restored")

        except Exception as e:
            logger.debug(f"Could not restore window state: {e}")

    def _save_window_state(self) -> None:
        """Save current window state."""
        try:
            settings = QSettings()

            # Save window geometry
            settings.setValue("window/geometry", self.saveGeometry())

            # Save window state
            settings.setValue("window/state", self.saveState())

            logger.debug("Window state saved")

        except Exception as e:
            logger.error(f"Failed to save window state: {e}")

    def _update_data(self) -> None:
        """Update real-time data display."""
        try:
            # Refresh market data model
            self.market_data_model.refresh_data()

            # Update widget displays
            self.market_data_widget.update_display()

        except Exception as e:
            logger.debug(f"Error updating data: {e}")

    def _update_status(self) -> None:
        """Update status bar information."""
        try:
            # Update connection status
            self.status_widget.update_connection_status()

            # Update performance metrics
            self.status_widget.update_performance_metrics()

        except Exception as e:
            logger.debug(f"Error updating status: {e}")

    def _refresh_data(self) -> None:
        """Manually refresh all data."""
        try:
            self.statusBar().showMessage("Refreshing data...", 2000)

            # Refresh symbol list
            self.symbol_widget.refresh_symbols()

            # Refresh market data
            self.market_data_widget.refresh_data()

            logger.info("Data refreshed manually")

        except Exception as e:
            logger.error(f"Failed to refresh data: {e}")
            self.statusBar().showMessage("Error refreshing data", 3000)

    def _switch_theme(self, theme_name: str) -> None:
        """Switch application theme."""
        if self.theme_manager:
            app = QApplication.instance()
            if hasattr(app, 'switch_theme'):
                app.switch_theme(theme_name)
            logger.info(f"Switched to {theme_name} theme")

    def _switch_language(self, language_code: str) -> None:
        """Switch application language."""
        if self.localization_manager:
            app = QApplication.instance()
            if hasattr(app, 'switch_language'):
                app.switch_language(language_code)
            logger.info(f"Switched to {language_code} language")

    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """
        <h3>MetaTrader Python Framework</h3>
        <p>Version 2.0.0</p>
        <p>A comprehensive framework for MetaTrader 5 integration with Python,
        featuring real-time market data, automated trading, and advanced analytics.</p>
        <p>Built with PyQt6 and modern Python technologies.</p>
        """

        QMessageBox.about(self, "About MetaTrader Framework", about_text)

    def retranslate_ui(self) -> None:
        """Retranslate UI elements after language change."""
        # This would update all translatable text in the UI
        # For now, just update window title
        self.setWindowTitle("MetaTrader Python Framework - Real-time Market Data")

        # Update child widgets
        if hasattr(self.symbol_widget, 'retranslate_ui'):
            self.symbol_widget.retranslate_ui()
        if hasattr(self.market_data_widget, 'retranslate_ui'):
            self.market_data_widget.retranslate_ui()
        if hasattr(self.status_widget, 'retranslate_ui'):
            self.status_widget.retranslate_ui()

    def save_settings(self) -> None:
        """Save application settings."""
        self._save_window_state()
        logger.debug("Settings saved")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event."""
        if self.is_closing:
            return

        self.is_closing = True

        try:
            # Save settings before closing
            self.save_settings()

            # Stop timers
            if hasattr(self, 'update_timer'):
                self.update_timer.stop()
            if hasattr(self, 'status_timer'):
                self.status_timer.stop()

            # Stop backend integration
            if hasattr(self, 'integrator') and self.integrator:
                self.integrator.stop_integration()

            # Emit custom signal
            self.windowClosed.emit()

            # Accept the close event
            event.accept()

            logger.info("Main window closed")

        except Exception as e:
            logger.error(f"Error during window close: {e}")
            event.accept()  # Close anyway

    def _start_integration(self) -> None:
        """Start backend integration."""
        try:
            if self.integrator and not self.integrator.is_running():
                success = self.integrator.start_integration()
                if success:
                    self.statusBar().showMessage("Backend integration started", 3000)
                    logger.info("Backend integration started from main window")
                else:
                    self.statusBar().showMessage("Failed to start backend integration", 5000)
                    logger.error("Failed to start backend integration")
        except Exception as e:
            logger.error(f"Error starting integration: {e}")
            self.statusBar().showMessage("Integration startup error", 5000)

    def _on_data_integrated(self, symbol: str, data: dict) -> None:
        """Handle integrated data updates."""
        try:
            # Data is already handled by the integrator -> model connection
            # This is mainly for logging and additional processing
            logger.debug(f"Data integrated for {symbol}")
        except Exception as e:
            logger.debug(f"Error handling integrated data: {e}")

    def _on_service_status_changed(self, service_name: str, is_running: bool) -> None:
        """Handle service status changes."""
        try:
            status = "running" if is_running else "stopped"
            message = f"Service {service_name} is {status}"
            self.statusBar().showMessage(message, 2000)
            logger.info(message)
        except Exception as e:
            logger.debug(f"Error handling service status change: {e}")