"""
Status bar widget for the MetaTrader Python Framework.

This module provides a comprehensive status bar widget that displays
connection status, performance metrics, and system information.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QProgressBar,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.symbols.manager import get_symbol_manager
from src.mt5.connection import get_mt5_session_manager

from ..themes import ThemeManager
from ..localization import LocalizationManager

logger = get_logger(__name__)


class StatusBarWidget(QWidget):
    """
    Comprehensive status bar widget.

    Features:
    - MT5 connection status indicator
    - Symbol manager status
    - Performance metrics display
    - Memory usage indicator
    - Update rate monitoring
    - Error/warning notifications
    """

    # Custom signals
    statusChanged = pyqtSignal(str)         # status message
    connectionChanged = pyqtSignal(bool)    # connected state
    errorOccurred = pyqtSignal(str)         # error message

    def __init__(
        self,
        theme_manager: Optional[ThemeManager] = None,
        localization_manager: Optional[LocalizationManager] = None,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize status bar widget.

        Args:
            theme_manager: Theme manager for styling
            localization_manager: Localization manager
            parent: Parent widget
        """
        super().__init__(parent)

        # Dependencies
        self.settings = get_settings()
        self.theme_manager = theme_manager
        self.localization_manager = localization_manager
        self.symbol_manager = get_symbol_manager()
        self.mt5_session_manager = get_mt5_session_manager()

        # Status tracking
        self._connection_status = False
        self._last_update_time = datetime.now()
        self._update_count = 0
        self._error_count = 0
        self._warning_count = 0

        # Performance metrics
        self._performance_metrics = {
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'update_rate': 0.0,
            'symbols_active': 0,
            'data_points_per_second': 0.0
        }

        # Create UI
        self._create_ui()
        self._setup_timers()

        logger.info("Status bar widget initialized")

    def _create_ui(self) -> None:
        """Create the user interface."""
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)

        # Create status sections
        self._create_connection_section(layout)
        self._create_symbols_section(layout)
        self._create_performance_section(layout)
        self._create_progress_section(layout)
        self._create_message_section(layout)

    def _create_connection_section(self, parent_layout: QHBoxLayout) -> None:
        """Create connection status section."""
        # Connection status indicator
        self.connection_label = QLabel("MT5: Disconnected")
        self.connection_label.setMinimumWidth(120)

        # Set initial styling
        self._update_connection_display(False)

        parent_layout.addWidget(self.connection_label)

        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.VLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        parent_layout.addWidget(separator1)

    def _create_symbols_section(self, parent_layout: QHBoxLayout) -> None:
        """Create symbols status section."""
        # Symbol count and status
        self.symbols_label = QLabel("Symbols: 0/0")
        self.symbols_label.setMinimumWidth(80)
        self.symbols_label.setToolTip("Active symbols / Total symbols")

        parent_layout.addWidget(self.symbols_label)

        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.VLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        parent_layout.addWidget(separator2)

    def _create_performance_section(self, parent_layout: QHBoxLayout) -> None:
        """Create performance metrics section."""
        # Update rate
        self.update_rate_label = QLabel("Rate: 0.0/s")
        self.update_rate_label.setMinimumWidth(70)
        self.update_rate_label.setToolTip("Data updates per second")

        # Memory usage
        self.memory_label = QLabel("Mem: 0 MB")
        self.memory_label.setMinimumWidth(70)
        self.memory_label.setToolTip("Memory usage")

        parent_layout.addWidget(self.update_rate_label)
        parent_layout.addWidget(self.memory_label)

        # Separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.VLine)
        separator3.setFrameShadow(QFrame.Shadow.Sunken)
        parent_layout.addWidget(separator3)

    def _create_progress_section(self, parent_layout: QHBoxLayout) -> None:
        """Create progress indicator section."""
        # Progress bar for operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)

        parent_layout.addWidget(self.progress_bar)

    def _create_message_section(self, parent_layout: QHBoxLayout) -> None:
        """Create message display section."""
        # Spacer to push message to the right
        parent_layout.addStretch()

        # Status message
        self.message_label = QLabel("Ready")
        self.message_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )

        # Time display
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(80)
        self._update_time_display()

        parent_layout.addWidget(self.message_label, 1)
        parent_layout.addWidget(self.time_label)

    def _setup_timers(self) -> None:
        """Setup update timers."""
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(2000)  # Update every 2 seconds

        # Time display timer
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self._update_time_display)
        self.time_timer.start(1000)  # Update every second

        # Performance metrics timer
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self._update_performance_metrics)
        self.metrics_timer.start(5000)  # Update every 5 seconds

    def _update_connection_display(self, connected: bool) -> None:
        """Update connection status display."""
        self._connection_status = connected

        if connected:
            self.connection_label.setText("MT5: Connected")
            self.connection_label.setStyleSheet("color: green; font-weight: bold;")
            self.connection_label.setToolTip("Connected to MetaTrader 5")
        else:
            self.connection_label.setText("MT5: Disconnected")
            self.connection_label.setStyleSheet("color: red; font-weight: bold;")
            self.connection_label.setToolTip("Disconnected from MetaTrader 5")

        self.connectionChanged.emit(connected)

    def _update_status(self) -> None:
        """Update all status information."""
        try:
            # Update connection status
            self._check_connection_status()

            # Update symbol status
            self._update_symbol_status()

            # Update performance metrics
            self._update_rate_metrics()

        except Exception as e:
            logger.debug(f"Error updating status: {e}")

    async def _check_connection_status(self) -> None:
        """Check MT5 connection status."""
        try:
            try:
                session = None  # Temporarily disabled due to async issues
                # session = await self.mt5_session_manager.get_session()
            except Exception as e:
                session = None
                logger.debug(f"MT5 session not available: {e}")

            connected = session is not None and session.terminal_info() is not None

            if connected != self._connection_status:
                self._update_connection_display(connected)

        except Exception as e:
            logger.debug(f"Error checking connection status: {e}")
            if self._connection_status:
                self._update_connection_display(False)

    def _update_symbol_status(self) -> None:
        """Update symbol status information."""
        try:
            if self.symbol_manager.is_running:
                stats = self.symbol_manager.get_system_stats()
                total_symbols = stats.get('total_symbols', 0)
                active_symbols = stats.get('active_symbols', 0)

                self.symbols_label.setText(f"Symbols: {active_symbols}/{total_symbols}")
                self.symbols_label.setToolTip(
                    f"Active: {active_symbols}, Total: {total_symbols}, "
                    f"Subscribed: {stats.get('subscribed_symbols', 0)}"
                )
            else:
                self.symbols_label.setText("Symbols: N/A")
                self.symbols_label.setToolTip("Symbol manager not running")

        except Exception as e:
            logger.debug(f"Error updating symbol status: {e}")
            self.symbols_label.setText("Symbols: Error")

    def _update_rate_metrics(self) -> None:
        """Update rate-based metrics."""
        try:
            current_time = datetime.now()
            time_delta = (current_time - self._last_update_time).total_seconds()

            if time_delta > 0:
                update_rate = self._update_count / time_delta
                self._performance_metrics['update_rate'] = update_rate
                self.update_rate_label.setText(f"Rate: {update_rate:.1f}/s")
            else:
                self.update_rate_label.setText("Rate: 0.0/s")

            # Reset counters
            self._update_count = 0
            self._last_update_time = current_time

        except Exception as e:
            logger.debug(f"Error updating rate metrics: {e}")

    def _update_performance_metrics(self) -> None:
        """Update system performance metrics."""
        try:
            import psutil
            import os

            # Get current process
            process = psutil.Process(os.getpid())

            # Memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self._performance_metrics['memory_usage_mb'] = memory_mb
            self.memory_label.setText(f"Mem: {memory_mb:.0f} MB")

            # CPU usage (if available)
            try:
                cpu_percent = process.cpu_percent()
                self._performance_metrics['cpu_usage_percent'] = cpu_percent
            except Exception:
                pass

        except ImportError:
            # psutil not available
            self.memory_label.setText("Mem: N/A")
        except Exception as e:
            logger.debug(f"Error updating performance metrics: {e}")

    def _update_time_display(self) -> None:
        """Update time display."""
        current_time = datetime.now()
        time_str = current_time.strftime("%H:%M:%S")
        self.time_label.setText(time_str)

    # Public interface methods

    def update_connection_status(self) -> None:
        """Update connection status (called externally)."""
        self._check_connection_status()

    def update_data_status(self) -> None:
        """Update data status (called when data is updated)."""
        self._update_count += 1

    def update_performance_metrics(self) -> None:
        """Update performance metrics (called externally)."""
        self._update_performance_metrics()

    def show_message(self, message: str, timeout_ms: int = 0) -> None:
        """
        Show status message.

        Args:
            message: Message to display
            timeout_ms: Message timeout in milliseconds (0 = permanent)
        """
        self.message_label.setText(message)
        self.statusChanged.emit(message)

        if timeout_ms > 0:
            QTimer.singleShot(timeout_ms, lambda: self.message_label.setText("Ready"))

    def show_error(self, error_message: str) -> None:
        """
        Show error message.

        Args:
            error_message: Error message to display
        """
        self._error_count += 1

        # Display error in red
        self.message_label.setText(f"Error: {error_message}")
        self.message_label.setStyleSheet("color: red;")

        # Emit error signal
        self.errorOccurred.emit(error_message)

        # Reset color after 5 seconds
        QTimer.singleShot(5000, self._reset_message_style)

        logger.error(f"Status bar error: {error_message}")

    def show_warning(self, warning_message: str) -> None:
        """
        Show warning message.

        Args:
            warning_message: Warning message to display
        """
        self._warning_count += 1

        # Display warning in orange
        self.message_label.setText(f"Warning: {warning_message}")
        self.message_label.setStyleSheet("color: orange;")

        # Reset color after 3 seconds
        QTimer.singleShot(3000, self._reset_message_style)

        logger.warning(f"Status bar warning: {warning_message}")

    def show_progress(self, show: bool, minimum: int = 0, maximum: int = 0) -> None:
        """
        Show or hide progress bar.

        Args:
            show: Whether to show progress bar
            minimum: Minimum progress value
            maximum: Maximum progress value
        """
        self.progress_bar.setVisible(show)

        if show:
            self.progress_bar.setRange(minimum, maximum)
            if maximum == 0:
                # Indeterminate progress
                self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)

    def set_progress_value(self, value: int) -> None:
        """
        Set progress bar value.

        Args:
            value: Progress value
        """
        if self.progress_bar.isVisible():
            self.progress_bar.setValue(value)

    def _reset_message_style(self) -> None:
        """Reset message label style to default."""
        self.message_label.setStyleSheet("")
        if self.message_label.text().startswith(("Error:", "Warning:")):
            self.message_label.setText("Ready")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self._performance_metrics.copy()

    def get_connection_status(self) -> bool:
        """Get connection status."""
        return self._connection_status

    def get_error_count(self) -> int:
        """Get error count."""
        return self._error_count

    def get_warning_count(self) -> int:
        """Get warning count."""
        return self._warning_count

    def reset_counters(self) -> None:
        """Reset error and warning counters."""
        self._error_count = 0
        self._warning_count = 0

    def retranslate_ui(self) -> None:
        """Retranslate UI elements."""
        # This would update translatable text
        # For now, it's a placeholder
        if not self._connection_status:
            self.connection_label.setText("MT5: Disconnected")
            self.connection_label.setToolTip("Disconnected from MetaTrader 5")