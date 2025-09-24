"""
Market data table widget for real-time market information display.

This module provides a high-performance table widget for displaying real-time
market data with color coding, trend indicators, and optimized updates.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView,
    QHeaderView, QLabel, QCheckBox, QSpinBox,
    QGroupBox, QFrame, QSizePolicy, QMenu,
    QMessageBox
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QModelIndex,
    QItemSelectionModel, QAbstractItemModel
)
from PyQt6.QtGui import QFont, QAction, QContextMenuEvent

from src.core.config import get_settings
from src.core.logging import get_logger

from ..models import MarketDataModel
from ..themes import ThemeManager
from ..localization import LocalizationManager

logger = get_logger(__name__)


class MarketDataTableWidget(QWidget):
    """
    High-performance market data table widget.

    Features:
    - Real-time bid/ask/spread display
    - Multi-timeframe trend indicators
    - Color-coded rows and cells
    - Performance-optimized updates
    - Configurable display options
    - Context menu actions
    """

    # Custom signals
    dataUpdated = pyqtSignal()
    symbolSelected = pyqtSignal(str)        # symbol
    errorOccurred = pyqtSignal(str)         # error message
    performanceUpdate = pyqtSignal(dict)    # performance metrics

    def __init__(
        self,
        model: Optional[MarketDataModel] = None,
        theme_manager: Optional[ThemeManager] = None,
        localization_manager: Optional[LocalizationManager] = None,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize market data table widget.

        Args:
            model: Market data model (creates new if None)
            theme_manager: Theme manager for styling
            localization_manager: Localization manager
            parent: Parent widget
        """
        super().__init__(parent)

        # Dependencies
        self.settings = get_settings()
        self.theme_manager = theme_manager
        self.localization_manager = localization_manager

        # Data model
        self.market_data_model = model if model else MarketDataModel()

        # Performance tracking
        self._update_count = 0
        self._last_update_time = datetime.now()
        self._performance_metrics = {
            'updates_per_second': 0.0,
            'displayed_symbols': 0,
            'total_updates': 0,
            'last_update_latency': 0.0
        }

        # Display configuration
        self._auto_resize = True
        self._show_milliseconds = False
        self._decimal_places = 5
        self._highlight_changes = True
        self._selected_symbol: Optional[str] = None

        # UI Components
        self._create_ui()
        self._setup_connections()
        self._setup_timers()
        self._configure_table()

        logger.info("Market data table widget initialized")

    def _create_ui(self) -> None:
        """Create the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Create sections
        self._create_control_section(main_layout)
        self._create_table_section(main_layout)
        self._create_info_section(main_layout)

    def _create_control_section(self, parent_layout: QVBoxLayout) -> None:
        """Create control section with display options."""
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(5, 5, 5, 5)

        # Auto-resize checkbox
        self.auto_resize_check = QCheckBox("Auto-resize columns")
        self.auto_resize_check.setChecked(self._auto_resize)

        # Show milliseconds checkbox
        self.milliseconds_check = QCheckBox("Show milliseconds")
        self.milliseconds_check.setChecked(self._show_milliseconds)

        # Decimal places spinbox
        decimal_label = QLabel("Decimal places:")
        self.decimal_spin = QSpinBox()
        self.decimal_spin.setRange(2, 8)
        self.decimal_spin.setValue(self._decimal_places)

        # Highlight changes checkbox
        self.highlight_check = QCheckBox("Highlight changes")
        self.highlight_check.setChecked(self._highlight_changes)

        # Add controls to layout
        control_layout.addWidget(self.auto_resize_check)
        control_layout.addWidget(self.milliseconds_check)
        control_layout.addWidget(decimal_label)
        control_layout.addWidget(self.decimal_spin)
        control_layout.addWidget(self.highlight_check)
        control_layout.addStretch()

        parent_layout.addWidget(control_frame)

    def _create_table_section(self, parent_layout: QVBoxLayout) -> None:
        """Create main table section."""
        # Table view
        self.table_view = QTableView()
        self.table_view.setModel(self.market_data_model)

        # Configure table appearance
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table_view.setSortingEnabled(True)
        self.table_view.setShowGrid(True)
        self.table_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # Font for data display
        font = QFont("Consolas", 9)
        self.table_view.setFont(font)

        # Size policy
        self.table_view.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        parent_layout.addWidget(self.table_view, 1)

    def _create_info_section(self, parent_layout: QVBoxLayout) -> None:
        """Create information section."""
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(5, 5, 5, 5)

        # Symbol count
        self.symbol_count_label = QLabel("Symbols: 0")

        # Update rate
        self.update_rate_label = QLabel("Updates/sec: 0.0")

        # Last update time
        self.last_update_label = QLabel("Last update: Never")

        # Performance indicator
        self.performance_label = QLabel("Performance: Good")

        # Add to layout
        info_layout.addWidget(self.symbol_count_label)
        info_layout.addWidget(self.update_rate_label)
        info_layout.addWidget(self.last_update_label)
        info_layout.addStretch()
        info_layout.addWidget(self.performance_label)

        parent_layout.addWidget(info_frame)

    def _setup_connections(self) -> None:
        """Setup signal connections."""
        try:
            # Control checkboxes and spinbox
            self.auto_resize_check.toggled.connect(self._on_auto_resize_changed)
            self.milliseconds_check.toggled.connect(self._on_milliseconds_changed)
            self.decimal_spin.valueChanged.connect(self._on_decimal_places_changed)
            self.highlight_check.toggled.connect(self._on_highlight_changed)

            # Table view signals
            self.table_view.selectionModel().selectionChanged.connect(
                self._on_selection_changed
            )
            self.table_view.doubleClicked.connect(self._on_double_click)
            self.table_view.customContextMenuRequested.connect(
                self._show_context_menu
            )

            # Model signals
            self.market_data_model.dataRefreshed.connect(self._on_data_refreshed)
            self.market_data_model.symbolDataUpdated.connect(self._on_symbol_updated)
            self.market_data_model.trendChanged.connect(self._on_trend_changed)
            self.market_data_model.errorOccurred.connect(self._on_model_error)

            logger.debug("Signal connections established")

        except Exception as e:
            logger.error(f"Error setting up connections: {e}")

    def _setup_timers(self) -> None:
        """Setup update timers."""
        # Performance metrics timer
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self._update_performance_metrics)
        self.metrics_timer.start(1000)  # Update every second

        # Auto-refresh timer (optional)
        if self.settings.gui.update_interval > 0:
            self.refresh_timer = QTimer()
            self.refresh_timer.timeout.connect(self._auto_refresh_data)
            self.refresh_timer.start(self.settings.gui.update_interval)

    def _configure_table(self) -> None:
        """Configure table appearance and behavior."""
        try:
            header = self.table_view.horizontalHeader()

            # Set initial column widths
            column_widths = {
                0: 80,   # Symbol
                1: 100,  # Bid
                2: 100,  # Ask
                3: 80,   # Spread
                4: 80,   # Spread (Pips)
                5: 80,   # Change %
                6: 80,   # Volume
                7: 60,   # Positions
                8: 100,  # Last Update
            }

            for col, width in column_widths.items():
                if col < self.market_data_model.columnCount():
                    self.table_view.setColumnWidth(col, width)

            # Configure resize modes
            if self._auto_resize:
                header.setStretchLastSection(True)
                header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            else:
                header.setStretchLastSection(False)
                header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

            # Set sorting
            self.table_view.sortByColumn(0, Qt.SortOrder.AscendingOrder)

            logger.debug("Table configuration applied")

        except Exception as e:
            logger.error(f"Error configuring table: {e}")

    def _on_auto_resize_changed(self, checked: bool) -> None:
        """Handle auto-resize option change."""
        self._auto_resize = checked
        self._configure_table()

    def _on_milliseconds_changed(self, checked: bool) -> None:
        """Handle milliseconds display option change."""
        self._show_milliseconds = checked
        self.table_view.update()

    def _on_decimal_places_changed(self, places: int) -> None:
        """Handle decimal places change."""
        self._decimal_places = places
        self.market_data_model.set_decimal_places(places)

    def _on_highlight_changed(self, checked: bool) -> None:
        """Handle highlight changes option."""
        self._highlight_changes = checked

    def _on_selection_changed(self) -> None:
        """Handle table selection changes."""
        selection_model = self.table_view.selectionModel()
        selected_rows = selection_model.selectedRows()

        if selected_rows:
            row = selected_rows[0].row()
            row_data = self.market_data_model.get_row_data(row)

            if row_data:
                symbol = row_data['Symbol']
                self._selected_symbol = symbol
                self.symbolSelected.emit(symbol)

    def _on_double_click(self, index: QModelIndex) -> None:
        """Handle double-click on table."""
        if index.isValid():
            row_data = self.market_data_model.get_row_data(index.row())
            if row_data:
                self._show_symbol_details(row_data['Symbol'])

    def _show_context_menu(self, position) -> None:
        """Show context menu."""
        index = self.table_view.indexAt(position)
        if not index.isValid():
            return

        row_data = self.market_data_model.get_row_data(index.row())
        if not row_data:
            return

        symbol = row_data['Symbol']

        # Create context menu
        menu = QMenu(self)

        # Symbol actions
        details_action = QAction(f"Symbol Details - {symbol}", self)
        details_action.triggered.connect(
            lambda: self._show_symbol_details(symbol)
        )

        remove_action = QAction(f"Remove {symbol}", self)
        remove_action.triggered.connect(
            lambda: self._remove_symbol(symbol)
        )

        # Chart action (placeholder)
        chart_action = QAction(f"Open Chart - {symbol}", self)
        chart_action.triggered.connect(
            lambda: self._open_chart(symbol)
        )
        chart_action.setEnabled(False)  # Not implemented yet

        # Add actions to menu
        menu.addAction(details_action)
        menu.addSeparator()
        menu.addAction(chart_action)
        menu.addSeparator()
        menu.addAction(remove_action)

        # Show menu
        menu.exec(self.table_view.mapToGlobal(position))

    def _show_symbol_details(self, symbol: str) -> None:
        """Show detailed symbol information."""
        symbol_data = self.market_data_model.get_symbol_data(symbol)
        if not symbol_data:
            return

        # Format details
        details = f"""
<h3>{symbol}</h3>
<table>
<tr><td><b>Bid:</b></td><td>{symbol_data.get('Bid', 'N/A')}</td></tr>
<tr><td><b>Ask:</b></td><td>{symbol_data.get('Ask', 'N/A')}</td></tr>
<tr><td><b>Spread:</b></td><td>{symbol_data.get('Spread', 'N/A')}</td></tr>
<tr><td><b>Spread (Pips):</b></td><td>{symbol_data.get('Spread (Pips)', 'N/A')}</td></tr>
<tr><td><b>Change %:</b></td><td>{symbol_data.get('Change %', 'N/A')}%</td></tr>
<tr><td><b>Volume:</b></td><td>{symbol_data.get('Volume', 'N/A')}</td></tr>
<tr><td><b>Positions:</b></td><td>{symbol_data.get('Positions', 'N/A')}</td></tr>
<tr><td><b>Last Update:</b></td><td>{symbol_data.get('Last Update', 'N/A')}</td></tr>
</table>
        """.strip()

        # Show message box
        msg = QMessageBox(self)
        msg.setWindowTitle(f"Market Data Details - {symbol}")
        msg.setText(details)
        msg.setIcon(QMessageBox.Icon.Information)

        if self.theme_manager:
            self.theme_manager.apply_theme_to_widget(msg)

        msg.exec()

    def _remove_symbol(self, symbol: str) -> None:
        """Remove symbol from display."""
        reply = QMessageBox.question(
            self,
            "Remove Symbol",
            f"Remove {symbol} from market data display?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            success = self.market_data_model.remove_symbol(symbol)
            if success:
                logger.info(f"Removed symbol {symbol} from market data")
            else:
                logger.error(f"Failed to remove symbol {symbol}")

    def _open_chart(self, symbol: str) -> None:
        """Open chart for symbol (placeholder)."""
        # This would open a chart window for the symbol
        QMessageBox.information(
            self,
            "Chart",
            f"Chart functionality for {symbol} not yet implemented"
        )

    def _on_data_refreshed(self) -> None:
        """Handle data refresh completion."""
        self.dataUpdated.emit()
        self._update_info_labels()

    def _on_symbol_updated(self, symbol: str) -> None:
        """Handle individual symbol update."""
        self._update_count += 1
        self._last_update_time = datetime.now()

        # Highlight the updated row if enabled
        if self._highlight_changes:
            self._highlight_symbol_row(symbol)

    def _on_trend_changed(self, symbol: str, timeframe: str, direction: str) -> None:
        """Handle trend change notification."""
        logger.debug(f"Trend changed for {symbol} on {timeframe}: {direction}")

    def _on_model_error(self, error_message: str) -> None:
        """Handle model errors."""
        self.errorOccurred.emit(error_message)

    def _highlight_symbol_row(self, symbol: str) -> None:
        """Highlight a symbol row briefly."""
        # This would implement row highlighting animation
        # For now, it's a placeholder
        pass

    def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        try:
            current_time = datetime.now()
            time_delta = (current_time - self._last_update_time).total_seconds()

            if time_delta > 0:
                self._performance_metrics['updates_per_second'] = self._update_count / max(1.0, time_delta)
            else:
                self._performance_metrics['updates_per_second'] = 0.0

            self._performance_metrics['displayed_symbols'] = self.market_data_model.rowCount()
            self._performance_metrics['total_updates'] += self._update_count

            # Emit performance update
            self.performanceUpdate.emit(self._performance_metrics.copy())

            # Reset counters
            self._update_count = 0

            # Update info labels
            self._update_info_labels()

        except Exception as e:
            logger.debug(f"Error updating performance metrics: {e}")

    def _update_info_labels(self) -> None:
        """Update information labels."""
        try:
            # Symbol count
            symbol_count = self.market_data_model.rowCount()
            self.symbol_count_label.setText(f"Symbols: {symbol_count}")

            # Update rate
            update_rate = self._performance_metrics['updates_per_second']
            self.update_rate_label.setText(f"Updates/sec: {update_rate:.1f}")

            # Last update
            last_update_str = self._last_update_time.strftime("%H:%M:%S")
            self.last_update_label.setText(f"Last update: {last_update_str}")

            # Performance status
            if update_rate > 50:
                performance = "High"
                color = "green"
            elif update_rate > 10:
                performance = "Good"
                color = "blue"
            elif update_rate > 1:
                performance = "Normal"
                color = "orange"
            else:
                performance = "Low"
                color = "red"

            self.performance_label.setText(f"Performance: {performance}")
            self.performance_label.setStyleSheet(f"color: {color};")

        except Exception as e:
            logger.debug(f"Error updating info labels: {e}")

    def _auto_refresh_data(self) -> None:
        """Auto-refresh market data."""
        try:
            self.market_data_model.refresh_data()
        except Exception as e:
            logger.debug(f"Error in auto-refresh: {e}")

    # Public interface methods

    def add_symbol(self, symbol: str) -> bool:
        """Add symbol to market data display."""
        return self.market_data_model.add_symbol(symbol)

    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from market data display."""
        return self.market_data_model.remove_symbol(symbol)

    def highlight_symbol(self, symbol: str) -> None:
        """Highlight a specific symbol."""
        # Find the symbol row and select it
        for row in range(self.market_data_model.rowCount()):
            row_data = self.market_data_model.get_row_data(row)
            if row_data and row_data['Symbol'] == symbol:
                index = self.market_data_model.index(row, 0)
                self.table_view.selectionModel().setCurrentIndex(
                    index,
                    QItemSelectionModel.SelectionFlag.ClearAndSelect |
                    QItemSelectionModel.SelectionFlag.Rows
                )
                self.table_view.scrollTo(index)
                break

    def refresh_data(self) -> None:
        """Refresh all market data."""
        self.market_data_model.refresh_data()

    def update_display(self) -> None:
        """Update the display (called by parent)."""
        self.table_view.update()

    def get_selected_symbol(self) -> Optional[str]:
        """Get currently selected symbol."""
        return self._selected_symbol

    def get_symbol_count(self) -> int:
        """Get number of symbols displayed."""
        return self.market_data_model.rowCount()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self._performance_metrics.copy()

    def set_update_interval(self, interval_ms: int) -> None:
        """Set update interval for auto-refresh."""
        if hasattr(self, 'refresh_timer'):
            if interval_ms > 0:
                self.refresh_timer.setInterval(interval_ms)
                if not self.refresh_timer.isActive():
                    self.refresh_timer.start()
            else:
                self.refresh_timer.stop()

    def retranslate_ui(self) -> None:
        """Retranslate UI elements."""
        # This would update translatable text
        # For now, it's a placeholder
        pass