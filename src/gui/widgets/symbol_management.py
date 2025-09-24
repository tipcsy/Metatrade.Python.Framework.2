"""
Symbol management widget for the MetaTrader Python Framework.

This module provides a comprehensive widget for managing trading symbols,
including selection, ordering, and real-time status monitoring.
"""

from __future__ import annotations

from typing import Optional, List, Set
import asyncio

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTableView, QHeaderView,
    QMessageBox, QLineEdit, QSplitter, QGroupBox,
    QCheckBox, QSpinBox, QProgressBar, QFrame
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QAbstractItemModel,
    QItemSelectionModel, QModelIndex
)
from PyQt6.QtGui import QFont, QPalette, QIcon

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.symbols.manager import get_symbol_manager
from src.mt5.connection import get_mt5_session_manager

from ..models import SymbolTableModel
from ..themes import ThemeManager
from ..localization import LocalizationManager

logger = get_logger(__name__)


class SymbolManagementWidget(QWidget):
    """
    Widget for comprehensive symbol management.

    Features:
    - MT5 symbol selection and search
    - Symbol list with drag-and-drop reordering
    - Add/Remove functionality with persistence
    - Up/Down priority management
    - Symbol information popup
    - Real-time status updates
    """

    # Custom signals
    symbolAdded = pyqtSignal(str)           # symbol
    symbolRemoved = pyqtSignal(str)         # symbol
    symbolSelectionChanged = pyqtSignal(str) # symbol
    symbolOrderChanged = pyqtSignal()
    symbolActivated = pyqtSignal(str, bool) # symbol, active

    def __init__(
        self,
        theme_manager: Optional[ThemeManager] = None,
        localization_manager: Optional[LocalizationManager] = None,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize symbol management widget.

        Args:
            theme_manager: Theme manager for styling
            localization_manager: Localization manager for translations
            parent: Parent widget
        """
        super().__init__(parent)

        # Dependencies
        self.settings = get_settings()
        self.theme_manager = theme_manager
        self.localization_manager = localization_manager
        self.symbol_manager = get_symbol_manager()
        self.mt5_session_manager = get_mt5_session_manager()

        # State
        self._is_loading = False
        self._selected_symbols: Set[str] = set()
        self._mt5_symbols: List[str] = []

        # Models
        self.symbol_model = SymbolTableModel()

        # UI Components
        self._create_ui()
        self._setup_connections()
        self._setup_timers()

        # Load initial data
        self._load_initial_data()

        logger.info("Symbol management widget initialized")

    def _create_ui(self) -> None:
        """Create the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Create sections
        self._create_symbol_selection_section(main_layout)
        self._create_symbol_list_section(main_layout)
        self._create_control_buttons_section(main_layout)
        self._create_status_section(main_layout)

    def _create_symbol_selection_section(self, parent_layout: QVBoxLayout) -> None:
        """Create symbol selection section."""
        # Group box for symbol selection
        selection_group = QGroupBox("Symbol Selection")
        selection_layout = QVBoxLayout(selection_group)

        # Search/filter row
        search_layout = QHBoxLayout()

        # Search label and input
        search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter symbol or search term...")
        self.search_input.setClearButtonEnabled(True)

        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input, 1)

        selection_layout.addLayout(search_layout)

        # Symbol selection row
        symbol_layout = QHBoxLayout()

        # Symbol dropdown
        symbol_label = QLabel("Symbol:")
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEditable(True)
        self.symbol_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.symbol_combo.setMinimumWidth(200)

        # Add button
        self.add_button = QPushButton("Add Symbol")
        self.add_button.setMinimumWidth(100)

        # Refresh MT5 symbols button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Refresh available symbols from MT5")

        symbol_layout.addWidget(symbol_label)
        symbol_layout.addWidget(self.symbol_combo, 1)
        symbol_layout.addWidget(self.add_button)
        symbol_layout.addWidget(self.refresh_button)

        selection_layout.addLayout(symbol_layout)

        parent_layout.addWidget(selection_group)

    def _create_symbol_list_section(self, parent_layout: QVBoxLayout) -> None:
        """Create symbol list section."""
        # Group box for symbol list
        list_group = QGroupBox("Active Symbols")
        list_layout = QVBoxLayout(list_group)

        # Table view for symbols
        self.symbol_table = QTableView()
        self.symbol_table.setModel(self.symbol_model)
        self.symbol_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.symbol_table.setDragDropMode(QTableView.DragDropMode.InternalMove)
        self.symbol_table.setDragEnabled(True)
        self.symbol_table.setAcceptDrops(True)
        self.symbol_table.setDropIndicatorShown(True)

        # Configure table headers
        header = self.symbol_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Active
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Priority
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)           # Symbol
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)           # Description

        # Hide less important columns by default
        self.symbol_table.hideColumn(4)  # Type
        self.symbol_table.hideColumn(5)  # Bid
        self.symbol_table.hideColumn(6)  # Ask
        self.symbol_table.hideColumn(7)  # Spread
        self.symbol_table.hideColumn(8)  # Status

        list_layout.addWidget(self.symbol_table)

        parent_layout.addWidget(list_group)

    def _create_control_buttons_section(self, parent_layout: QVBoxLayout) -> None:
        """Create control buttons section."""
        # Button layout
        button_layout = QHBoxLayout()

        # Remove button
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.setEnabled(False)

        # Order control buttons
        self.up_button = QPushButton("↑ Up")
        self.up_button.setToolTip("Move selected symbol up")
        self.up_button.setEnabled(False)

        self.down_button = QPushButton("↓ Down")
        self.down_button.setToolTip("Move selected symbol down")
        self.down_button.setEnabled(False)

        # Spacer
        button_layout.addStretch()

        # Info button
        self.info_button = QPushButton("Symbol Info")
        self.info_button.setToolTip("Show detailed symbol information")
        self.info_button.setEnabled(False)

        # Add buttons to layout
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.up_button)
        button_layout.addWidget(self.down_button)
        button_layout.addStretch()
        button_layout.addWidget(self.info_button)

        parent_layout.addLayout(button_layout)

    def _create_status_section(self, parent_layout: QVBoxLayout) -> None:
        """Create status section."""
        # Status frame
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Shape.Box)
        status_layout = QVBoxLayout(status_frame)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Statistics
        stats_layout = QHBoxLayout()

        self.total_symbols_label = QLabel("Total: 0")
        self.active_symbols_label = QLabel("Active: 0")
        self.mt5_connection_label = QLabel("MT5: Disconnected")

        stats_layout.addWidget(self.total_symbols_label)
        stats_layout.addWidget(self.active_symbols_label)
        stats_layout.addStretch()
        stats_layout.addWidget(self.mt5_connection_label)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        status_layout.addLayout(stats_layout)

        parent_layout.addWidget(status_frame)

    def _setup_connections(self) -> None:
        """Setup signal connections."""
        try:
            # Search input
            self.search_input.textChanged.connect(self._filter_symbols)

            # Symbol combo and buttons
            self.add_button.clicked.connect(self._add_selected_symbol)
            self.refresh_button.clicked.connect(self._refresh_mt5_symbols)

            # Control buttons
            self.remove_button.clicked.connect(self._remove_selected_symbols)
            self.up_button.clicked.connect(self._move_symbol_up)
            self.down_button.clicked.connect(self._move_symbol_down)
            self.info_button.clicked.connect(self._show_symbol_info)

            # Table selection
            selection_model = self.symbol_table.selectionModel()
            selection_model.selectionChanged.connect(self._on_selection_changed)
            self.symbol_table.doubleClicked.connect(self._on_symbol_double_clicked)

            # Model signals
            self.symbol_model.symbolActivated.connect(self._on_symbol_activated)
            self.symbol_model.symbolOrderChanged.connect(self._on_symbol_order_changed)
            self.symbol_model.errorOccurred.connect(self._show_error_message)

            logger.debug("Signal connections established")

        except Exception as e:
            logger.error(f"Error setting up connections: {e}")

    def _setup_timers(self) -> None:
        """Setup update timers."""
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(5000)  # Update every 5 seconds

        # Symbol refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_symbol_data)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds

    def _load_initial_data(self) -> None:
        """Load initial data."""
        try:
            # Load symbols from symbol manager
            self.symbol_model.load_symbols_from_manager()

            # Load MT5 symbols asynchronously
            QTimer.singleShot(1000, self._refresh_mt5_symbols)

            logger.debug("Initial data loaded")

        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
            self._show_error_message(f"Failed to load initial data: {e}")

    def _filter_symbols(self, filter_text: str) -> None:
        """Filter available symbols based on search text."""
        if not filter_text:
            self.symbol_combo.clear()
            self.symbol_combo.addItems(self._mt5_symbols)
            return

        filter_text = filter_text.upper()
        filtered_symbols = [
            symbol for symbol in self._mt5_symbols
            if filter_text in symbol.upper()
        ]

        self.symbol_combo.clear()
        self.symbol_combo.addItems(filtered_symbols)

    def _add_selected_symbol(self) -> None:
        """Add the selected symbol to the active list."""
        symbol = self.symbol_combo.currentText().strip().upper()

        if not symbol:
            self._show_error_message("Please select or enter a symbol")
            return

        if self._is_loading:
            return

        try:
            self._set_loading(True, f"Adding symbol {symbol}...")

            # Add to symbol model
            success = self.symbol_model.add_symbol(symbol)

            if success:
                self.symbolAdded.emit(symbol)
                self.status_label.setText(f"Added symbol: {symbol}")

                # Clear the combo box
                self.symbol_combo.setCurrentText("")

                logger.info(f"Added symbol: {symbol}")
            else:
                self._show_error_message(f"Failed to add symbol: {symbol}")

        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
            self._show_error_message(f"Error adding symbol: {e}")

        finally:
            self._set_loading(False)

    def _remove_selected_symbols(self) -> None:
        """Remove selected symbols from the active list."""
        selection_model = self.symbol_table.selectionModel()
        selected_rows = selection_model.selectedRows()

        if not selected_rows:
            self._show_error_message("Please select symbols to remove")
            return

        try:
            # Get symbols to remove
            symbols_to_remove = []
            for index in selected_rows:
                row_data = self.symbol_model.get_row_data(index.row())
                if row_data:
                    symbols_to_remove.append(row_data['Symbol'])

            if not symbols_to_remove:
                return

            # Confirm removal
            reply = QMessageBox.question(
                self,
                "Remove Symbols",
                f"Remove {len(symbols_to_remove)} symbol(s)?\n\n" +
                "\n".join(symbols_to_remove),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Remove symbols
                for symbol in symbols_to_remove:
                    row = self.symbol_model.find_row('Symbol', symbol)
                    if row is not None:
                        self.symbol_model.remove_row(row)
                        self.symbolRemoved.emit(symbol)

                self.status_label.setText(f"Removed {len(symbols_to_remove)} symbols")
                logger.info(f"Removed symbols: {symbols_to_remove}")

        except Exception as e:
            logger.error(f"Error removing symbols: {e}")
            self._show_error_message(f"Error removing symbols: {e}")

    def _move_symbol_up(self) -> None:
        """Move selected symbol up in priority."""
        current_row = self._get_selected_row()
        if current_row is None:
            return

        row_data = self.symbol_model.get_row_data(current_row)
        if row_data:
            symbol = row_data['Symbol']
            if self.symbol_model.move_symbol_up(symbol):
                # Update selection to moved position
                new_index = self.symbol_model.index(current_row - 1, 0)
                self.symbol_table.selectionModel().setCurrentIndex(
                    new_index, QItemSelectionModel.SelectionFlag.ClearAndSelect
                )

    def _move_symbol_down(self) -> None:
        """Move selected symbol down in priority."""
        current_row = self._get_selected_row()
        if current_row is None:
            return

        row_data = self.symbol_model.get_row_data(current_row)
        if row_data:
            symbol = row_data['Symbol']
            if self.symbol_model.move_symbol_down(symbol):
                # Update selection to moved position
                new_index = self.symbol_model.index(current_row + 1, 0)
                self.symbol_table.selectionModel().setCurrentIndex(
                    new_index, QItemSelectionModel.SelectionFlag.ClearAndSelect
                )

    def _get_selected_row(self) -> Optional[int]:
        """Get currently selected row."""
        selection_model = self.symbol_table.selectionModel()
        selected_rows = selection_model.selectedRows()
        return selected_rows[0].row() if selected_rows else None

    def _show_symbol_info(self) -> None:
        """Show detailed symbol information."""
        current_row = self._get_selected_row()
        if current_row is None:
            return

        row_data = self.symbol_model.get_row_data(current_row)
        if not row_data:
            return

        symbol = row_data['Symbol']

        try:
            # Get detailed symbol info from symbol manager
            symbol_info = self.symbol_manager.get_symbol(symbol)
            if not symbol_info:
                self._show_error_message(f"Symbol information not available for {symbol}")
                return

            # Create info dialog
            info_text = self._format_symbol_info(symbol_info)

            msg = QMessageBox(self)
            msg.setWindowTitle(f"Symbol Information - {symbol}")
            msg.setText(f"<h3>{symbol}</h3>")
            msg.setDetailedText(info_text)
            msg.setIcon(QMessageBox.Icon.Information)

            if self.theme_manager:
                self.theme_manager.apply_theme_to_widget(msg)

            msg.exec()

        except Exception as e:
            logger.error(f"Error showing symbol info for {symbol}: {e}")
            self._show_error_message(f"Error loading symbol information: {e}")

    def _format_symbol_info(self, symbol_info) -> str:
        """Format symbol information for display."""
        return f"""
Symbol: {symbol_info.symbol}
Description: {symbol_info.description or 'N/A'}
Type: {symbol_info.symbol_type.value if symbol_info.symbol_type else 'Unknown'}
Status: {symbol_info.status.value if symbol_info.status else 'Unknown'}

Market Data:
- Bid: {symbol_info.last_bid or 'N/A'}
- Ask: {symbol_info.last_ask or 'N/A'}
- Volume: {symbol_info.last_volume or 'N/A'}

Trading Info:
- Tradeable: {'Yes' if symbol_info.is_tradable else 'No'}
- Visible: {'Yes' if symbol_info.is_visible else 'No'}

Timestamps:
- Created: {symbol_info.created_at}
- Updated: {symbol_info.updated_at}
        """.strip()

    def _refresh_mt5_symbols(self) -> None:
        """Refresh available symbols from MT5."""
        if self._is_loading:
            return

        try:
            self._set_loading(True, "Refreshing MT5 symbols...")

            # Get MT5 session
            session = self.mt5_session_manager.get_active_session()
            if not session:
                self._show_error_message("No active MT5 connection")
                return

            # Get available symbols
            symbols_info = session.symbols_get()
            if symbols_info is None:
                self._show_error_message("Failed to retrieve MT5 symbols")
                return

            # Extract symbol names
            self._mt5_symbols = [info.name for info in symbols_info if info.visible]
            self._mt5_symbols.sort()

            # Update combo box
            current_text = self.symbol_combo.currentText()
            self.symbol_combo.clear()
            self.symbol_combo.addItems(self._mt5_symbols)

            if current_text:
                index = self.symbol_combo.findText(current_text)
                if index >= 0:
                    self.symbol_combo.setCurrentIndex(index)

            self.status_label.setText(f"Loaded {len(self._mt5_symbols)} MT5 symbols")
            logger.info(f"Refreshed {len(self._mt5_symbols)} MT5 symbols")

        except Exception as e:
            logger.error(f"Error refreshing MT5 symbols: {e}")
            self._show_error_message(f"Error refreshing symbols: {e}")

        finally:
            self._set_loading(False)

    def _refresh_symbol_data(self) -> None:
        """Refresh data for all symbols in the table."""
        try:
            active_symbols = self.symbol_model.get_active_symbols()
            if active_symbols:
                # Trigger data refresh in symbol manager
                for symbol in active_symbols:
                    self.symbol_manager.update_symbol_quote(symbol)

                logger.debug(f"Refreshed data for {len(active_symbols)} symbols")

        except Exception as e:
            logger.debug(f"Error refreshing symbol data: {e}")

    def _on_selection_changed(self) -> None:
        """Handle table selection changes."""
        selection_model = self.symbol_table.selectionModel()
        has_selection = selection_model.hasSelection()
        selected_rows = selection_model.selectedRows()

        # Update button states
        self.remove_button.setEnabled(has_selection)
        self.info_button.setEnabled(len(selected_rows) == 1)
        self.up_button.setEnabled(len(selected_rows) == 1 and selected_rows[0].row() > 0)
        self.down_button.setEnabled(
            len(selected_rows) == 1 and
            selected_rows[0].row() < self.symbol_model.rowCount() - 1
        )

        # Emit selection change signal
        if len(selected_rows) == 1:
            row_data = self.symbol_model.get_row_data(selected_rows[0].row())
            if row_data:
                self.symbolSelectionChanged.emit(row_data['Symbol'])

    def _on_symbol_double_clicked(self, index: QModelIndex) -> None:
        """Handle symbol double-click."""
        if index.isValid():
            self._show_symbol_info()

    def _on_symbol_activated(self, symbol: str, active: bool) -> None:
        """Handle symbol activation/deactivation."""
        self.symbolActivated.emit(symbol, active)

    def _on_symbol_order_changed(self) -> None:
        """Handle symbol order change."""
        self.symbolOrderChanged.emit()

    def _update_status(self) -> None:
        """Update status information."""
        try:
            # Update symbol counts
            total_symbols = self.symbol_model.rowCount()
            active_symbols = len(self.symbol_model.get_active_symbols())

            self.total_symbols_label.setText(f"Total: {total_symbols}")
            self.active_symbols_label.setText(f"Active: {active_symbols}")

            # Update MT5 connection status
            session = self.mt5_session_manager.get_active_session()
            if session and session.terminal_info():
                self.mt5_connection_label.setText("MT5: Connected")
                self.mt5_connection_label.setStyleSheet("color: green;")
            else:
                self.mt5_connection_label.setText("MT5: Disconnected")
                self.mt5_connection_label.setStyleSheet("color: red;")

        except Exception as e:
            logger.debug(f"Error updating status: {e}")

    def _set_loading(self, loading: bool, message: str = "") -> None:
        """Set loading state."""
        self._is_loading = loading

        # Update UI elements
        self.add_button.setEnabled(not loading)
        self.refresh_button.setEnabled(not loading)
        self.remove_button.setEnabled(not loading and self.symbol_table.selectionModel().hasSelection())

        self.progress_bar.setVisible(loading)
        if loading:
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            if message:
                self.status_label.setText(message)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)

    def _show_error_message(self, message: str) -> None:
        """Show error message to user."""
        QMessageBox.critical(self, "Error", message)

    def refresh_symbols(self) -> None:
        """Public method to refresh symbols."""
        self.symbol_model.load_symbols_from_manager()
        self._refresh_mt5_symbols()

    def get_active_symbols(self) -> List[str]:
        """Get list of active symbols."""
        return self.symbol_model.get_active_symbols()

    def add_symbol(self, symbol: str) -> bool:
        """Add symbol programmatically."""
        return self.symbol_model.add_symbol(symbol)

    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol programmatically."""
        row = self.symbol_model.find_row('Symbol', symbol)
        if row is not None:
            success = self.symbol_model.remove_row(row)
            if success:
                self.symbolRemoved.emit(symbol)
            return success
        return False

    def retranslate_ui(self) -> None:
        """Retranslate UI elements."""
        # This would be implemented with actual translations
        # For now, it's a placeholder
        pass