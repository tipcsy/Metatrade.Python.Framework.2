"""
Symbol table model for displaying and managing trading symbols.

This module provides a PyQt6 table model for symbol selection and management
with drag-and-drop support, persistence, and integration with the symbol manager.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from decimal import Decimal

from PyQt6.QtCore import Qt, QMimeData, pyqtSignal
from PyQt6.QtGui import QColor, QFont

from src.core.logging import get_logger
from src.core.symbols.manager import get_symbol_manager
from src.core.symbols.models import SymbolInfo, SymbolStatus, SymbolType

from .base_model import BaseTableModel

logger = get_logger(__name__)


class SymbolTableModel(BaseTableModel):
    """
    Table model for symbol management and selection.

    Features:
    - Symbol list display with details
    - Drag and drop support
    - Symbol ordering and priority
    - Tradability indicators
    - Symbol type categorization
    """

    # Custom signals
    symbolSelected = pyqtSignal(str)  # symbol
    symbolOrderChanged = pyqtSignal()
    symbolActivated = pyqtSignal(str, bool)  # symbol, active

    def __init__(self, parent=None):
        """Initialize symbol table model."""
        super().__init__(parent)

        # Dependencies
        self.symbol_manager = get_symbol_manager()

        # Model configuration
        self._headers = self.get_column_headers()
        self._column_types = self.get_column_types()

        # Selection tracking
        self._selected_symbols: set = set()

        # Symbol ordering
        self._symbol_order: List[str] = []

        logger.info("Symbol table model initialized")

    def get_column_headers(self) -> List[str]:
        """Get column headers for symbol display."""
        return [
            'Active',
            'Priority',
            'Symbol',
            'Description',
            'Type',
            'Bid',
            'Ask',
            'Spread',
            'Status',
            'Tradeable'
        ]

    def get_column_types(self) -> Dict[str, type]:
        """Get column data types."""
        return {
            'Active': bool,
            'Priority': int,
            'Symbol': str,
            'Description': str,
            'Type': str,
            'Bid': Decimal,
            'Ask': Decimal,
            'Spread': Decimal,
            'Status': str,
            'Tradeable': bool
        }

    def flags(self, index) -> Qt.ItemFlag:
        """Return item flags including drag/drop support."""
        base_flags = super().flags(index)

        if not index.isValid():
            return base_flags

        # Make Active column editable
        if index.column() == 0:  # Active column
            base_flags |= Qt.ItemFlag.ItemIsUserCheckable

        # Enable drag and drop
        base_flags |= (
            Qt.ItemFlag.ItemIsDragEnabled |
            Qt.ItemFlag.ItemIsDropEnabled
        )

        return base_flags

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Override data method for checkbox and custom formatting."""
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        row_data = self.get_row_data(row)
        if not row_data:
            return None

        column_name = self._headers[col]

        # Handle checkbox for Active column
        if column_name == 'Active':
            if role == Qt.ItemDataRole.CheckStateRole:
                active = row_data.get('Active', False)
                return Qt.CheckState.Checked if active else Qt.CheckState.Unchecked
            elif role == Qt.ItemDataRole.DisplayRole:
                return ""  # No text for checkbox column

        # Handle priority display
        elif column_name == 'Priority':
            if role == Qt.ItemDataRole.DisplayRole:
                priority = row_data.get('Priority', 0)
                return f"#{priority}" if priority > 0 else ""

        return super().data(index, role)

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        """Handle data editing, particularly for checkboxes."""
        if not index.isValid():
            return False

        column_name = self._headers[index.column()]

        # Handle Active column checkbox
        if column_name == 'Active' and role == Qt.ItemDataRole.CheckStateRole:
            row_data = self.get_row_data(index.row())
            if row_data:
                symbol = row_data['Symbol']
                active = value == Qt.CheckState.Checked

                # Update model data
                row_data['Active'] = active
                self.update_row(index.row(), row_data)

                # Emit signal
                self.symbolActivated.emit(symbol, active)

                return True

        return super().setData(index, value, role)

    def _get_foreground_color(self, value: Any, column_name: str) -> Optional[QColor]:
        """Get foreground color based on column and value."""
        if column_name == 'Status':
            status_colors = {
                'ACTIVE': QColor(34, 139, 34),      # Green
                'INACTIVE': QColor(128, 128, 128),  # Gray
                'ERROR': QColor(220, 20, 60),       # Red
                'DISABLED': QColor(169, 169, 169),  # Dark Gray
            }
            return status_colors.get(str(value).upper())

        elif column_name == 'Tradeable':
            if isinstance(value, bool):
                return QColor(34, 139, 34) if value else QColor(220, 20, 60)

        elif column_name == 'Type':
            type_colors = {
                'FOREX': QColor(0, 123, 255),       # Blue
                'CRYPTO': QColor(255, 193, 7),      # Amber
                'STOCK': QColor(40, 167, 69),       # Green
                'COMMODITY': QColor(220, 53, 69),   # Red
                'INDEX': QColor(108, 117, 125),     # Gray
            }
            return type_colors.get(str(value).upper())

        return super()._get_foreground_color(value, column_name)

    def _get_font(self, value: Any, column_name: str) -> Optional[QFont]:
        """Get font styling for specific cells."""
        if column_name == 'Symbol':
            font = QFont()
            font.setBold(True)
            return font

        elif column_name in ['Bid', 'Ask', 'Spread']:
            return QFont("Consolas", 9)

        return super()._get_font(value, column_name)

    def _format_display_value(self, value: Any, column_name: str) -> str:
        """Format display values for symbols."""
        if column_name in ['Bid', 'Ask', 'Spread'] and value:
            return f"{float(value):.5f}"

        elif column_name == 'Type' and isinstance(value, str):
            return value.replace('_', ' ').title()

        elif column_name == 'Status' and isinstance(value, str):
            return value.replace('_', ' ').title()

        return super()._format_display_value(value, column_name)

    def supportedDropActions(self) -> Qt.DropAction:
        """Return supported drop actions."""
        return Qt.DropAction.MoveAction

    def mimeTypes(self) -> List[str]:
        """Return supported MIME types for drag and drop."""
        return ['application/x-symbol-list']

    def mimeData(self, indexes) -> QMimeData:
        """Create MIME data for drag operation."""
        mime_data = QMimeData()

        # Get symbols from selected rows
        symbols = []
        rows = set()
        for index in indexes:
            if index.column() == 2:  # Symbol column
                rows.add(index.row())

        for row in sorted(rows):
            row_data = self.get_row_data(row)
            if row_data:
                symbols.append(row_data['Symbol'])

        # Encode symbol list
        symbol_data = '\n'.join(symbols)
        mime_data.setData('application/x-symbol-list', symbol_data.encode())

        return mime_data

    def dropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        row: int,
        column: int,
        parent
    ) -> bool:
        """Handle drop operation for reordering symbols."""
        if action != Qt.DropAction.MoveAction:
            return False

        if not data.hasFormat('application/x-symbol-list'):
            return False

        # Decode dropped symbols
        symbol_data = data.data('application/x-symbol-list').data().decode()
        dropped_symbols = symbol_data.split('\n')

        # Determine drop position
        drop_row = row if row >= 0 else self.rowCount()

        try:
            # Reorder symbols in the model
            success = self._reorder_symbols(dropped_symbols, drop_row)

            if success:
                self.symbolOrderChanged.emit()
                logger.info(f"Reordered symbols: {dropped_symbols}")

            return success

        except Exception as e:
            logger.error(f"Error handling drop operation: {e}")
            return False

    def _reorder_symbols(self, symbols: List[str], target_position: int) -> bool:
        """Reorder symbols in the model."""
        try:
            # Find current positions of symbols
            symbol_rows = {}
            for symbol in symbols:
                row = self.find_row('Symbol', symbol)
                if row is not None:
                    symbol_rows[symbol] = row

            if not symbol_rows:
                return False

            # Extract row data for symbols to move
            symbols_data = []
            for symbol in symbols:
                row = symbol_rows[symbol]
                row_data = self.get_row_data(row)
                if row_data:
                    symbols_data.append(row_data)

            # Remove symbols from current positions (in reverse order)
            for row in sorted(symbol_rows.values(), reverse=True):
                self.remove_row(row)

            # Insert at target position
            for i, symbol_data in enumerate(symbols_data):
                # Update priority
                symbol_data['Priority'] = target_position + i + 1

                # Insert row
                self.beginInsertRows(
                    parent=None,
                    first=target_position + i,
                    last=target_position + i
                )

                with self._lock:
                    self._data.insert(target_position + i, symbol_data)

                self.endInsertRows()

            # Update priorities for all symbols
            self._update_all_priorities()

            return True

        except Exception as e:
            logger.error(f"Error reordering symbols: {e}")
            return False

    def _update_all_priorities(self) -> None:
        """Update priority numbers for all symbols."""
        try:
            with self._lock:
                for i, row_data in enumerate(self._data):
                    row_data['Priority'] = i + 1

            # Emit data changed for priority column
            priority_col = self._headers.index('Priority')
            self.dataChanged.emit(
                self.index(0, priority_col),
                self.index(self.rowCount() - 1, priority_col)
            )

        except Exception as e:
            logger.error(f"Error updating priorities: {e}")

    def load_symbols_from_manager(self) -> None:
        """Load symbols from the symbol manager."""
        try:
            symbols = self.symbol_manager.list_symbols()
            symbol_data = []

            for i, symbol_info in enumerate(symbols):
                row_data = {
                    'Active': symbol_info.status == SymbolStatus.ACTIVE,
                    'Priority': i + 1,
                    'Symbol': symbol_info.symbol,
                    'Description': symbol_info.description or '',
                    'Type': symbol_info.symbol_type.value if symbol_info.symbol_type else 'UNKNOWN',
                    'Bid': symbol_info.last_bid or Decimal('0'),
                    'Ask': symbol_info.last_ask or Decimal('0'),
                    'Spread': (symbol_info.last_ask or Decimal('0')) - (symbol_info.last_bid or Decimal('0')),
                    'Status': symbol_info.status.value if symbol_info.status else 'UNKNOWN',
                    'Tradeable': symbol_info.is_tradable
                }

                symbol_data.append(row_data)

            self.set_data(symbol_data)
            logger.info(f"Loaded {len(symbol_data)} symbols from symbol manager")

        except Exception as e:
            logger.error(f"Error loading symbols from manager: {e}")
            self.errorOccurred.emit(f"Failed to load symbols: {e}")

    def add_symbol(self, symbol: str) -> bool:
        """Add a symbol to the table."""
        try:
            # Check if already exists
            if self.find_row('Symbol', symbol) is not None:
                logger.debug(f"Symbol {symbol} already exists in table")
                return True

            # Get symbol info from manager
            symbol_info = self.symbol_manager.get_symbol(symbol)
            if not symbol_info:
                # Try to add to symbol manager first
                if not self.symbol_manager.add_symbol(symbol):
                    logger.error(f"Failed to add symbol {symbol} to symbol manager")
                    return False

                symbol_info = self.symbol_manager.get_symbol(symbol)
                if not symbol_info:
                    return False

            # Create row data
            row_data = {
                'Active': True,  # New symbols are active by default
                'Priority': self.rowCount() + 1,
                'Symbol': symbol,
                'Description': symbol_info.description or '',
                'Type': symbol_info.symbol_type.value if symbol_info.symbol_type else 'UNKNOWN',
                'Bid': symbol_info.last_bid or Decimal('0'),
                'Ask': symbol_info.last_ask or Decimal('0'),
                'Spread': (symbol_info.last_ask or Decimal('0')) - (symbol_info.last_bid or Decimal('0')),
                'Status': SymbolStatus.ACTIVE.value,
                'Tradeable': symbol_info.is_tradable
            }

            # Add to model
            self.add_row(row_data)

            # Emit activation signal
            self.symbolActivated.emit(symbol, True)

            logger.info(f"Added symbol {symbol} to symbol table")
            return True

        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
            return False

    def remove_selected_symbols(self) -> List[str]:
        """Remove selected symbols from the table."""
        removed_symbols = []

        try:
            # Find selected symbols
            symbols_to_remove = []
            for row in range(self.rowCount()):
                row_data = self.get_row_data(row)
                if row_data and row_data.get('Active'):
                    symbol = row_data['Symbol']
                    if symbol in self._selected_symbols:
                        symbols_to_remove.append((row, symbol))

            # Remove in reverse order
            for row, symbol in reversed(symbols_to_remove):
                if self.remove_row(row):
                    removed_symbols.append(symbol)
                    self.symbolActivated.emit(symbol, False)

            # Update priorities
            if removed_symbols:
                self._update_all_priorities()

            logger.info(f"Removed symbols: {removed_symbols}")

        except Exception as e:
            logger.error(f"Error removing selected symbols: {e}")

        return removed_symbols

    def get_active_symbols(self) -> List[str]:
        """Get list of active symbols."""
        active_symbols = []

        for row in range(self.rowCount()):
            row_data = self.get_row_data(row)
            if row_data and row_data.get('Active'):
                active_symbols.append(row_data['Symbol'])

        return active_symbols

    def set_symbol_selection(self, symbols: set) -> None:
        """Set selected symbols."""
        self._selected_symbols = symbols.copy()

    def get_symbol_selection(self) -> set:
        """Get selected symbols."""
        return self._selected_symbols.copy()

    def move_symbol_up(self, symbol: str) -> bool:
        """Move symbol up in priority."""
        row = self.find_row('Symbol', symbol)
        if row is None or row == 0:
            return False

        return self._swap_symbols(row, row - 1)

    def move_symbol_down(self, symbol: str) -> bool:
        """Move symbol down in priority."""
        row = self.find_row('Symbol', symbol)
        if row is None or row >= self.rowCount() - 1:
            return False

        return self._swap_symbols(row, row + 1)

    def _swap_symbols(self, row1: int, row2: int) -> bool:
        """Swap two symbols in the table."""
        try:
            data1 = self.get_row_data(row1)
            data2 = self.get_row_data(row2)

            if not data1 or not data2:
                return False

            # Swap priorities
            data1['Priority'], data2['Priority'] = data2['Priority'], data1['Priority']

            # Update rows
            self.update_row(row1, data2)
            self.update_row(row2, data1)

            self.symbolOrderChanged.emit()
            return True

        except Exception as e:
            logger.error(f"Error swapping symbols at rows {row1}, {row2}: {e}")
            return False