"""
Base model classes for GUI data models.

This module provides base classes for PyQt6 table models with common
functionality like sorting, filtering, and performance optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from decimal import Decimal

from PyQt6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt,
    pyqtSignal, QTimer, QMutex, QMutexLocker
)
from PyQt6.QtGui import QColor, QFont

from src.core.logging import get_logger

logger = get_logger(__name__)


class BaseTableModel(QAbstractTableModel, ABC):
    """
    Base class for table models with common functionality.

    Provides thread-safe data access, performance optimizations,
    and common table model operations.
    """

    # Custom signals
    dataRefreshed = pyqtSignal()
    errorOccurred = pyqtSignal(str)

    def __init__(self, parent=None):
        """
        Initialize base table model.

        Args:
            parent: Parent QObject
        """
        super().__init__(parent)

        # Data storage
        self._data: List[Dict[str, Any]] = []
        self._headers: List[str] = []
        self._column_types: Dict[str, type] = {}

        # Thread safety
        self._mutex = QMutex()

        # Performance optimization
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._emit_data_changed)
        self._pending_updates: set = set()

        # Display options
        self._decimal_places = 5
        self._date_format = "yyyy-MM-dd hh:mm:ss"

        logger.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def get_column_headers(self) -> List[str]:
        """Get column headers for the model."""
        pass

    @abstractmethod
    def get_column_types(self) -> Dict[str, type]:
        """Get column data types."""
        pass

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return number of rows."""
        with QMutexLocker(self._mutex):
            return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return number of columns."""
        return len(self._headers)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        """Return header data."""
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                if 0 <= section < len(self._headers):
                    return self._headers[section]
            else:
                return str(section + 1)
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return data for the given index and role."""
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        with QMutexLocker(self._mutex):
            if not (0 <= row < len(self._data) and 0 <= col < len(self._headers)):
                return None

            row_data = self._data[row]
            column_name = self._headers[col]
            value = row_data.get(column_name)

        return self._format_data(value, column_name, role)

    def _format_data(self, value: Any, column_name: str, role: int) -> Any:
        """Format data based on role and column type."""
        if role == Qt.ItemDataRole.DisplayRole:
            return self._format_display_value(value, column_name)

        elif role == Qt.ItemDataRole.ForegroundRole:
            return self._get_foreground_color(value, column_name)

        elif role == Qt.ItemDataRole.BackgroundRole:
            return self._get_background_color(value, column_name)

        elif role == Qt.ItemDataRole.FontRole:
            return self._get_font(value, column_name)

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return self._get_alignment(column_name)

        elif role == Qt.ItemDataRole.ToolTipRole:
            return self._get_tooltip(value, column_name)

        return None

    def _format_display_value(self, value: Any, column_name: str) -> str:
        """Format value for display."""
        if value is None:
            return ""

        column_type = self._column_types.get(column_name, str)

        if column_type == Decimal or isinstance(value, Decimal):
            return f"{float(value):.{self._decimal_places}f}"

        elif column_type == float or isinstance(value, float):
            return f"{value:.{self._decimal_places}f}"

        elif column_type == int or isinstance(value, int):
            return f"{value:,}"

        elif column_type == bool or isinstance(value, bool):
            return "Yes" if value else "No"

        return str(value)

    def _get_foreground_color(self, value: Any, column_name: str) -> Optional[QColor]:
        """Get foreground color for value."""
        # Override in subclasses for custom coloring
        return None

    def _get_background_color(self, value: Any, column_name: str) -> Optional[QColor]:
        """Get background color for value."""
        # Override in subclasses for custom coloring
        return None

    def _get_font(self, value: Any, column_name: str) -> Optional[QFont]:
        """Get font for value."""
        # Override in subclasses for custom fonts
        return None

    def _get_alignment(self, column_name: str) -> Qt.AlignmentFlag:
        """Get alignment for column."""
        column_type = self._column_types.get(column_name, str)

        if column_type in (int, float, Decimal):
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

    def _get_tooltip(self, value: Any, column_name: str) -> Optional[str]:
        """Get tooltip for value."""
        # Override in subclasses for custom tooltips
        return None

    def setData(
        self,
        index: QModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole
    ) -> bool:
        """Set data for the given index."""
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False

        row = index.row()
        col = index.column()

        with QMutexLocker(self._mutex):
            if not (0 <= row < len(self._data) and 0 <= col < len(self._headers)):
                return False

            column_name = self._headers[col]
            self._data[row][column_name] = value

        self.dataChanged.emit(index, index, [role])
        return True

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Return item flags."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags

        return (
            Qt.ItemFlag.ItemIsEnabled |
            Qt.ItemFlag.ItemIsSelectable
        )

    def sort(
        self,
        column: int,
        order: Qt.SortOrder = Qt.SortOrder.AscendingOrder
    ) -> None:
        """Sort the model data."""
        if not (0 <= column < len(self._headers)):
            return

        column_name = self._headers[column]
        reverse = (order == Qt.SortOrder.DescendingOrder)

        self.layoutAboutToBeChanged.emit()

        with QMutexLocker(self._mutex):
            try:
                self._data.sort(
                    key=lambda x: x.get(column_name, ''),
                    reverse=reverse
                )
            except Exception as e:
                logger.error(f"Error sorting column {column_name}: {e}")

        self.layoutChanged.emit()

    def clear_data(self) -> None:
        """Clear all data."""
        self.beginResetModel()

        with QMutexLocker(self._mutex):
            self._data.clear()

        self.endResetModel()
        logger.debug("Model data cleared")

    def set_data(self, data: List[Dict[str, Any]]) -> None:
        """Set model data."""
        self.beginResetModel()

        with QMutexLocker(self._mutex):
            self._data = data.copy()

        self.endResetModel()
        self.dataRefreshed.emit()
        logger.debug(f"Model data set: {len(data)} rows")

    def add_row(self, row_data: Dict[str, Any]) -> None:
        """Add a single row to the model."""
        row_count = self.rowCount()

        self.beginInsertRows(QModelIndex(), row_count, row_count)

        with QMutexLocker(self._mutex):
            self._data.append(row_data)

        self.endInsertRows()

    def remove_row(self, row: int) -> bool:
        """Remove a row from the model."""
        if not (0 <= row < len(self._data)):
            return False

        self.beginRemoveRows(QModelIndex(), row, row)

        with QMutexLocker(self._mutex):
            del self._data[row]

        self.endRemoveRows()
        return True

    def update_row(self, row: int, row_data: Dict[str, Any]) -> bool:
        """Update a specific row."""
        with QMutexLocker(self._mutex):
            if not (0 <= row < len(self._data)):
                return False

            self._data[row].update(row_data)

        # Schedule update notification
        self._schedule_update(row)
        return True

    def find_row(self, key_column: str, key_value: Any) -> Optional[int]:
        """Find row index by key column value."""
        with QMutexLocker(self._mutex):
            for i, row_data in enumerate(self._data):
                if row_data.get(key_column) == key_value:
                    return i
        return None

    def get_row_data(self, row: int) -> Optional[Dict[str, Any]]:
        """Get data for a specific row."""
        with QMutexLocker(self._mutex):
            if 0 <= row < len(self._data):
                return self._data[row].copy()
        return None

    def _schedule_update(self, row: int) -> None:
        """Schedule update notification for performance."""
        self._pending_updates.add(row)

        if not self._update_timer.isActive():
            self._update_timer.start(50)  # 50ms batch delay

    def _emit_data_changed(self) -> None:
        """Emit dataChanged for pending updates."""
        if not self._pending_updates:
            return

        min_row = min(self._pending_updates)
        max_row = max(self._pending_updates)

        top_left = self.index(min_row, 0)
        bottom_right = self.index(max_row, self.columnCount() - 1)

        self.dataChanged.emit(top_left, bottom_right)
        self._pending_updates.clear()

    def set_decimal_places(self, places: int) -> None:
        """Set number of decimal places for numeric display."""
        self._decimal_places = max(0, places)
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(self.rowCount() - 1, self.columnCount() - 1)
        )

    def refresh_data(self) -> None:
        """Refresh the model data from source."""
        # Override in subclasses to implement data refresh
        pass