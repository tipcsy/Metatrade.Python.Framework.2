"""
Market data model for displaying real-time market information in tables.

This module provides a PyQt6 table model for displaying market data with
real-time updates, color coding, and performance optimization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from decimal import Decimal
from datetime import datetime
from enum import Enum

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.symbols.manager import get_symbol_manager
from src.core.data.collector import get_data_collection_manager

from .base_model import BaseTableModel

logger = get_logger(__name__)


class TrendDirection(Enum):
    """Trend direction enumeration."""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


class MarketDataModel(BaseTableModel):
    """
    Table model for displaying real-time market data.

    Features:
    - Real-time bid/ask/spread display
    - Position count tracking
    - Multi-timeframe trend indicators
    - Color-coded rows and cells
    - Performance-optimized updates
    """

    # Custom signals
    symbolDataUpdated = pyqtSignal(str)  # symbol
    trendChanged = pyqtSignal(str, str, str)  # symbol, timeframe, direction

    def __init__(self, parent=None):
        """Initialize market data model."""
        super().__init__(parent)

        # Configuration
        self.settings = get_settings()

        # Dependencies
        self.symbol_manager = get_symbol_manager()
        self.data_collector = get_data_collection_manager()

        # Model configuration
        self._headers = self.get_column_headers()
        self._column_types = self.get_column_types()

        # Trend analysis configuration
        self.trend_timeframes = [
            'M1', 'M5', 'M15', 'H1', 'H4', 'D1'
        ]
        self._trend_data: Dict[str, Dict[str, TrendDirection]] = {}

        # Color coding thresholds
        self._high_spread_threshold = Decimal('0.0010')  # 1 pip for major pairs
        self._price_change_threshold = Decimal('0.0001')

        # Update optimization
        self._last_update_time: Dict[str, datetime] = {}
        self._update_batch_size = 50

        # Performance metrics
        self._update_count = 0
        self._performance_timer = QTimer()
        self._performance_timer.timeout.connect(self._log_performance_metrics)
        self._performance_timer.start(30000)  # Log every 30 seconds

        logger.info("Market data model initialized")

    def get_column_headers(self) -> List[str]:
        """Get column headers for market data display."""
        base_headers = [
            'Symbol',
            'Bid',
            'Ask',
            'Spread',
            'Spread (Pips)',
            'Change %',
            'Volume',
            'Positions',
            'Last Update'
        ]

        # Add trend columns for configured timeframes
        for timeframe in self.trend_timeframes:
            base_headers.append(f'Trend {timeframe}')

        return base_headers

    def get_column_types(self) -> Dict[str, type]:
        """Get column data types."""
        types = {
            'Symbol': str,
            'Bid': Decimal,
            'Ask': Decimal,
            'Spread': Decimal,
            'Spread (Pips)': Decimal,
            'Change %': float,
            'Volume': int,
            'Positions': int,
            'Last Update': datetime,
        }

        # Add trend column types
        for timeframe in self.trend_timeframes:
            types[f'Trend {timeframe}'] = str

        return types

    def _format_display_value(self, value: Any, column_name: str) -> str:
        """Format value for display with market-specific formatting."""
        if value is None:
            return ""

        if column_name in ['Bid', 'Ask']:
            return f"{float(value):.{self._decimal_places}f}"

        elif column_name == 'Spread':
            return f"{float(value):.{self._decimal_places + 1}f}"

        elif column_name == 'Spread (Pips)':
            return f"{float(value):.1f}"

        elif column_name == 'Change %':
            sign = '+' if value > 0 else ''
            return f"{sign}{value:.2f}%"

        elif column_name == 'Volume':
            return f"{value:,}" if value else "0"

        elif column_name == 'Positions':
            return str(value) if value else "0"

        elif column_name == 'Last Update':
            if isinstance(value, datetime):
                return value.strftime("%H:%M:%S")

        elif column_name.startswith('Trend '):
            return self._get_trend_arrow(value)

        return super()._format_display_value(value, column_name)

    def _get_trend_arrow(self, direction: Any) -> str:
        """Get trend arrow for display."""
        if isinstance(direction, TrendDirection):
            direction = direction.value

        trend_arrows = {
            'up': '⬆️',
            'down': '⬇️',
            'sideways': '➡️'
        }

        return trend_arrows.get(str(direction).lower(), '❓')

    def _get_foreground_color(self, value: Any, column_name: str) -> Optional[QColor]:
        """Get foreground color based on value and column."""
        if column_name == 'Change %' and isinstance(value, (int, float)):
            if value > 0:
                return QColor(34, 139, 34)  # Forest Green
            elif value < 0:
                return QColor(220, 20, 60)  # Crimson

        elif column_name.startswith('Trend '):
            if isinstance(value, TrendDirection):
                trend_colors = {
                    TrendDirection.UP: QColor(34, 139, 34),     # Green
                    TrendDirection.DOWN: QColor(220, 20, 60),   # Red
                    TrendDirection.SIDEWAYS: QColor(255, 165, 0) # Orange
                }
                return trend_colors.get(value)

        return None

    def _get_background_color(self, value: Any, column_name: str) -> Optional[QColor]:
        """Get background color for row highlighting."""
        # This is handled at row level in _get_row_background_color
        return None

    def _get_font(self, value: Any, column_name: str) -> Optional[QFont]:
        """Get font styling for specific cells."""
        if column_name in ['Bid', 'Ask', 'Spread']:
            font = QFont("Consolas", 9)
            font.setBold(True)
            return font

        elif column_name.startswith('Trend '):
            font = QFont()
            font.setPointSize(12)  # Larger for trend arrows
            return font

        return None

    def _get_row_background_color(self, row_data: Dict[str, Any]) -> Optional[QColor]:
        """Get background color for entire row based on conditions."""
        try:
            spread = row_data.get('Spread')
            if spread and isinstance(spread, (Decimal, float)):
                spread_decimal = Decimal(str(spread))
                if spread_decimal > self._high_spread_threshold:
                    return QColor(139, 69, 19, 50)  # Light brown for high spread

            # Normal condition - very light gray
            return QColor(248, 248, 255, 25)

        except Exception as e:
            logger.debug(f"Error determining row background color: {e}")
            return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        """Override data method to add row-level background coloring."""
        if role == Qt.ItemDataRole.BackgroundRole:
            row = index.row()
            row_data = self.get_row_data(row)
            if row_data:
                return self._get_row_background_color(row_data)

        return super().data(index, role)

    def add_symbol(self, symbol: str) -> bool:
        """Add symbol to market data display."""
        try:
            # Check if symbol already exists
            if self.find_row('Symbol', symbol) is not None:
                logger.debug(f"Symbol {symbol} already in model")
                return True

            # Get symbol info from symbol manager
            symbol_info = self.symbol_manager.get_symbol(symbol)
            if not symbol_info:
                logger.warning(f"Symbol {symbol} not found in symbol manager")
                return False

            # Create initial row data
            row_data = {
                'Symbol': symbol,
                'Bid': Decimal('0.0'),
                'Ask': Decimal('0.0'),
                'Spread': Decimal('0.0'),
                'Spread (Pips)': Decimal('0.0'),
                'Change %': 0.0,
                'Volume': 0,
                'Positions': 0,
                'Last Update': datetime.now()
            }

            # Initialize trend data
            for timeframe in self.trend_timeframes:
                row_data[f'Trend {timeframe}'] = TrendDirection.SIDEWAYS

            # Add row to model
            self.add_row(row_data)

            # Initialize trend tracking
            self._trend_data[symbol] = {
                tf: TrendDirection.SIDEWAYS for tf in self.trend_timeframes
            }

            # Subscribe to real-time updates
            self.symbol_manager.subscribe_to_symbol(symbol, "quote")

            logger.info(f"Added symbol {symbol} to market data model")
            return True

        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
            return False

    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from market data display."""
        try:
            row = self.find_row('Symbol', symbol)
            if row is None:
                logger.debug(f"Symbol {symbol} not found in model")
                return False

            # Remove from model
            success = self.remove_row(row)

            if success:
                # Clean up trend data
                self._trend_data.pop(symbol, None)
                self._last_update_time.pop(symbol, None)

                # Unsubscribe from updates
                self.symbol_manager.unsubscribe_from_symbol(symbol)

                logger.info(f"Removed symbol {symbol} from market data model")

            return success

        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {e}")
            return False

    def update_symbol_data(
        self,
        symbol: str,
        bid: Optional[Decimal] = None,
        ask: Optional[Decimal] = None,
        volume: Optional[int] = None
    ) -> bool:
        """Update market data for a symbol."""
        try:
            row = self.find_row('Symbol', symbol)
            if row is None:
                logger.debug(f"Symbol {symbol} not found for update")
                return False

            # Get current data
            current_data = self.get_row_data(row)
            if not current_data:
                return False

            # Prepare update data
            update_data = {
                'Last Update': datetime.now()
            }

            # Calculate spread and percentage change
            if bid is not None:
                old_bid = current_data.get('Bid', Decimal('0'))
                update_data['Bid'] = bid

                # Calculate percentage change
                if old_bid and old_bid > 0:
                    change_pct = float((bid - old_bid) / old_bid * 100)
                    update_data['Change %'] = change_pct

            if ask is not None:
                update_data['Ask'] = ask

            # Calculate spread if both bid and ask are available
            current_bid = update_data.get('Bid', current_data.get('Bid'))
            current_ask = update_data.get('Ask', current_data.get('Ask'))

            if current_bid and current_ask:
                spread = current_ask - current_bid
                update_data['Spread'] = spread

                # Calculate spread in pips (assuming 5-digit pricing)
                # This is a simplified calculation - real implementation should
                # get tick size from symbol information
                pip_value = Decimal('0.00001')  # 5-digit pricing
                spread_pips = spread / pip_value
                update_data['Spread (Pips)'] = spread_pips

            if volume is not None:
                update_data['Volume'] = volume

            # Update position count (would come from trading system)
            # For now, this is placeholder
            update_data['Positions'] = current_data.get('Positions', 0)

            # Update trend analysis
            self._update_trend_analysis(symbol, update_data)

            # Update the row
            success = self.update_row(row, update_data)

            if success:
                self._update_count += 1
                self._last_update_time[symbol] = datetime.now()
                self.symbolDataUpdated.emit(symbol)

            return success

        except Exception as e:
            logger.error(f"Error updating symbol data for {symbol}: {e}")
            return False

    def _update_trend_analysis(self, symbol: str, update_data: Dict[str, Any]) -> None:
        """Update trend analysis for symbol."""
        try:
            # This is a placeholder for trend analysis
            # In the real implementation, this would:
            # 1. Get OHLC data for different timeframes
            # 2. Calculate MACD, moving averages, etc.
            # 3. Determine trend direction
            # 4. Update trend columns

            # For now, simulate trend changes occasionally
            import random

            for timeframe in self.trend_timeframes:
                current_trend = self._trend_data.get(symbol, {}).get(timeframe, TrendDirection.SIDEWAYS)

                # 5% chance to change trend on each update
                if random.random() < 0.05:
                    new_trend = random.choice(list(TrendDirection))
                    self._trend_data.setdefault(symbol, {})[timeframe] = new_trend
                    update_data[f'Trend {timeframe}'] = new_trend

                    # Emit trend change signal
                    self.trendChanged.emit(symbol, timeframe, new_trend.value)
                else:
                    update_data[f'Trend {timeframe}'] = current_trend

        except Exception as e:
            logger.debug(f"Error updating trend analysis for {symbol}: {e}")

    def refresh_data(self) -> None:
        """Refresh all market data from sources."""
        try:
            logger.debug("Refreshing market data...")

            # Get all symbols currently in the model
            symbols = []
            for row in range(self.rowCount()):
                row_data = self.get_row_data(row)
                if row_data and 'Symbol' in row_data:
                    symbols.append(row_data['Symbol'])

            # Update each symbol with fresh data
            updated_count = 0
            for symbol in symbols:
                if self._refresh_symbol_data(symbol):
                    updated_count += 1

            logger.info(f"Refreshed data for {updated_count}/{len(symbols)} symbols")
            self.dataRefreshed.emit()

        except Exception as e:
            logger.error(f"Error refreshing market data: {e}")
            self.errorOccurred.emit(f"Failed to refresh data: {e}")

    def _refresh_symbol_data(self, symbol: str) -> bool:
        """Refresh data for a specific symbol."""
        try:
            # Get latest quote from symbol manager
            symbol_info = self.symbol_manager.get_symbol(symbol)
            if not symbol_info:
                return False

            # Update with latest data
            return self.update_symbol_data(
                symbol=symbol,
                bid=symbol_info.last_bid,
                ask=symbol_info.last_ask,
                volume=symbol_info.last_volume or 0
            )

        except Exception as e:
            logger.debug(f"Error refreshing data for {symbol}: {e}")
            return False

    def get_symbols(self) -> List[str]:
        """Get list of symbols currently in the model."""
        symbols = []
        for row in range(self.rowCount()):
            row_data = self.get_row_data(row)
            if row_data and 'Symbol' in row_data:
                symbols.append(row_data['Symbol'])
        return symbols

    def get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current data for a specific symbol."""
        row = self.find_row('Symbol', symbol)
        if row is not None:
            return self.get_row_data(row)
        return None

    def set_spread_threshold(self, threshold: Decimal) -> None:
        """Set the high spread warning threshold."""
        self._high_spread_threshold = threshold
        # Refresh display to update colors
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(self.rowCount() - 1, self.columnCount() - 1)
        )

    def _log_performance_metrics(self) -> None:
        """Log performance metrics."""
        logger.debug(
            f"Market data model performance: "
            f"{self._update_count} updates in last 30s, "
            f"{self.rowCount()} symbols, "
            f"{len(self._last_update_time)} active updates"
        )
        self._update_count = 0