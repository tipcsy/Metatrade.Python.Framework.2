"""
Time Machine - Event-driven simulation engine for backtesting

The Time Machine replays historical data bar-by-bar in chronological order,
ensuring no look-ahead bias. It simulates a real-time environment where only
past data is available at each decision point.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class TimeMachine:
    """
    Event-driven time machine that replays historical data sequentially.

    This ensures accurate backtesting by preventing look-ahead bias - at any
    point in the simulation, only data up to the current simulated time is
    available to the strategy.
    """

    def __init__(self, bars: List[Dict[str, Any]]):
        """
        Initialize time machine with historical bar data.

        Args:
            bars: List of OHLC bars sorted chronologically
                  Each bar must have: time, open, high, low, close, volume
        """
        self.bars = bars
        self.current_index = 0
        self.current_time = None
        self.total_bars = len(bars)

        if self.total_bars == 0:
            raise ValueError("Cannot initialize TimeMachine with empty bar data")

        # Verify bars are sorted chronologically
        self._verify_chronological_order()

        logger.info(f"TimeMachine initialized with {self.total_bars} bars")
        logger.info(f"Date range: {self._format_time(bars[0]['time'])} to {self._format_time(bars[-1]['time'])}")

    def _verify_chronological_order(self):
        """Verify that bars are in chronological order"""
        for i in range(1, len(self.bars)):
            if self.bars[i]['time'] <= self.bars[i-1]['time']:
                raise ValueError(
                    f"Bars must be in chronological order. "
                    f"Bar {i} time ({self.bars[i]['time']}) <= Bar {i-1} time ({self.bars[i-1]['time']})"
                )

    def _format_time(self, timestamp: int) -> str:
        """Format Unix timestamp to readable string"""
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

    def has_next(self) -> bool:
        """Check if there are more bars to process"""
        return self.current_index < self.total_bars

    def next(self) -> Optional[Dict[str, Any]]:
        """
        Get next bar and advance time.

        Returns:
            Next bar data or None if no more bars
        """
        if not self.has_next():
            return None

        bar = self.bars[self.current_index]
        self.current_time = bar['time']
        self.current_index += 1

        return bar

    def peek_next(self) -> Optional[Dict[str, Any]]:
        """
        Peek at next bar without advancing time.

        WARNING: This should only be used for validation purposes.
        Using future data for trading decisions will cause look-ahead bias!

        Returns:
            Next bar data or None if no more bars
        """
        if not self.has_next():
            return None

        return self.bars[self.current_index]

    def get_current_bar(self) -> Optional[Dict[str, Any]]:
        """
        Get the current bar (last processed bar).

        Returns:
            Current bar data or None if no bars processed yet
        """
        if self.current_index == 0:
            return None

        return self.bars[self.current_index - 1]

    def get_historical_bars(self, lookback: int) -> List[Dict[str, Any]]:
        """
        Get historical bars up to current time.

        This is safe for strategy use as it only returns past data.

        Args:
            lookback: Number of bars to look back (including current)

        Returns:
            List of historical bars
        """
        if self.current_index == 0:
            return []

        start_index = max(0, self.current_index - lookback)
        end_index = self.current_index

        return self.bars[start_index:end_index]

    def get_all_past_bars(self) -> List[Dict[str, Any]]:
        """
        Get all bars up to current time.

        Returns:
            List of all processed bars
        """
        if self.current_index == 0:
            return []

        return self.bars[:self.current_index]

    def reset(self):
        """Reset time machine to beginning"""
        self.current_index = 0
        self.current_time = None
        logger.info("TimeMachine reset to beginning")

    def get_progress(self) -> float:
        """
        Get simulation progress.

        Returns:
            Progress as percentage (0-100)
        """
        if self.total_bars == 0:
            return 100.0

        return (self.current_index / self.total_bars) * 100.0

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status information.

        Returns:
            Status dictionary
        """
        return {
            "current_index": self.current_index,
            "total_bars": self.total_bars,
            "progress_percent": self.get_progress(),
            "current_time": self._format_time(self.current_time) if self.current_time else None,
            "has_next": self.has_next()
        }

    async def run_simulation(
        self,
        on_bar_callback: Callable[[Dict[str, Any], int], None],
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        """
        Run complete simulation by iterating through all bars.

        This is the main loop for event-driven backtesting.

        Args:
            on_bar_callback: Async function called for each bar (bar_data, bar_index)
            progress_callback: Optional async function called with progress updates
        """
        logger.info("Starting time machine simulation")

        self.reset()
        bar_count = 0

        while self.has_next():
            bar = self.next()

            # Call the strategy's bar handler
            await on_bar_callback(bar, self.current_index - 1)

            bar_count += 1

            # Progress updates every 100 bars
            if progress_callback and bar_count % 100 == 0:
                await progress_callback(self.get_progress())

        logger.info(f"Simulation completed. Processed {bar_count} bars")

        # Final progress update
        if progress_callback:
            await progress_callback(100.0)


class BarBuilder:
    """
    Helper class to convert raw data into standardized bar format.
    """

    @staticmethod
    def from_dict_list(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert list of dictionaries to standardized bar format.

        Args:
            bars: List of bar dictionaries

        Returns:
            Standardized bars
        """
        standardized = []

        for bar in bars:
            standardized.append({
                'time': int(bar.get('time', bar.get('timestamp', 0))),
                'open': float(bar['open']),
                'high': float(bar['high']),
                'low': float(bar['low']),
                'close': float(bar['close']),
                'volume': float(bar.get('volume', bar.get('tick_volume', 0)))
            })

        # Sort by time
        standardized.sort(key=lambda x: x['time'])

        return standardized

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert pandas DataFrame to bar format.

        Args:
            df: DataFrame with columns: time, open, high, low, close, volume

        Returns:
            List of bar dictionaries
        """
        # Ensure required columns exist
        required = ['time', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0

        # Sort by time
        df = df.sort_values('time')

        # Convert to list of dicts
        bars = df[['time', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')

        # Ensure time is integer
        for bar in bars:
            bar['time'] = int(bar['time'])

        return bars
