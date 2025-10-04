"""
Pattern Detector Engine
Pure Python/Pandas/Numpy implementation of candlestick and chart pattern detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Detects candlestick and chart patterns using pure Python/Pandas/Numpy.
    No external TA libraries required.
    """

    def __init__(self):
        """Initialize the pattern detector"""
        logger.info("Pattern Detector initialized")

    # ==================== Helper Methods ====================

    def _get_body_size(self, row: pd.Series) -> float:
        """Calculate the size of the candle body"""
        return abs(row['close'] - row['open'])

    def _get_total_size(self, row: pd.Series) -> float:
        """Calculate the total size of the candle (high to low)"""
        return row['high'] - row['low']

    def _get_upper_shadow(self, row: pd.Series) -> float:
        """Calculate the size of the upper shadow/wick"""
        return row['high'] - max(row['open'], row['close'])

    def _get_lower_shadow(self, row: pd.Series) -> float:
        """Calculate the size of the lower shadow/wick"""
        return min(row['open'], row['close']) - row['low']

    def _is_bullish(self, row: pd.Series) -> bool:
        """Check if candle is bullish (close > open)"""
        return row['close'] > row['open']

    def _is_bearish(self, row: pd.Series) -> bool:
        """Check if candle is bearish (close < open)"""
        return row['close'] < row['open']

    # ==================== Candlestick Patterns ====================

    def detect_doji(self, data: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """
        Detect Doji pattern

        A Doji occurs when open and close are very close (small body).
        Indicates indecision in the market.

        Args:
            data: DataFrame with OHLC data
            threshold: Maximum body size as percentage of total range (default: 0.1 = 10%)

        Returns:
            pd.Series of boolean values indicating Doji presence
        """
        body_size = data.apply(self._get_body_size, axis=1)
        total_size = data.apply(self._get_total_size, axis=1)

        # Doji: body is less than threshold% of total range
        is_doji = (body_size / total_size.replace(0, np.nan)) < threshold

        return is_doji.fillna(False)

    def detect_hammer(self, data: pd.DataFrame, body_threshold: float = 0.3,
                     shadow_ratio: float = 2.0) -> pd.Series:
        """
        Detect Hammer pattern

        Bullish reversal pattern:
        - Small body at the top
        - Long lower shadow (at least 2x body size)
        - Little to no upper shadow

        Args:
            data: DataFrame with OHLC data
            body_threshold: Maximum body as % of total range (default: 0.3 = 30%)
            shadow_ratio: Minimum lower shadow to body ratio (default: 2.0)

        Returns:
            pd.Series of boolean values indicating Hammer presence
        """
        body_size = data.apply(self._get_body_size, axis=1)
        total_size = data.apply(self._get_total_size, axis=1)
        lower_shadow = data.apply(self._get_lower_shadow, axis=1)
        upper_shadow = data.apply(self._get_upper_shadow, axis=1)

        # Conditions for Hammer
        small_body = (body_size / total_size.replace(0, np.nan)) <= body_threshold
        long_lower_shadow = lower_shadow >= (shadow_ratio * body_size)
        small_upper_shadow = upper_shadow <= (body_size * 0.5)

        is_hammer = small_body & long_lower_shadow & small_upper_shadow

        return is_hammer.fillna(False)

    def detect_shooting_star(self, data: pd.DataFrame, body_threshold: float = 0.3,
                            shadow_ratio: float = 2.0) -> pd.Series:
        """
        Detect Shooting Star pattern

        Bearish reversal pattern:
        - Small body at the bottom
        - Long upper shadow (at least 2x body size)
        - Little to no lower shadow

        Args:
            data: DataFrame with OHLC data
            body_threshold: Maximum body as % of total range (default: 0.3 = 30%)
            shadow_ratio: Minimum upper shadow to body ratio (default: 2.0)

        Returns:
            pd.Series of boolean values indicating Shooting Star presence
        """
        body_size = data.apply(self._get_body_size, axis=1)
        total_size = data.apply(self._get_total_size, axis=1)
        lower_shadow = data.apply(self._get_lower_shadow, axis=1)
        upper_shadow = data.apply(self._get_upper_shadow, axis=1)

        # Conditions for Shooting Star
        small_body = (body_size / total_size.replace(0, np.nan)) <= body_threshold
        long_upper_shadow = upper_shadow >= (shadow_ratio * body_size)
        small_lower_shadow = lower_shadow <= (body_size * 0.5)

        is_shooting_star = small_body & long_upper_shadow & small_lower_shadow

        return is_shooting_star.fillna(False)

    def detect_engulfing(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detect Bullish and Bearish Engulfing patterns

        Bullish Engulfing: Large bullish candle completely engulfs previous bearish candle
        Bearish Engulfing: Large bearish candle completely engulfs previous bullish candle

        Args:
            data: DataFrame with OHLC data

        Returns:
            Dict with 'bullish_engulfing' and 'bearish_engulfing' Series
        """
        if len(data) < 2:
            return {
                'bullish_engulfing': pd.Series([False] * len(data)),
                'bearish_engulfing': pd.Series([False] * len(data))
            }

        bullish_engulfing = pd.Series([False] * len(data), index=data.index)
        bearish_engulfing = pd.Series([False] * len(data), index=data.index)

        for i in range(1, len(data)):
            curr = data.iloc[i]
            prev = data.iloc[i-1]

            # Bullish Engulfing
            if (self._is_bearish(prev) and self._is_bullish(curr) and
                curr['open'] < prev['close'] and curr['close'] > prev['open']):
                bullish_engulfing.iloc[i] = True

            # Bearish Engulfing
            if (self._is_bullish(prev) and self._is_bearish(curr) and
                curr['open'] > prev['close'] and curr['close'] < prev['open']):
                bearish_engulfing.iloc[i] = True

        return {
            'bullish_engulfing': bullish_engulfing,
            'bearish_engulfing': bearish_engulfing
        }

    def detect_morning_star(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect Morning Star pattern

        Bullish reversal pattern (3 candles):
        1. Large bearish candle
        2. Small body candle (gap down)
        3. Large bullish candle (closes above midpoint of first candle)

        Args:
            data: DataFrame with OHLC data

        Returns:
            pd.Series of boolean values indicating Morning Star presence
        """
        if len(data) < 3:
            return pd.Series([False] * len(data), index=data.index)

        morning_star = pd.Series([False] * len(data), index=data.index)

        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]

            # First candle: large bearish
            first_bearish = self._is_bearish(first)
            first_large = self._get_body_size(first) > (self._get_total_size(first) * 0.6)

            # Second candle: small body
            second_small = self._get_body_size(second) < (self._get_body_size(first) * 0.3)

            # Third candle: large bullish
            third_bullish = self._is_bullish(third)
            third_large = self._get_body_size(third) > (self._get_total_size(third) * 0.6)
            third_closes_high = third['close'] > (first['open'] + first['close']) / 2

            if (first_bearish and first_large and second_small and
                third_bullish and third_large and third_closes_high):
                morning_star.iloc[i] = True

        return morning_star

    def detect_evening_star(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect Evening Star pattern

        Bearish reversal pattern (3 candles):
        1. Large bullish candle
        2. Small body candle (gap up)
        3. Large bearish candle (closes below midpoint of first candle)

        Args:
            data: DataFrame with OHLC data

        Returns:
            pd.Series of boolean values indicating Evening Star presence
        """
        if len(data) < 3:
            return pd.Series([False] * len(data), index=data.index)

        evening_star = pd.Series([False] * len(data), index=data.index)

        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]

            # First candle: large bullish
            first_bullish = self._is_bullish(first)
            first_large = self._get_body_size(first) > (self._get_total_size(first) * 0.6)

            # Second candle: small body
            second_small = self._get_body_size(second) < (self._get_body_size(first) * 0.3)

            # Third candle: large bearish
            third_bearish = self._is_bearish(third)
            third_large = self._get_body_size(third) > (self._get_total_size(third) * 0.6)
            third_closes_low = third['close'] < (first['open'] + first['close']) / 2

            if (first_bullish and first_large and second_small and
                third_bearish and third_large and third_closes_low):
                evening_star.iloc[i] = True

        return evening_star

    def detect_three_white_soldiers(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect Three White Soldiers pattern

        Bullish reversal pattern (3 candles):
        Three consecutive large bullish candles with higher closes

        Args:
            data: DataFrame with OHLC data

        Returns:
            pd.Series of boolean values indicating Three White Soldiers presence
        """
        if len(data) < 3:
            return pd.Series([False] * len(data), index=data.index)

        three_white_soldiers = pd.Series([False] * len(data), index=data.index)

        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]

            # All three bullish
            all_bullish = (self._is_bullish(first) and
                          self._is_bullish(second) and
                          self._is_bullish(third))

            # All three have large bodies
            all_large = (self._get_body_size(first) > self._get_total_size(first) * 0.6 and
                        self._get_body_size(second) > self._get_total_size(second) * 0.6 and
                        self._get_body_size(third) > self._get_total_size(third) * 0.6)

            # Each closes higher than previous
            higher_closes = (second['close'] > first['close'] and
                           third['close'] > second['close'])

            if all_bullish and all_large and higher_closes:
                three_white_soldiers.iloc[i] = True

        return three_white_soldiers

    def detect_three_black_crows(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect Three Black Crows pattern

        Bearish reversal pattern (3 candles):
        Three consecutive large bearish candles with lower closes

        Args:
            data: DataFrame with OHLC data

        Returns:
            pd.Series of boolean values indicating Three Black Crows presence
        """
        if len(data) < 3:
            return pd.Series([False] * len(data), index=data.index)

        three_black_crows = pd.Series([False] * len(data), index=data.index)

        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]

            # All three bearish
            all_bearish = (self._is_bearish(first) and
                          self._is_bearish(second) and
                          self._is_bearish(third))

            # All three have large bodies
            all_large = (self._get_body_size(first) > self._get_total_size(first) * 0.6 and
                        self._get_body_size(second) > self._get_total_size(second) * 0.6 and
                        self._get_body_size(third) > self._get_total_size(third) * 0.6)

            # Each closes lower than previous
            lower_closes = (second['close'] < first['close'] and
                          third['close'] < second['close'])

            if all_bearish and all_large and lower_closes:
                three_black_crows.iloc[i] = True

        return three_black_crows

    # ==================== Chart Patterns ====================

    def detect_support_resistance(self, data: pd.DataFrame,
                                  window: int = 20,
                                  tolerance: float = 0.02) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels

        Uses local extrema (swing highs/lows) to identify key levels

        Args:
            data: DataFrame with OHLC data
            window: Window size for finding local extrema (default: 20)
            tolerance: Price tolerance for clustering levels (default: 0.02 = 2%)

        Returns:
            Dict with 'support' and 'resistance' level lists
        """
        if len(data) < window * 2:
            return {'support': [], 'resistance': []}

        # Find local lows (support)
        local_lows = []
        for i in range(window, len(data) - window):
            window_data = data.iloc[i-window:i+window+1]
            if data.iloc[i]['low'] == window_data['low'].min():
                local_lows.append(data.iloc[i]['low'])

        # Find local highs (resistance)
        local_highs = []
        for i in range(window, len(data) - window):
            window_data = data.iloc[i-window:i+window+1]
            if data.iloc[i]['high'] == window_data['high'].max():
                local_highs.append(data.iloc[i]['high'])

        # Cluster similar levels
        def cluster_levels(levels, tolerance):
            if not levels:
                return []

            levels = sorted(levels)
            clustered = []
            current_cluster = [levels[0]]

            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                    current_cluster.append(level)
                else:
                    clustered.append(np.mean(current_cluster))
                    current_cluster = [level]

            clustered.append(np.mean(current_cluster))
            return clustered

        support_levels = cluster_levels(local_lows, tolerance)
        resistance_levels = cluster_levels(local_highs, tolerance)

        return {
            'support': support_levels,
            'resistance': resistance_levels
        }

    def detect_trend(self, data: pd.DataFrame, period: int = 20) -> str:
        """
        Detect overall trend direction

        Uses moving average slope and price position

        Args:
            data: DataFrame with OHLC data
            period: Period for trend calculation (default: 20)

        Returns:
            'UPTREND', 'DOWNTREND', or 'SIDEWAYS'
        """
        if len(data) < period:
            return 'SIDEWAYS'

        # Calculate SMA
        sma = data['close'].rolling(window=period).mean()

        # Get recent slope
        recent_sma = sma.iloc[-10:]
        if len(recent_sma) < 2:
            return 'SIDEWAYS'

        x = np.arange(len(recent_sma))
        slope = np.polyfit(x, recent_sma.values, 1)[0]

        # Current price vs SMA
        current_price = data['close'].iloc[-1]
        current_sma = sma.iloc[-1]

        # Determine trend
        if slope > 0 and current_price > current_sma:
            return 'UPTREND'
        elif slope < 0 and current_price < current_sma:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'

    def detect_all_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect all candlestick patterns at once

        Args:
            data: DataFrame with OHLC data

        Returns:
            Dict with all detected patterns (latest values)
        """
        if data.empty:
            return {}

        result = {}

        try:
            # Single candle patterns
            doji = self.detect_doji(data)
            result['doji'] = bool(doji.iloc[-1]) if len(doji) > 0 else False

            hammer = self.detect_hammer(data)
            result['hammer'] = bool(hammer.iloc[-1]) if len(hammer) > 0 else False

            shooting_star = self.detect_shooting_star(data)
            result['shooting_star'] = bool(shooting_star.iloc[-1]) if len(shooting_star) > 0 else False

            # Two candle patterns
            engulfing = self.detect_engulfing(data)
            result['bullish_engulfing'] = bool(engulfing['bullish_engulfing'].iloc[-1]) if len(data) >= 2 else False
            result['bearish_engulfing'] = bool(engulfing['bearish_engulfing'].iloc[-1]) if len(data) >= 2 else False

            # Three candle patterns
            if len(data) >= 3:
                morning_star = self.detect_morning_star(data)
                result['morning_star'] = bool(morning_star.iloc[-1])

                evening_star = self.detect_evening_star(data)
                result['evening_star'] = bool(evening_star.iloc[-1])

                three_white_soldiers = self.detect_three_white_soldiers(data)
                result['three_white_soldiers'] = bool(three_white_soldiers.iloc[-1])

                three_black_crows = self.detect_three_black_crows(data)
                result['three_black_crows'] = bool(three_black_crows.iloc[-1])

        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}", exc_info=True)
            raise

        return result
