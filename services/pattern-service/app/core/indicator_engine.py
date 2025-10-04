"""
Technical Indicator Engine
Pure Python/Pandas/Numpy implementation of technical indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """
    Core engine for calculating technical indicators using pure Python/Pandas/Numpy.
    No external TA libraries required.
    """

    def __init__(self):
        """Initialize the indicator engine"""
        logger.info("Indicator Engine initialized")

    def calculate_sma(self, data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Calculate Simple Moving Average (SMA)

        Args:
            data: DataFrame with OHLC data
            period: Period for SMA calculation
            column: Column to calculate SMA on (default: 'close')

        Returns:
            pd.Series with SMA values
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        if len(data) < period:
            logger.warning(f"Insufficient data for SMA({period}): {len(data)} bars")

        return data[column].rolling(window=period, min_periods=period).mean()

    def calculate_ema(self, data: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA)

        EMA = Price(t) * k + EMA(y) * (1 - k)
        where k = 2 / (N + 1)

        Args:
            data: DataFrame with OHLC data
            period: Period for EMA calculation
            column: Column to calculate EMA on (default: 'close')

        Returns:
            pd.Series with EMA values
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        if len(data) < period:
            logger.warning(f"Insufficient data for EMA({period}): {len(data)} bars")

        return data[column].ewm(span=period, adjust=False, min_periods=period).mean()

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss

        Args:
            data: DataFrame with OHLC data
            period: Period for RSI calculation (default: 14)
            column: Column to calculate RSI on (default: 'close')

        Returns:
            pd.Series with RSI values (0-100)
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        if len(data) < period + 1:
            logger.warning(f"Insufficient data for RSI({period}): {len(data)} bars")

        # Calculate price changes
        delta = data[column].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Calculate average gain and loss using EWM (Wilder's smoothing)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26,
                      signal: int = 9, column: str = 'close') -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD, signal)
        Histogram = MACD - Signal

        Args:
            data: DataFrame with OHLC data
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            column: Column to calculate MACD on (default: 'close')

        Returns:
            Dict with 'macd', 'signal', and 'histogram' Series
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        if len(data) < slow + signal:
            logger.warning(f"Insufficient data for MACD({fast},{slow},{signal}): {len(data)} bars")

        # Calculate fast and slow EMAs
        ema_fast = self.calculate_ema(data, period=fast, column=column)
        ema_slow = self.calculate_ema(data, period=slow, column=column)

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate signal line
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20,
                                  std_dev: float = 2.0, column: str = 'close') -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands

        Middle Band = SMA(period)
        Upper Band = Middle Band + (std_dev * standard deviation)
        Lower Band = Middle Band - (std_dev * standard deviation)

        Args:
            data: DataFrame with OHLC data
            period: Period for SMA and std deviation (default: 20)
            std_dev: Number of standard deviations (default: 2.0)
            column: Column to calculate bands on (default: 'close')

        Returns:
            Dict with 'upper', 'middle', and 'lower' Series
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")

        if len(data) < period:
            logger.warning(f"Insufficient data for Bollinger Bands({period}): {len(data)} bars")

        # Calculate middle band (SMA)
        middle_band = self.calculate_sma(data, period=period, column=column)

        # Calculate standard deviation
        rolling_std = data[column].rolling(window=period, min_periods=period).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * rolling_std)
        lower_band = middle_band - (std_dev * rolling_std)

        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)

        True Range = max[(high - low), abs(high - close_prev), abs(low - close_prev)]
        ATR = EMA of True Range

        Args:
            data: DataFrame with OHLC data
            period: Period for ATR calculation (default: 14)

        Returns:
            pd.Series with ATR values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        if len(data) < period + 1:
            logger.warning(f"Insufficient data for ATR({period}): {len(data)} bars")

        # Calculate True Range components
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))

        # True Range is the maximum of the three
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate ATR using EMA (Wilder's smoothing)
        atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        return atr

    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14,
                            d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator

        %K = 100 * (Close - Low(k_period)) / (High(k_period) - Low(k_period))
        %D = SMA(%K, d_period)

        Args:
            data: DataFrame with OHLC data
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D smoothing (default: 3)

        Returns:
            Dict with 'k' and 'd' Series (0-100)
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        if len(data) < k_period + d_period:
            logger.warning(f"Insufficient data for Stochastic({k_period},{d_period}): {len(data)} bars")

        # Calculate lowest low and highest high over k_period
        lowest_low = data['low'].rolling(window=k_period, min_periods=k_period).min()
        highest_high = data['high'].rolling(window=k_period, min_periods=k_period).max()

        # Calculate %K
        k_percent = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)

        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()

        return {
            'k': k_percent,
            'd': d_percent
        }

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Average Directional Index (ADX)

        Measures trend strength (0-100)
        Also calculates +DI and -DI

        Args:
            data: DataFrame with OHLC data
            period: Period for ADX calculation (default: 14)

        Returns:
            Dict with 'adx', 'plus_di', and 'minus_di' Series
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        if len(data) < period * 2:
            logger.warning(f"Insufficient data for ADX({period}): {len(data)} bars")

        # Calculate +DM and -DM
        high_diff = data['high'].diff()
        low_diff = -data['low'].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

        # Calculate ATR
        atr = self.calculate_atr(data, period=period)

        # Calculate smoothed +DM and -DM
        plus_dm_smooth = plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr

        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

        # Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }

    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)

        OBV = OBV_prev + volume (if close > close_prev)
        OBV = OBV_prev - volume (if close < close_prev)
        OBV = OBV_prev (if close == close_prev)

        Args:
            data: DataFrame with OHLC and volume data

        Returns:
            pd.Series with OBV values
        """
        required_cols = ['close', 'tick_volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        # Calculate price direction
        direction = np.sign(data['close'].diff())

        # Calculate volume flow
        volume_flow = direction * data['tick_volume']

        # Calculate cumulative OBV
        obv = volume_flow.cumsum()

        return obv

    def calculate_all_indicators(self, data: pd.DataFrame,
                                 indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate multiple indicators at once

        Args:
            data: DataFrame with OHLC data
            indicators: List of indicator names to calculate (None = all)

        Returns:
            Dict with all calculated indicators
        """
        if data.empty:
            raise ValueError("Empty DataFrame provided")

        result = {}

        # Default to all indicators if none specified
        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'macd', 'bbands', 'atr', 'stochastic', 'adx', 'obv']

        try:
            if 'sma' in indicators:
                result['sma_20'] = self.calculate_sma(data, period=20).iloc[-1] if len(data) >= 20 else None
                result['sma_50'] = self.calculate_sma(data, period=50).iloc[-1] if len(data) >= 50 else None
                result['sma_200'] = self.calculate_sma(data, period=200).iloc[-1] if len(data) >= 200 else None

            if 'ema' in indicators:
                result['ema_12'] = self.calculate_ema(data, period=12).iloc[-1] if len(data) >= 12 else None
                result['ema_26'] = self.calculate_ema(data, period=26).iloc[-1] if len(data) >= 26 else None

            if 'rsi' in indicators:
                rsi = self.calculate_rsi(data, period=14)
                result['rsi'] = rsi.iloc[-1] if len(data) >= 15 else None

            if 'macd' in indicators:
                macd_data = self.calculate_macd(data)
                result['macd'] = {
                    'macd': macd_data['macd'].iloc[-1] if len(data) >= 35 else None,
                    'signal': macd_data['signal'].iloc[-1] if len(data) >= 35 else None,
                    'histogram': macd_data['histogram'].iloc[-1] if len(data) >= 35 else None
                }

            if 'bbands' in indicators:
                bbands = self.calculate_bollinger_bands(data)
                result['bollinger_bands'] = {
                    'upper': bbands['upper'].iloc[-1] if len(data) >= 20 else None,
                    'middle': bbands['middle'].iloc[-1] if len(data) >= 20 else None,
                    'lower': bbands['lower'].iloc[-1] if len(data) >= 20 else None
                }

            if 'atr' in indicators:
                atr = self.calculate_atr(data, period=14)
                result['atr'] = atr.iloc[-1] if len(data) >= 15 else None

            if 'stochastic' in indicators:
                stoch = self.calculate_stochastic(data)
                result['stochastic'] = {
                    'k': stoch['k'].iloc[-1] if len(data) >= 17 else None,
                    'd': stoch['d'].iloc[-1] if len(data) >= 17 else None
                }

            if 'adx' in indicators:
                adx_data = self.calculate_adx(data)
                result['adx'] = {
                    'adx': adx_data['adx'].iloc[-1] if len(data) >= 28 else None,
                    'plus_di': adx_data['plus_di'].iloc[-1] if len(data) >= 28 else None,
                    'minus_di': adx_data['minus_di'].iloc[-1] if len(data) >= 28 else None
                }

            if 'obv' in indicators and 'tick_volume' in data.columns:
                obv = self.calculate_obv(data)
                result['obv'] = obv.iloc[-1] if len(data) >= 2 else None

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            raise

        return result
