"""
Pattern Scanner Engine
Scans multiple symbols and timeframes for technical patterns and signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import requests
from datetime import datetime, timedelta

from .indicator_engine import IndicatorEngine
from .pattern_detector import PatternDetector

logger = logging.getLogger(__name__)


class PatternScanner:
    """
    Scans multiple symbols and timeframes for patterns and technical signals.
    Coordinates between indicator calculation and pattern detection.
    """

    def __init__(self, data_service_url: str = "http://localhost:5001"):
        """
        Initialize the pattern scanner

        Args:
            data_service_url: URL of the data service for fetching OHLC data
        """
        self.data_service_url = data_service_url
        self.indicator_engine = IndicatorEngine()
        self.pattern_detector = PatternDetector()
        logger.info(f"Pattern Scanner initialized with data service: {data_service_url}")

    def fetch_ohlc_data(self, symbol: str, timeframe: str,
                       bars: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch OHLC data from data service

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1, etc.)
            bars: Number of bars to fetch

        Returns:
            DataFrame with OHLC data or None if error
        """
        try:
            # Calculate time range
            now = datetime.now()
            from_time = int((now - timedelta(days=30)).timestamp())
            to_time = int(now.timestamp())

            # Query data service
            url = f"{self.data_service_url}/ohlc/query"
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'from_time': from_time,
                'to_time': to_time,
                'limit': bars
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if not data.get('success'):
                logger.error(f"Failed to fetch data for {symbol} {timeframe}")
                return None

            bars_data = data.get('data', {}).get('bars', [])

            if not bars_data:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(bars_data)

            # Ensure required columns exist
            required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in data for {symbol} {timeframe}")
                return None

            # Set time as index
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time')

            # Sort by time
            df = df.sort_index()

            logger.info(f"Fetched {len(df)} bars for {symbol} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLC data for {symbol} {timeframe}: {e}", exc_info=True)
            return None

    def analyze_symbol(self, symbol: str, timeframe: str,
                      indicators: Optional[List[str]] = None,
                      patterns: bool = True) -> Dict[str, Any]:
        """
        Analyze a single symbol/timeframe for indicators and patterns

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicators: List of indicators to calculate (None = all)
            patterns: Whether to detect patterns

        Returns:
            Dict with analysis results
        """
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'indicators': {},
            'patterns': {},
            'signals': []
        }

        try:
            # Fetch data
            data = self.fetch_ohlc_data(symbol, timeframe)

            if data is None or data.empty:
                result['error'] = 'No data available'
                return result

            # Calculate indicators
            if indicators is not None or indicators != []:
                try:
                    result['indicators'] = self.indicator_engine.calculate_all_indicators(
                        data, indicators=indicators
                    )
                except Exception as e:
                    logger.error(f"Error calculating indicators: {e}")
                    result['indicators'] = {'error': str(e)}

            # Detect patterns
            if patterns:
                try:
                    result['patterns'] = self.pattern_detector.detect_all_candlestick_patterns(data)

                    # Add chart patterns
                    sr_levels = self.pattern_detector.detect_support_resistance(data)
                    result['patterns']['support_levels'] = sr_levels['support']
                    result['patterns']['resistance_levels'] = sr_levels['resistance']

                    trend = self.pattern_detector.detect_trend(data)
                    result['patterns']['trend'] = trend

                except Exception as e:
                    logger.error(f"Error detecting patterns: {e}")
                    result['patterns'] = {'error': str(e)}

            # Generate signals based on patterns and indicators
            result['signals'] = self._generate_signals(result)

        except Exception as e:
            logger.error(f"Error analyzing {symbol} {timeframe}: {e}", exc_info=True)
            result['error'] = str(e)

        return result

    def scan_multiple_symbols(self, symbols: List[str], timeframes: List[str],
                             indicators: Optional[List[str]] = None,
                             patterns: bool = True) -> List[Dict[str, Any]]:
        """
        Scan multiple symbols and timeframes

        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes
            indicators: List of indicators to calculate
            patterns: Whether to detect patterns

        Returns:
            List of analysis results for each symbol/timeframe combination
        """
        results = []

        logger.info(f"Scanning {len(symbols)} symbols across {len(timeframes)} timeframes")

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    analysis = self.analyze_symbol(
                        symbol=symbol,
                        timeframe=timeframe,
                        indicators=indicators,
                        patterns=patterns
                    )
                    results.append(analysis)

                except Exception as e:
                    logger.error(f"Error scanning {symbol} {timeframe}: {e}")
                    results.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'error': str(e)
                    })

        logger.info(f"Scan completed: {len(results)} results")
        return results

    def find_trading_opportunities(self, symbols: List[str], timeframes: List[str],
                                  min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """
        Scan for high-confidence trading opportunities

        Args:
            symbols: List of symbols to scan
            timeframes: List of timeframes to scan
            min_confidence: Minimum confidence score (0.0 to 1.0)

        Returns:
            List of trading opportunities with signals
        """
        all_results = self.scan_multiple_symbols(
            symbols=symbols,
            timeframes=timeframes,
            indicators=['sma', 'ema', 'rsi', 'macd'],
            patterns=True
        )

        # Filter for opportunities with signals
        opportunities = []

        for result in all_results:
            if 'error' in result:
                continue

            signals = result.get('signals', [])

            # Filter by confidence
            high_confidence_signals = [
                sig for sig in signals
                if sig.get('confidence', 0) >= min_confidence
            ]

            if high_confidence_signals:
                opportunities.append({
                    'symbol': result['symbol'],
                    'timeframe': result['timeframe'],
                    'timestamp': result['timestamp'],
                    'signals': high_confidence_signals,
                    'indicators': result.get('indicators', {}),
                    'patterns': result.get('patterns', {})
                })

        logger.info(f"Found {len(opportunities)} trading opportunities")
        return opportunities

    def _generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on indicators and patterns

        Args:
            analysis: Analysis result with indicators and patterns

        Returns:
            List of trading signals
        """
        signals = []
        indicators = analysis.get('indicators', {})
        patterns = analysis.get('patterns', {})

        # RSI signals
        if 'rsi' in indicators and indicators['rsi'] is not None:
            rsi = indicators['rsi']

            if rsi < 30:
                signals.append({
                    'type': 'BUY',
                    'reason': 'RSI Oversold',
                    'indicator': 'RSI',
                    'value': rsi,
                    'confidence': 0.7
                })
            elif rsi > 70:
                signals.append({
                    'type': 'SELL',
                    'reason': 'RSI Overbought',
                    'indicator': 'RSI',
                    'value': rsi,
                    'confidence': 0.7
                })

        # MACD signals
        if 'macd' in indicators and indicators['macd'].get('histogram') is not None:
            histogram = indicators['macd']['histogram']
            macd_line = indicators['macd']['macd']
            signal_line = indicators['macd']['signal']

            if macd_line > signal_line and histogram > 0:
                signals.append({
                    'type': 'BUY',
                    'reason': 'MACD Bullish Crossover',
                    'indicator': 'MACD',
                    'value': histogram,
                    'confidence': 0.75
                })
            elif macd_line < signal_line and histogram < 0:
                signals.append({
                    'type': 'SELL',
                    'reason': 'MACD Bearish Crossover',
                    'indicator': 'MACD',
                    'value': histogram,
                    'confidence': 0.75
                })

        # Bullish candlestick patterns
        bullish_patterns = ['hammer', 'bullish_engulfing', 'morning_star', 'three_white_soldiers']
        for pattern_name in bullish_patterns:
            if patterns.get(pattern_name):
                signals.append({
                    'type': 'BUY',
                    'reason': f'{pattern_name.replace("_", " ").title()} Pattern',
                    'indicator': 'Pattern',
                    'value': pattern_name,
                    'confidence': 0.8
                })

        # Bearish candlestick patterns
        bearish_patterns = ['shooting_star', 'bearish_engulfing', 'evening_star', 'three_black_crows']
        for pattern_name in bearish_patterns:
            if patterns.get(pattern_name):
                signals.append({
                    'type': 'SELL',
                    'reason': f'{pattern_name.replace("_", " ").title()} Pattern',
                    'indicator': 'Pattern',
                    'value': pattern_name,
                    'confidence': 0.8
                })

        # Trend signals
        if patterns.get('trend'):
            trend = patterns['trend']
            if trend == 'UPTREND':
                signals.append({
                    'type': 'BUY',
                    'reason': 'Uptrend Detected',
                    'indicator': 'Trend',
                    'value': trend,
                    'confidence': 0.65
                })
            elif trend == 'DOWNTREND':
                signals.append({
                    'type': 'SELL',
                    'reason': 'Downtrend Detected',
                    'indicator': 'Trend',
                    'value': trend,
                    'confidence': 0.65
                })

        return signals
