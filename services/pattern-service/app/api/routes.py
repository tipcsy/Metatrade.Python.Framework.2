from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import pandas as pd

router = APIRouter()
logger = logging.getLogger(__name__)

# This will be set by main.py
get_pattern_analyzer = None


class IndicatorRequest(BaseModel):
    """Request for calculating indicators"""
    symbol: str
    timeframe: str
    indicators: Optional[List[str]] = None
    bars: int = 500


class PatternRequest(BaseModel):
    """Request for detecting patterns"""
    symbol: str
    timeframe: str
    bars: int = 500


class ScanRequest(BaseModel):
    """Request for scanning multiple symbols"""
    symbols: List[str]
    timeframes: List[str]
    indicators: Optional[List[str]] = None
    patterns: bool = True


class OHLCData(BaseModel):
    """OHLC data for custom analysis"""
    data: List[Dict[str, Any]]


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "pattern-service",
        "port": 5004
    }


@router.get("/indicators/{type}")
async def calculate_indicator(
    type: str,
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe (M1, M5, H1, etc.)"),
    period: int = Query(14, description="Period for indicator calculation")
):
    """
    Calculate a specific indicator

    Supported types:
    - sma: Simple Moving Average
    - ema: Exponential Moving Average
    - rsi: Relative Strength Index
    - macd: MACD
    - bbands: Bollinger Bands
    - atr: Average True Range
    - stochastic: Stochastic Oscillator
    - adx: Average Directional Index
    - obv: On-Balance Volume
    """
    try:
        analyzer = get_pattern_analyzer()
        if not analyzer:
            raise HTTPException(status_code=500, detail="PatternAnalyzer not initialized")

        # Fetch data
        scanner = analyzer.get_pattern_scanner()
        data = scanner.fetch_ohlc_data(symbol, timeframe, bars=500)

        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol} {timeframe}")

        # Calculate indicator
        engine = analyzer.get_indicator_engine()
        result = None

        if type == 'sma':
            series = engine.calculate_sma(data, period=period)
            result = {
                'values': series.dropna().tail(100).to_dict(),
                'latest': float(series.iloc[-1]) if not pd.isna(series.iloc[-1]) else None
            }

        elif type == 'ema':
            series = engine.calculate_ema(data, period=period)
            result = {
                'values': series.dropna().tail(100).to_dict(),
                'latest': float(series.iloc[-1]) if not pd.isna(series.iloc[-1]) else None
            }

        elif type == 'rsi':
            series = engine.calculate_rsi(data, period=period)
            result = {
                'values': series.dropna().tail(100).to_dict(),
                'latest': float(series.iloc[-1]) if not pd.isna(series.iloc[-1]) else None
            }

        elif type == 'macd':
            macd_data = engine.calculate_macd(data)
            result = {
                'macd': macd_data['macd'].dropna().tail(100).to_dict(),
                'signal': macd_data['signal'].dropna().tail(100).to_dict(),
                'histogram': macd_data['histogram'].dropna().tail(100).to_dict(),
                'latest': {
                    'macd': float(macd_data['macd'].iloc[-1]) if not pd.isna(macd_data['macd'].iloc[-1]) else None,
                    'signal': float(macd_data['signal'].iloc[-1]) if not pd.isna(macd_data['signal'].iloc[-1]) else None,
                    'histogram': float(macd_data['histogram'].iloc[-1]) if not pd.isna(macd_data['histogram'].iloc[-1]) else None
                }
            }

        elif type == 'bbands':
            bbands = engine.calculate_bollinger_bands(data, period=period)
            result = {
                'upper': bbands['upper'].dropna().tail(100).to_dict(),
                'middle': bbands['middle'].dropna().tail(100).to_dict(),
                'lower': bbands['lower'].dropna().tail(100).to_dict(),
                'latest': {
                    'upper': float(bbands['upper'].iloc[-1]) if not pd.isna(bbands['upper'].iloc[-1]) else None,
                    'middle': float(bbands['middle'].iloc[-1]) if not pd.isna(bbands['middle'].iloc[-1]) else None,
                    'lower': float(bbands['lower'].iloc[-1]) if not pd.isna(bbands['lower'].iloc[-1]) else None
                }
            }

        elif type == 'atr':
            series = engine.calculate_atr(data, period=period)
            result = {
                'values': series.dropna().tail(100).to_dict(),
                'latest': float(series.iloc[-1]) if not pd.isna(series.iloc[-1]) else None
            }

        elif type == 'stochastic':
            stoch = engine.calculate_stochastic(data, k_period=period)
            result = {
                'k': stoch['k'].dropna().tail(100).to_dict(),
                'd': stoch['d'].dropna().tail(100).to_dict(),
                'latest': {
                    'k': float(stoch['k'].iloc[-1]) if not pd.isna(stoch['k'].iloc[-1]) else None,
                    'd': float(stoch['d'].iloc[-1]) if not pd.isna(stoch['d'].iloc[-1]) else None
                }
            }

        elif type == 'adx':
            adx_data = engine.calculate_adx(data, period=period)
            result = {
                'adx': adx_data['adx'].dropna().tail(100).to_dict(),
                'plus_di': adx_data['plus_di'].dropna().tail(100).to_dict(),
                'minus_di': adx_data['minus_di'].dropna().tail(100).to_dict(),
                'latest': {
                    'adx': float(adx_data['adx'].iloc[-1]) if not pd.isna(adx_data['adx'].iloc[-1]) else None,
                    'plus_di': float(adx_data['plus_di'].iloc[-1]) if not pd.isna(adx_data['plus_di'].iloc[-1]) else None,
                    'minus_di': float(adx_data['minus_di'].iloc[-1]) if not pd.isna(adx_data['minus_di'].iloc[-1]) else None
                }
            }

        elif type == 'obv':
            series = engine.calculate_obv(data)
            result = {
                'values': series.dropna().tail(100).to_dict(),
                'latest': float(series.iloc[-1]) if not pd.isna(series.iloc[-1]) else None
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown indicator type: {type}")

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator": type,
                "period": period,
                "result": result
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating {type} indicator: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/candlestick")
async def detect_candlestick_patterns(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe")
):
    """Detect candlestick patterns"""
    try:
        analyzer = get_pattern_analyzer()
        if not analyzer:
            raise HTTPException(status_code=500, detail="PatternAnalyzer not initialized")

        # Fetch data
        scanner = analyzer.get_pattern_scanner()
        data = scanner.fetch_ohlc_data(symbol, timeframe, bars=500)

        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol} {timeframe}")

        # Detect patterns
        detector = analyzer.get_pattern_detector()
        patterns = detector.detect_all_candlestick_patterns(data)

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "patterns": patterns
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting candlestick patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/chart")
async def detect_chart_patterns(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe")
):
    """Detect chart patterns (support/resistance, trend)"""
    try:
        analyzer = get_pattern_analyzer()
        if not analyzer:
            raise HTTPException(status_code=500, detail="PatternAnalyzer not initialized")

        # Fetch data
        scanner = analyzer.get_pattern_scanner()
        data = scanner.fetch_ohlc_data(symbol, timeframe, bars=500)

        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol} {timeframe}")

        # Detect patterns
        detector = analyzer.get_pattern_detector()

        # Support/Resistance
        sr_levels = detector.detect_support_resistance(data)

        # Trend
        trend = detector.detect_trend(data)

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "support_levels": sr_levels['support'],
                "resistance_levels": sr_levels['resistance'],
                "trend": trend
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting chart patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan")
async def scan_symbols(request: ScanRequest):
    """
    Scan multiple symbols and timeframes for patterns and indicators
    """
    try:
        analyzer = get_pattern_analyzer()
        if not analyzer:
            raise HTTPException(status_code=500, detail="PatternAnalyzer not initialized")

        logger.info(f"Scanning {len(request.symbols)} symbols across {len(request.timeframes)} timeframes")

        # Perform scan
        scanner = analyzer.get_pattern_scanner()
        results = scanner.scan_multiple_symbols(
            symbols=request.symbols,
            timeframes=request.timeframes,
            indicators=request.indicators,
            patterns=request.patterns
        )

        return {
            "success": True,
            "data": {
                "total_scanned": len(results),
                "results": results
            }
        }

    except Exception as e:
        logger.error(f"Error scanning symbols: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan/opportunities")
async def find_opportunities(request: ScanRequest, min_confidence: float = Query(0.7)):
    """
    Scan for high-confidence trading opportunities
    """
    try:
        analyzer = get_pattern_analyzer()
        if not analyzer:
            raise HTTPException(status_code=500, detail="PatternAnalyzer not initialized")

        logger.info(f"Finding opportunities in {len(request.symbols)} symbols")

        # Find opportunities
        scanner = analyzer.get_pattern_scanner()
        opportunities = scanner.find_trading_opportunities(
            symbols=request.symbols,
            timeframes=request.timeframes,
            min_confidence=min_confidence
        )

        return {
            "success": True,
            "data": {
                "total_opportunities": len(opportunities),
                "opportunities": opportunities
            }
        }

    except Exception as e:
        logger.error(f"Error finding opportunities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{symbol}/{timeframe}")
async def analyze_symbol(
    symbol: str,
    timeframe: str,
    indicators: Optional[str] = Query(None, description="Comma-separated list of indicators")
):
    """
    Perform complete analysis on a symbol/timeframe
    """
    try:
        analyzer = get_pattern_analyzer()
        if not analyzer:
            raise HTTPException(status_code=500, detail="PatternAnalyzer not initialized")

        # Parse indicators
        indicator_list = None
        if indicators:
            indicator_list = [i.strip() for i in indicators.split(',')]

        # Perform analysis
        scanner = analyzer.get_pattern_scanner()
        result = scanner.analyze_symbol(
            symbol=symbol,
            timeframe=timeframe,
            indicators=indicator_list,
            patterns=True
        )

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error analyzing {symbol} {timeframe}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
