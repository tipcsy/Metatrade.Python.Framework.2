"""
Data gap detection for historical data synchronization.

This module provides comprehensive gap detection capabilities to identify
missing data periods and classify gap types for appropriate handling.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set

from pydantic import BaseModel, Field
from sqlalchemy import text

from src.core.config import get_settings
from src.core.logging import get_logger
from src.database import get_session
from src.database.models import TickData as DBTickData, OHLCData as DBoHLCData

logger = get_logger(__name__)
settings = get_settings()


class GapType(Enum):
    """Types of data gaps."""
    MISSING_TICK = "missing_tick"
    MISSING_OHLC = "missing_ohlc"
    MARKET_CLOSED = "market_closed"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    DATA_ERROR = "data_error"
    CONNECTION_LOSS = "connection_loss"


class DataGap(BaseModel):
    """Represents a detected data gap."""

    symbol: str = Field(description="Trading symbol")
    gap_type: GapType = Field(description="Type of gap")
    start_time: datetime = Field(description="Gap start time")
    end_time: datetime = Field(description="Gap end time")
    expected_points: int = Field(description="Expected number of data points")
    actual_points: int = Field(description="Actual number of data points")
    gap_duration_minutes: float = Field(description="Gap duration in minutes")
    severity: str = Field(description="Gap severity: low, medium, high, critical")
    requires_backfill: bool = Field(default=True, description="Whether gap requires backfill")
    metadata: Dict[str, any] = Field(default_factory=dict, description="Additional gap metadata")

    @property
    def gap_duration(self) -> timedelta:
        """Get gap duration as timedelta."""
        return self.end_time - self.start_time

    @property
    def missing_points(self) -> int:
        """Get number of missing data points."""
        return self.expected_points - self.actual_points

    def to_dict(self) -> Dict[str, any]:
        """Convert gap to dictionary."""
        return {
            "symbol": self.symbol,
            "gap_type": self.gap_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "expected_points": self.expected_points,
            "actual_points": self.actual_points,
            "missing_points": self.missing_points,
            "gap_duration_minutes": self.gap_duration_minutes,
            "severity": self.severity,
            "requires_backfill": self.requires_backfill,
            "metadata": self.metadata
        }


class GapDetector:
    """
    Detects gaps in historical market data.

    Provides comprehensive gap detection for both tick and OHLC data
    with intelligent classification and severity assessment.
    """

    def __init__(self):
        """Initialize gap detector."""
        self._detection_stats = {
            "total_scans": 0,
            "gaps_detected": 0,
            "symbols_scanned": 0,
            "critical_gaps": 0,
            "backfill_required": 0
        }

        # Market schedule configuration
        self._market_hours = {
            "forex": {
                "open_hour": 0,  # 24/7 except weekends
                "close_hour": 23,
                "trading_days": [0, 1, 2, 3, 4]  # Monday to Friday
            },
            "stocks": {
                "open_hour": 9,
                "close_hour": 16,
                "trading_days": [0, 1, 2, 3, 4]  # Monday to Friday
            }
        }

        # Gap detection thresholds
        self._tick_gap_threshold = timedelta(seconds=30)  # Missing ticks
        self._ohlc_gap_threshold = timedelta(minutes=5)   # Missing OHLC bars
        self._connection_loss_threshold = timedelta(hours=1)  # Connection issues

        logger.info("Gap detector initialized")

    def detect_tick_gaps(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        expected_interval: timedelta = timedelta(seconds=1)
    ) -> List[DataGap]:
        """
        Detect gaps in tick data.

        Args:
            symbol: Trading symbol
            start_time: Start of detection period
            end_time: End of detection period
            expected_interval: Expected time between ticks

        Returns:
            List of detected gaps
        """
        gaps = []

        try:
            with get_session() as session:
                # Query tick data in time range
                query = text("""
                    SELECT timestamp, bid, ask, volume
                    FROM tick_data
                    WHERE symbol = :symbol
                    AND timestamp BETWEEN :start_time AND :end_time
                    ORDER BY timestamp
                """)

                result = session.execute(query, {
                    "symbol": symbol,
                    "start_time": start_time,
                    "end_time": end_time
                })

                ticks = result.fetchall()

                if not ticks:
                    # Complete data missing
                    gap = self._create_complete_missing_gap(
                        symbol, GapType.MISSING_TICK, start_time, end_time, expected_interval
                    )
                    gaps.append(gap)
                    return gaps

                # Check for gaps between consecutive ticks
                previous_tick = None
                for tick in ticks:
                    current_time = tick[0]

                    if previous_tick:
                        time_diff = current_time - previous_tick[0]

                        if time_diff > self._tick_gap_threshold:
                            # Check if gap is due to market closure
                            if self._is_market_closed_period(previous_tick[0], current_time, symbol):
                                gap_type = self._classify_market_closure(previous_tick[0], current_time)
                                gap = DataGap(
                                    symbol=symbol,
                                    gap_type=gap_type,
                                    start_time=previous_tick[0],
                                    end_time=current_time,
                                    expected_points=0,  # Market closed, no data expected
                                    actual_points=0,
                                    gap_duration_minutes=time_diff.total_seconds() / 60,
                                    severity="low",
                                    requires_backfill=False
                                )
                            else:
                                # Actual data gap
                                expected_ticks = int(time_diff.total_seconds() / expected_interval.total_seconds())
                                gap = DataGap(
                                    symbol=symbol,
                                    gap_type=GapType.MISSING_TICK,
                                    start_time=previous_tick[0],
                                    end_time=current_time,
                                    expected_points=expected_ticks,
                                    actual_points=0,
                                    gap_duration_minutes=time_diff.total_seconds() / 60,
                                    severity=self._calculate_gap_severity(time_diff, expected_ticks),
                                    requires_backfill=True
                                )

                            gaps.append(gap)

                    previous_tick = tick

                # Check for gap at the end
                if ticks:
                    last_tick_time = ticks[-1][0]
                    if end_time - last_tick_time > self._tick_gap_threshold:
                        time_diff = end_time - last_tick_time
                        expected_ticks = int(time_diff.total_seconds() / expected_interval.total_seconds())

                        gap = DataGap(
                            symbol=symbol,
                            gap_type=GapType.MISSING_TICK,
                            start_time=last_tick_time,
                            end_time=end_time,
                            expected_points=expected_ticks,
                            actual_points=0,
                            gap_duration_minutes=time_diff.total_seconds() / 60,
                            severity=self._calculate_gap_severity(time_diff, expected_ticks),
                            requires_backfill=True
                        )
                        gaps.append(gap)

            self._update_detection_stats(len(gaps))
            logger.info(f"Detected {len(gaps)} tick gaps for {symbol}")

        except Exception as e:
            logger.error(f"Error detecting tick gaps for {symbol}: {e}")

        return gaps

    def detect_ohlc_gaps(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[DataGap]:
        """
        Detect gaps in OHLC data.

        Args:
            symbol: Trading symbol
            timeframe: OHLC timeframe (M1, M5, H1, etc.)
            start_time: Start of detection period
            end_time: End of detection period

        Returns:
            List of detected gaps
        """
        gaps = []

        try:
            # Calculate expected interval from timeframe
            expected_interval = self._get_timeframe_interval(timeframe)
            if not expected_interval:
                logger.warning(f"Unknown timeframe: {timeframe}")
                return gaps

            with get_session() as session:
                # Query OHLC data in time range
                query = text("""
                    SELECT bar_timestamp, open_price, high_price, low_price, close_price, volume
                    FROM ohlc_data
                    WHERE symbol = :symbol
                    AND timeframe = :timeframe
                    AND bar_timestamp BETWEEN :start_time AND :end_time
                    ORDER BY bar_timestamp
                """)

                result = session.execute(query, {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_time": start_time,
                    "end_time": end_time
                })

                bars = result.fetchall()

                if not bars:
                    # Complete data missing
                    gap = self._create_complete_missing_gap(
                        symbol, GapType.MISSING_OHLC, start_time, end_time, expected_interval
                    )
                    gaps.append(gap)
                    return gaps

                # Check for gaps between consecutive bars
                previous_bar = None
                for bar in bars:
                    current_time = bar[0]

                    if previous_bar:
                        expected_next_time = previous_bar[0] + expected_interval
                        if current_time > expected_next_time + self._ohlc_gap_threshold:
                            # Gap detected
                            time_diff = current_time - previous_bar[0]

                            if self._is_market_closed_period(previous_bar[0], current_time, symbol):
                                gap_type = self._classify_market_closure(previous_bar[0], current_time)
                                gap = DataGap(
                                    symbol=symbol,
                                    gap_type=gap_type,
                                    start_time=previous_bar[0],
                                    end_time=current_time,
                                    expected_points=0,
                                    actual_points=0,
                                    gap_duration_minutes=time_diff.total_seconds() / 60,
                                    severity="low",
                                    requires_backfill=False,
                                    metadata={"timeframe": timeframe}
                                )
                            else:
                                expected_bars = int(time_diff.total_seconds() / expected_interval.total_seconds())
                                gap = DataGap(
                                    symbol=symbol,
                                    gap_type=GapType.MISSING_OHLC,
                                    start_time=previous_bar[0],
                                    end_time=current_time,
                                    expected_points=expected_bars,
                                    actual_points=0,
                                    gap_duration_minutes=time_diff.total_seconds() / 60,
                                    severity=self._calculate_gap_severity(time_diff, expected_bars),
                                    requires_backfill=True,
                                    metadata={"timeframe": timeframe}
                                )

                            gaps.append(gap)

                    previous_bar = bar

                # Check for gap at the end
                if bars:
                    last_bar_time = bars[-1][0]
                    expected_next_time = last_bar_time + expected_interval
                    if end_time > expected_next_time + self._ohlc_gap_threshold:
                        time_diff = end_time - last_bar_time
                        expected_bars = int(time_diff.total_seconds() / expected_interval.total_seconds())

                        gap = DataGap(
                            symbol=symbol,
                            gap_type=GapType.MISSING_OHLC,
                            start_time=last_bar_time,
                            end_time=end_time,
                            expected_points=expected_bars,
                            actual_points=0,
                            gap_duration_minutes=time_diff.total_seconds() / 60,
                            severity=self._calculate_gap_severity(time_diff, expected_bars),
                            requires_backfill=True,
                            metadata={"timeframe": timeframe}
                        )
                        gaps.append(gap)

            self._update_detection_stats(len(gaps))
            logger.info(f"Detected {len(gaps)} OHLC gaps for {symbol} {timeframe}")

        except Exception as e:
            logger.error(f"Error detecting OHLC gaps for {symbol} {timeframe}: {e}")

        return gaps

    def detect_all_gaps(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_time: datetime,
        end_time: datetime,
        include_ticks: bool = True
    ) -> Dict[str, List[DataGap]]:
        """
        Detect all gaps across multiple symbols and timeframes.

        Args:
            symbols: List of trading symbols
            timeframes: List of OHLC timeframes
            start_time: Start of detection period
            end_time: End of detection period
            include_ticks: Whether to include tick gap detection

        Returns:
            Dictionary mapping symbol_timeframe to gaps
        """
        all_gaps = {}

        for symbol in symbols:
            # Detect tick gaps
            if include_ticks:
                tick_gaps = self.detect_tick_gaps(symbol, start_time, end_time)
                if tick_gaps:
                    all_gaps[f"{symbol}_TICK"] = tick_gaps

            # Detect OHLC gaps
            for timeframe in timeframes:
                ohlc_gaps = self.detect_ohlc_gaps(symbol, timeframe, start_time, end_time)
                if ohlc_gaps:
                    all_gaps[f"{symbol}_{timeframe}"] = ohlc_gaps

        self._detection_stats["symbols_scanned"] = len(symbols)
        logger.info(f"Completed gap detection for {len(symbols)} symbols across {len(timeframes)} timeframes")

        return all_gaps

    def get_critical_gaps(self, gaps: List[DataGap]) -> List[DataGap]:
        """Get gaps marked as critical severity."""
        return [gap for gap in gaps if gap.severity == "critical"]

    def get_backfill_required_gaps(self, gaps: List[DataGap]) -> List[DataGap]:
        """Get gaps that require backfill."""
        return [gap for gap in gaps if gap.requires_backfill]

    def get_detection_stats(self) -> Dict[str, any]:
        """Get gap detection statistics."""
        return self._detection_stats.copy()

    def _create_complete_missing_gap(
        self,
        symbol: str,
        gap_type: GapType,
        start_time: datetime,
        end_time: datetime,
        expected_interval: timedelta
    ) -> DataGap:
        """Create gap for completely missing data."""
        time_diff = end_time - start_time
        expected_points = int(time_diff.total_seconds() / expected_interval.total_seconds())

        return DataGap(
            symbol=symbol,
            gap_type=gap_type,
            start_time=start_time,
            end_time=end_time,
            expected_points=expected_points,
            actual_points=0,
            gap_duration_minutes=time_diff.total_seconds() / 60,
            severity="critical",
            requires_backfill=True
        )

    def _is_market_closed_period(self, start_time: datetime, end_time: datetime, symbol: str) -> bool:
        """Check if time period falls during market closure."""
        # Determine market type (simplified)
        market_type = "forex"  # Default to forex (24/7 except weekends)
        if any(term in symbol.upper() for term in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]):
            market_type = "forex"
        else:
            market_type = "stocks"

        market_hours = self._market_hours[market_type]

        # Check if period spans weekend
        if market_type == "forex":
            # Forex closes Friday evening and reopens Sunday evening
            start_weekday = start_time.weekday()
            end_weekday = end_time.weekday()

            # Friday after 21:00 UTC to Sunday 21:00 UTC
            if (start_weekday == 4 and start_time.hour >= 21) or start_weekday == 5 or start_weekday == 6:
                return True
            if (end_weekday == 4 and end_time.hour >= 21) or end_weekday == 5 or end_weekday == 6:
                return True

        elif market_type == "stocks":
            # Check if weekday and within trading hours
            start_weekday = start_time.weekday()
            end_weekday = end_time.weekday()

            if start_weekday not in market_hours["trading_days"] or end_weekday not in market_hours["trading_days"]:
                return True

            # Check trading hours
            if (start_time.hour < market_hours["open_hour"] or
                start_time.hour >= market_hours["close_hour"] or
                end_time.hour < market_hours["open_hour"] or
                end_time.hour >= market_hours["close_hour"]):
                return True

        return False

    def _classify_market_closure(self, start_time: datetime, end_time: datetime) -> GapType:
        """Classify type of market closure gap."""
        start_weekday = start_time.weekday()
        end_weekday = end_time.weekday()

        # Weekend
        if start_weekday >= 5 or end_weekday >= 5:
            return GapType.WEEKEND

        # Market closure (daily)
        return GapType.MARKET_CLOSED

    def _calculate_gap_severity(self, gap_duration: timedelta, missing_points: int) -> str:
        """Calculate gap severity based on duration and missing points."""
        duration_hours = gap_duration.total_seconds() / 3600

        if duration_hours >= 24 or missing_points >= 1000:
            return "critical"
        elif duration_hours >= 4 or missing_points >= 100:
            return "high"
        elif duration_hours >= 1 or missing_points >= 10:
            return "medium"
        else:
            return "low"

    def _get_timeframe_interval(self, timeframe: str) -> Optional[timedelta]:
        """Get interval for timeframe."""
        timeframe_intervals = {
            "M1": timedelta(minutes=1),
            "M3": timedelta(minutes=3),
            "M5": timedelta(minutes=5),
            "M15": timedelta(minutes=15),
            "M30": timedelta(minutes=30),
            "H1": timedelta(hours=1),
            "H4": timedelta(hours=4),
            "D1": timedelta(days=1),
            "W1": timedelta(weeks=1),
            "MN1": timedelta(days=30)  # Approximate
        }

        return timeframe_intervals.get(timeframe)

    def _update_detection_stats(self, gaps_found: int) -> None:
        """Update detection statistics."""
        self._detection_stats["total_scans"] += 1
        self._detection_stats["gaps_detected"] += gaps_found

        # Count critical and backfill required gaps would be done in the calling context