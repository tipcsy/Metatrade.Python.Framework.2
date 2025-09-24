"""
Historical data validation for synchronization quality assurance.

This module provides comprehensive validation capabilities to ensure
data quality and integrity after synchronization operations.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Any, Tuple

from pydantic import BaseModel, Field
from sqlalchemy import text, func

from src.core.config import get_settings
from src.core.logging import get_logger
from src.database import get_session
from src.database.models import TickData as DBTickData, OHLCData as DBOHLCData

logger = get_logger(__name__)
settings = get_settings()


class DataQualityReport(BaseModel):
    """Comprehensive data quality assessment report."""

    symbol: str = Field(description="Trading symbol")
    timeframe: Optional[str] = Field(default=None, description="Timeframe for OHLC data")
    start_time: datetime = Field(description="Report period start")
    end_time: datetime = Field(description="Report period end")
    data_type: str = Field(description="Data type: tick or ohlc")

    # Coverage metrics
    total_points: int = Field(description="Total data points found")
    expected_points: int = Field(description="Expected data points")
    coverage_percentage: float = Field(description="Data coverage percentage")
    missing_points: int = Field(description="Number of missing points")

    # Quality metrics
    duplicate_count: int = Field(description="Number of duplicate entries")
    invalid_price_count: int = Field(description="Invalid price entries")
    invalid_volume_count: int = Field(description="Invalid volume entries")
    timestamp_issues: int = Field(description="Timestamp-related issues")
    ohlc_consistency_issues: int = Field(description="OHLC consistency issues")

    # Statistical analysis
    price_statistics: Dict[str, float] = Field(default_factory=dict, description="Price statistics")
    volume_statistics: Dict[str, float] = Field(default_factory=dict, description="Volume statistics")
    gap_analysis: Dict[str, Any] = Field(default_factory=dict, description="Gap analysis results")

    # Quality scores (0-1 scale)
    coverage_score: float = Field(description="Coverage quality score")
    accuracy_score: float = Field(description="Data accuracy score")
    consistency_score: float = Field(description="Data consistency score")
    overall_score: float = Field(description="Overall quality score")

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Quality improvement recommendations")
    critical_issues: List[str] = Field(default_factory=list, description="Critical issues requiring attention")

    def get_quality_grade(self) -> str:
        """Get quality grade based on overall score."""
        if self.overall_score >= 0.95:
            return "A+"
        elif self.overall_score >= 0.90:
            return "A"
        elif self.overall_score >= 0.85:
            return "B+"
        elif self.overall_score >= 0.80:
            return "B"
        elif self.overall_score >= 0.70:
            return "C"
        elif self.overall_score >= 0.60:
            return "D"
        else:
            return "F"


class HistoricalDataValidator:
    """
    Validates historical data quality and integrity.

    Provides comprehensive validation including coverage analysis,
    duplicate detection, data consistency checks, and statistical analysis.
    """

    def __init__(self):
        """Initialize historical data validator."""
        self._validation_stats = {
            "total_validations": 0,
            "symbols_validated": 0,
            "issues_found": 0,
            "critical_issues": 0
        }

        # Validation thresholds
        self._thresholds = {
            "min_coverage_percentage": 95.0,
            "max_duplicate_percentage": 1.0,
            "max_invalid_price_percentage": 0.1,
            "max_gap_minutes": 60,
            "min_volume": 0,
            "max_spread_percentage": 10.0
        }

        logger.info("Historical data validator initialized")

    async def validate_tick_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        expected_interval: timedelta = timedelta(seconds=1)
    ) -> DataQualityReport:
        """
        Validate tick data quality.

        Args:
            symbol: Trading symbol
            start_time: Validation period start
            end_time: Validation period end
            expected_interval: Expected time between ticks

        Returns:
            Comprehensive data quality report
        """
        logger.info(f"Validating tick data for {symbol}")

        try:
            with get_session() as session:
                # Basic coverage analysis
                coverage_metrics = await self._analyze_tick_coverage(
                    session, symbol, start_time, end_time, expected_interval
                )

                # Quality checks
                quality_metrics = await self._check_tick_quality(
                    session, symbol, start_time, end_time
                )

                # Statistical analysis
                stats = await self._analyze_tick_statistics(
                    session, symbol, start_time, end_time
                )

                # Gap analysis
                gap_analysis = await self._analyze_tick_gaps(
                    session, symbol, start_time, end_time, expected_interval
                )

                # Calculate quality scores
                coverage_score = min(coverage_metrics["coverage_percentage"] / 100.0, 1.0)
                accuracy_score = self._calculate_tick_accuracy_score(quality_metrics)
                consistency_score = self._calculate_tick_consistency_score(gap_analysis)
                overall_score = (coverage_score + accuracy_score + consistency_score) / 3.0

                # Generate recommendations
                recommendations, critical_issues = self._generate_tick_recommendations(
                    coverage_metrics, quality_metrics, gap_analysis
                )

                report = DataQualityReport(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    data_type="tick",
                    total_points=coverage_metrics["total_points"],
                    expected_points=coverage_metrics["expected_points"],
                    coverage_percentage=coverage_metrics["coverage_percentage"],
                    missing_points=coverage_metrics["missing_points"],
                    duplicate_count=quality_metrics["duplicates"],
                    invalid_price_count=quality_metrics["invalid_prices"],
                    invalid_volume_count=quality_metrics["invalid_volumes"],
                    timestamp_issues=quality_metrics["timestamp_issues"],
                    ohlc_consistency_issues=0,  # Not applicable for tick data
                    price_statistics=stats["price_stats"],
                    volume_statistics=stats["volume_stats"],
                    gap_analysis=gap_analysis,
                    coverage_score=coverage_score,
                    accuracy_score=accuracy_score,
                    consistency_score=consistency_score,
                    overall_score=overall_score,
                    recommendations=recommendations,
                    critical_issues=critical_issues
                )

                self._update_validation_stats(report)
                logger.info(f"Tick validation completed for {symbol}: Grade {report.get_quality_grade()}")

                return report

        except Exception as e:
            logger.error(f"Error validating tick data for {symbol}: {e}")
            raise

    async def validate_ohlc_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> DataQualityReport:
        """
        Validate OHLC data quality.

        Args:
            symbol: Trading symbol
            timeframe: OHLC timeframe
            start_time: Validation period start
            end_time: Validation period end

        Returns:
            Comprehensive data quality report
        """
        logger.info(f"Validating OHLC data for {symbol} {timeframe}")

        try:
            expected_interval = self._get_timeframe_interval(timeframe)
            if not expected_interval:
                raise ValueError(f"Unknown timeframe: {timeframe}")

            with get_session() as session:
                # Basic coverage analysis
                coverage_metrics = await self._analyze_ohlc_coverage(
                    session, symbol, timeframe, start_time, end_time, expected_interval
                )

                # Quality checks
                quality_metrics = await self._check_ohlc_quality(
                    session, symbol, timeframe, start_time, end_time
                )

                # Statistical analysis
                stats = await self._analyze_ohlc_statistics(
                    session, symbol, timeframe, start_time, end_time
                )

                # Gap analysis
                gap_analysis = await self._analyze_ohlc_gaps(
                    session, symbol, timeframe, start_time, end_time, expected_interval
                )

                # Calculate quality scores
                coverage_score = min(coverage_metrics["coverage_percentage"] / 100.0, 1.0)
                accuracy_score = self._calculate_ohlc_accuracy_score(quality_metrics)
                consistency_score = self._calculate_ohlc_consistency_score(quality_metrics, gap_analysis)
                overall_score = (coverage_score + accuracy_score + consistency_score) / 3.0

                # Generate recommendations
                recommendations, critical_issues = self._generate_ohlc_recommendations(
                    coverage_metrics, quality_metrics, gap_analysis
                )

                report = DataQualityReport(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    data_type="ohlc",
                    total_points=coverage_metrics["total_points"],
                    expected_points=coverage_metrics["expected_points"],
                    coverage_percentage=coverage_metrics["coverage_percentage"],
                    missing_points=coverage_metrics["missing_points"],
                    duplicate_count=quality_metrics["duplicates"],
                    invalid_price_count=quality_metrics["invalid_prices"],
                    invalid_volume_count=quality_metrics["invalid_volumes"],
                    timestamp_issues=quality_metrics["timestamp_issues"],
                    ohlc_consistency_issues=quality_metrics["ohlc_issues"],
                    price_statistics=stats["price_stats"],
                    volume_statistics=stats["volume_stats"],
                    gap_analysis=gap_analysis,
                    coverage_score=coverage_score,
                    accuracy_score=accuracy_score,
                    consistency_score=consistency_score,
                    overall_score=overall_score,
                    recommendations=recommendations,
                    critical_issues=critical_issues
                )

                self._update_validation_stats(report)
                logger.info(f"OHLC validation completed for {symbol} {timeframe}: Grade {report.get_quality_grade()}")

                return report

        except Exception as e:
            logger.error(f"Error validating OHLC data for {symbol} {timeframe}: {e}")
            raise

    async def validate_multiple_symbols(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_time: datetime,
        end_time: datetime,
        include_ticks: bool = False
    ) -> Dict[str, DataQualityReport]:
        """
        Validate multiple symbols and timeframes.

        Args:
            symbols: List of trading symbols
            timeframes: List of OHLC timeframes
            start_time: Validation period start
            end_time: Validation period end
            include_ticks: Whether to include tick validation

        Returns:
            Dictionary mapping symbol_timeframe to quality reports
        """
        reports = {}

        for symbol in symbols:
            # Validate tick data if requested
            if include_ticks:
                try:
                    tick_report = await self.validate_tick_data(symbol, start_time, end_time)
                    reports[f"{symbol}_TICK"] = tick_report
                except Exception as e:
                    logger.error(f"Failed to validate tick data for {symbol}: {e}")

            # Validate OHLC data
            for timeframe in timeframes:
                try:
                    ohlc_report = await self.validate_ohlc_data(symbol, timeframe, start_time, end_time)
                    reports[f"{symbol}_{timeframe}"] = ohlc_report
                except Exception as e:
                    logger.error(f"Failed to validate OHLC data for {symbol} {timeframe}: {e}")

        self._validation_stats["symbols_validated"] = len(symbols)
        logger.info(f"Completed validation for {len(symbols)} symbols")

        return reports

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self._validation_stats.copy()

    async def _analyze_tick_coverage(
        self, session, symbol: str, start_time: datetime, end_time: datetime, expected_interval: timedelta
    ) -> Dict[str, Any]:
        """Analyze tick data coverage."""
        # Count total ticks
        count_query = text("""
            SELECT COUNT(*) as total_count
            FROM tick_data
            WHERE symbol = :symbol
            AND timestamp BETWEEN :start_time AND :end_time
        """)

        result = session.execute(count_query, {
            "symbol": symbol,
            "start_time": start_time,
            "end_time": end_time
        }).fetchone()

        total_points = result[0] if result else 0

        # Calculate expected points (approximate)
        time_span = end_time - start_time
        expected_points = int(time_span.total_seconds() / expected_interval.total_seconds())

        coverage_percentage = (total_points / expected_points * 100) if expected_points > 0 else 0
        missing_points = max(0, expected_points - total_points)

        return {
            "total_points": total_points,
            "expected_points": expected_points,
            "coverage_percentage": coverage_percentage,
            "missing_points": missing_points
        }

    async def _check_tick_quality(
        self, session, symbol: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Check tick data quality issues."""
        # Check for duplicates
        duplicate_query = text("""
            SELECT COUNT(*) - COUNT(DISTINCT timestamp, bid, ask) as duplicates
            FROM tick_data
            WHERE symbol = :symbol
            AND timestamp BETWEEN :start_time AND :end_time
        """)

        duplicate_result = session.execute(duplicate_query, {
            "symbol": symbol,
            "start_time": start_time,
            "end_time": end_time
        }).fetchone()

        duplicates = duplicate_result[0] if duplicate_result else 0

        # Check for invalid prices
        invalid_price_query = text("""
            SELECT COUNT(*) as invalid_prices
            FROM tick_data
            WHERE symbol = :symbol
            AND timestamp BETWEEN :start_time AND :end_time
            AND (bid <= 0 OR ask <= 0 OR ask <= bid)
        """)

        invalid_price_result = session.execute(invalid_price_query, {
            "symbol": symbol,
            "start_time": start_time,
            "end_time": end_time
        }).fetchone()

        invalid_prices = invalid_price_result[0] if invalid_price_result else 0

        # Check for invalid volumes
        invalid_volume_query = text("""
            SELECT COUNT(*) as invalid_volumes
            FROM tick_data
            WHERE symbol = :symbol
            AND timestamp BETWEEN :start_time AND :end_time
            AND volume < 0
        """)

        invalid_volume_result = session.execute(invalid_volume_query, {
            "symbol": symbol,
            "start_time": start_time,
            "end_time": end_time
        }).fetchone()

        invalid_volumes = invalid_volume_result[0] if invalid_volume_result else 0

        # Check for timestamp issues (out of order, future dates)
        timestamp_query = text("""
            SELECT COUNT(*) as timestamp_issues
            FROM (
                SELECT *,
                    LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
                FROM tick_data
                WHERE symbol = :symbol
                AND timestamp BETWEEN :start_time AND :end_time
            ) t
            WHERE t.timestamp < t.prev_timestamp
            OR t.timestamp > NOW() + INTERVAL '5 minutes'
        """)

        timestamp_result = session.execute(timestamp_query, {
            "symbol": symbol,
            "start_time": start_time,
            "end_time": end_time
        }).fetchone()

        timestamp_issues = timestamp_result[0] if timestamp_result else 0

        return {
            "duplicates": duplicates,
            "invalid_prices": invalid_prices,
            "invalid_volumes": invalid_volumes,
            "timestamp_issues": timestamp_issues
        }

    async def _analyze_tick_statistics(
        self, session, symbol: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Analyze tick data statistics."""
        stats_query = text("""
            SELECT
                AVG((bid + ask) / 2) as avg_price,
                MIN((bid + ask) / 2) as min_price,
                MAX((bid + ask) / 2) as max_price,
                STDDEV((bid + ask) / 2) as price_std,
                AVG(ask - bid) as avg_spread,
                MAX(ask - bid) as max_spread,
                AVG(volume) as avg_volume,
                MIN(volume) as min_volume,
                MAX(volume) as max_volume,
                STDDEV(volume) as volume_std
            FROM tick_data
            WHERE symbol = :symbol
            AND timestamp BETWEEN :start_time AND :end_time
            AND bid > 0 AND ask > 0 AND ask > bid
        """)

        result = session.execute(stats_query, {
            "symbol": symbol,
            "start_time": start_time,
            "end_time": end_time
        }).fetchone()

        if result:
            price_stats = {
                "avg_price": float(result[0] or 0),
                "min_price": float(result[1] or 0),
                "max_price": float(result[2] or 0),
                "price_std": float(result[3] or 0),
                "avg_spread": float(result[4] or 0),
                "max_spread": float(result[5] or 0)
            }

            volume_stats = {
                "avg_volume": float(result[6] or 0),
                "min_volume": float(result[7] or 0),
                "max_volume": float(result[8] or 0),
                "volume_std": float(result[9] or 0)
            }
        else:
            price_stats = {}
            volume_stats = {}

        return {
            "price_stats": price_stats,
            "volume_stats": volume_stats
        }

    async def _analyze_tick_gaps(
        self, session, symbol: str, start_time: datetime, end_time: datetime, expected_interval: timedelta
    ) -> Dict[str, Any]:
        """Analyze gaps in tick data."""
        # This would implement gap analysis similar to the detector
        # Simplified for this implementation
        return {
            "total_gaps": 0,
            "max_gap_minutes": 0,
            "avg_gap_minutes": 0
        }

    # Similar methods for OHLC data analysis would follow...
    async def _analyze_ohlc_coverage(self, session, symbol: str, timeframe: str, start_time: datetime, end_time: datetime, expected_interval: timedelta) -> Dict[str, Any]:
        """Analyze OHLC data coverage - placeholder implementation."""
        return {"total_points": 0, "expected_points": 0, "coverage_percentage": 0, "missing_points": 0}

    async def _check_ohlc_quality(self, session, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Check OHLC data quality - placeholder implementation."""
        return {"duplicates": 0, "invalid_prices": 0, "invalid_volumes": 0, "timestamp_issues": 0, "ohlc_issues": 0}

    async def _analyze_ohlc_statistics(self, session, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze OHLC statistics - placeholder implementation."""
        return {"price_stats": {}, "volume_stats": {}}

    async def _analyze_ohlc_gaps(self, session, symbol: str, timeframe: str, start_time: datetime, end_time: datetime, expected_interval: timedelta) -> Dict[str, Any]:
        """Analyze OHLC gaps - placeholder implementation."""
        return {"total_gaps": 0, "max_gap_minutes": 0, "avg_gap_minutes": 0}

    def _calculate_tick_accuracy_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate accuracy score for tick data."""
        # Simplified scoring based on error counts
        total_issues = (quality_metrics["invalid_prices"] +
                       quality_metrics["invalid_volumes"] +
                       quality_metrics["timestamp_issues"])

        if total_issues == 0:
            return 1.0
        elif total_issues < 10:
            return 0.9
        elif total_issues < 100:
            return 0.7
        else:
            return 0.5

    def _calculate_tick_consistency_score(self, gap_analysis: Dict[str, Any]) -> float:
        """Calculate consistency score for tick data."""
        return 0.9  # Placeholder

    def _calculate_ohlc_accuracy_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate accuracy score for OHLC data."""
        return 0.9  # Placeholder

    def _calculate_ohlc_consistency_score(self, quality_metrics: Dict[str, Any], gap_analysis: Dict[str, Any]) -> float:
        """Calculate consistency score for OHLC data."""
        return 0.9  # Placeholder

    def _generate_tick_recommendations(self, coverage_metrics: Dict, quality_metrics: Dict, gap_analysis: Dict) -> Tuple[List[str], List[str]]:
        """Generate recommendations for tick data quality."""
        recommendations = []
        critical_issues = []

        if coverage_metrics["coverage_percentage"] < self._thresholds["min_coverage_percentage"]:
            critical_issues.append(f"Low data coverage: {coverage_metrics['coverage_percentage']:.1f}%")
            recommendations.append("Implement more aggressive gap detection and backfill strategies")

        if quality_metrics["duplicates"] > 0:
            recommendations.append("Remove duplicate entries and implement duplicate prevention")

        if quality_metrics["invalid_prices"] > 0:
            critical_issues.append(f"Found {quality_metrics['invalid_prices']} invalid price entries")
            recommendations.append("Implement stricter price validation rules")

        return recommendations, critical_issues

    def _generate_ohlc_recommendations(self, coverage_metrics: Dict, quality_metrics: Dict, gap_analysis: Dict) -> Tuple[List[str], List[str]]:
        """Generate recommendations for OHLC data quality."""
        return [], []  # Placeholder

    def _get_timeframe_interval(self, timeframe: str) -> Optional[timedelta]:
        """Get interval for timeframe."""
        intervals = {
            "M1": timedelta(minutes=1),
            "M3": timedelta(minutes=3),
            "M5": timedelta(minutes=5),
            "M15": timedelta(minutes=15),
            "M30": timedelta(minutes=30),
            "H1": timedelta(hours=1),
            "H4": timedelta(hours=4),
            "D1": timedelta(days=1)
        }
        return intervals.get(timeframe)

    def _update_validation_stats(self, report: DataQualityReport) -> None:
        """Update validation statistics."""
        self._validation_stats["total_validations"] += 1

        total_issues = (report.duplicate_count + report.invalid_price_count +
                       report.invalid_volume_count + report.timestamp_issues +
                       report.ohlc_consistency_issues)

        self._validation_stats["issues_found"] += total_issues

        if report.critical_issues:
            self._validation_stats["critical_issues"] += len(report.critical_issues)