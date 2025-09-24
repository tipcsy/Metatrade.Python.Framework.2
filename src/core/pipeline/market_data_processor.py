"""
Market data processing pipeline with validation and enrichment.

This module provides a comprehensive data processing pipeline for market data
with validation, enrichment, transformation, and quality assessment capabilities.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable, Set, Union, Tuple
import threading
import uuid
from dataclasses import dataclass

from core.config.settings import Settings
from core.exceptions import (
    DataValidationError,
    PerformanceError,
    ConfigurationError,
)
from core.logging import get_logger
from core.data.models import (
    TickData,
    OHLCData,
    MarketEvent,
    MarketEventType,
    DataQuality,
)
from core.data import get_buffer_manager, get_event_publisher
from core.tasks import get_task_manager

logger = get_logger(__name__)


@dataclass
class ProcessingMetrics:
    """Processing pipeline metrics."""
    items_processed: int = 0
    items_validated: int = 0
    items_enriched: int = 0
    items_failed: int = 0
    processing_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    enrichment_time_ms: float = 0.0


@dataclass
class ValidationResult:
    """Data validation result."""
    is_valid: bool
    quality: DataQuality
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class DataProcessor(ABC):
    """Base class for data processors in the pipeline."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.metrics = ProcessingMetrics()
        self._lock = threading.RLock()

    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Process data item."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics."""
        pass

    async def process_batch(self, data_items: List[Any]) -> List[Any]:
        """Process batch of data items."""
        results = []
        for item in data_items:
            try:
                if self.enabled:
                    result = await self.process(item)
                    results.append(result)
                else:
                    results.append(item)
            except Exception as e:
                logger.error(f"Error processing item in {self.name}: {e}")
                self.metrics.items_failed += 1
        return results

    def reset_metrics(self) -> None:
        """Reset processor metrics."""
        with self._lock:
            self.metrics = ProcessingMetrics()


class TickDataValidator(DataProcessor):
    """
    Tick data validator with comprehensive quality checks.

    Features:
    - Price validation (range, sanity checks)
    - Timestamp validation
    - Spread analysis
    - Volume validation
    - Market session validation
    - Duplicate detection
    """

    def __init__(self, settings: Settings):
        super().__init__("TickDataValidator")
        self.settings = settings

        # Validation thresholds
        self.max_spread_percentage = 0.05  # 5% max spread
        self.max_price_change_percentage = 0.10  # 10% max price change
        self.max_latency_ms = 5000  # 5 seconds max latency
        self.min_volume = 0

        # Recent data for validation
        self.recent_ticks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.price_ranges: Dict[str, Tuple[float, float]] = {}

    async def process(self, data: TickData) -> TickData:
        """Validate tick data."""
        start_time = time.perf_counter()

        try:
            validation_result = await self._validate_tick(data)

            # Update data quality based on validation
            data.quality = validation_result.quality

            # Add validation metadata
            if hasattr(data, 'validation_info'):
                data.validation_info = validation_result.metadata

            # Update metrics
            with self._lock:
                self.metrics.items_processed += 1
                if validation_result.is_valid:
                    self.metrics.items_validated += 1
                else:
                    self.metrics.items_failed += 1

            # Store recent tick for future validation
            self.recent_ticks[data.symbol].append(data)

            return data

        except Exception as e:
            logger.error(f"Error validating tick for {data.symbol}: {e}")
            data.quality = DataQuality.INVALID
            self.metrics.items_failed += 1
            raise

        finally:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.metrics.validation_time_ms += processing_time

    async def _validate_tick(self, tick: TickData) -> ValidationResult:
        """Perform comprehensive tick validation."""
        issues = []
        warnings = []
        quality = DataQuality.HIGH

        # 1. Basic price validation
        if tick.bid <= 0 or tick.ask <= 0:
            issues.append("Invalid price: bid or ask is zero or negative")
            quality = DataQuality.INVALID

        if tick.ask <= tick.bid:
            issues.append("Invalid price: ask price is not greater than bid")
            quality = DataQuality.INVALID

        # 2. Spread validation
        spread_percentage = (tick.spread / tick.mid_price) if tick.mid_price > 0 else 0
        if spread_percentage > self.max_spread_percentage:
            warnings.append(f"Large spread: {spread_percentage:.2%}")
            if quality == DataQuality.HIGH:
                quality = DataQuality.MEDIUM

        # 3. Price change validation
        recent = self.recent_ticks.get(tick.symbol)
        if recent:
            last_tick = recent[-1]
            price_change = abs(tick.mid_price - last_tick.mid_price) / last_tick.mid_price
            if price_change > self.max_price_change_percentage:
                warnings.append(f"Large price change: {price_change:.2%}")
                if quality == DataQuality.HIGH:
                    quality = DataQuality.MEDIUM

        # 4. Timestamp validation
        now = datetime.now(timezone.utc)
        age_seconds = (now - tick.timestamp).total_seconds()

        if age_seconds < 0:
            issues.append("Future timestamp")
            quality = DataQuality.SUSPECT
        elif age_seconds > 300:  # 5 minutes
            warnings.append(f"Old data: {age_seconds:.1f} seconds old")
            if quality == DataQuality.HIGH:
                quality = DataQuality.LOW

        # 5. Latency validation
        if tick.latency_ms and tick.latency_ms > self.max_latency_ms:
            warnings.append(f"High latency: {tick.latency_ms:.1f}ms")
            if quality == DataQuality.HIGH:
                quality = DataQuality.LOW

        # 6. Volume validation
        if tick.volume < self.min_volume:
            warnings.append("Zero or negative volume")

        # 7. Price range validation
        symbol_range = self.price_ranges.get(tick.symbol)
        if symbol_range:
            min_price, max_price = symbol_range
            if not (min_price <= tick.mid_price <= max_price):
                warnings.append("Price outside expected range")
                if quality == DataQuality.HIGH:
                    quality = DataQuality.LOW

        # Update price range
        if symbol_range:
            min_price = min(symbol_range[0], float(tick.mid_price))
            max_price = max(symbol_range[1], float(tick.mid_price))
        else:
            min_price = max_price = float(tick.mid_price)
        self.price_ranges[tick.symbol] = (min_price, max_price)

        # Determine overall validity
        is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            quality=quality,
            issues=issues,
            warnings=warnings,
            metadata={
                'validation_timestamp': now.isoformat(),
                'spread_percentage': spread_percentage,
                'age_seconds': age_seconds,
                'price_range': self.price_ranges.get(tick.symbol),
            }
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get validator metrics."""
        with self._lock:
            success_rate = (
                self.metrics.items_validated / max(self.metrics.items_processed, 1)
            ) * 100

            return {
                'processor_name': self.name,
                'enabled': self.enabled,
                'items_processed': self.metrics.items_processed,
                'items_validated': self.metrics.items_validated,
                'items_failed': self.metrics.items_failed,
                'success_rate_percent': success_rate,
                'avg_validation_time_ms': (
                    self.metrics.validation_time_ms / max(self.metrics.items_processed, 1)
                ),
                'symbols_tracked': len(self.recent_ticks),
                'price_ranges_tracked': len(self.price_ranges),
            }


class OHLCDataValidator(DataProcessor):
    """
    OHLC data validator with comprehensive quality checks.

    Features:
    - OHLC relationship validation
    - Volume validation
    - Timeframe consistency
    - Gap detection
    - Pattern validation
    """

    def __init__(self, settings: Settings):
        super().__init__("OHLCDataValidator")
        self.settings = settings

        # Recent OHLC data for validation
        self.recent_ohlc: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=50)))

    async def process(self, data: OHLCData) -> OHLCData:
        """Validate OHLC data."""
        start_time = time.perf_counter()

        try:
            validation_result = await self._validate_ohlc(data)

            # Update data quality
            data.quality = validation_result.quality

            # Add validation metadata
            if hasattr(data, 'validation_info'):
                data.validation_info = validation_result.metadata

            # Update metrics
            with self._lock:
                self.metrics.items_processed += 1
                if validation_result.is_valid:
                    self.metrics.items_validated += 1
                else:
                    self.metrics.items_failed += 1

            # Store recent OHLC
            self.recent_ohlc[data.symbol][data.timeframe].append(data)

            return data

        except Exception as e:
            logger.error(f"Error validating OHLC for {data.symbol}: {e}")
            data.quality = DataQuality.INVALID
            self.metrics.items_failed += 1
            raise

        finally:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.metrics.validation_time_ms += processing_time

    async def _validate_ohlc(self, ohlc: OHLCData) -> ValidationResult:
        """Perform comprehensive OHLC validation."""
        issues = []
        warnings = []
        quality = DataQuality.HIGH

        # 1. OHLC relationship validation
        prices = [float(ohlc.open), float(ohlc.high), float(ohlc.low), float(ohlc.close)]

        if not all(p > 0 for p in prices):
            issues.append("Invalid price: zero or negative OHLC values")
            quality = DataQuality.INVALID

        if ohlc.high < ohlc.low:
            issues.append("Invalid OHLC: high is less than low")
            quality = DataQuality.INVALID

        if not (ohlc.low <= ohlc.open <= ohlc.high):
            issues.append("Invalid OHLC: open price outside high-low range")
            quality = DataQuality.SUSPECT

        if not (ohlc.low <= ohlc.close <= ohlc.high):
            issues.append("Invalid OHLC: close price outside high-low range")
            quality = DataQuality.SUSPECT

        # 2. Volume validation
        if ohlc.volume < 0:
            issues.append("Negative volume")
            quality = DataQuality.INVALID
        elif ohlc.volume == 0:
            warnings.append("Zero volume")

        # 3. Tick count validation
        if ohlc.tick_count < 0:
            issues.append("Negative tick count")
            quality = DataQuality.INVALID
        elif ohlc.tick_count == 0 and ohlc.volume > 0:
            warnings.append("Volume without ticks")

        # 4. Gap detection
        recent = self.recent_ohlc[ohlc.symbol][ohlc.timeframe]
        if recent:
            last_ohlc = recent[-1]

            # Check for price gaps
            gap_percentage = abs(ohlc.open - last_ohlc.close) / last_ohlc.close
            if gap_percentage > 0.05:  # 5% gap threshold
                warnings.append(f"Price gap detected: {gap_percentage:.2%}")
                if quality == DataQuality.HIGH:
                    quality = DataQuality.MEDIUM

            # Check for time gaps
            expected_interval = self._get_timeframe_interval(ohlc.timeframe)
            if expected_interval:
                time_diff = (ohlc.timestamp - last_ohlc.timestamp).total_seconds()
                if time_diff > expected_interval * 1.5:  # 50% tolerance
                    warnings.append(f"Time gap detected: {time_diff:.0f}s")
                    if quality == DataQuality.HIGH:
                        quality = DataQuality.MEDIUM

        # 5. Timestamp validation
        now = datetime.now(timezone.utc)
        age_seconds = (now - ohlc.timestamp).total_seconds()

        if age_seconds < 0:
            issues.append("Future timestamp")
            quality = DataQuality.SUSPECT

        # 6. Pattern validation (basic sanity checks)
        body_size = abs(ohlc.close - ohlc.open)
        range_size = ohlc.high - ohlc.low

        if range_size == 0:
            warnings.append("Zero price range")
        elif body_size > range_size:
            issues.append("Invalid pattern: body larger than range")
            quality = DataQuality.SUSPECT

        # Determine overall validity
        is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            quality=quality,
            issues=issues,
            warnings=warnings,
            metadata={
                'validation_timestamp': now.isoformat(),
                'body_size': float(body_size),
                'range_size': float(range_size),
                'age_seconds': age_seconds,
                'has_gaps': len([w for w in warnings if 'gap' in w.lower()]) > 0,
            }
        )

    def _get_timeframe_interval(self, timeframe: str) -> Optional[int]:
        """Get interval in seconds for timeframe."""
        intervals = {
            'M1': 60, 'M2': 120, 'M3': 180, 'M4': 240, 'M5': 300,
            'M6': 360, 'M10': 600, 'M12': 720, 'M15': 900, 'M20': 1200,
            'M30': 1800, 'H1': 3600, 'H2': 7200, 'H3': 10800, 'H4': 14400,
            'H6': 21600, 'H8': 28800, 'H12': 43200, 'D1': 86400,
        }
        return intervals.get(timeframe)

    def get_metrics(self) -> Dict[str, Any]:
        """Get validator metrics."""
        with self._lock:
            success_rate = (
                self.metrics.items_validated / max(self.metrics.items_processed, 1)
            ) * 100

            return {
                'processor_name': self.name,
                'enabled': self.enabled,
                'items_processed': self.metrics.items_processed,
                'items_validated': self.metrics.items_validated,
                'items_failed': self.metrics.items_failed,
                'success_rate_percent': success_rate,
                'avg_validation_time_ms': (
                    self.metrics.validation_time_ms / max(self.metrics.items_processed, 1)
                ),
                'symbols_tracked': len(self.recent_ohlc),
                'timeframes_tracked': sum(len(tfs) for tfs in self.recent_ohlc.values()),
            }


class DataEnricher(DataProcessor):
    """
    Data enrichment processor.

    Features:
    - Calculate additional price metrics
    - Add technical indicators
    - Market session information
    - Statistical analysis
    - Correlation data
    """

    def __init__(self, settings: Settings):
        super().__init__("DataEnricher")
        self.settings = settings

        # Market sessions configuration
        self.market_sessions = {
            'forex': {
                'sydney': {'start': '22:00', 'end': '07:00', 'timezone': 'UTC'},
                'tokyo': {'start': '00:00', 'end': '09:00', 'timezone': 'UTC'},
                'london': {'start': '08:00', 'end': '17:00', 'timezone': 'UTC'},
                'new_york': {'start': '13:00', 'end': '22:00', 'timezone': 'UTC'},
            }
        }

    async def process(self, data: Union[TickData, OHLCData]) -> Union[TickData, OHLCData]:
        """Enrich data with additional information."""
        start_time = time.perf_counter()

        try:
            if isinstance(data, TickData):
                enriched_data = await self._enrich_tick_data(data)
            elif isinstance(data, OHLCData):
                enriched_data = await self._enrich_ohlc_data(data)
            else:
                enriched_data = data

            with self._lock:
                self.metrics.items_processed += 1
                self.metrics.items_enriched += 1

            return enriched_data

        except Exception as e:
            logger.error(f"Error enriching data: {e}")
            self.metrics.items_failed += 1
            return data

        finally:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.metrics.enrichment_time_ms += processing_time

    async def _enrich_tick_data(self, tick: TickData) -> TickData:
        """Enrich tick data with additional metrics."""
        # Add market session information
        session_info = self._get_market_session_info(tick.timestamp)

        # Add enrichment data as metadata
        enrichment_data = {
            'market_session': session_info,
            'is_market_open': session_info.get('active_sessions', []) != [],
            'spread_percentage': float(tick.spread / tick.mid_price) if tick.mid_price > 0 else 0,
            'enriched_at': datetime.now(timezone.utc).isoformat(),
        }

        # Add enrichment info to tick if possible
        if hasattr(tick, 'enrichment_info'):
            tick.enrichment_info = enrichment_data

        return tick

    async def _enrich_ohlc_data(self, ohlc: OHLCData) -> OHLCData:
        """Enrich OHLC data with additional metrics."""
        # Calculate additional price metrics
        typical_price = ohlc.typical_price
        weighted_price = ohlc.weighted_price
        range_size = ohlc.high - ohlc.low
        body_size = abs(ohlc.close - ohlc.open)

        # Determine candlestick pattern
        pattern_info = self._analyze_candlestick_pattern(ohlc)

        # Market session information
        session_info = self._get_market_session_info(ohlc.timestamp)

        # Add enrichment data
        enrichment_data = {
            'typical_price': float(typical_price),
            'weighted_price': float(weighted_price),
            'range_size': float(range_size),
            'body_size': float(body_size),
            'body_to_range_ratio': float(body_size / range_size) if range_size > 0 else 0,
            'pattern_info': pattern_info,
            'market_session': session_info,
            'is_market_open': session_info.get('active_sessions', []) != [],
            'enriched_at': datetime.now(timezone.utc).isoformat(),
        }

        # Add enrichment info if possible
        if hasattr(ohlc, 'enrichment_info'):
            ohlc.enrichment_info = enrichment_data

        return ohlc

    def _get_market_session_info(self, timestamp: datetime) -> Dict[str, Any]:
        """Get market session information for timestamp."""
        current_time = timestamp.time()
        active_sessions = []

        # Check forex sessions (simplified)
        for session_name, session_info in self.market_sessions['forex'].items():
            start_time = datetime.strptime(session_info['start'], '%H:%M').time()
            end_time = datetime.strptime(session_info['end'], '%H:%M').time()

            # Handle overnight sessions
            if start_time <= end_time:
                if start_time <= current_time <= end_time:
                    active_sessions.append(session_name)
            else:
                if current_time >= start_time or current_time <= end_time:
                    active_sessions.append(session_name)

        return {
            'timestamp': timestamp.isoformat(),
            'active_sessions': active_sessions,
            'is_major_session': any(s in ['london', 'new_york'] for s in active_sessions),
        }

    def _analyze_candlestick_pattern(self, ohlc: OHLCData) -> Dict[str, Any]:
        """Basic candlestick pattern analysis."""
        body_size = abs(ohlc.close - ohlc.open)
        range_size = ohlc.high - ohlc.low

        upper_shadow = ohlc.high - max(ohlc.open, ohlc.close)
        lower_shadow = min(ohlc.open, ohlc.close) - ohlc.low

        pattern_type = "unknown"

        if body_size == 0:
            pattern_type = "doji"
        elif body_size < range_size * 0.1:
            pattern_type = "spinning_top"
        elif body_size > range_size * 0.7:
            pattern_type = "marubozu"
        elif ohlc.close > ohlc.open:
            pattern_type = "bullish"
        else:
            pattern_type = "bearish"

        return {
            'pattern_type': pattern_type,
            'is_bullish': ohlc.close > ohlc.open,
            'is_bearish': ohlc.close < ohlc.open,
            'is_doji': ohlc.is_doji,
            'upper_shadow_size': float(upper_shadow),
            'lower_shadow_size': float(lower_shadow),
            'body_percentage': float(body_size / range_size) if range_size > 0 else 0,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get enricher metrics."""
        with self._lock:
            enrichment_rate = (
                self.metrics.items_enriched / max(self.metrics.items_processed, 1)
            ) * 100

            return {
                'processor_name': self.name,
                'enabled': self.enabled,
                'items_processed': self.metrics.items_processed,
                'items_enriched': self.metrics.items_enriched,
                'items_failed': self.metrics.items_failed,
                'enrichment_rate_percent': enrichment_rate,
                'avg_enrichment_time_ms': (
                    self.metrics.enrichment_time_ms / max(self.metrics.items_processed, 1)
                ),
            }


class MarketDataPipeline:
    """
    Comprehensive market data processing pipeline.

    Features:
    - Multi-stage processing with validation and enrichment
    - Configurable processor chain
    - Real-time processing with buffering
    - Error handling and retry logic
    - Performance monitoring
    - Quality assessment and reporting
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.buffer_manager = get_buffer_manager()
        self.event_publisher = get_event_publisher()
        self.task_manager = get_task_manager()

        # Processing pipeline stages
        self.tick_processors: List[DataProcessor] = []
        self.ohlc_processors: List[DataProcessor] = []

        # Initialize default processors
        self._initialize_processors()

        # Pipeline state
        self.is_running = False
        self.processing_lock = threading.RLock()

        # Performance metrics
        self.pipeline_metrics = {
            'items_processed': 0,
            'processing_errors': 0,
            'avg_processing_time_ms': 0.0,
            'throughput_per_second': 0.0,
        }

        # Processing queues
        self.tick_queue = asyncio.Queue(maxsize=1000)
        self.ohlc_queue = asyncio.Queue(maxsize=1000)

        # Worker tasks
        self.worker_tasks: List[asyncio.Task] = []

    def _initialize_processors(self) -> None:
        """Initialize default processing pipeline."""
        # Tick data processors
        self.tick_processors = [
            TickDataValidator(self.settings),
            DataEnricher(self.settings),
        ]

        # OHLC data processors
        self.ohlc_processors = [
            OHLCDataValidator(self.settings),
            DataEnricher(self.settings),
        ]

        logger.info(
            f"Initialized pipeline with {len(self.tick_processors)} tick processors "
            f"and {len(self.ohlc_processors)} OHLC processors"
        )

    async def start_pipeline(self) -> None:
        """Start the data processing pipeline."""
        if self.is_running:
            logger.warning("Pipeline already running")
            return

        try:
            self.is_running = True

            # Start worker tasks
            self.worker_tasks = [
                asyncio.create_task(self._tick_processor_worker(), name="tick-processor"),
                asyncio.create_task(self._ohlc_processor_worker(), name="ohlc-processor"),
                asyncio.create_task(self._metrics_monitor(), name="metrics-monitor"),
            ]

            logger.info("Market data pipeline started")

        except Exception as e:
            self.is_running = False
            logger.error(f"Failed to start pipeline: {e}")
            raise

    async def stop_pipeline(self) -> None:
        """Stop the data processing pipeline."""
        if not self.is_running:
            return

        try:
            self.is_running = False

            # Cancel worker tasks
            for task in self.worker_tasks:
                task.cancel()

            # Wait for tasks to complete
            if self.worker_tasks:
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)

            self.worker_tasks.clear()

            logger.info("Market data pipeline stopped")

        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")

    async def process_tick_data(self, tick: TickData) -> Optional[TickData]:
        """Process tick data through the pipeline."""
        if not self.is_running:
            return tick

        try:
            # Add to processing queue
            await self.tick_queue.put(tick)
            return tick

        except asyncio.QueueFull:
            logger.warning("Tick processing queue is full, dropping data")
            self.pipeline_metrics['processing_errors'] += 1
            return None
        except Exception as e:
            logger.error(f"Error queueing tick data: {e}")
            self.pipeline_metrics['processing_errors'] += 1
            return None

    async def process_ohlc_data(self, ohlc: OHLCData) -> Optional[OHLCData]:
        """Process OHLC data through the pipeline."""
        if not self.is_running:
            return ohlc

        try:
            # Add to processing queue
            await self.ohlc_queue.put(ohlc)
            return ohlc

        except asyncio.QueueFull:
            logger.warning("OHLC processing queue is full, dropping data")
            self.pipeline_metrics['processing_errors'] += 1
            return None
        except Exception as e:
            logger.error(f"Error queueing OHLC data: {e}")
            self.pipeline_metrics['processing_errors'] += 1
            return None

    async def _tick_processor_worker(self) -> None:
        """Worker for processing tick data."""
        logger.debug("Started tick processor worker")

        while self.is_running:
            try:
                # Get tick from queue with timeout
                tick = await asyncio.wait_for(self.tick_queue.get(), timeout=1.0)

                start_time = time.perf_counter()

                # Process through pipeline stages
                processed_tick = tick
                for processor in self.tick_processors:
                    if processor.enabled:
                        processed_tick = await processor.process(processed_tick)

                # Update metrics
                processing_time = (time.perf_counter() - start_time) * 1000
                await self._update_pipeline_metrics(processing_time)

                # Publish processed data event
                await self._publish_processed_data_event(processed_tick)

                # Mark task as done
                self.tick_queue.task_done()

            except asyncio.TimeoutError:
                continue  # No data available, continue loop
            except Exception as e:
                logger.error(f"Error in tick processor worker: {e}")
                self.pipeline_metrics['processing_errors'] += 1
                await asyncio.sleep(0.1)  # Brief pause on error

    async def _ohlc_processor_worker(self) -> None:
        """Worker for processing OHLC data."""
        logger.debug("Started OHLC processor worker")

        while self.is_running:
            try:
                # Get OHLC from queue with timeout
                ohlc = await asyncio.wait_for(self.ohlc_queue.get(), timeout=1.0)

                start_time = time.perf_counter()

                # Process through pipeline stages
                processed_ohlc = ohlc
                for processor in self.ohlc_processors:
                    if processor.enabled:
                        processed_ohlc = await processor.process(processed_ohlc)

                # Update metrics
                processing_time = (time.perf_counter() - start_time) * 1000
                await self._update_pipeline_metrics(processing_time)

                # Publish processed data event
                await self._publish_processed_data_event(processed_ohlc)

                # Mark task as done
                self.ohlc_queue.task_done()

            except asyncio.TimeoutError:
                continue  # No data available, continue loop
            except Exception as e:
                logger.error(f"Error in OHLC processor worker: {e}")
                self.pipeline_metrics['processing_errors'] += 1
                await asyncio.sleep(0.1)  # Brief pause on error

    async def _update_pipeline_metrics(self, processing_time_ms: float) -> None:
        """Update pipeline performance metrics."""
        with self.processing_lock:
            self.pipeline_metrics['items_processed'] += 1

            # Update average processing time
            count = self.pipeline_metrics['items_processed']
            current_avg = self.pipeline_metrics['avg_processing_time_ms']
            self.pipeline_metrics['avg_processing_time_ms'] = (
                (current_avg * (count - 1) + processing_time_ms) / count
            )

    async def _publish_processed_data_event(self, data: Union[TickData, OHLCData]) -> None:
        """Publish processed data event."""
        try:
            event_type = MarketEventType.TICK_RECEIVED if isinstance(data, TickData) else MarketEventType.OHLC_UPDATED

            event = MarketEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                symbol=data.symbol,
                data={
                    'processed': True,
                    'quality': data.quality.value,
                    'timestamp': data.timestamp.isoformat(),
                }
            )

            await self.event_publisher.publish(event)

        except Exception as e:
            logger.debug(f"Error publishing processed data event: {e}")

    async def _metrics_monitor(self) -> None:
        """Monitor pipeline performance metrics."""
        last_count = 0

        while self.is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute

                current_count = self.pipeline_metrics['items_processed']
                throughput = (current_count - last_count) / 60  # items per second
                self.pipeline_metrics['throughput_per_second'] = throughput

                # Log performance metrics
                logger.info(
                    f"Pipeline Performance: "
                    f"{throughput:.1f} items/sec, "
                    f"{self.pipeline_metrics['avg_processing_time_ms']:.2f}ms avg latency, "
                    f"{self.pipeline_metrics['processing_errors']} errors"
                )

                last_count = current_count

            except Exception as e:
                logger.error(f"Error in metrics monitor: {e}")

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline metrics."""
        with self.processing_lock:
            # Get processor metrics
            processor_metrics = {
                'tick_processors': [p.get_metrics() for p in self.tick_processors],
                'ohlc_processors': [p.get_metrics() for p in self.ohlc_processors],
            }

            return {
                'is_running': self.is_running,
                'pipeline_metrics': self.pipeline_metrics.copy(),
                'queue_sizes': {
                    'tick_queue': self.tick_queue.qsize(),
                    'ohlc_queue': self.ohlc_queue.qsize(),
                },
                'processor_metrics': processor_metrics,
                'worker_tasks': len(self.worker_tasks),
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on pipeline."""
        health = {
            'status': 'healthy' if self.is_running else 'stopped',
            'is_running': self.is_running,
            'queue_health': 'good',
            'processor_health': 'good',
        }

        # Check queue sizes
        if self.is_running:
            tick_queue_usage = self.tick_queue.qsize() / self.tick_queue.maxsize
            ohlc_queue_usage = self.ohlc_queue.qsize() / self.ohlc_queue.maxsize

            if tick_queue_usage > 0.8 or ohlc_queue_usage > 0.8:
                health['queue_health'] = 'degraded'
                health['status'] = 'degraded'

        # Check processor health
        error_rate = (
            self.pipeline_metrics['processing_errors'] /
            max(self.pipeline_metrics['items_processed'], 1)
        )

        if error_rate > 0.05:  # 5% error rate threshold
            health['processor_health'] = 'degraded'
            health['status'] = 'degraded'

        return health


# Global pipeline instance
_market_data_pipeline: Optional[MarketDataPipeline] = None


def get_market_data_pipeline() -> MarketDataPipeline:
    """Get the global market data pipeline instance."""
    global _market_data_pipeline

    if _market_data_pipeline is None:
        from core.config.settings import Settings
        settings = Settings()
        _market_data_pipeline = MarketDataPipeline(settings)

    return _market_data_pipeline