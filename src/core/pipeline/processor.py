"""
Core data processing components for the market data pipeline.

This module provides the foundation for processing market data with
validation, transformation, enrichment, and error handling.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.data.models import TickData, OHLCData, MarketEvent

logger = get_logger(__name__)
settings = get_settings()


class ProcessingStage(str, Enum):
    """Data processing stages."""

    INGESTION = "ingestion"
    VALIDATION = "validation"
    ENRICHMENT = "enrichment"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    OUTPUT = "output"


class ProcessingStatus(str, Enum):
    """Processing status codes."""

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class ProcessingResult(BaseModel):
    """Result of a data processing operation."""

    stage: ProcessingStage = Field(description="Processing stage")
    status: ProcessingStatus = Field(description="Processing status")
    message: str = Field(default="", description="Processing message")

    # Timing information
    started_at: datetime = Field(description="Processing start time")
    completed_at: datetime = Field(description="Processing completion time")
    duration_ms: float = Field(description="Processing duration in milliseconds")

    # Data information
    input_count: int = Field(default=0, description="Number of input items")
    output_count: int = Field(default=0, description="Number of output items")
    error_count: int = Field(default=0, description="Number of errors")

    # Additional context
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing context"
    )

    def is_successful(self) -> bool:
        """Check if processing was successful."""
        return self.status == ProcessingStatus.SUCCESS

    def has_warnings(self) -> bool:
        """Check if processing had warnings."""
        return self.status == ProcessingStatus.WARNING

    def has_errors(self) -> bool:
        """Check if processing had errors."""
        return self.status == ProcessingStatus.ERROR

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "message": self.message,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_ms": self.duration_ms,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "error_count": self.error_count,
            "context": self.context
        }

    # Pydantic v2 configuration moved to model_config

class DataProcessor(ABC):
    """
    Base class for all data processors.

    Provides common functionality for processing market data
    with error handling, performance tracking, and result reporting.
    """

    def __init__(self, name: str, stage: ProcessingStage):
        """
        Initialize data processor.

        Args:
            name: Processor name
            stage: Processing stage
        """
        self.name = name
        self.stage = stage

        # Performance metrics
        self._processed_count = 0
        self._error_count = 0
        self._total_processing_time = 0.0
        self._last_processing_time = 0.0

        logger.debug(f"Initialized {self.stage.value} processor: {self.name}")

    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any] = None) -> ProcessingResult:
        """
        Process input data.

        Args:
            data: Input data to process
            context: Processing context

        Returns:
            ProcessingResult with processing outcome
        """
        pass

    def process_batch(
        self,
        data_batch: List[Any],
        context: Dict[str, Any] = None
    ) -> List[ProcessingResult]:
        """
        Process batch of data items.

        Args:
            data_batch: List of data items to process
            context: Processing context

        Returns:
            List of processing results
        """
        results = []

        for item in data_batch:
            try:
                result = self.process(item, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing batch item in {self.name}: {e}")
                results.append(self._create_error_result(
                    f"Batch processing error: {e}"
                ))

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get processor performance statistics."""
        avg_processing_time = (
            self._total_processing_time / max(self._processed_count, 1)
        )

        return {
            "processor_name": self.name,
            "stage": self.stage.value,
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._processed_count - self._error_count) /
                max(self._processed_count, 1) * 100
            ),
            "avg_processing_time_ms": avg_processing_time * 1000,
            "last_processing_time_ms": self._last_processing_time * 1000,
            "throughput_per_second": (
                1.0 / avg_processing_time if avg_processing_time > 0 else 0
            )
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._processed_count = 0
        self._error_count = 0
        self._total_processing_time = 0.0
        self._last_processing_time = 0.0

    def _create_result(
        self,
        status: ProcessingStatus,
        message: str = "",
        started_at: datetime = None,
        completed_at: datetime = None,
        input_count: int = 1,
        output_count: int = 1,
        error_count: int = 0,
        context: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Create processing result."""
        now = datetime.now(timezone.utc)
        started_at = started_at or now
        completed_at = completed_at or now

        duration_ms = (completed_at - started_at).total_seconds() * 1000

        return ProcessingResult(
            stage=self.stage,
            status=status,
            message=message,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            input_count=input_count,
            output_count=output_count,
            error_count=error_count,
            context=context or {}
        )

    def _create_success_result(
        self,
        message: str = "Processing completed successfully",
        **kwargs
    ) -> ProcessingResult:
        """Create success result."""
        return self._create_result(ProcessingStatus.SUCCESS, message, **kwargs)

    def _create_warning_result(
        self,
        message: str = "Processing completed with warnings",
        **kwargs
    ) -> ProcessingResult:
        """Create warning result."""
        return self._create_result(ProcessingStatus.WARNING, message, **kwargs)

    def _create_error_result(
        self,
        message: str = "Processing failed",
        **kwargs
    ) -> ProcessingResult:
        """Create error result."""
        kwargs.setdefault("error_count", 1)
        kwargs.setdefault("output_count", 0)
        return self._create_result(ProcessingStatus.ERROR, message, **kwargs)

    def _track_performance(self, processing_time: float, success: bool) -> None:
        """Track processor performance metrics."""
        self._processed_count += 1
        self._total_processing_time += processing_time
        self._last_processing_time = processing_time

        if not success:
            self._error_count += 1


class TickDataProcessor(DataProcessor):
    """
    Processor specifically for tick data.

    Handles tick data validation, enrichment, and transformation
    with optimizations for high-frequency data processing.
    """

    def __init__(self, name: str = "TickDataProcessor"):
        """Initialize tick data processor."""
        super().__init__(name, ProcessingStage.TRANSFORMATION)

        # Tick-specific configuration
        self.max_spread_percent = 10.0  # Maximum spread percentage
        self.max_price_change_percent = 5.0  # Maximum price change percentage
        self.min_price_threshold = 0.00001  # Minimum price threshold

    def process(self, data: Any, context: Dict[str, Any] = None) -> ProcessingResult:
        """
        Process tick data.

        Args:
            data: TickData instance
            context: Processing context

        Returns:
            ProcessingResult
        """
        start_time = time.time()
        started_at = datetime.now(timezone.utc)

        try:
            if not isinstance(data, TickData):
                return self._create_error_result(
                    f"Invalid data type: expected TickData, got {type(data).__name__}",
                    started_at=started_at
                )

            # Validate tick data
            validation_issues = self._validate_tick(data, context)

            # Enrich tick data
            self._enrich_tick(data, context)

            # Transform tick data
            self._transform_tick(data, context)

            processing_time = time.time() - start_time
            self._track_performance(processing_time, len(validation_issues) == 0)

            if validation_issues:
                return self._create_warning_result(
                    f"Tick processed with {len(validation_issues)} issues: {', '.join(validation_issues)}",
                    started_at=started_at,
                    context={"validation_issues": validation_issues}
                )
            else:
                return self._create_success_result(
                    "Tick processed successfully",
                    started_at=started_at
                )

        except Exception as e:
            processing_time = time.time() - start_time
            self._track_performance(processing_time, False)

            logger.error(f"Error processing tick data: {e}")
            return self._create_error_result(
                f"Processing error: {e}",
                started_at=started_at
            )

    def _validate_tick(self, tick: TickData, context: Dict[str, Any] = None) -> List[str]:
        """
        Validate tick data quality.

        Args:
            tick: Tick data to validate
            context: Processing context

        Returns:
            List of validation issues
        """
        issues = []

        # Validate prices
        if tick.bid <= 0 or tick.ask <= 0:
            issues.append("invalid_prices")

        if tick.bid > tick.ask:
            issues.append("inverted_spread")

        # Check spread
        if tick.spread and tick.mid_price:
            spread_percent = (tick.spread / tick.mid_price) * 100
            if spread_percent > self.max_spread_percent:
                issues.append("excessive_spread")

        # Check for price jumps (if previous price available in context)
        if context and "previous_tick" in context:
            prev_tick = context["previous_tick"]
            if isinstance(prev_tick, TickData):
                price_change = abs(tick.mid_price - prev_tick.mid_price)
                change_percent = (price_change / prev_tick.mid_price) * 100

                if change_percent > self.max_price_change_percent:
                    issues.append("large_price_jump")

        # Check timestamp
        if tick.timestamp > datetime.now(timezone.utc):
            issues.append("future_timestamp")

        return issues

    def _enrich_tick(self, tick: TickData, context: Dict[str, Any] = None) -> None:
        """
        Enrich tick data with additional information.

        Args:
            tick: Tick data to enrich
            context: Processing context
        """
        # Calculate latency if not already set
        if tick.latency_ms is None:
            tick.latency_ms = tick.calculate_latency()

        # Add market session information if available
        if context and "market_sessions" in context:
            # This would be implemented based on trading session data
            pass

        # Add statistical information
        if context and "price_statistics" in context:
            stats = context["price_statistics"]
            # Could add percentile information, volatility measures, etc.
            pass

    def _transform_tick(self, tick: TickData, context: Dict[str, Any] = None) -> None:
        """
        Transform tick data for downstream processing.

        Args:
            tick: Tick data to transform
            context: Processing context
        """
        # Normalize prices if needed
        if context and "price_normalization" in context:
            # This would apply price normalization rules
            pass

        # Apply currency conversion if needed
        if context and "currency_conversion" in context:
            # This would apply currency conversion
            pass


class OHLCDataProcessor(DataProcessor):
    """
    Processor specifically for OHLC data.

    Handles OHLC data validation, aggregation, and indicator calculation
    with support for multiple timeframes.
    """

    def __init__(self, name: str = "OHLCDataProcessor"):
        """Initialize OHLC data processor."""
        super().__init__(name, ProcessingStage.AGGREGATION)

        # OHLC-specific configuration
        self.validate_ohlc_relationships = True
        self.calculate_derived_values = True

    def process(self, data: Any, context: Dict[str, Any] = None) -> ProcessingResult:
        """
        Process OHLC data.

        Args:
            data: OHLCData instance
            context: Processing context

        Returns:
            ProcessingResult
        """
        start_time = time.time()
        started_at = datetime.now(timezone.utc)

        try:
            if not isinstance(data, OHLCData):
                return self._create_error_result(
                    f"Invalid data type: expected OHLCData, got {type(data).__name__}",
                    started_at=started_at
                )

            # Validate OHLC data
            validation_issues = self._validate_ohlc(data, context)

            # Calculate derived values
            if self.calculate_derived_values:
                self._calculate_derived_values(data, context)

            # Aggregate with previous data if available
            self._aggregate_ohlc(data, context)

            processing_time = time.time() - start_time
            self._track_performance(processing_time, len(validation_issues) == 0)

            if validation_issues:
                return self._create_warning_result(
                    f"OHLC processed with {len(validation_issues)} issues: {', '.join(validation_issues)}",
                    started_at=started_at,
                    context={"validation_issues": validation_issues}
                )
            else:
                return self._create_success_result(
                    "OHLC processed successfully",
                    started_at=started_at
                )

        except Exception as e:
            processing_time = time.time() - start_time
            self._track_performance(processing_time, False)

            logger.error(f"Error processing OHLC data: {e}")
            return self._create_error_result(
                f"Processing error: {e}",
                started_at=started_at
            )

    def _validate_ohlc(self, ohlc: OHLCData, context: Dict[str, Any] = None) -> List[str]:
        """
        Validate OHLC data relationships.

        Args:
            ohlc: OHLC data to validate
            context: Processing context

        Returns:
            List of validation issues
        """
        issues = []

        if not self.validate_ohlc_relationships:
            return issues

        # Validate OHLC relationships
        if ohlc.high < ohlc.low:
            issues.append("high_less_than_low")

        if ohlc.high < ohlc.open or ohlc.high < ohlc.close:
            issues.append("high_less_than_open_close")

        if ohlc.low > ohlc.open or ohlc.low > ohlc.close:
            issues.append("low_greater_than_open_close")

        # Check for zero values
        if any(price <= 0 for price in [ohlc.open, ohlc.high, ohlc.low, ohlc.close]):
            issues.append("zero_or_negative_prices")

        # Check volume
        if ohlc.volume < 0:
            issues.append("negative_volume")

        # Check tick count
        if ohlc.tick_count < 0:
            issues.append("negative_tick_count")

        return issues

    def _calculate_derived_values(self, ohlc: OHLCData, context: Dict[str, Any] = None) -> None:
        """
        Calculate derived OHLC values.

        Args:
            ohlc: OHLC data to enhance
            context: Processing context
        """
        # Values are calculated as properties in OHLCData model
        # This method could be used to add additional derived metrics
        pass

    def _aggregate_ohlc(self, ohlc: OHLCData, context: Dict[str, Any] = None) -> None:
        """
        Aggregate OHLC data with historical data.

        Args:
            ohlc: OHLC data to aggregate
            context: Processing context
        """
        # This would implement aggregation logic
        # For example, rolling averages, cumulative volumes, etc.
        if context and "aggregation_buffer" in context:
            # Add to aggregation buffer for rolling calculations
            pass


class EventProcessor(DataProcessor):
    """
    Processor for market events.

    Handles event validation, routing, and transformation
    for real-time event processing.
    """

    def __init__(self, name: str = "EventProcessor"):
        """Initialize event processor."""
        super().__init__(name, ProcessingStage.OUTPUT)

        # Event-specific configuration
        self.max_event_age_seconds = 300  # 5 minutes
        self.duplicate_detection = True

    def process(self, data: Any, context: Dict[str, Any] = None) -> ProcessingResult:
        """
        Process market event.

        Args:
            data: MarketEvent instance
            context: Processing context

        Returns:
            ProcessingResult
        """
        start_time = time.time()
        started_at = datetime.now(timezone.utc)

        try:
            if not isinstance(data, MarketEvent):
                return self._create_error_result(
                    f"Invalid data type: expected MarketEvent, got {type(data).__name__}",
                    started_at=started_at
                )

            # Validate event
            validation_issues = self._validate_event(data, context)

            # Route event
            routing_result = self._route_event(data, context)

            processing_time = time.time() - start_time
            self._track_performance(processing_time, len(validation_issues) == 0)

            result_context = {
                "validation_issues": validation_issues,
                "routing_result": routing_result
            }

            if validation_issues:
                return self._create_warning_result(
                    f"Event processed with {len(validation_issues)} issues: {', '.join(validation_issues)}",
                    started_at=started_at,
                    context=result_context
                )
            else:
                return self._create_success_result(
                    "Event processed successfully",
                    started_at=started_at,
                    context=result_context
                )

        except Exception as e:
            processing_time = time.time() - start_time
            self._track_performance(processing_time, False)

            logger.error(f"Error processing event: {e}")
            return self._create_error_result(
                f"Processing error: {e}",
                started_at=started_at
            )

    def _validate_event(self, event: MarketEvent, context: Dict[str, Any] = None) -> List[str]:
        """
        Validate market event.

        Args:
            event: Event to validate
            context: Processing context

        Returns:
            List of validation issues
        """
        issues = []

        # Check event age
        now = datetime.now(timezone.utc)
        age_seconds = (now - event.timestamp).total_seconds()

        if age_seconds > self.max_event_age_seconds:
            issues.append("stale_event")

        # Check for duplicate events
        if self.duplicate_detection and context and "processed_events" in context:
            processed_events = context["processed_events"]
            if event.event_id in processed_events:
                issues.append("duplicate_event")

        # Validate event data
        if not event.event_id:
            issues.append("missing_event_id")

        if not event.event_type:
            issues.append("missing_event_type")

        return issues

    def _route_event(self, event: MarketEvent, context: Dict[str, Any] = None) -> str:
        """
        Route event to appropriate handlers.

        Args:
            event: Event to route
            context: Processing context

        Returns:
            Routing result description
        """
        # This would implement event routing logic
        # For now, just return a description
        return f"Event {event.event_id} routed to {event.event_type} handlers"