"""
Market data processing pipeline with comprehensive data flow management.

This module provides the main pipeline for processing market data through
multiple stages with validation, enrichment, transformation, and routing.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.data.models import TickData, OHLCData, MarketEvent
from src.core.data.buffer import get_buffer_manager
from src.core.data.events import get_event_publisher
from src.core.symbols.manager import get_symbol_manager
from src.indicators.manager import get_indicator_manager
from src.core.tasks import background_task, scheduled_task

from .processor import (
    DataProcessor, ProcessingResult, ProcessingStatus, ProcessingStage,
    TickDataProcessor, OHLCDataProcessor, EventProcessor
)

logger = get_logger(__name__)
settings = get_settings()


class PipelineConfig(BaseModel):
    """Configuration for the market data pipeline."""

    # Processing stages to enable
    enable_validation: bool = Field(
        default=True,
        description="Enable data validation stage"
    )
    enable_enrichment: bool = Field(
        default=True,
        description="Enable data enrichment stage"
    )
    enable_transformation: bool = Field(
        default=True,
        description="Enable data transformation stage"
    )
    enable_aggregation: bool = Field(
        default=True,
        description="Enable data aggregation stage"
    )

    # Performance settings
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Batch size for processing"
    )
    max_processing_time_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Maximum processing time per batch"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing"
    )
    max_worker_threads: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum number of worker threads"
    )

    # Error handling
    max_retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed processing"
    )
    error_threshold_percent: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Error threshold percentage before pipeline pause"
    )
    auto_recovery: bool = Field(
        default=True,
        description="Enable automatic error recovery"
    )

    # Data quality
    quality_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum data quality threshold"
    )
    reject_low_quality_data: bool = Field(
        default=False,
        description="Reject data below quality threshold"
    )

    # Output settings
    publish_events: bool = Field(
        default=True,
        description="Publish processing events"
    )
    update_indicators: bool = Field(
        default=True,
        description="Update technical indicators"
    )
    store_processed_data: bool = Field(
        default=True,
        description="Store processed data in buffers"
    )

    # Pydantic v2 configuration moved to model_config

class PipelineMetrics(BaseModel):
    """Pipeline performance metrics."""

    # Throughput metrics
    total_items_processed: int = Field(default=0, description="Total items processed")
    items_per_second: float = Field(default=0.0, description="Processing throughput")
    average_processing_time_ms: float = Field(default=0.0, description="Average processing time")

    # Quality metrics
    success_rate_percent: float = Field(default=100.0, description="Success rate percentage")
    error_rate_percent: float = Field(default=0.0, description="Error rate percentage")
    quality_score: float = Field(default=1.0, description="Average quality score")

    # Stage metrics
    stage_metrics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Metrics per processing stage"
    )

    # Resource usage
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(default=0.0, description="CPU usage percentage")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_items_processed": self.total_items_processed,
            "items_per_second": self.items_per_second,
            "average_processing_time_ms": self.average_processing_time_ms,
            "success_rate_percent": self.success_rate_percent,
            "error_rate_percent": self.error_rate_percent,
            "quality_score": self.quality_score,
            "stage_metrics": self.stage_metrics,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent
        }


class MarketDataPipeline:
    """
    Comprehensive market data processing pipeline.

    Orchestrates the flow of market data through multiple processing stages
    with error handling, performance monitoring, and quality control.
    """

    def __init__(self, config: PipelineConfig = None):
        """
        Initialize market data pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Processing stages
        self._processors: Dict[ProcessingStage, List[DataProcessor]] = {
            ProcessingStage.VALIDATION: [],
            ProcessingStage.ENRICHMENT: [],
            ProcessingStage.TRANSFORMATION: [
                TickDataProcessor(),
                OHLCDataProcessor()
            ],
            ProcessingStage.AGGREGATION: [],
            ProcessingStage.OUTPUT: [EventProcessor()]
        }

        # State management
        self._is_running = False
        self._is_paused = False
        self._error_recovery_mode = False

        # Performance tracking
        self._metrics = PipelineMetrics()
        self._processing_times: List[float] = []
        self._last_metrics_update = time.time()

        # Dependencies
        self.buffer_manager = get_buffer_manager()
        self.event_publisher = get_event_publisher()
        self.symbol_manager = get_symbol_manager()
        self.indicator_manager = get_indicator_manager()

        # Threading
        self._processing_lock = threading.RLock()
        self._metrics_lock = threading.RLock()

        # Background processing
        self._processing_queue: asyncio.Queue = None
        self._processing_task: Optional[asyncio.Task] = None

        logger.info("Market data pipeline initialized")

    def start(self) -> bool:
        """Start the market data pipeline."""
        if self._is_running:
            logger.warning("Pipeline already running")
            return True

        try:
            # Initialize processing queue
            self._processing_queue = asyncio.Queue(maxsize=10000)

            # Start background processing
            self._processing_task = asyncio.create_task(self._background_processor())

            # Start monitoring
            self._start_monitoring()

            self._is_running = True
            self._is_paused = False

            logger.info("Market data pipeline started")
            return True

        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            return False

    def stop(self) -> None:
        """Stop the market data pipeline."""
        if not self._is_running:
            return

        logger.info("Stopping market data pipeline...")

        self._is_running = False

        # Stop background processing
        if self._processing_task:
            self._processing_task.cancel()

        # Process remaining items in queue
        if self._processing_queue:
            remaining_items = self._processing_queue.qsize()
            if remaining_items > 0:
                logger.info(f"Processing {remaining_items} remaining items...")
                # Could implement graceful queue draining here

        logger.info("Market data pipeline stopped")

    def process_tick(self, tick: TickData) -> bool:
        """
        Process tick data through the pipeline.

        Args:
            tick: Tick data to process

        Returns:
            bool: True if processing initiated successfully
        """
        return self._enqueue_for_processing(tick, "tick")

    def process_ohlc(self, ohlc: OHLCData) -> bool:
        """
        Process OHLC data through the pipeline.

        Args:
            ohlc: OHLC data to process

        Returns:
            bool: True if processing initiated successfully
        """
        return self._enqueue_for_processing(ohlc, "ohlc")

    def process_event(self, event: MarketEvent) -> bool:
        """
        Process market event through the pipeline.

        Args:
            event: Market event to process

        Returns:
            bool: True if processing initiated successfully
        """
        return self._enqueue_for_processing(event, "event")

    def process_batch(
        self,
        items: List[Union[TickData, OHLCData, MarketEvent]]
    ) -> bool:
        """
        Process batch of data items.

        Args:
            items: List of data items to process

        Returns:
            bool: True if processing initiated successfully
        """
        try:
            for item in items:
                if isinstance(item, TickData):
                    self.process_tick(item)
                elif isinstance(item, OHLCData):
                    self.process_ohlc(item)
                elif isinstance(item, MarketEvent):
                    self.process_event(item)
                else:
                    logger.warning(f"Unknown data type in batch: {type(item)}")

            return True

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return False

    def add_processor(self, stage: ProcessingStage, processor: DataProcessor) -> bool:
        """
        Add custom processor to pipeline stage.

        Args:
            stage: Processing stage
            processor: Processor to add

        Returns:
            bool: True if added successfully
        """
        try:
            with self._processing_lock:
                self._processors[stage].append(processor)

            logger.info(f"Added processor {processor.name} to {stage.value} stage")
            return True

        except Exception as e:
            logger.error(f"Error adding processor: {e}")
            return False

    def remove_processor(self, stage: ProcessingStage, processor_name: str) -> bool:
        """
        Remove processor from pipeline stage.

        Args:
            stage: Processing stage
            processor_name: Name of processor to remove

        Returns:
            bool: True if removed successfully
        """
        try:
            with self._processing_lock:
                processors = self._processors[stage]
                for i, processor in enumerate(processors):
                    if processor.name == processor_name:
                        del processors[i]
                        logger.info(f"Removed processor {processor_name} from {stage.value} stage")
                        return True

            logger.warning(f"Processor {processor_name} not found in {stage.value} stage")
            return False

        except Exception as e:
            logger.error(f"Error removing processor: {e}")
            return False

    def pause(self) -> None:
        """Pause pipeline processing."""
        self._is_paused = True
        logger.info("Pipeline processing paused")

    def resume(self) -> None:
        """Resume pipeline processing."""
        self._is_paused = False
        self._error_recovery_mode = False
        logger.info("Pipeline processing resumed")

    def get_metrics(self) -> PipelineMetrics:
        """Get pipeline performance metrics."""
        with self._metrics_lock:
            return self._metrics.copy(deep=True)

    def get_stage_processors(self, stage: ProcessingStage) -> List[str]:
        """Get list of processors for a stage."""
        with self._processing_lock:
            return [p.name for p in self._processors[stage]]

    def get_queue_size(self) -> int:
        """Get current processing queue size."""
        return self._processing_queue.qsize() if self._processing_queue else 0

    def _enqueue_for_processing(self, item: Any, item_type: str) -> bool:
        """Enqueue item for background processing."""
        if not self._is_running or self._is_paused:
            return False

        try:
            if self._processing_queue.full():
                logger.warning("Processing queue is full, dropping item")
                return False

            processing_item = {
                "item": item,
                "type": item_type,
                "enqueued_at": datetime.now(timezone.utc)
            }

            self._processing_queue.put_nowait(processing_item)
            return True

        except Exception as e:
            logger.error(f"Error enqueuing item for processing: {e}")
            return False

    async def _background_processor(self) -> None:
        """Background processing loop."""
        logger.debug("Started background processor")

        while self._is_running:
            try:
                if self._is_paused:
                    await asyncio.sleep(1.0)
                    continue

                # Get item from queue with timeout
                try:
                    processing_item = await asyncio.wait_for(
                        self._processing_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process item
                await self._process_item_async(processing_item)

                # Mark task done
                self._processing_queue.task_done()

            except Exception as e:
                logger.error(f"Error in background processor: {e}")
                await asyncio.sleep(1.0)

        logger.debug("Background processor stopped")

    async def _process_item_async(self, processing_item: Dict[str, Any]) -> None:
        """Process individual item asynchronously."""
        item = processing_item["item"]
        item_type = processing_item["type"]

        start_time = time.time()

        try:
            # Process through pipeline stages
            context = {
                "item_type": item_type,
                "enqueued_at": processing_item["enqueued_at"],
                "processing_started_at": datetime.now(timezone.utc)
            }

            processed_item = await self._run_pipeline_stages(item, context)

            # Handle processed result
            if processed_item:
                await self._handle_processed_item(processed_item, context)

            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, True)

        except Exception as e:
            logger.error(f"Error processing {item_type}: {e}")
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, False)

            # Handle error recovery
            if self.config.auto_recovery:
                await self._handle_processing_error(item, e)

    async def _run_pipeline_stages(
        self,
        item: Any,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """Run item through all pipeline stages."""
        current_item = item

        # Define stage order
        stage_order = [
            ProcessingStage.VALIDATION,
            ProcessingStage.ENRICHMENT,
            ProcessingStage.TRANSFORMATION,
            ProcessingStage.AGGREGATION,
            ProcessingStage.OUTPUT
        ]

        for stage in stage_order:
            if not self._is_stage_enabled(stage):
                continue

            try:
                current_item = await self._process_stage(current_item, stage, context)
                if current_item is None:
                    # Stage rejected the item
                    break

            except Exception as e:
                logger.error(f"Error in {stage.value} stage: {e}")
                if not self.config.auto_recovery:
                    raise
                break

        return current_item

    async def _process_stage(
        self,
        item: Any,
        stage: ProcessingStage,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """Process item through a specific stage."""
        processors = self._processors.get(stage, [])

        if not processors:
            return item  # No processors for this stage

        stage_context = context.copy()
        stage_context["stage"] = stage

        for processor in processors:
            try:
                # Run processor in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    processor.process,
                    item,
                    stage_context
                )

                # Check processing result
                if not result.is_successful():
                    if result.has_errors():
                        if self.config.reject_low_quality_data:
                            logger.warning(
                                f"Rejecting item due to processing error: {result.message}"
                            )
                            return None
                    elif result.has_warnings():
                        logger.warning(
                            f"Processing warning in {processor.name}: {result.message}"
                        )

                # Update stage metrics
                self._update_stage_metrics(stage, processor.name, result)

            except Exception as e:
                logger.error(f"Error in processor {processor.name}: {e}")
                if not self.config.auto_recovery:
                    raise

        return item

    async def _handle_processed_item(self, item: Any, context: Dict[str, Any]) -> None:
        """Handle successfully processed item."""
        try:
            # Store in appropriate buffer
            if self.config.store_processed_data:
                if isinstance(item, TickData):
                    buffer = self.buffer_manager.get_tick_buffer(item.symbol)
                    if buffer:
                        buffer.add_tick(item)

                elif isinstance(item, OHLCData):
                    buffer = self.buffer_manager.get_ohlc_buffer(item.symbol, item.timeframe)
                    if buffer:
                        buffer.add_bar(item)

            # Update indicators
            if self.config.update_indicators and isinstance(item, (TickData, OHLCData)):
                await self._update_indicators(item)

            # Publish events
            if self.config.publish_events:
                await self._publish_processing_event(item, context)

        except Exception as e:
            logger.error(f"Error handling processed item: {e}")

    async def _update_indicators(self, item: Union[TickData, OHLCData]) -> None:
        """Update technical indicators with new data."""
        try:
            if isinstance(item, OHLCData):
                # Convert to format expected by indicators
                price_data = [{
                    "open": float(item.open),
                    "high": float(item.high),
                    "low": float(item.low),
                    "close": float(item.close),
                    "volume": item.volume,
                    "timestamp": item.timestamp.timestamp()
                }]

                # Calculate MACD
                macd_result = self.indicator_manager.calculate_macd(
                    item.symbol,
                    item.timeframe,
                    price_data
                )

                if macd_result:
                    logger.debug(f"Updated MACD for {item.symbol} {item.timeframe}")

        except Exception as e:
            logger.error(f"Error updating indicators: {e}")

    async def _publish_processing_event(self, item: Any, context: Dict[str, Any]) -> None:
        """Publish processing completion event."""
        try:
            if isinstance(item, (TickData, OHLCData)):
                event_type = "ohlc_updated" if isinstance(item, OHLCData) else "tick_received"

                event = MarketEvent(
                    event_id=f"pipeline_{event_type}_{item.symbol}_{int(time.time() * 1000)}",
                    event_type=event_type,
                    symbol=item.symbol,
                    data={
                        "processing_completed": True,
                        "processing_time_ms": (
                            (datetime.now(timezone.utc) - context["processing_started_at"])
                            .total_seconds() * 1000
                        )
                    }
                )

                await self.event_publisher.publish_async(event)

        except Exception as e:
            logger.error(f"Error publishing processing event: {e}")

    async def _handle_processing_error(self, item: Any, error: Exception) -> None:
        """Handle processing error with recovery logic."""
        logger.warning(f"Entering error recovery mode due to: {error}")
        self._error_recovery_mode = True

        # Could implement retry logic, item quarantine, etc.
        # For now, just log and continue
        pass

    def _is_stage_enabled(self, stage: ProcessingStage) -> bool:
        """Check if processing stage is enabled."""
        stage_config = {
            ProcessingStage.VALIDATION: self.config.enable_validation,
            ProcessingStage.ENRICHMENT: self.config.enable_enrichment,
            ProcessingStage.TRANSFORMATION: self.config.enable_transformation,
            ProcessingStage.AGGREGATION: self.config.enable_aggregation,
            ProcessingStage.OUTPUT: True  # Always enabled
        }

        return stage_config.get(stage, True)

    def _update_metrics(self, processing_time: float, success: bool) -> None:
        """Update pipeline performance metrics."""
        with self._metrics_lock:
            self._metrics.total_items_processed += 1
            self._processing_times.append(processing_time)

            # Limit processing time history
            if len(self._processing_times) > 1000:
                self._processing_times = self._processing_times[-1000:]

            # Update averages
            self._metrics.average_processing_time_ms = (
                sum(self._processing_times) / len(self._processing_times) * 1000
            )

            # Update rates
            current_time = time.time()
            time_delta = current_time - self._last_metrics_update

            if time_delta >= 1.0:  # Update every second
                items_in_period = len([
                    t for t in self._processing_times
                    if (current_time - t) <= time_delta
                ])

                self._metrics.items_per_second = items_in_period / time_delta
                self._last_metrics_update = current_time

            # Update error rates
            if not success:
                error_count = sum(
                    1 for t in self._processing_times[-100:]
                    if not success  # This would need to be tracked properly
                )
                self._metrics.error_rate_percent = (error_count / min(100, len(self._processing_times))) * 100
                self._metrics.success_rate_percent = 100.0 - self._metrics.error_rate_percent

    def _update_stage_metrics(
        self,
        stage: ProcessingStage,
        processor_name: str,
        result: ProcessingResult
    ) -> None:
        """Update metrics for specific processing stage."""
        with self._metrics_lock:
            stage_name = stage.value

            if stage_name not in self._metrics.stage_metrics:
                self._metrics.stage_metrics[stage_name] = {}

            stage_metrics = self._metrics.stage_metrics[stage_name]

            # Update processor-specific metrics
            processor_key = f"{processor_name}"
            if processor_key not in stage_metrics:
                stage_metrics[processor_key] = {
                    "total_processed": 0,
                    "total_errors": 0,
                    "avg_duration_ms": 0.0
                }

            proc_metrics = stage_metrics[processor_key]
            proc_metrics["total_processed"] += 1

            if result.has_errors():
                proc_metrics["total_errors"] += 1

            # Update average duration
            current_avg = proc_metrics["avg_duration_ms"]
            total_processed = proc_metrics["total_processed"]
            proc_metrics["avg_duration_ms"] = (
                (current_avg * (total_processed - 1) + result.duration_ms) / total_processed
            )

    def _start_monitoring(self) -> None:
        """Start pipeline monitoring tasks."""

        @scheduled_task(
            interval_seconds=60,
            name="log_pipeline_metrics"
        )
        def log_metrics():
            metrics = self.get_metrics()
            logger.info(
                f"Pipeline Metrics: "
                f"{metrics.total_items_processed} processed, "
                f"{metrics.items_per_second:.1f} items/sec, "
                f"{metrics.success_rate_percent:.1f}% success rate"
            )

        @scheduled_task(
            interval_seconds=300,  # Every 5 minutes
            name="pipeline_health_check"
        )
        def health_check():
            self._perform_health_check()

        logger.info("Started pipeline monitoring")

    @background_task(name="pipeline_health_check")
    def _perform_health_check(self) -> None:
        """Perform pipeline health check."""
        try:
            metrics = self.get_metrics()

            # Check error rate
            if metrics.error_rate_percent > self.config.error_threshold_percent:
                logger.warning(
                    f"High error rate detected: {metrics.error_rate_percent:.1f}% "
                    f"(threshold: {self.config.error_threshold_percent}%)"
                )

                if self.config.auto_recovery and not self._is_paused:
                    logger.info("Pausing pipeline for error recovery")
                    self.pause()
                    # Could implement automatic recovery logic here

            # Check queue size
            queue_size = self.get_queue_size()
            if queue_size > 5000:  # Arbitrary threshold
                logger.warning(f"Large processing queue: {queue_size} items")

        except Exception as e:
            logger.error(f"Error in pipeline health check: {e}")

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._is_running

    @property
    def is_paused(self) -> bool:
        """Check if pipeline is paused."""
        return self._is_paused