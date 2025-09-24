"""
High-Performance Tick Data Processing Pipeline.

This module provides advanced tick data processing with real-time filtering,
validation, transformation, and aggregation capabilities optimized for
high-frequency trading scenarios.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue, Empty

import numpy as np
import pandas as pd

from src.core.config.settings import PerformanceSettings, MarketDataSettings
from src.core.exceptions import TickProcessingError, ValidationError
from src.core.logging import get_logger
from src.core.data.models import TickData, ProcessedTick, TickFilter, TickMetrics

logger = get_logger(__name__)


class ProcessingStage(Enum):
    """Tick processing pipeline stages."""
    RAW = "raw"
    VALIDATED = "validated"
    FILTERED = "filtered"
    TRANSFORMED = "transformed"
    AGGREGATED = "aggregated"


class ProcessorStatus(Enum):
    """Tick processor status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ProcessingConfig:
    """Tick processing configuration."""
    enable_validation: bool = True
    enable_filtering: bool = True
    enable_transformation: bool = True
    enable_aggregation: bool = True
    batch_size: int = 1000
    buffer_size: int = 10000
    worker_threads: int = 4
    use_multiprocessing: bool = False
    process_count: int = 2
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    filter_rules: List[TickFilter] = field(default_factory=list)


@dataclass
class ProcessingMetrics:
    """Processing pipeline metrics."""
    ticks_received: int = 0
    ticks_processed: int = 0
    ticks_filtered_out: int = 0
    ticks_invalid: int = 0
    processing_time_total: float = 0.0
    processing_time_avg: float = 0.0
    throughput_per_second: float = 0.0
    queue_depth: int = 0
    error_count: int = 0
    last_update: float = field(default_factory=time.time)


class TickValidator:
    """High-performance tick data validator."""

    def __init__(self, validation_rules: Dict[str, Any]) -> None:
        """Initialize validator with rules.

        Args:
            validation_rules: Dictionary of validation rules
        """
        self.rules = validation_rules
        self.logger = get_logger(f"{__name__}.TickValidator")

    def validate_tick(self, tick: TickData) -> Tuple[bool, Optional[str]]:
        """Validate a single tick.

        Args:
            tick: Tick data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic data integrity checks
            if not tick.symbol or len(tick.symbol) == 0:
                return False, "Missing or empty symbol"

            if tick.time <= 0:
                return False, "Invalid timestamp"

            if tick.bid <= 0 and tick.ask <= 0 and tick.last <= 0:
                return False, "No valid price data"

            # Price validation
            if tick.bid > 0 and tick.ask > 0:
                if tick.ask < tick.bid:
                    return False, "Ask price lower than bid price"

                spread = tick.ask - tick.bid
                max_spread = self.rules.get("max_spread", 0.1)
                if spread > max_spread:
                    return False, f"Spread too large: {spread} > {max_spread}"

            # Volume validation
            min_volume = self.rules.get("min_volume", 0)
            if tick.volume < min_volume:
                return False, f"Volume too low: {tick.volume} < {min_volume}"

            max_volume = self.rules.get("max_volume", float('inf'))
            if tick.volume > max_volume:
                return False, f"Volume too high: {tick.volume} > {max_volume}"

            # Time validation
            current_time = time.time()
            max_age = self.rules.get("max_tick_age", 300)  # 5 minutes
            if current_time - tick.time > max_age:
                return False, f"Tick too old: age {current_time - tick.time} seconds"

            future_tolerance = self.rules.get("future_tolerance", 60)  # 1 minute
            if tick.time > current_time + future_tolerance:
                return False, f"Tick from future: {tick.time - current_time} seconds ahead"

            return True, None

        except Exception as e:
            self.logger.error(
                "Error validating tick",
                extra={"tick": tick, "error": str(e)},
                exc_info=True
            )
            return False, f"Validation error: {e}"

    def validate_batch(self, ticks: List[TickData]) -> Tuple[List[TickData], List[Tuple[TickData, str]]]:
        """Validate a batch of ticks.

        Args:
            ticks: List of ticks to validate

        Returns:
            Tuple of (valid_ticks, invalid_ticks_with_reasons)
        """
        valid_ticks = []
        invalid_ticks = []

        for tick in ticks:
            is_valid, error_msg = self.validate_tick(tick)
            if is_valid:
                valid_ticks.append(tick)
            else:
                invalid_ticks.append((tick, error_msg))

        return valid_ticks, invalid_ticks


class TickFilter:
    """High-performance tick data filter."""

    def __init__(self, filter_rules: List[TickFilter]) -> None:
        """Initialize filter with rules.

        Args:
            filter_rules: List of filter rules
        """
        self.rules = filter_rules
        self.logger = get_logger(f"{__name__}.TickFilter")

    def filter_tick(self, tick: TickData) -> bool:
        """Check if tick should be filtered out.

        Args:
            tick: Tick data to check

        Returns:
            True if tick should be kept, False if filtered out
        """
        try:
            for rule in self.rules:
                if not rule.should_keep(tick):
                    return False
            return True

        except Exception as e:
            self.logger.error(
                "Error filtering tick",
                extra={"tick": tick, "error": str(e)},
                exc_info=True
            )
            return False

    def filter_batch(self, ticks: List[TickData]) -> List[TickData]:
        """Filter a batch of ticks.

        Args:
            ticks: List of ticks to filter

        Returns:
            List of ticks that passed filters
        """
        return [tick for tick in ticks if self.filter_tick(tick)]


class TickTransformer:
    """High-performance tick data transformer."""

    def __init__(self) -> None:
        """Initialize transformer."""
        self.logger = get_logger(f"{__name__}.TickTransformer")

    def transform_tick(self, tick: TickData) -> ProcessedTick:
        """Transform a single tick.

        Args:
            tick: Raw tick data

        Returns:
            Processed tick data
        """
        try:
            # Calculate derived values
            mid_price = None
            if tick.bid > 0 and tick.ask > 0:
                mid_price = (tick.bid + tick.ask) / 2

            spread = None
            if tick.bid > 0 and tick.ask > 0:
                spread = tick.ask - tick.bid
                spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0
            else:
                spread_pct = 0

            # Create processed tick
            processed = ProcessedTick(
                symbol=tick.symbol,
                time=tick.time,
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                volume=tick.volume,
                mid_price=mid_price,
                spread=spread,
                spread_pct=spread_pct,
                flags=tick.flags,
                processing_time=time.time()
            )

            return processed

        except Exception as e:
            self.logger.error(
                "Error transforming tick",
                extra={"tick": tick, "error": str(e)},
                exc_info=True
            )
            raise TickProcessingError(f"Transform failed: {e}") from e

    def transform_batch(self, ticks: List[TickData]) -> List[ProcessedTick]:
        """Transform a batch of ticks.

        Args:
            ticks: List of raw ticks

        Returns:
            List of processed ticks
        """
        processed_ticks = []

        for tick in ticks:
            try:
                processed = self.transform_tick(tick)
                processed_ticks.append(processed)
            except Exception as e:
                self.logger.warning(
                    "Failed to transform tick, skipping",
                    extra={"tick": tick, "error": str(e)}
                )

        return processed_ticks


class TickProcessor:
    """
    High-performance tick data processing pipeline.

    Features:
    - Multi-threaded processing
    - Real-time validation and filtering
    - Configurable transformation stages
    - Performance monitoring
    - Error handling and recovery
    """

    def __init__(
        self,
        config: ProcessingConfig,
        performance_settings: PerformanceSettings
    ) -> None:
        """Initialize tick processor.

        Args:
            config: Processing configuration
            performance_settings: Performance settings
        """
        self.config = config
        self.performance_settings = performance_settings

        # Processing components
        self.validator = TickValidator(config.validation_rules) if config.enable_validation else None
        self.filter = TickFilter(config.filter_rules) if config.enable_filtering else None
        self.transformer = TickTransformer() if config.enable_transformation else None

        # Processing queues
        self._input_queue: Queue = Queue(maxsize=config.buffer_size)
        self._output_queue: Queue = Queue(maxsize=config.buffer_size)
        self._error_queue: Queue = Queue(maxsize=1000)

        # Threading
        self._worker_threads: List[threading.Thread] = []
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None

        # State management
        self._status = ProcessorStatus.STOPPED
        self._stop_event = threading.Event()
        self._metrics = ProcessingMetrics()
        self._metrics_lock = threading.RLock()

        # Callbacks
        self._output_callbacks: List[Callable[[List[ProcessedTick]], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []

        # Monitoring
        self._last_metrics_update = time.time()
        self._metrics_update_interval = 1.0

        logger.info(
            "Tick processor initialized",
            extra={
                "validation_enabled": config.enable_validation,
                "filtering_enabled": config.enable_filtering,
                "transformation_enabled": config.enable_transformation,
                "batch_size": config.batch_size,
                "buffer_size": config.buffer_size,
                "worker_threads": config.worker_threads,
                "use_multiprocessing": config.use_multiprocessing,
            }
        )

    async def start(self) -> None:
        """Start the tick processing pipeline."""
        if self._status == ProcessorStatus.RUNNING:
            logger.warning("Tick processor already running")
            return

        try:
            self._status = ProcessorStatus.STARTING
            logger.info("Starting tick processor")

            # Reset stop event
            self._stop_event.clear()

            # Initialize thread/process pools
            if self.config.use_multiprocessing:
                self._process_pool = ProcessPoolExecutor(
                    max_workers=self.config.process_count,
                    mp_context=mp.get_context('spawn')
                )
            else:
                self._thread_pool = ThreadPoolExecutor(
                    max_workers=self.config.worker_threads,
                    thread_name_prefix="TickProcessor"
                )

            # Start worker threads
            for i in range(self.config.worker_threads):
                worker_thread = threading.Thread(
                    target=self._worker_loop,
                    name=f"TickProcessor-Worker-{i}",
                    daemon=True
                )
                worker_thread.start()
                self._worker_threads.append(worker_thread)

            # Start metrics thread
            metrics_thread = threading.Thread(
                target=self._metrics_loop,
                name="TickProcessor-Metrics",
                daemon=True
            )
            metrics_thread.start()
            self._worker_threads.append(metrics_thread)

            self._status = ProcessorStatus.RUNNING
            logger.info("Tick processor started successfully")

        except Exception as e:
            self._status = ProcessorStatus.ERROR
            logger.error(
                "Failed to start tick processor",
                extra={"error": str(e)},
                exc_info=True
            )
            await self._cleanup_resources()
            raise TickProcessingError(f"Failed to start processor: {e}") from e

    async def stop(self) -> None:
        """Stop the tick processing pipeline."""
        if self._status == ProcessorStatus.STOPPED:
            return

        logger.info("Stopping tick processor")

        try:
            # Signal stop
            self._stop_event.set()

            # Wait for worker threads
            for thread in self._worker_threads:
                if thread.is_alive():
                    thread.join(timeout=5.0)

            # Clean up resources
            await self._cleanup_resources()

            self._status = ProcessorStatus.STOPPED
            logger.info("Tick processor stopped")

        except Exception as e:
            logger.error(
                "Error stopping tick processor",
                extra={"error": str(e)},
                exc_info=True
            )

    def process_tick(self, tick: TickData) -> bool:
        """Process a single tick.

        Args:
            tick: Tick data to process

        Returns:
            True if tick was queued for processing
        """
        try:
            if self._status != ProcessorStatus.RUNNING:
                return False

            self._input_queue.put_nowait(tick)
            return True

        except Exception as e:
            logger.error(
                "Failed to queue tick for processing",
                extra={"tick": tick, "error": str(e)},
                exc_info=True
            )
            return False

    def process_batch(self, ticks: List[TickData]) -> int:
        """Process a batch of ticks.

        Args:
            ticks: List of ticks to process

        Returns:
            Number of ticks successfully queued
        """
        queued_count = 0

        for tick in ticks:
            if self.process_tick(tick):
                queued_count += 1

        return queued_count

    def add_output_callback(self, callback: Callable[[List[ProcessedTick]], None]) -> None:
        """Add callback for processed ticks.

        Args:
            callback: Function to call with processed ticks
        """
        self._output_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add callback for processing errors.

        Args:
            callback: Function to call with errors
        """
        self._error_callbacks.append(callback)

    def get_metrics(self) -> ProcessingMetrics:
        """Get processing metrics.

        Returns:
            Current processing metrics
        """
        with self._metrics_lock:
            return ProcessingMetrics(**self._metrics.__dict__)

    def get_status(self) -> Dict[str, Any]:
        """Get processor status.

        Returns:
            Dictionary containing processor status
        """
        with self._metrics_lock:
            return {
                "status": self._status.value,
                "input_queue_size": self._input_queue.qsize(),
                "output_queue_size": self._output_queue.qsize(),
                "error_queue_size": self._error_queue.qsize(),
                "worker_threads": len(self._worker_threads),
                "metrics": self._metrics.__dict__,
            }

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        logger.debug("Worker thread started")

        batch = []
        last_batch_time = time.time()

        try:
            while not self._stop_event.is_set():
                try:
                    # Get tick from input queue
                    try:
                        tick = self._input_queue.get(timeout=0.1)
                        batch.append(tick)
                    except Empty:
                        pass

                    current_time = time.time()

                    # Process batch when full or timeout reached
                    should_process_batch = (
                        len(batch) >= self.config.batch_size or
                        (batch and current_time - last_batch_time >= 0.1)
                    )

                    if should_process_batch and batch:
                        self._process_batch(batch)
                        batch.clear()
                        last_batch_time = current_time

                except Exception as e:
                    logger.error(
                        "Error in worker loop",
                        extra={"error": str(e)},
                        exc_info=True
                    )
                    self._handle_error(e)

        except Exception as e:
            logger.error(
                "Fatal error in worker thread",
                extra={"error": str(e)},
                exc_info=True
            )

        logger.debug("Worker thread ended")

    def _process_batch(self, batch: List[TickData]) -> None:
        """Process a batch of ticks."""
        start_time = time.time()
        processed_count = 0

        try:
            current_batch = batch[:]

            # Validation stage
            if self.validator:
                valid_ticks, invalid_ticks = self.validator.validate_batch(current_batch)
                current_batch = valid_ticks

                # Log invalid ticks
                for invalid_tick, reason in invalid_ticks:
                    logger.debug(
                        "Invalid tick filtered out",
                        extra={"tick": invalid_tick, "reason": reason}
                    )

            # Filtering stage
            if self.filter:
                current_batch = self.filter.filter_batch(current_batch)

            # Transformation stage
            if self.transformer:
                processed_ticks = self.transformer.transform_batch(current_batch)
            else:
                # Convert to ProcessedTick without transformation
                processed_ticks = [
                    ProcessedTick(
                        symbol=tick.symbol,
                        time=tick.time,
                        bid=tick.bid,
                        ask=tick.ask,
                        last=tick.last,
                        volume=tick.volume,
                        processing_time=time.time()
                    )
                    for tick in current_batch
                ]

            processed_count = len(processed_ticks)

            # Send to output callbacks
            if processed_ticks:
                for callback in self._output_callbacks:
                    try:
                        callback(processed_ticks)
                    except Exception as e:
                        logger.error(
                            "Error in output callback",
                            extra={"error": str(e)},
                            exc_info=True
                        )

            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(len(batch), processed_count, processing_time)

        except Exception as e:
            logger.error(
                "Error processing batch",
                extra={
                    "batch_size": len(batch),
                    "error": str(e),
                },
                exc_info=True
            )
            self._handle_error(e)

    def _update_metrics(self, received: int, processed: int, processing_time: float) -> None:
        """Update processing metrics."""
        with self._metrics_lock:
            self._metrics.ticks_received += received
            self._metrics.ticks_processed += processed
            self._metrics.ticks_filtered_out += (received - processed)
            self._metrics.processing_time_total += processing_time

            if self._metrics.ticks_processed > 0:
                self._metrics.processing_time_avg = (
                    self._metrics.processing_time_total / self._metrics.ticks_processed
                )

            self._metrics.queue_depth = self._input_queue.qsize()
            self._metrics.last_update = time.time()

    def _metrics_loop(self) -> None:
        """Metrics calculation loop."""
        last_processed = 0
        last_time = time.time()

        while not self._stop_event.is_set():
            try:
                time.sleep(self._metrics_update_interval)

                current_time = time.time()
                with self._metrics_lock:
                    current_processed = self._metrics.ticks_processed
                    time_diff = current_time - last_time

                    if time_diff > 0:
                        self._metrics.throughput_per_second = (
                            (current_processed - last_processed) / time_diff
                        )

                    last_processed = current_processed
                    last_time = current_time

            except Exception as e:
                logger.error(
                    "Error in metrics loop",
                    extra={"error": str(e)},
                    exc_info=True
                )

    def _handle_error(self, error: Exception) -> None:
        """Handle processing error."""
        try:
            with self._metrics_lock:
                self._metrics.error_count += 1

            # Add to error queue
            try:
                self._error_queue.put_nowait(error)
            except Exception:
                pass  # Error queue full

            # Call error callbacks
            for callback in self._error_callbacks:
                try:
                    callback(error)
                except Exception as e:
                    logger.error(
                        "Error in error callback",
                        extra={"original_error": str(error), "callback_error": str(e)},
                        exc_info=True
                    )

        except Exception as e:
            logger.error(
                "Error handling error",
                extra={"original_error": str(error), "handler_error": str(e)},
                exc_info=True
            )

    async def _cleanup_resources(self) -> None:
        """Clean up processing resources."""
        try:
            # Shutdown thread pool
            if self._thread_pool:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None

            # Shutdown process pool
            if self._process_pool:
                self._process_pool.shutdown(wait=True)
                self._process_pool = None

            # Clear queues
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except Empty:
                    break

            while not self._output_queue.empty():
                try:
                    self._output_queue.get_nowait()
                except Empty:
                    break

            self._worker_threads.clear()

        except Exception as e:
            logger.error(
                "Error cleaning up resources",
                extra={"error": str(e)},
                exc_info=True
            )

    async def __aenter__(self) -> TickProcessor:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()