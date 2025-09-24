"""
Market data processing pipeline for the MetaTrader Python Framework.

This module provides comprehensive data processing pipelines with validation,
enrichment, transformation, and real-time processing capabilities.
"""

from .processor import DataProcessor, ProcessingStage, ProcessingResult
from .pipeline import MarketDataPipeline, PipelineConfig
from .market_data_processor import (
    MarketDataPipeline as MarketDataProcessorPipeline,
    DataProcessor as BaseDataProcessor,
    TickDataValidator,
    OHLCDataValidator,
    DataEnricher,
    ValidationResult,
    ProcessingMetrics,
    get_market_data_pipeline,
)
from .validators import (
    DataValidator, ValidationRule, ValidationResult,
    SymbolValidationRule, PriceValidationRule, TimestampValidationRule,
    VolumeValidationRule, OHLCConsistencyRule
)
from .enrichers import (
    DataEnricher, EnrichmentRule,
    SpreadEnrichmentRule, VolumeEnrichmentRule, OHLCEnrichmentRule,
    TechnicalIndicatorEnrichmentRule, MarketConditionEnrichmentRule,
    TimingEnrichmentRule
)
from .transformers import (
    DataTransformer, TransformationRule,
    NormalizationRule, AggregationRule, FilteringRule,
    EnrichmentIntegrationRule, TypeConversionRule
)

__all__ = [
    # Core processing
    "DataProcessor",
    "ProcessingStage",
    "ProcessingResult",

    # Pipeline
    "MarketDataPipeline",
    "PipelineConfig",

    # Phase 3 Market Data Processor
    "MarketDataProcessorPipeline",
    "BaseDataProcessor",
    "TickDataValidator",
    "OHLCDataValidator",
    "ProcessingMetrics",
    "get_market_data_pipeline",

    # Validation
    "DataValidator",
    "ValidationRule",
    "ValidationResult",
    "SymbolValidationRule",
    "PriceValidationRule",
    "TimestampValidationRule",
    "VolumeValidationRule",
    "OHLCConsistencyRule",

    # Enrichment
    "DataEnricher",
    "EnrichmentRule",
    "SpreadEnrichmentRule",
    "VolumeEnrichmentRule",
    "OHLCEnrichmentRule",
    "TechnicalIndicatorEnrichmentRule",
    "MarketConditionEnrichmentRule",
    "TimingEnrichmentRule",

    # Transformation
    "DataTransformer",
    "TransformationRule",
    "NormalizationRule",
    "AggregationRule",
    "FilteringRule",
    "EnrichmentIntegrationRule",
    "TypeConversionRule",
]