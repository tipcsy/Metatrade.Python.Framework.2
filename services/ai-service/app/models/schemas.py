"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# Training request models
class TrainingParameters(BaseModel):
    """Model training hyperparameters"""
    sequence_length: int = Field(default=60, ge=10, le=200, description="Sequence length for LSTM")
    lstm_units: int = Field(default=50, ge=10, le=200, description="Number of LSTM units per layer")
    num_lstm_layers: int = Field(default=3, ge=1, le=5, description="Number of LSTM layers")
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.5, description="Dropout rate")
    epochs: int = Field(default=50, ge=10, le=200, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=8, le=128, description="Batch size")
    train_ratio: float = Field(default=0.8, ge=0.5, le=0.95, description="Train/validation split ratio")
    add_indicators: bool = Field(default=True, description="Add technical indicators as features")


class TrainModelRequest(BaseModel):
    """Request to train a new model"""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(..., description="Timeframe (e.g., M15)")
    start_date: str = Field(..., description="Training data start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Training data end date (YYYY-MM-DD)")
    parameters: Optional[TrainingParameters] = Field(default_factory=TrainingParameters)


class TrainModelResponse(BaseModel):
    """Response from training request"""
    success: bool
    job_id: Optional[str] = None
    status: Optional[str] = None
    estimated_duration: Optional[int] = None
    error: Optional[str] = None


# Prediction request models
class PredictRequest(BaseModel):
    """Request for price prediction"""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    steps_ahead: int = Field(default=1, ge=1, le=10, description="Number of steps to predict")
    use_confidence: bool = Field(default=True, description="Calculate confidence score")


class PredictionResult(BaseModel):
    """Prediction result"""
    price: float = Field(..., description="Predicted price")
    current_price: float = Field(..., description="Current price")
    price_change: float = Field(..., description="Price change")
    price_change_pct: float = Field(..., description="Price change percentage")
    direction: str = Field(..., description="Direction (UP/DOWN)")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    steps_ahead: int = Field(..., description="Steps ahead predicted")
    timestamp: str = Field(..., description="Prediction timestamp")


class PredictResponse(BaseModel):
    """Response from prediction request"""
    success: bool
    model_id: Optional[str] = None
    prediction: Optional[PredictionResult] = None
    input_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Model management models
class ModelInfo(BaseModel):
    """Model information"""
    model_id: str
    symbol: str
    timeframe: str
    model_type: str
    status: str
    training_date: str
    accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_mae: Optional[float] = None


class ModelListResponse(BaseModel):
    """Response for model list"""
    success: bool
    models: List[ModelInfo]
    total: int


class ModelDetailResponse(BaseModel):
    """Detailed model information"""
    success: bool
    model: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DeleteModelResponse(BaseModel):
    """Response from model deletion"""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None


class ActivateModelResponse(BaseModel):
    """Response from model activation"""
    success: bool
    model_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


# Training job status models
class TrainingJobStatus(BaseModel):
    """Training job status information"""
    job_id: str
    status: str
    progress: int = 0
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TrainingJobStatusResponse(BaseModel):
    """Response for training job status"""
    success: bool
    job: Optional[TrainingJobStatus] = None
    error: Optional[str] = None


# Health check model
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    port: int
    loaded_models: int = 0
    gpu_available: bool = False
    tensorflow_version: Optional[str] = None
