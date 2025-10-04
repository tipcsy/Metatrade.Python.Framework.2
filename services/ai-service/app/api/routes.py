"""
API Routes for AI Service
Defines all REST endpoints for model training, prediction, and management
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from ..models.schemas import (
    TrainModelRequest, TrainModelResponse,
    PredictRequest, PredictResponse,
    ModelListResponse, ModelDetailResponse,
    DeleteModelResponse, ActivateModelResponse,
    TrainingJobStatusResponse, HealthResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Global AI service instance (will be initialized in main.py)
ai_service = None


def set_ai_service(service):
    """Set the global AI service instance"""
    global ai_service
    ai_service = service


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")

    if ai_service is None:
        return HealthResponse(
            status="unhealthy",
            service="ai-service",
            port=5005,
            loaded_models=0,
            gpu_available=False
        )

    stats = ai_service.get_service_stats()

    return HealthResponse(
        status="healthy",
        service="ai-service",
        port=5005,
        loaded_models=stats.get('loaded_models', 0),
        gpu_available=stats.get('gpu_available', False),
        tensorflow_version=stats.get('tensorflow_version')
    )


# Model Training Endpoints
@router.post("/models/train", response_model=TrainModelResponse)
async def train_model(request: TrainModelRequest):
    """
    Train a new LSTM model

    This endpoint queues a training job and returns immediately.
    Use the job_id to check training status.
    """
    logger.info(f"Training request: {request.symbol} {request.timeframe}")

    if ai_service is None:
        raise HTTPException(status_code=500, detail="AI Service not initialized")

    try:
        config = request.dict()
        result = await ai_service.train_model(config)

        if result['success']:
            return TrainModelResponse(
                success=True,
                job_id=result['job_id'],
                status=result['status'],
                estimated_duration=result.get('estimated_duration')
            )
        else:
            return TrainModelResponse(
                success=False,
                error=result.get('error', 'Unknown error')
            )

    except Exception as e:
        logger.error(f"Error in train_model endpoint: {e}")
        return TrainModelResponse(
            success=False,
            error=str(e)
        )


@router.get("/models/train/{job_id}/status", response_model=TrainingJobStatusResponse)
async def get_training_status(job_id: str):
    """Get training job status"""
    logger.info(f"Training status requested: {job_id}")

    if ai_service is None:
        raise HTTPException(status_code=500, detail="AI Service not initialized")

    try:
        result = ai_service.get_training_status(job_id)

        if result['success']:
            return TrainingJobStatusResponse(
                success=True,
                job=result['job']
            )
        else:
            return TrainingJobStatusResponse(
                success=False,
                error=result.get('error', 'Unknown error')
            )

    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return TrainingJobStatusResponse(
            success=False,
            error=str(e)
        )


# Prediction Endpoints
@router.post("/models/{model_id}/predict", response_model=PredictResponse)
async def predict(model_id: str, request: PredictRequest):
    """
    Make price prediction with a trained model

    Args:
        model_id: Model identifier
        request: Prediction request parameters
    """
    logger.info(f"Prediction request: model={model_id}, symbol={request.symbol}")

    if ai_service is None:
        raise HTTPException(status_code=500, detail="AI Service not initialized")

    try:
        result = await ai_service.predict(
            model_id=model_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            steps_ahead=request.steps_ahead,
            use_confidence=request.use_confidence
        )

        if result['success']:
            return PredictResponse(
                success=True,
                model_id=result['model_id'],
                prediction=result['prediction'],
                input_data=result.get('input_data')
            )
        else:
            return PredictResponse(
                success=False,
                error=result.get('error', 'Unknown error')
            )

    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return PredictResponse(
            success=False,
            error=str(e)
        )


# Model Management Endpoints
@router.get("/models", response_model=ModelListResponse)
async def list_models(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """
    List all available models

    Optional filters:
    - symbol: Filter by trading symbol
    - timeframe: Filter by timeframe
    - status: Filter by status (active/inactive)
    """
    logger.info(f"List models request: symbol={symbol}, timeframe={timeframe}, status={status}")

    if ai_service is None:
        raise HTTPException(status_code=500, detail="AI Service not initialized")

    try:
        result = ai_service.list_models(
            symbol=symbol,
            timeframe=timeframe,
            status=status
        )

        # Convert to response format
        models = []
        for model in result['models']:
            models.append({
                'model_id': model['model_id'],
                'symbol': model['symbol'],
                'timeframe': model['timeframe'],
                'model_type': model['model_type'],
                'status': model['status'],
                'training_date': model['training_date'],
                'accuracy': model.get('metrics', {}).get('val_mae'),
                'val_loss': model.get('metrics', {}).get('val_loss'),
                'val_mae': model.get('metrics', {}).get('val_mae')
            })

        return ModelListResponse(
            success=True,
            models=models,
            total=result['total']
        )

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}", response_model=ModelDetailResponse)
async def get_model(model_id: str):
    """Get detailed information about a specific model"""
    logger.info(f"Get model request: {model_id}")

    if ai_service is None:
        raise HTTPException(status_code=500, detail="AI Service not initialized")

    try:
        result = ai_service.get_model(model_id)

        if result['success']:
            return ModelDetailResponse(
                success=True,
                model=result['model']
            )
        else:
            return ModelDetailResponse(
                success=False,
                error=result.get('error', 'Model not found')
            )

    except Exception as e:
        logger.error(f"Error getting model: {e}")
        return ModelDetailResponse(
            success=False,
            error=str(e)
        )


@router.delete("/models/{model_id}", response_model=DeleteModelResponse)
async def delete_model(model_id: str):
    """Delete a model"""
    logger.info(f"Delete model request: {model_id}")

    if ai_service is None:
        raise HTTPException(status_code=500, detail="AI Service not initialized")

    try:
        result = ai_service.delete_model(model_id)

        if result['success']:
            return DeleteModelResponse(
                success=True,
                message=result['message']
            )
        else:
            return DeleteModelResponse(
                success=False,
                error=result.get('error', 'Failed to delete model')
            )

    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return DeleteModelResponse(
            success=False,
            error=str(e)
        )


@router.post("/models/{model_id}/activate", response_model=ActivateModelResponse)
async def activate_model(model_id: str):
    """
    Activate a model for production use

    This will deactivate other models for the same symbol/timeframe
    """
    logger.info(f"Activate model request: {model_id}")

    if ai_service is None:
        raise HTTPException(status_code=500, detail="AI Service not initialized")

    try:
        result = ai_service.activate_model(model_id)

        if result['success']:
            return ActivateModelResponse(
                success=True,
                model_id=result['model_id'],
                message=result['message']
            )
        else:
            return ActivateModelResponse(
                success=False,
                error=result.get('error', 'Failed to activate model')
            )

    except Exception as e:
        logger.error(f"Error activating model: {e}")
        return ActivateModelResponse(
            success=False,
            error=str(e)
        )
