from fastapi import APIRouter
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ai-service",
        "port": 5005
    }

@router.get("/models")
async def get_models():
    """Get all AI models"""
    logger.info("Models list requested")
    return {
        "success": True,
        "data": []
    }

@router.post("/models/{id}/predict")
async def predict(id: int):
    """Make prediction with model"""
    logger.info(f"Prediction requested for model {id}")
    return {
        "success": True,
        "data": {
            "prediction": None
        }
    }
