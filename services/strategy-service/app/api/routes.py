from fastapi import APIRouter
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "strategy-service",
        "port": 5004
    }

@router.get("/strategies")
async def get_strategies():
    """Get all strategies"""
    logger.info("Strategies list requested")
    return {
        "success": True,
        "data": []
    }

@router.post("/strategies/{id}/start")
async def start_strategy(id: int):
    """Start a strategy"""
    logger.info(f"Starting strategy {id}")
    return {
        "success": True,
        "message": f"Strategy {id} started"
    }
