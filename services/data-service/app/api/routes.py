from fastapi import APIRouter
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data-service",
        "port": 5001
    }

@router.post("/gap-fill")
async def start_gap_fill():
    """Start gap fill process"""
    logger.info("Gap fill requested")
    return {
        "success": True,
        "message": "Gap fill started"
    }

@router.get("/statistics")
async def get_statistics():
    """Get data collection statistics"""
    return {
        "success": True,
        "data": {
            "tick_count": 0,
            "ohlc_count": 0
        }
    }
