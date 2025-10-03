from fastapi import APIRouter
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "backtesting-service",
        "port": 5006
    }

@router.post("/backtest/start")
async def start_backtest():
    """Start a new backtest"""
    logger.info("Backtest start requested")
    return {
        "success": True,
        "data": {
            "backtest_id": 1
        }
    }

@router.get("/backtest/{id}/status")
async def get_backtest_status(id: int):
    """Get backtest status"""
    return {
        "success": True,
        "data": {
            "backtest_id": id,
            "status": "pending"
        }
    }
