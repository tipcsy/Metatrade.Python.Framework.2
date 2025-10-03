from fastapi import APIRouter
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "pattern-service",
        "port": 5003
    }

@router.get("/patterns")
async def get_patterns():
    """Get all available patterns"""
    logger.info("Patterns list requested")
    return {
        "success": True,
        "data": []
    }

@router.get("/indicators/{symbol}/{timeframe}")
async def get_indicators(symbol: str, timeframe: str):
    """Get technical indicators for symbol/timeframe"""
    return {
        "success": True,
        "data": {
            "symbol": symbol,
            "timeframe": timeframe,
            "indicators": {}
        }
    }
