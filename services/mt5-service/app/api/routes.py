from fastapi import APIRouter
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mt5-service",
        "port": 5002,
        "mt5_connected": False
    }

@router.post("/connect")
async def connect_mt5():
    """Connect to MT5 terminal"""
    logger.info("MT5 connection requested")
    return {
        "success": True,
        "message": "MT5 connection initiated"
    }

@router.get("/account")
async def get_account_info():
    """Get MT5 account information"""
    return {
        "success": True,
        "data": {
            "balance": 0.0,
            "equity": 0.0
        }
    }
