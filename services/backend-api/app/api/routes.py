from fastapi import APIRouter
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "backend-api",
        "port": 5000
    }

@router.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "success": True,
        "data": {
            "system": "online",
            "services": []
        }
    }
