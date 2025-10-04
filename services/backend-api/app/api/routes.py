from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)

# This will be set by main.py
get_service_manager = None


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
    """Get system status and all services"""
    try:
        manager = get_service_manager()
        if not manager:
            raise HTTPException(status_code=500, detail="ServiceManager not initialized")

        services = []
        for service_info in manager.get_all_services_status():
            services.append({
                "name": service_info.name,
                "status": service_info.status.value,
                "port": service_info.port,
                "pid": service_info.pid,
                "restart_count": service_info.restart_count,
                "error": service_info.error,
                "last_check": service_info.last_check.isoformat() if service_info.last_check else None
            })

        return {
            "success": True,
            "data": {
                "system": "online",
                "services": services
            }
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/services/{service_name}")
async def get_service_status(service_name: str):
    """Get status of a specific service"""
    try:
        manager = get_service_manager()
        if not manager:
            raise HTTPException(status_code=500, detail="ServiceManager not initialized")

        service_info = manager.get_service_status(service_name)
        if not service_info:
            raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")

        return {
            "success": True,
            "data": {
                "name": service_info.name,
                "status": service_info.status.value,
                "port": service_info.port,
                "pid": service_info.pid,
                "restart_count": service_info.restart_count,
                "error": service_info.error,
                "last_check": service_info.last_check.isoformat() if service_info.last_check else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting service status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/services/{service_name}/start")
async def start_service(service_name: str):
    """Start a specific service"""
    try:
        manager = get_service_manager()
        if not manager:
            raise HTTPException(status_code=500, detail="ServiceManager not initialized")

        success, message = manager.start_service(service_name)

        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            return {
                "success": False,
                "message": message
            }
    except Exception as e:
        logger.error(f"Error starting service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/services/{service_name}/stop")
async def stop_service(service_name: str):
    """Stop a specific service"""
    try:
        manager = get_service_manager()
        if not manager:
            raise HTTPException(status_code=500, detail="ServiceManager not initialized")

        success, message = manager.stop_service(service_name)

        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            return {
                "success": False,
                "message": message
            }
    except Exception as e:
        logger.error(f"Error stopping service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/services/{service_name}/restart")
async def restart_service(service_name: str):
    """Restart a specific service"""
    try:
        manager = get_service_manager()
        if not manager:
            raise HTTPException(status_code=500, detail="ServiceManager not initialized")

        success, message = manager.restart_service(service_name)

        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            return {
                "success": False,
                "message": message
            }
    except Exception as e:
        logger.error(f"Error restarting service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
