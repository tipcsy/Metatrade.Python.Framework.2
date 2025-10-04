from fastapi import FastAPI
from app.api import routes
from app.core.service_manager import ServiceManager
import uvicorn
import logging
import sys
import asyncio
from threading import Thread

# Service configuration
SERVICE_NAME = "backend-api"
SERVICE_PORT = 5000

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] [%(levelname)s] [{SERVICE_NAME}] [%(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"../../logs/{SERVICE_NAME}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(SERVICE_NAME)

# Create FastAPI app
app = FastAPI(
    title="Backend API Service",
    version="1.0.0",
    description="Main orchestrator and frontend API gateway"
)

# Global service manager instance
service_manager: ServiceManager = None
health_monitor_task = None

# Include routes
app.include_router(routes.router)

# Make service_manager accessible to routes
def get_service_manager() -> ServiceManager:
    """Get global service manager instance"""
    return service_manager

# Attach to routes module
routes.get_service_manager = get_service_manager


async def health_monitoring_loop():
    """
    Health monitoring loop that checks all services every 5 seconds
    and auto-restarts failed services if configured
    """
    logger.info("Health monitoring loop started")

    while True:
        try:
            await asyncio.sleep(5)

            for service_info in service_manager.get_all_services_status():
                # Skip offline services that are not auto-start
                if service_info.status.value == "offline" and not service_info.config.auto_start:
                    continue

                # Health check for running services
                if service_info.status.value == "online":
                    is_healthy = service_manager._check_health(service_info)

                    if not is_healthy:
                        logger.warning(f"Service {service_info.name} health check failed")

                        # Auto-restart if configured
                        if service_info.config.auto_restart:
                            logger.info(f"Auto-restarting service {service_info.name}")
                            service_info.restart_count += 1

                            # Max 3 restart attempts
                            if service_info.restart_count < 3:
                                success, message = service_manager.restart_service(service_info.name)
                                if success:
                                    logger.info(f"Service {service_info.name} restarted successfully")
                                else:
                                    logger.error(f"Failed to restart {service_info.name}: {message}")
                            else:
                                logger.error(f"Service {service_info.name} exceeded max restart attempts")
                                service_info.status = service_info.status.__class__.ERROR
                                service_info.error = "Max restart attempts exceeded"
                    else:
                        # Reset restart count on successful health check
                        service_info.restart_count = 0

        except Exception as e:
            logger.error(f"Error in health monitoring loop: {e}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    global service_manager, health_monitor_task

    logger.info(f"{SERVICE_NAME} starting on port {SERVICE_PORT}")

    # Initialize service manager
    service_manager = ServiceManager()
    logger.info("ServiceManager initialized")

    # Load service configurations
    try:
        service_manager.load_service_configs()
        logger.info("Service configurations loaded")
    except Exception as e:
        logger.error(f"Failed to load service configurations: {e}")
        return

    # Auto-start configured services
    service_manager.auto_start_services()

    # Start health monitoring loop
    health_monitor_task = asyncio.create_task(health_monitoring_loop())
    logger.info("Health monitoring loop started")


@app.on_event("shutdown")
async def shutdown_event():
    global health_monitor_task

    logger.info(f"{SERVICE_NAME} shutting down")

    # Cancel health monitoring
    if health_monitor_task:
        health_monitor_task.cancel()
        try:
            await health_monitor_task
        except asyncio.CancelledError:
            pass

    # Shutdown all services
    if service_manager:
        service_manager.shutdown_all_services()
        logger.info("All services shut down")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
