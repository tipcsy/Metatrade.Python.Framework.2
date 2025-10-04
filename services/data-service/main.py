from fastapi import FastAPI
from app.api import routes
from app.core.database_manager import DatabaseManager
import uvicorn
import logging
import sys

# Service configuration
SERVICE_NAME = "data-service"
SERVICE_PORT = 5001

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
    title="Data Service",
    version="1.0.0",
    description="Data collection, storage, and gap fill service"
)

# Global database manager instance
db_manager: DatabaseManager = None

# Include routes
app.include_router(routes.router)

# Make db_manager accessible to routes
def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    return db_manager

# Attach to routes module
routes.get_db_manager = get_db_manager


@app.on_event("startup")
async def startup_event():
    global db_manager

    logger.info(f"{SERVICE_NAME} starting on port {SERVICE_PORT}")

    # Initialize database manager
    db_manager = DatabaseManager()
    logger.info("DatabaseManager initialized")
    logger.info(f"Database base directory: {db_manager.base_dir}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"{SERVICE_NAME} shutting down")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
