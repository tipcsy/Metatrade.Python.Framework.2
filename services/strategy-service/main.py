from fastapi import FastAPI
from app.api import routes
import uvicorn
import logging
import sys

# Service configuration
SERVICE_NAME = "strategy-service"
SERVICE_PORT = 5004

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
    title="Strategy Service",
    version="1.0.0",
    description="Strategy execution and position management service"
)

# Include routes
app.include_router(routes.router)

@app.on_event("startup")
async def startup_event():
    logger.info(f"{SERVICE_NAME} starting on port {SERVICE_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"{SERVICE_NAME} shutting down")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
