from fastapi import FastAPI
from app.api import routes
from app.core.mt5_manager import get_mt5_manager
import uvicorn
import logging
import sys
import os

# Service configuration
SERVICE_NAME = "mt5-service"
SERVICE_PORT = 5002

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
    title="MT5 Service",
    version="1.0.0",
    description="MetaTrader 5 connection and trading service"
)

# Include routes
app.include_router(routes.router)

# MT5 Manager instance
mt5_manager = None


@app.on_event("startup")
async def startup_event():
    global mt5_manager

    logger.info(f"{SERVICE_NAME} starting on port {SERVICE_PORT}")

    # Get MT5 Manager instance
    mt5_manager = get_mt5_manager()

    # Initialize MT5 connection
    # Read credentials from environment if available
    account = os.getenv("MT5_ACCOUNT")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if account and password and server:
        logger.info(f"Initializing MT5 with account: {account}")
        success = mt5_manager.initialize(
            account=int(account),
            password=password,
            server=server
        )
    else:
        logger.info("Initializing MT5 with last logged account")
        success = mt5_manager.initialize()

    if success:
        logger.info("MT5 connection established successfully")

        # Start auto-reconnect monitoring
        mt5_manager.start_auto_reconnect()
        logger.info("MT5 auto-reconnect monitoring started")
    else:
        logger.error("Failed to establish MT5 connection")
        logger.warning("Service will continue running and retry connection...")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"{SERVICE_NAME} shutting down")

    # Stop auto-reconnect
    if mt5_manager:
        mt5_manager.stop_auto_reconnect()
        logger.info("Auto-reconnect monitoring stopped")

        # Shutdown MT5 connection
        mt5_manager.shutdown()
        logger.info("MT5 connection closed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
