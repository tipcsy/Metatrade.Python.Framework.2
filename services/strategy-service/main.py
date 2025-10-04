from fastapi import FastAPI
from app.api import routes
from app.core.position_manager import PositionManager
from app.core.risk_manager import RiskManager
from app.core.strategy_engine import StrategyEngine
import uvicorn
import logging
import sys
import os

# Service configuration
SERVICE_NAME = "strategy-service"
SERVICE_PORT = 5003

# Setup logging directory
log_dir = "../../logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] [%(levelname)s] [{SERVICE_NAME}] [%(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"{log_dir}/{SERVICE_NAME}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(SERVICE_NAME)

# Create FastAPI app
app = FastAPI(
    title="Strategy Service",
    version="1.0.0",
    description="Strategy execution and position management service for MT5 trading platform"
)

# Include routes
app.include_router(routes.router)


@app.on_event("startup")
async def startup_event():
    """Initialize all service components on startup"""
    logger.info(f"{SERVICE_NAME} starting on port {SERVICE_PORT}")

    try:
        # Initialize Risk Manager
        logger.info("Initializing Risk Manager...")
        risk_manager = RiskManager(
            account_balance=10000.0,  # Starting balance
            max_risk_per_trade=2.0,   # 2% per trade
            max_positions=5,           # Max 5 concurrent positions
            max_daily_loss=5.0,        # Max 5% daily loss
            max_drawdown=20.0          # Max 20% drawdown
        )
        app.state.risk_manager = risk_manager
        logger.info("Risk Manager initialized successfully")

        # Initialize Position Manager
        logger.info("Initializing Position Manager...")
        position_manager = PositionManager()
        app.state.position_manager = position_manager
        logger.info("Position Manager initialized successfully")

        # Initialize Strategy Engine
        logger.info("Initializing Strategy Engine...")
        strategy_engine = StrategyEngine(
            position_manager=position_manager,
            risk_manager=risk_manager
        )
        app.state.strategy_engine = strategy_engine
        logger.info("Strategy Engine initialized successfully")

        logger.info(f"{SERVICE_NAME} started successfully on port {SERVICE_PORT}")
        logger.info("Service is running in MOCK MODE (no MT5 connection required)")

    except Exception as e:
        logger.error(f"Failed to initialize {SERVICE_NAME}: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info(f"{SERVICE_NAME} shutting down")

    try:
        # Close all open positions if any
        if hasattr(app.state, 'position_manager'):
            position_manager = app.state.position_manager
            open_positions = position_manager.get_open_positions()
            if open_positions:
                logger.warning(f"Service shutting down with {len(open_positions)} open positions")
                # In a real scenario, you might want to close them or save state
                # For now, just log the warning

        logger.info(f"{SERVICE_NAME} shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


if __name__ == "__main__":
    logger.info(f"Starting {SERVICE_NAME} on port {SERVICE_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
