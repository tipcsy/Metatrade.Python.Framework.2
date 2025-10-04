from fastapi import FastAPI
from app.api import routes
from app.core.service import PatternAnalyzer
import uvicorn
import logging
import sys
import os

# Service configuration
SERVICE_NAME = "pattern-service"
SERVICE_PORT = 5004
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://localhost:5001")

# Setup logging
os.makedirs("../../logs", exist_ok=True)
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
    title="Pattern & Indicator Service",
    version="1.0.0",
    description="Technical indicators and pattern recognition service for MT5 trading platform"
)

# Global pattern analyzer instance
pattern_analyzer = None


def get_pattern_analyzer():
    """Get the pattern analyzer instance"""
    return pattern_analyzer


# Set the getter function in routes
routes.get_pattern_analyzer = get_pattern_analyzer

# Include routes
app.include_router(routes.router)


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global pattern_analyzer

    logger.info(f"{SERVICE_NAME} starting on port {SERVICE_PORT}")
    logger.info(f"Data Service URL: {DATA_SERVICE_URL}")

    try:
        # Initialize pattern analyzer
        pattern_analyzer = PatternAnalyzer(data_service_url=DATA_SERVICE_URL)
        logger.info("Pattern Analyzer initialized successfully")

        logger.info("Available endpoints:")
        logger.info("  GET  /health - Health check")
        logger.info("  GET  /indicators/{type} - Calculate specific indicator")
        logger.info("  GET  /patterns/candlestick - Detect candlestick patterns")
        logger.info("  GET  /patterns/chart - Detect chart patterns")
        logger.info("  POST /scan - Scan multiple symbols/timeframes")
        logger.info("  POST /scan/opportunities - Find trading opportunities")
        logger.info("  GET  /analyze/{symbol}/{timeframe} - Complete analysis")

    except Exception as e:
        logger.error(f"Failed to initialize Pattern Analyzer: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global pattern_analyzer

    logger.info(f"{SERVICE_NAME} shutting down")

    if pattern_analyzer:
        pattern_analyzer = None
        logger.info("Pattern Analyzer cleaned up")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Pattern & Indicator Service")
    logger.info("=" * 80)
    logger.info(f"Service: {SERVICE_NAME}")
    logger.info(f"Port: {SERVICE_PORT}")
    logger.info(f"Data Service: {DATA_SERVICE_URL}")
    logger.info("=" * 80)

    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
