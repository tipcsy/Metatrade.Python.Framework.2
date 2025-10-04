"""
AI Service Main Application
Initializes and runs the AI/ML service for forex prediction
"""

from fastapi import FastAPI
from app.api import routes
from app.core.service import AIService
import uvicorn
import logging
import sys
import os
import tensorflow as tf

# Service configuration
SERVICE_NAME = "ai-service"
SERVICE_PORT = 5005
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://localhost:5002")

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] [%(levelname)s] [{SERVICE_NAME}] [%(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"{SERVICE_NAME}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(SERVICE_NAME)

# Configure TensorFlow
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable GPU memory growth (if GPU available)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU available: {len(gpus)} device(s)")
    else:
        logger.info("No GPU available, using CPU")
except Exception as e:
    logger.warning(f"Error configuring GPU: {e}")

# Create FastAPI app
app = FastAPI(
    title="AI Service",
    version="1.0.0",
    description="AI/ML model training and inference service for forex trading",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include routes
app.include_router(routes.router)

# Global AI service instance
ai_service_instance = None


@app.on_event("startup")
async def startup_event():
    """Initialize AI Service on startup"""
    global ai_service_instance

    logger.info(f"{SERVICE_NAME} starting on port {SERVICE_PORT}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Data Service URL: {DATA_SERVICE_URL}")

    try:
        # Initialize AI Service
        ai_service_instance = AIService(data_service_url=DATA_SERVICE_URL)

        # Set service instance in routes
        routes.set_ai_service(ai_service_instance)

        logger.info("AI Service initialized successfully")

        # Log GPU availability
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        logger.info(f"GPU available: {gpu_available}")

    except Exception as e:
        logger.error(f"Error initializing AI Service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info(f"{SERVICE_NAME} shutting down")

    # Cleanup resources if needed
    if ai_service_instance:
        logger.info("Cleaning up AI Service resources")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVICE_PORT,
        log_level="info"
    )
