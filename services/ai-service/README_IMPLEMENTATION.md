# AI Service Implementation Summary

## Overview
Complete implementation of the AI Service for LSTM-based forex price prediction and model management.

**Service Port:** 5005
**Technology:** TensorFlow 2.14 + Keras, FastAPI, Python 3.8+

---

## Implemented Components

### 1. Core Modules

#### `/app/core/lstm_model.py`
**Purpose:** LSTM neural network architecture for time series forecasting

**Features:**
- 3-layer LSTM architecture (configurable)
- 50 LSTM units per layer (default)
- 0.2 dropout rate between layers
- Adam optimizer with MSE loss
- Early stopping and model checkpointing
- Monte Carlo Dropout for confidence estimation
- Model save/load functionality

**Key Methods:**
- `build_model()` - Construct LSTM architecture
- `train()` - Train model with validation
- `predict()` - Make predictions
- `predict_with_confidence()` - Predictions with uncertainty quantification
- `evaluate()` - Calculate test metrics

#### `/app/core/data_preprocessor.py`
**Purpose:** Data preparation and feature engineering

**Features:**
- MinMaxScaler normalization (0-1 range)
- Sliding window sequence generation
- Technical indicators:
  - EMA (9, 21, 50 periods)
  - RSI (14 periods)
  - MACD with signal line
  - ATR (14 periods)
  - Bollinger Bands (20 periods, 2 std)
- Train/validation splitting
- Data validation (minimum 10,000 rows)
- NaN handling and forward fill

**Key Methods:**
- `prepare_data()` - Normalize and add indicators
- `create_sequences()` - Generate LSTM input sequences
- `split_data()` - Train/validation split
- `inverse_transform()` - Convert predictions back to original scale
- `validate_data()` - Quality checks

#### `/app/core/model_trainer.py`
**Purpose:** End-to-end model training pipeline

**Features:**
- Async training job management
- Data fetching from Data Service
- Complete training workflow:
  1. Fetch historical data
  2. Validate data quality
  3. Preprocess and normalize
  4. Create sequences
  5. Build LSTM model
  6. Train with early stopping
  7. Evaluate on validation set
  8. Save model, scaler, and metadata
- Training job status tracking
- Model metadata generation with full training history

**Key Methods:**
- `train_model()` - Complete training pipeline
- `_fetch_historical_data()` - Get data from Data Service
- `_generate_model_id()` - Create unique model identifier

**TrainingJobManager:**
- Job queue management
- Status tracking (queued, running, completed, failed)
- Progress monitoring

#### `/app/core/predictor.py`
**Purpose:** Model inference and prediction

**Features:**
- Model loading and caching
- Multi-model support (load multiple models in memory)
- Confidence score calculation
- Recent data fetching
- Prediction preprocessing pipeline
- Direction detection (UP/DOWN)
- Price change calculation
- Timeout handling (5 seconds)

**Key Methods:**
- `load_model()` - Load model into memory
- `predict()` - Make price predictions
- `unload_model()` - Free memory
- `get_loaded_models()` - List cached models

#### `/app/core/model_manager.py`
**Purpose:** Model lifecycle and registry management

**Features:**
- Model versioning and tracking
- Status management (active/inactive/deprecated)
- Model filtering (by symbol, timeframe, status)
- Model activation/deactivation
- Model deletion with file cleanup
- Statistics and reporting
- Old model cleanup

**Key Methods:**
- `list_models()` - Get all models with filters
- `get_model()` - Get model metadata
- `delete_model()` - Remove model files
- `activate_model()` - Set model as active
- `get_model_statistics()` - Service statistics
- `cleanup_old_models()` - Remove old inactive models

#### `/app/core/service.py`
**Purpose:** Main AI Service orchestrator

**Features:**
- Component initialization and coordination
- Async training job creation
- Background training execution
- Model prediction routing
- Service statistics aggregation
- GPU availability checking

**Key Methods:**
- `train_model()` - Create training job
- `predict()` - Route prediction requests
- `list_models()` - Get model list
- `activate_model()` - Activate model for production
- `get_service_stats()` - Service health stats

### 2. API Endpoints

#### Health Check
- **GET /health** - Service health status
  - Returns: status, loaded_models, GPU availability, TensorFlow version

#### Model Training
- **POST /models/train** - Train new LSTM model (async)
  - Request: symbol, timeframe, date range, hyperparameters
  - Returns: job_id, status, estimated_duration

- **GET /models/train/{job_id}/status** - Get training job status
  - Returns: job status, progress, current_epoch, losses

#### Model Inference
- **POST /models/{model_id}/predict** - Make price prediction
  - Request: symbol, timeframe, steps_ahead, use_confidence
  - Returns: predicted_price, confidence, direction, price_change

#### Model Management
- **GET /models** - List all models (with filters)
  - Query params: symbol, timeframe, status
  - Returns: array of model info

- **GET /models/{model_id}** - Get model details
  - Returns: full model metadata

- **DELETE /models/{model_id}** - Delete a model
  - Returns: success status

- **POST /models/{model_id}/activate** - Activate model
  - Returns: activation status

### 3. Request/Response Models (`/app/models/schemas.py`)

**Pydantic models for validation:**
- `TrainModelRequest` - Training request with hyperparameters
- `TrainModelResponse` - Training job creation response
- `PredictRequest` - Prediction request parameters
- `PredictResponse` - Prediction results with confidence
- `ModelListResponse` - List of models
- `ModelDetailResponse` - Detailed model information
- `TrainingJobStatus` - Job status tracking
- `HealthResponse` - Health check response

### 4. Main Application (`main.py`)

**Features:**
- FastAPI application setup
- TensorFlow GPU configuration
- Memory growth enabled for GPU
- Logging to file and stdout
- Service initialization on startup
- Automatic API documentation (/docs, /redoc)

---

## LSTM Model Specifications

### Default Architecture
```
Input: (batch_size, 60, n_features)
├── LSTM Layer 1: 50 units, return_sequences=True
├── Dropout: 0.2
├── LSTM Layer 2: 50 units, return_sequences=True
├── Dropout: 0.2
├── LSTM Layer 3: 50 units
├── Dropout: 0.2
└── Dense Output: 1 unit (predicted close price)
```

### Training Parameters
- **Sequence Length:** 60 bars (default)
- **Features:** OHLC + Volume + Technical Indicators (~16 features)
- **Epochs:** 50 (default, with early stopping)
- **Batch Size:** 32 (default)
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Validation Split:** 80/20

### Confidence Scoring
Uses **Monte Carlo Dropout**:
1. Run model 10 times with dropout enabled during inference
2. Calculate mean and standard deviation of predictions
3. Confidence = 1.0 - tanh(std) → [0, 1]
4. Threshold: 0.8 for high-confidence signals

---

## Model Versioning

### Naming Convention
```
{symbol}_{timeframe}_lstm_{timestamp}
Example: eurusd_m15_lstm_20251004_143022
```

### Files Stored
```
/app/models/saved_models/
├── {model_id}.h5              # Keras model
├── {model_id}_scaler.pkl      # MinMaxScaler
└── {model_id}_metadata.json   # Training metadata
```

### Metadata Structure
```json
{
  "model_id": "eurusd_m15_lstm_20251004_143022",
  "symbol": "EURUSD",
  "timeframe": "M15",
  "model_type": "LSTM",
  "training_date": "2025-10-04T14:30:22",
  "training_data_period": {
    "start": "2023-01-01",
    "end": "2024-12-31"
  },
  "data_points": 45000,
  "training_samples": 36000,
  "validation_samples": 9000,
  "parameters": {
    "sequence_length": 60,
    "lstm_units": 50,
    "num_lstm_layers": 3,
    "dropout_rate": 0.2,
    "epochs": 50,
    "batch_size": 32
  },
  "metrics": {
    "val_loss": 0.0023,
    "val_mae": 0.0015,
    "train_loss": 0.0018,
    "train_mae": 0.0012
  },
  "n_features": 16,
  "status": "active"
}
```

---

## Integration with Other Services

### Data Service (Port 5002)
- **GET /data/historical** - Fetch training data
- **GET /data/ohlc** - Fetch recent data for predictions

### Future Integration
- **Strategy Service** - Consume AI predictions for strategy signals
- **Backend API** - Route requests from frontend

---

## Usage Examples

### 1. Train a Model
```bash
curl -X POST "http://localhost:5005/models/train" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "timeframe": "M15",
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "parameters": {
      "sequence_length": 60,
      "lstm_units": 50,
      "epochs": 50,
      "batch_size": 32
    }
  }'

# Response:
{
  "success": true,
  "job_id": "train-1",
  "status": "queued",
  "estimated_duration": 3600
}
```

### 2. Check Training Status
```bash
curl "http://localhost:5005/models/train/train-1/status"

# Response:
{
  "success": true,
  "job": {
    "job_id": "train-1",
    "status": "running",
    "progress": 60,
    "current_epoch": 30,
    "total_epochs": 50,
    "train_loss": 0.0023,
    "val_loss": 0.0031
  }
}
```

### 3. Make Prediction
```bash
curl -X POST "http://localhost:5005/models/eurusd_m15_lstm_20251004_143022/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "timeframe": "M15",
    "steps_ahead": 1,
    "use_confidence": true
  }'

# Response:
{
  "success": true,
  "model_id": "eurusd_m15_lstm_20251004_143022",
  "prediction": {
    "price": 1.10550,
    "current_price": 1.10500,
    "price_change": 0.00050,
    "price_change_pct": 0.045,
    "direction": "UP",
    "confidence": 0.82,
    "timestamp": "2025-10-04T14:35:00Z"
  }
}
```

### 4. List Models
```bash
curl "http://localhost:5005/models?symbol=EURUSD&status=active"

# Response:
{
  "success": true,
  "models": [
    {
      "model_id": "eurusd_m15_lstm_20251004_143022",
      "symbol": "EURUSD",
      "timeframe": "M15",
      "model_type": "LSTM",
      "status": "active",
      "training_date": "2025-10-04T14:30:22",
      "val_loss": 0.0023,
      "val_mae": 0.0015
    }
  ],
  "total": 1
}
```

---

## Performance Optimization

### GPU Configuration
- Automatic GPU detection
- Memory growth enabled (prevents OOM)
- Falls back to CPU if no GPU available

### Inference Optimization
- Model caching (loaded models kept in memory)
- Batch inference support
- 5-second timeout for predictions
- Lazy loading (models loaded on first use)

### Training Optimization
- Early stopping (patience=5 epochs)
- Model checkpointing (saves best model only)
- Async background training (non-blocking API)
- Data validation before training

---

## Error Handling

### Common Errors
1. **Insufficient Data** - Minimum 10,000 rows required
2. **Model Not Found** - Returns 404 with error message
3. **Training Failed** - Job status set to 'failed' with error details
4. **Prediction Timeout** - Returns error after 5 seconds
5. **GPU OOM** - Automatic fallback to CPU

### Logging
- All operations logged with timestamps
- Separate log file: `logs/ai-service.log`
- Console output for real-time monitoring

---

## Testing

### Health Check
```bash
curl http://localhost:5005/health
```

### API Documentation
- **Swagger UI:** http://localhost:5005/docs
- **ReDoc:** http://localhost:5005/redoc

---

## Dependencies

See `requirements.txt`:
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- tensorflow==2.14.0
- scikit-learn==1.3.2
- pandas==2.1.1
- numpy==1.26.0
- requests==2.31.0

---

## Important Notes

1. **Data Service Dependency:** AI Service requires Data Service (port 5002) to be running for training and predictions

2. **Model Storage:** Models are stored in `/app/models/saved_models/` - ensure sufficient disk space

3. **Training Time:** Training can take 30-60 minutes depending on data size and hardware

4. **GPU Recommended:** While CPU works, GPU significantly speeds up training (10-20x faster)

5. **Confidence Threshold:** Use predictions with confidence > 0.8 for trading decisions

6. **Model Retraining:** Recommended monthly retraining to adapt to market changes

7. **Async Training:** Training is non-blocking - use job status endpoint to monitor progress

---

## File Structure
```
ai-service/
├── main.py                          # FastAPI application entry point
├── requirements.txt                 # Python dependencies
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py               # REST API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── lstm_model.py           # LSTM architecture
│   │   ├── data_preprocessor.py   # Data preprocessing
│   │   ├── model_trainer.py       # Training pipeline
│   │   ├── predictor.py           # Inference engine
│   │   ├── model_manager.py       # Model lifecycle
│   │   └── service.py             # Main orchestrator
│   └── models/
│       ├── __init__.py
│       ├── schemas.py             # Pydantic models
│       └── saved_models/          # Trained models storage
│           └── .gitkeep
└── README_IMPLEMENTATION.md       # This file
```

---

**Implementation Date:** October 4, 2025
**Status:** Complete and ready for testing
**Next Steps:** Integration testing with Data Service and end-to-end training/prediction workflow
