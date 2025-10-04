# AI Service API Reference

**Base URL:** `http://localhost:5005`

---

## Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Service health check |
| POST | /models/train | Train a new LSTM model |
| GET | /models/train/{job_id}/status | Get training job status |
| POST | /models/{model_id}/predict | Make price prediction |
| GET | /models | List all models |
| GET | /models/{model_id} | Get model details |
| DELETE | /models/{model_id} | Delete a model |
| POST | /models/{model_id}/activate | Activate a model |

---

## 1. Health Check

### GET /health

Check service health and status.

**Response:**
```json
{
  "status": "healthy",
  "service": "ai-service",
  "port": 5005,
  "loaded_models": 2,
  "gpu_available": false,
  "tensorflow_version": "2.14.0"
}
```

---

## 2. Model Training

### POST /models/train

Train a new LSTM model for forex price prediction.

**Request Body:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "M15",
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "parameters": {
    "sequence_length": 60,
    "lstm_units": 50,
    "num_lstm_layers": 3,
    "dropout_rate": 0.2,
    "epochs": 50,
    "batch_size": 32,
    "train_ratio": 0.8,
    "add_indicators": true
  }
}
```

**Parameters:**
- `symbol` (required): Trading symbol (e.g., "EURUSD", "GBPUSD")
- `timeframe` (required): Timeframe (e.g., "M15", "H1", "H4")
- `start_date` (required): Training data start date (YYYY-MM-DD)
- `end_date` (required): Training data end date (YYYY-MM-DD)
- `parameters` (optional): Training hyperparameters
  - `sequence_length`: Lookback window (default: 60)
  - `lstm_units`: LSTM units per layer (default: 50)
  - `num_lstm_layers`: Number of LSTM layers (default: 3)
  - `dropout_rate`: Dropout rate (default: 0.2)
  - `epochs`: Training epochs (default: 50)
  - `batch_size`: Batch size (default: 32)
  - `train_ratio`: Train/validation split (default: 0.8)
  - `add_indicators`: Add technical indicators (default: true)

**Response:**
```json
{
  "success": true,
  "job_id": "train-1",
  "status": "queued",
  "estimated_duration": 3600
}
```

**Status Codes:**
- `200`: Training job created successfully
- `500`: Service error

---

### GET /models/train/{job_id}/status

Get the status of a training job.

**Path Parameters:**
- `job_id`: Job identifier returned from train endpoint

**Response:**
```json
{
  "success": true,
  "job": {
    "job_id": "train-1",
    "status": "running",
    "progress": 60,
    "current_epoch": 30,
    "total_epochs": 50,
    "train_loss": 0.0023,
    "val_loss": 0.0031,
    "created_at": "2025-10-04T14:00:00Z",
    "started_at": "2025-10-04T14:00:05Z",
    "completed_at": null
  }
}
```

**Job Status Values:**
- `queued`: Job created, waiting to start
- `running`: Training in progress
- `completed`: Training finished successfully
- `failed`: Training failed

---

## 3. Prediction

### POST /models/{model_id}/predict

Make price prediction using a trained model.

**Path Parameters:**
- `model_id`: Model identifier (e.g., "eurusd_m15_lstm_20251004_143022")

**Request Body:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "M15",
  "steps_ahead": 1,
  "use_confidence": true
}
```

**Parameters:**
- `symbol` (required): Trading symbol
- `timeframe` (required): Timeframe
- `steps_ahead`: Number of steps to predict (default: 1, max: 10)
- `use_confidence`: Calculate confidence score (default: true)

**Response:**
```json
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
    "steps_ahead": 1,
    "timestamp": "2025-10-04T14:35:00Z"
  },
  "input_data": {
    "symbol": "EURUSD",
    "timeframe": "M15",
    "last_close": 1.10500,
    "sequence_length": 60
  }
}
```

**Prediction Fields:**
- `price`: Predicted close price
- `current_price`: Current close price
- `price_change`: Absolute price change
- `price_change_pct`: Percentage price change
- `direction`: "UP" or "DOWN"
- `confidence`: Confidence score (0-1), null if use_confidence=false

**Status Codes:**
- `200`: Prediction successful
- `404`: Model not found
- `500`: Prediction failed

---

## 4. Model Management

### GET /models

List all available models with optional filters.

**Query Parameters:**
- `symbol`: Filter by symbol (optional)
- `timeframe`: Filter by timeframe (optional)
- `status`: Filter by status ("active", "inactive") (optional)

**Example:**
```
GET /models?symbol=EURUSD&status=active
```

**Response:**
```json
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
      "accuracy": null,
      "val_loss": 0.0023,
      "val_mae": 0.0015
    }
  ],
  "total": 1
}
```

---

### GET /models/{model_id}

Get detailed information about a specific model.

**Path Parameters:**
- `model_id`: Model identifier

**Response:**
```json
{
  "success": true,
  "model": {
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
      "batch_size": 32,
      "add_indicators": true
    },
    "metrics": {
      "val_loss": 0.0023,
      "val_mae": 0.0015,
      "train_loss": 0.0018,
      "train_mae": 0.0012
    },
    "n_features": 16,
    "feature_columns": ["open", "high", "low", "close", "volume", "ema_9", ...],
    "status": "active",
    "model_path": "/path/to/model.h5",
    "scaler_path": "/path/to/scaler.pkl"
  }
}
```

**Status Codes:**
- `200`: Model found
- `404`: Model not found

---

### DELETE /models/{model_id}

Delete a model and all associated files.

**Path Parameters:**
- `model_id`: Model identifier

**Response:**
```json
{
  "success": true,
  "message": "Model deleted: eurusd_m15_lstm_20251004_143022"
}
```

**Status Codes:**
- `200`: Model deleted
- `404`: Model not found
- `500`: Deletion failed

---

### POST /models/{model_id}/activate

Activate a model for production use. This will deactivate other models for the same symbol/timeframe.

**Path Parameters:**
- `model_id`: Model identifier

**Response:**
```json
{
  "success": true,
  "model_id": "eurusd_m15_lstm_20251004_143022",
  "message": "Model activated successfully"
}
```

**Status Codes:**
- `200`: Model activated
- `404`: Model not found
- `500`: Activation failed

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

**Common Error Messages:**
- "AI Service not initialized"
- "Model not found: {model_id}"
- "Failed to load model: {model_id}"
- "Insufficient recent data for prediction"
- "Job not found: {job_id}"
- "No data received from Data Service"
- "Data validation failed"

---

## Interactive Documentation

The AI Service provides interactive API documentation:

- **Swagger UI:** http://localhost:5005/docs
- **ReDoc:** http://localhost:5005/redoc

These interfaces allow you to:
- Browse all endpoints
- See request/response schemas
- Test API calls directly from the browser
- View detailed parameter descriptions

---

## Code Examples

### Python (using requests)

```python
import requests

# Train a model
response = requests.post('http://localhost:5005/models/train', json={
    'symbol': 'EURUSD',
    'timeframe': 'M15',
    'start_date': '2023-01-01',
    'end_date': '2024-12-31'
})
job_id = response.json()['job_id']

# Check training status
status = requests.get(f'http://localhost:5005/models/train/{job_id}/status')
print(status.json())

# Make prediction
prediction = requests.post(
    f'http://localhost:5005/models/{model_id}/predict',
    json={
        'symbol': 'EURUSD',
        'timeframe': 'M15',
        'steps_ahead': 1,
        'use_confidence': True
    }
)
print(prediction.json()['prediction'])

# List models
models = requests.get('http://localhost:5005/models?status=active')
print(models.json())
```

### cURL

```bash
# Train model
curl -X POST "http://localhost:5005/models/train" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"EURUSD","timeframe":"M15","start_date":"2023-01-01","end_date":"2024-12-31"}'

# Make prediction
curl -X POST "http://localhost:5005/models/eurusd_m15_lstm_20251004_143022/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"EURUSD","timeframe":"M15","steps_ahead":1,"use_confidence":true}'

# List models
curl "http://localhost:5005/models?symbol=EURUSD&status=active"
```

---

## Best Practices

1. **Training:**
   - Use at least 1 year of historical data
   - Monitor job status regularly during training
   - Validate model performance before activation

2. **Prediction:**
   - Always use confidence scores
   - Only trade on predictions with confidence > 0.8
   - Verify symbol/timeframe matches model

3. **Model Management:**
   - Activate only one model per symbol/timeframe
   - Regularly retrain models (monthly recommended)
   - Clean up old inactive models to save disk space

4. **Error Handling:**
   - Always check `success` field in responses
   - Implement retry logic for failed predictions
   - Log all API calls for debugging

---

**Last Updated:** October 4, 2025
