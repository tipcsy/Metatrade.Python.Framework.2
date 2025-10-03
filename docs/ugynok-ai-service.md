# AI Service Ügynök Leírás

**Verzió:** 1.0.0
**Létrehozva:** 2025. október 3.
**Service Port:** 5005

---

## 1. Ügynök Áttekintés

### 1.1 Ügynök Neve
**AI Service Ügynök**

### 1.2 Fő Felelősségek

Az AI Service Ügynök felelős a **mesterséges intelligencia alapú idősor elemzésért és előrejelzésért**. Ez a service gépi tanulási modelleket használ a historikus tick/OHLC adatok elemzésére és jövőbeli árfolyam előrejelzésére.

### 1.3 Service Típus
- **Machine Learning Service**
- **Time Series Forecasting**
- **Model Training & Inference**

### 1.4 Technológia

**Fő Könyvtár:** TensorFlow 2.x + Keras API

**Modell Típusok:**
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Transformer (opcionális, fejlettebb)

---

## 2. Részletes Felelősségek

### 2.1 Idősor Előrejelzés (Time Series Forecasting)

**Cél:** Múltbeli árfolyam adatok alapján jövőbeli árak előrejelzése

**Működés:**

#### 2.1.1 Adat Előkészítés

**Input Adatok:**
- OHLC (Open, High, Low, Close)
- Volume (Tick/Real volume)
- Technikai indikátorok (EMA, RSI, MACD, ATR stb.)

**Normalizálás:**
```python
from sklearn.preprocessing import MinMaxScaler

# Ár adatok normalizálása 0-1 tartományra
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(ohlc_data)
```

**Sequence Generálás:**
```python
# Sliding window approach
sequence_length = 60  # Utolsó 60 bar alapján jósol

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])  # Input: 60 bar
        y.append(data[i+seq_length])    # Output: következő bar (close ár)
    return np.array(X), np.array(y)
```

#### 2.1.2 Model Architektúra

**LSTM Model (Példa):**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    # 1. LSTM réteg
    LSTM(units=50, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.2),

    # 2. LSTM réteg
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),

    # 3. LSTM réteg
    LSTM(units=50),
    Dropout(0.2),

    # Output réteg
    Dense(units=1)  # Előrejelzés: következő close ár
])

model.compile(optimizer='adam', loss='mean_squared_error')
```

**GRU Model (Alternatíva):**
- Hasonló az LSTM-hez, de kevesebb paraméter
- Gyorsabb training
- Kissé kevésbé pontos, de elég jó

#### 2.1.3 Model Tanítás (Training)

**Training Folyamat:**
```
1. Historikus adatok betöltése (pl. 2023-2024 EURUSD M15)
2. Adat előkészítés (normalizálás, sequence generálás)
3. Train/Validation split (80% train, 20% validation)
4. Model training
   - Epochs: 50-100
   - Batch size: 32-64
   - Loss: Mean Squared Error (MSE)
   - Validation loss figyelése (early stopping)
5. Model mentés
```

**Training Paraméterek:**
```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(patience=5),  # Ha 5 epoch-on nem javul, stop
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)
```

#### 2.1.4 Inference (Előrejelzés)

**Előrejelzés Lépései:**
```python
# 1. Betöltjük az utolsó 60 bar-t
last_60_bars = get_last_n_bars(symbol, timeframe, 60)

# 2. Normalizálás
scaled_data = scaler.transform(last_60_bars)

# 3. Reshape (modell input formátum)
input_data = scaled_data.reshape(1, 60, n_features)

# 4. Prediction
predicted_scaled = model.predict(input_data)

# 5. De-normalizálás
predicted_price = scaler.inverse_transform(predicted_scaled)

# 6. Confidence score (opcionális)
confidence = calculate_confidence(predicted_price, last_60_bars)
```

**Előrejelzési Típusok:**

1. **Egy lépéses előrejelzés (1-step ahead):**
   - Következő 1 bar előrejelzése
   - Legpontosabb

2. **Több lépéses előrejelzés (multi-step ahead):**
   - Következő N bar előrejelzése (pl. 5, 10, 20 bar)
   - Kevésbé pontos, de hasznos hosszabb távú trendhez

### 2.2 Stratégia Optimalizáció

**Cél:** Stratégia paraméterek optimalizálása AI segítségével

#### 2.2.1 Genetic Algorithm (Genetikus Algoritmus)

**Működés:**
```
1. Populáció inicializálás (N db stratégia paraméter kombináció)
2. Backtesting minden kombinációra
3. Fitness érték számítás (pl. Sharpe ratio)
4. Szelekció: legjobb kombinációk kiválasztása
5. Crossover: kombinációk "keresztezése"
6. Mutáció: random változtatások
7. Ismétlés 2-6. lépések G generációra
8. Legjobb paraméter kombináció kiválasztása
```

**Példa:**
```
Stratégia: EMA Crossover
Paraméterek:
  - EMA Fast: [10, 50] tartomány
  - EMA Slow: [50, 200] tartomány
  - SL pips: [20, 100] tartomány

Genetikus algoritmus optimalizálja ezeket a paramétereket
Eredmény: EMA Fast=22, EMA Slow=87, SL=45 (legjobb Sharpe ratio)
```

#### 2.2.2 Reinforcement Learning (Megerősítéses Tanulás)

**Cél:** AI ügynök tanítása kereskedésre

**Működés:**
```
Környezet: Forex piac (szimulált backtest)
Ügynök: AI model
Állapot: Aktuális piaci adatok (OHLC, indikátorok)
Akciók: BUY, SELL, HOLD
Jutalom: Profit/Loss

1. Ügynök megfigyeli az állapotot
2. Döntés: BUY/SELL/HOLD
3. Piaci szimuláció (pozíció nyitás/zárás)
4. Jutalom számítás (profit = pozitív jutalom, loss = negatív jutalom)
5. Ügynök tanul a jutalomból
6. Ismétlés több ezer epizódon
```

**Algoritmusok:**
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- A3C (Asynchronous Actor-Critic)

**Előny:**
- Nem kell explicit szabályokat írni
- AI maga tanulja meg a kereskedési stratégiát

**Hátrány:**
- Nagyon hosszú training idő
- Instabil lehet
- Overfitting veszély

### 2.3 Pattern Felismerés AI-val

**Cél:** Chartpattern-ek automatikus felismerése gépi tanulással

#### 2.3.1 Convolutional Neural Network (CNN)

**Működés:**
```
1. Chart képek generálása (candlestick chartok)
2. Labelezés: "Head & Shoulders", "Triangle", "Double Top" stb.
3. CNN model training a képeken
4. Inference: új chart → pattern osztályozás
```

**Példa:**
```python
# CNN model chart pattern felismerésre
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(n_patterns, activation='softmax')  # Output: pattern osztályok
])
```

#### 2.3.2 Time Series Classification

**Cél:** OHLC adatok közvetlenül osztályozása (kép nélkül)

**Modellek:**
- LSTM Classifier
- 1D CNN
- Transformer

**Előny:**
- Gyorsabb, mint képfeldolgozás
- Nem kell chart képet generálni

### 2.4 Model Kezelés

**Model Lifecycle:**

1. **Training:**
   - Data preprocessing
   - Model training
   - Validation
   - Model mentés (`.h5` vagy `.keras` fájl)

2. **Deployment:**
   - Model betöltés memóriába
   - Inference endpoint elérhetővé tétele

3. **Monitoring:**
   - Model teljesítmény mérés (valós adatokon)
   - Drift detection (ha a modell romlik)

4. **Retraining:**
   - Periodikus újra-training (pl. havonta)
   - Új adatokkal való frissítés

**Model Verziókezelés:**
```
models/
├── eurusd_m15_lstm_v1.0.h5
├── eurusd_m15_lstm_v1.1.h5
├── eurusd_m15_lstm_v2.0.h5  (jelenleg aktív)
└── metadata.json
```

**Metadata:**
```json
{
  "model_id": "eurusd_m15_lstm_v2.0",
  "symbol": "EURUSD",
  "timeframe": "M15",
  "model_type": "LSTM",
  "training_date": "2025-09-01",
  "training_data_period": "2023-01-01 to 2024-12-31",
  "accuracy": 0.67,
  "mae": 0.0012,
  "status": "active"
}
```

### 2.5 Confidence Score

**Cél:** AI előrejelzés megbízhatóságának mérése

**Módszerek:**

1. **Prediction Variance:**
   - Model többször fut ugyanazon input-on (dropout engedélyezve)
   - Ha az eredmények szórása kicsi → magas confidence

2. **Ensemble Confidence:**
   - Több model előrejelzése
   - Ha konszenzus van → magas confidence

3. **Historical Accuracy:**
   - Elmúlt N előrejelzés pontossága
   - Ha múltban pontos volt → magas confidence

**Használat:**
```python
prediction, confidence = model.predict_with_confidence(data)

if confidence > 0.8:
    # Magas confidence → stratégia használhatja az előrejelzést
    use_ai_signal(prediction)
else:
    # Alacsony confidence → ne használja
    ignore_ai_signal()
```

---

## 3. REST API Endpointok

### 3.1 Health Check

#### GET /health
**Leírás:** Service állapot ellenőrzés

**Válasz:**
```json
{
  "status": "healthy",
  "service": "ai-service",
  "loaded_models": 3,
  "gpu_available": false
}
```

### 3.2 Model Training

#### POST /models/train
**Leírás:** Új model tanítása

**Body:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "M15",
  "model_type": "LSTM",
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "parameters": {
    "sequence_length": 60,
    "lstm_units": 50,
    "epochs": 50,
    "batch_size": 32
  }
}
```

**Válasz:**
```json
{
  "success": true,
  "job_id": "train-12345",
  "status": "queued",
  "estimated_duration": 3600
}
```

#### GET /models/train/{job_id}/status
**Leírás:** Training állapot

**Válasz:**
```json
{
  "job_id": "train-12345",
  "status": "running",
  "progress": 60,
  "current_epoch": 30,
  "total_epochs": 50,
  "train_loss": 0.0023,
  "val_loss": 0.0031
}
```

### 3.3 Model Inference

#### POST /models/{model_id}/predict
**Leírás:** Előrejelzés

**Body:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "M15",
  "steps_ahead": 1
}
```

**Válasz:**
```json
{
  "success": true,
  "model_id": "eurusd_m15_lstm_v2.0",
  "prediction": {
    "price": 1.10550,
    "confidence": 0.82,
    "direction": "UP",
    "timestamp": "2025-10-03T14:30:00Z"
  },
  "input_data": {
    "last_close": 1.10500,
    "last_60_bars": "..."
  }
}
```

### 3.4 Model Kezelés

#### GET /models
**Leírás:** Összes model listája

**Válasz:**
```json
{
  "models": [
    {
      "model_id": "eurusd_m15_lstm_v2.0",
      "symbol": "EURUSD",
      "timeframe": "M15",
      "model_type": "LSTM",
      "status": "active",
      "accuracy": 0.67,
      "trained_at": "2025-09-01"
    }
  ]
}
```

#### GET /models/{model_id}
**Leírás:** Model részletei

#### DELETE /models/{model_id}
**Leírás:** Model törlése

#### POST /models/{model_id}/activate
**Leírás:** Model aktiválása (inference-hez használva)

### 3.5 Strategy Optimization

#### POST /optimize/genetic
**Leírás:** Genetikus algoritmus futtatása

**Body:**
```json
{
  "strategy_id": 1,
  "symbol": "EURUSD",
  "timeframe": "M15",
  "parameter_ranges": {
    "ema_fast": [10, 50],
    "ema_slow": [50, 200],
    "sl_pips": [20, 100]
  },
  "population_size": 50,
  "generations": 20
}
```

**Válasz:**
```json
{
  "success": true,
  "job_id": "opt-456",
  "status": "queued"
}
```

#### GET /optimize/{job_id}/results
**Leírás:** Optimalizáció eredményei

**Válasz:**
```json
{
  "job_id": "opt-456",
  "status": "completed",
  "best_parameters": {
    "ema_fast": 22,
    "ema_slow": 87,
    "sl_pips": 45
  },
  "best_fitness": 1.85,
  "backtest_result": {
    "net_profit": 5000,
    "sharpe_ratio": 1.85
  }
}
```

---

## 4. Implementációs Útmutató

### 4.1 Projekt Struktúra

```
ai-service/
├── main.py
├── config.py
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── models.py
│   │   ├── inference.py
│   │   └── optimization.py
│   ├── core/
│   │   ├── model_trainer.py         # Model training
│   │   ├── predictor.py             # Inference
│   │   ├── data_preprocessor.py     # Adat előkészítés
│   │   └── genetic_optimizer.py     # Genetikus algoritmus
│   ├── models/
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   └── model_loader.py
│   ├── database/
│   │   └── data_loader.py           # Historikus adatok betöltése
│   └── utils/
│       └── metrics.py
└── saved_models/                    # Mentett modellek
```

### 4.2 Főbb Modulok

#### 4.2.1 Model Trainer
**Felelősség:** Model tanítás

**Főbb metódusok:**
- `train_model(symbol, timeframe, params)` - Training indítás
- `validate_model(model, test_data)` - Validálás
- `save_model(model, metadata)` - Model mentés

#### 4.2.2 Predictor
**Felelősség:** Inference

**Főbb metódusok:**
- `load_model(model_id)` - Model betöltés
- `predict(symbol, timeframe)` - Előrejelzés
- `calculate_confidence(prediction)` - Confidence számítás

#### 4.2.3 Data Preprocessor
**Felelősség:** Adat előkészítés

**Főbb metódusok:**
- `normalize_data(data)` - Normalizálás
- `create_sequences(data, seq_length)` - Sequence generálás
- `add_technical_indicators(ohlc)` - Indikátorok hozzáadása

---

## 5. Teljesítmény Optimalizáció

### 5.1 GPU Használat

**TensorFlow GPU támogatás:**
```python
import tensorflow as tf

# GPU ellenőrzés
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

# GPU memória növekedés engedélyezése (ne foglalja le az összeset)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### 5.2 Model Quantization

**Cél:** Model méret csökkentés, gyorsabb inference

```python
# Model konvertálás TensorFlow Lite-ra (kisebb, gyorsabb)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### 5.3 Batch Inference

**Több előrejelzés egyszerre:**
```python
# Több symbol előrejelzése egy batch-ben
symbols = ["EURUSD", "GBPUSD", "USDJPY"]
batch_data = [get_data(s) for s in symbols]
predictions = model.predict(np.array(batch_data))
```

---

## 6. Hibakezelés

### 6.1 Model Load Hiba
- Ha model fájl sérült → fallback régebbi verzióra
- Ha nincs model → figyelmeztetés, inference nem elérhető

### 6.2 Training Hiba
- Ellenőrzés: elég adat van-e (min. 10,000 bar)
- Validation loss divergál → early stopping

### 6.3 Inference Hiba
- Ha input adat hiányos → error visszaadás
- Timeout 5 másodperc (ha model lassú)

---

## 7. Tesztelés

### 7.1 Unit Tesztek
- Data preprocessing helyesség
- Sequence generation
- Model output formátum

### 7.2 Model Performance Tesztek
- Accuracy mérés test data-n
- Backtesting AI előrejelzésekkel

### 7.3 Load Tesztek
- Párhuzamos inference kérések
- Model switching (több model párhuzamosan)

---

**Dokumentum vége**
