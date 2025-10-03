# Kiegészítés - Configuration, Real-time, Performance

Ez a dokumentum a projekt-dokumentacio.md 11-13. sz

ekciói, amelyeket be kell illeszteni a 10. Error Recovery után.

---

## 11. Configuration Management

### 11.1 Központi Konfiguráció

**Fájl:** `config.json` (projekt gyökérben)

**Struktúra:**

```json
{
  "system": {
    "environment": "development",
    "log_level": "INFO",
    "data_directory": "./database"
  },
  "services": {
    "backend-api": {
      "port": 5000,
      "host": "localhost",
      "auto_start": true,
      "path": "services/backend-api/main.py"
    },
    "data-service": {
      "port": 5001,
      "host": "localhost",
      "auto_start": true,
      "auto_restart": true,
      "path": "services/data-service/main.py",
      "gap_fill_on_start": true,
      "tick_collection_interval_ms": 100,
      "batch_size": 1000
    },
    "mt5-service": {
      "port": 5002,
      "host": "localhost",
      "auto_start": true,
      "auto_restart": true,
      "path": "services/mt5-service/main.py",
      "mt5_terminal_path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
      "connection_timeout_seconds": 10
    },
    "pattern-service": {
      "port": 5003,
      "auto_start": true,
      "auto_restart": true,
      "path": "services/pattern-service/main.py"
    },
    "strategy-service": {
      "port": 5004,
      "auto_start": false,
      "auto_restart": true,
      "path": "services/strategy-service/main.py"
    },
    "backtesting-service": {
      "port": 5006,
      "auto_start": false,
      "auto_restart": false,
      "path": "services/backtesting-service/main.py",
      "max_parallel_backtests": 3
    },
    "ai-service": {
      "port": 5005,
      "auto_start": false,
      "auto_restart": false,
      "path": "services/ai-service/main.py"
    }
  },
  "database": {
    "type": "sqlite",
    "tick_partition_by": "month",
    "ohlc_partition_by": "symbol"
  },
  "trading": {
    "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
    "timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"]
  },
  "monitoring": {
    "health_check_interval_seconds": 5,
    "metrics_collection_enabled": true,
    "alert_email": "trader@example.com"
  }
}
```

### 11.2 Environment Variables

**Sensitive adatok (jelszavak, API kulcsok) környezeti változókban:**

**.env fájl:**
```
MT5_ACCOUNT_NUMBER=12345678
MT5_ACCOUNT_PASSWORD=SecretPassword123
SMTP_EMAIL_PASSWORD=EmailPassword456
DATABASE_ENCRYPTION_KEY=EncryptionKey789
```

**Python betöltés:**
```python
from dotenv import load_dotenv
import os

load_dotenv()

mt5_account = os.getenv("MT5_ACCOUNT_NUMBER")
mt5_password = os.getenv("MT5_ACCOUNT_PASSWORD")
```

### 11.3 Config Hot-Reload

**Cél:** Konfiguráció változtatása újraindítás nélkül

**File Watcher:**
```python
import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("config.json"):
            logger.info("Config file changed, reloading...")
            reload_config()
            notify_services_config_changed()

observer = Observer()
observer.schedule(ConfigFileHandler(), path=".", recursive=False)
observer.start()
```

**Dinamikusan változtatható értékek:**
- Log level
- Health check interval
- Tick collection interval
- Auto-restart beállítások

**NEM változtatható futás közben (újraindítás kell):**
- Port számok
- Service paths
- Database paths

---

## 12. Real-time Communication Részletesen

### 12.1 WebSocket vs Server-Sent Events (SSE)

**Döntés: WebSocket** ✅

**Indoklás:**
- **Kétirányú** kommunikáció (Frontend → Backend és Backend → Frontend)
- Alacsony latency
- Széles browser támogatás
- Python library egyszerű (websockets, python-socketio)

**SSE hátrányai:**
- Csak **egyirányú** (Backend → Frontend)
- HTTP/1.1 connection limit (max 6 egyidejű connection/domain)

### 12.2 WebSocket Implementáció

**Backend API WebSocket Server:**

```python
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import json

app = FastAPI()

# Connected clients
connected_clients = []

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        while True:
            # Fogadás Frontend-től (ha kell)
            data = await websocket.receive_text()
            message = json.loads(data)

            # Feldolgozás
            handle_client_message(message)

    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def broadcast_event(event_type: str, data: dict):
    """Event küldés minden kapcsolódott kliensnek"""
    message = json.dumps({
        "type": event_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    })

    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send to client: {e}")
```

**Frontend WebSocket Kliens (Angular):**

```typescript
export class WebSocketService {
  private socket: WebSocket;
  public events$: Subject<any> = new Subject();

  connect() {
    this.socket = new WebSocket('ws://localhost:5000/ws/events');

    this.socket.onopen = () => {
      console.log('WebSocket connected');
    };

    this.socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.events$.next(message);
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.socket.onclose = () => {
      console.log('WebSocket disconnected, reconnecting...');
      setTimeout(() => this.connect(), 3000);  // Auto-reconnect
    };
  }

  send(message: any) {
    this.socket.send(JSON.stringify(message));
  }
}
```

### 12.3 Message Formátumok

**1. Service Status Change:**
```json
{
  "type": "service_status_change",
  "data": {
    "service": "data-service",
    "status": "ONLINE",
    "uptime": 3600
  },
  "timestamp": "2025-10-03T14:35:22Z"
}
```

**2. New Tick:**
```json
{
  "type": "new_tick",
  "data": {
    "symbol": "EURUSD",
    "bid": 1.10523,
    "ask": 1.10525,
    "timestamp": 1696337696000
  },
  "timestamp": "2025-10-03T14:35:22Z"
}
```

**3. Strategy Signal:**
```json
{
  "type": "strategy_signal",
  "data": {
    "strategy_id": 1,
    "strategy_name": "EMA Crossover",
    "symbol": "EURUSD",
    "signal": "BUY",
    "confidence": 0.85,
    "entry_price": 1.10520
  },
  "timestamp": "2025-10-03T14:35:22Z"
}
```

**4. Gap Fill Progress:**
```json
{
  "type": "gap_fill_progress",
  "data": {
    "symbol": "EURUSD",
    "progress": 65,
    "ticks_downloaded": 1250000,
    "estimated_remaining_time": 120
  },
  "timestamp": "2025-10-03T14:35:22Z"
}
```

**5. Alert:**
```json
{
  "type": "alert",
  "data": {
    "severity": "CRITICAL",
    "title": "MT5 Disconnected",
    "message": "MT5 connection lost, trading halted",
    "action_required": true
  },
  "timestamp": "2025-10-03T14:35:22Z"
}
```

### 12.4 Reconnection Strategy

**Frontend Auto-reconnect:**

```typescript
class WebSocketService {
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;

  connect() {
    // ... connection code ...

    this.socket.onclose = () => {
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
        console.log(`Reconnecting in ${delay}ms...`);

        setTimeout(() => {
          this.reconnectAttempts++;
          this.connect();
        }, delay);
      } else {
        console.error('Max reconnect attempts reached');
        this.showReconnectError();
      }
    };

    this.socket.onopen = () => {
      this.reconnectAttempts = 0;  // Reset on successful connection
    };
  }
}
```

### 12.5 Message Rate Limiting

**Cél:** Túl sok üzenet ne terhelje le a Frontend-et

**Backend Throttling:**

```python
from collections import deque
import time

class MessageThrottler:
    def __init__(self, max_messages_per_second=100):
        self.max_rate = max_messages_per_second
        self.message_times = deque()

    def can_send(self):
        now = time.time()

        # Töröljük a régi üzeneteket (1 másodpercnél régebbi)
        while self.message_times and self.message_times[0] < now - 1:
            self.message_times.popleft()

        if len(self.message_times) < self.max_rate:
            self.message_times.append(now)
            return True
        else:
            return False

throttler = MessageThrottler(max_messages_per_second=100)

async def send_tick_event(tick_data):
    if throttler.can_send():
        await broadcast_event("new_tick", tick_data)
    # else: skip (túl gyors)
```

---

## 13. Performance Tuning és Optimization

### 13.1 Database Optimalizáció

**1. Index-ek:**

```sql
-- Tick adatbázis
CREATE INDEX idx_timestamp ON ticks(timestamp);
CREATE INDEX idx_symbol_timestamp ON ticks(symbol, timestamp);

-- OHLC adatbázis
CREATE INDEX idx_symbol_timeframe_timestamp ON ohlc_data(symbol, timeframe, timestamp);

-- Completeness
CREATE INDEX idx_symbol_date ON tick_data_completeness(symbol, date_readable);
```

**2. Batch Írás (már megvan):**
- 1000 tick egyszerre INSERT
- Transaction használat

**3. PRAGMA beállítások (SQLite):**

```python
# Gyorsabb írás (kockázatosabb)
db.execute("PRAGMA synchronous = NORMAL")  # Default: FULL
db.execute("PRAGMA journal_mode = WAL")    # Write-Ahead Logging
db.execute("PRAGMA cache_size = -64000")   # 64 MB cache
```

**4. Connection Pool:**

```python
from sqlalchemy import create_engine, pool

engine = create_engine(
    'sqlite:///database/eurusd_ticks.db',
    poolclass=pool.QueuePool,
    pool_size=5,
    max_overflow=10
)
```

### 13.2 Caching Stratégia

**Redis (opcionális, később):**
- Indikátor értékek cache-elése (TTL: 60s)
- Pattern jelzések cache (TTL: 30s)
- Account info cache (TTL: 5s)

**In-Memory Cache (egyszerűbb, első verzió):**

```python
from functools import lru_cache
import time

class TimedCache:
    def __init__(self, ttl=60):
        self.cache = {}
        self.ttl = ttl

    def get(self, key):
        if key in self.cache:
            value, expiry = self.cache[key]
            if time.time() < expiry:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key, value):
        self.cache[key] = (value, time.time() + self.ttl)

# Használat
indicator_cache = TimedCache(ttl=60)

def get_ema(symbol, timeframe, period):
    cache_key = f"ema_{symbol}_{timeframe}_{period}"
    cached_value = indicator_cache.get(cache_key)

    if cached_value:
        return cached_value

    # Számítás
    ema = calculate_ema(symbol, timeframe, period)

    # Cache-elés
    indicator_cache.set(cache_key, ema)

    return ema
```

### 13.3 Async/Await és Parallel Processing

**FastAPI Async Endpoints:**

```python
@app.get("/ticks/{symbol}")
async def get_ticks(symbol: str, from_time: int, to_time: int):
    """Async endpoint - nem blokkolja a többi kérést"""

    # Async DB query
    ticks = await db.fetch_ticks_async(symbol, from_time, to_time)

    return {"ticks": ticks}
```

**Parallel Backtest:**

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def run_backtest_parallel(strategy_id, param_combinations):
    """Több backtest párhuzamosan"""

    cpu_count = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=cpu_count - 1) as executor:
        futures = [
            executor.submit(run_single_backtest, strategy_id, params)
            for params in param_combinations
        ]

        results = [future.result() for future in futures]

    return results
```

### 13.4 Profiling és Bottleneck Azonosítás

**cProfile:**

```python
import cProfile
import pstats

def profile_function(func):
    profiler = cProfile.Profile()
    profiler.enable()

    result = func()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 leglassabb

    return result
```

**line_profiler (részletesebb):**

```python
from line_profiler import LineProfiler

@profile
def gap_fill_slow_function():
    # ... kód ...
    pass

# Futtatás: kernprof -l -v script.py
```

**Memory Profiler:**

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # ... kód ...
    pass
```

### 13.5 Query Optimization Tips

**Rossz:**
```python
# N+1 query probléma
for symbol in symbols:
    ticks = db.query(f"SELECT * FROM ticks WHERE symbol = '{symbol}'")
```

**Jó:**
```python
# Egy query
symbols_str = "','".join(symbols)
all_ticks = db.query(f"SELECT * FROM ticks WHERE symbol IN ('{symbols_str}')")
```

**Még jobb (Batch + Indexed):**
```python
# Batch + index használat
query = """
SELECT * FROM ticks
WHERE symbol IN (?)
  AND timestamp BETWEEN ? AND ?
"""
result = db.execute(query, (symbols, start_time, end_time))
```

---

**Dokumentum vége**
