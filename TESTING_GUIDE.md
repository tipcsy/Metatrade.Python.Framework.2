# 🧪 MT5 Trading System - Tesztelési Útmutató

**Verzió:** 1.0
**Utolsó frissítés:** 2025-10-04

---

## 📋 Tartalomjegyzék

1. [Gyors Start](#1-gyors-start)
2. [Service Health Check](#2-service-health-check)
3. [Backtesting Service Részletes Teszt](#3-backtesting-service-részletes-teszt)
4. [Frontend Tesztelés](#4-frontend-tesztelés)
5. [API Tesztelés curl-lel](#5-api-tesztelés-curl-lel)
6. [End-to-End Teszt](#6-end-to-end-teszt)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Gyors Start

### 1.1 Szolgáltatások Indítása

```bash
# Projekt root könyvtárban
cd /home/tipcsy/Metatrade.Python.Framework.2

# Összes backend service indítása
./start_all_services.sh

# Várj 5-10 másodpercet, amíg minden elindul
```

### 1.2 Gyors Health Check

```bash
# Minden service tesztelése egyszerre
./test_all_services.sh
```

**Várt kimenet:**
```
🧪 Testing All Services...
================================
Testing Backend API (port 5000)... ✓ OK
Testing Data Service (port 5001)... ✓ OK
Testing MT5 Service (port 5002)... ✓ OK
Testing Pattern Service (port 5003)... ✓ OK
Testing Strategy Service (port 5004)... ✓ OK
Testing AI Service (port 5005)... ✓ OK
Testing Backtesting Service (port 5006)... ✓ OK
```

---

## 2. Service Health Check

### 2.1 Manuális Tesztelés (curl)

```bash
# Backend API
curl http://localhost:5000/health

# Data Service
curl http://localhost:5001/health

# MT5 Service
curl http://localhost:5002/health

# Pattern Service
curl http://localhost:5003/health

# Strategy Service
curl http://localhost:5004/health

# AI Service
curl http://localhost:5005/health

# Backtesting Service
curl http://localhost:5006/health
```

### 2.2 Várt Válasz Formátum

```json
{
  "status": "healthy",
  "service": "backtesting-service",
  "port": 5006,
  "version": "1.0.0"
}
```

---

## 3. Backtesting Service Részletes Teszt

### 3.1 Automatikus Test Suite

```bash
# Python test script futtatása
python3 test_backtesting_service.py
```

**Ez teszteli:**
- ✅ Health check
- ✅ Elérhető stratégiák listázása
- ✅ Performance metrikák definíciói
- ✅ Backtest indítása (MA Crossover stratégiával)
- ✅ Backtest státusz lekérdezése
- ✅ Backtest eredmények lekérdezése
- ✅ Összes backtest listázása

### 3.2 Manuális API Tesztelés

#### 3.2.1 Elérhető Stratégiák

```bash
curl http://localhost:5006/strategies | jq
```

**Válasz:**
```json
{
  "success": true,
  "data": {
    "strategies": [
      {
        "type": "MA_CROSSOVER",
        "name": "Moving Average Crossover",
        "description": "Simple MA crossover strategy",
        "parameters": {
          "fast_period": {"type": "int", "min": 5, "max": 50, "default": 10},
          "slow_period": {"type": "int", "min": 20, "max": 200, "default": 30}
        }
      },
      {
        "type": "RSI",
        "name": "RSI Strategy",
        "description": "RSI overbought/oversold strategy"
      }
    ]
  }
}
```

#### 3.2.2 Backtest Indítása

```bash
curl -X POST http://localhost:5006/backtest/start \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_type": "MA_CROSSOVER",
    "symbol": "EURUSD",
    "timeframe": "H1",
    "from_time": 1704067200,
    "to_time": 1706745600,
    "initial_balance": 10000.0,
    "parameters": {
      "fast_period": 10,
      "slow_period": 30,
      "stop_loss_pips": 50.0,
      "take_profit_pips": 100.0
    },
    "commission": 0.0,
    "spread_pips": 1.0,
    "slippage_pips": 0.5,
    "position_size": 0.01
  }' | jq
```

**Válasz:**
```json
{
  "success": true,
  "data": {
    "backtest_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "message": "Backtest started successfully"
  }
}
```

#### 3.2.3 Backtest Státusz

```bash
# Használd az előző válaszból kapott backtest_id-t
BACKTEST_ID="a1b2c3d4-e5f6-7890-abcd-ef1234567890"

curl http://localhost:5006/backtest/$BACKTEST_ID/status | jq
```

**Válasz:**
```json
{
  "success": true,
  "data": {
    "backtest_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "running",
    "progress": 45,
    "error_message": null
  }
}
```

#### 3.2.4 Backtest Eredmények

```bash
curl http://localhost:5006/backtest/$BACKTEST_ID | jq
```

**Válasz (rövidített):**
```json
{
  "success": true,
  "data": {
    "backtest_id": "...",
    "status": "completed",
    "strategy_type": "MA_CROSSOVER",
    "symbol": "EURUSD",
    "performance": {
      "total_trades": 25,
      "winning_trades": 15,
      "losing_trades": 10,
      "win_rate": 60.0,
      "net_profit": 1250.50,
      "sharpe_ratio": 1.85,
      "max_drawdown": 8.5
    },
    "trades": [...],
    "equity_curve": [...]
  }
}
```

#### 3.2.5 Összes Backtest Listázása

```bash
curl "http://localhost:5006/backtests?limit=10&status=completed" | jq
```

#### 3.2.6 Backtest Törlése

```bash
curl -X DELETE http://localhost:5006/backtest/$BACKTEST_ID | jq
```

---

## 4. Frontend Tesztelés

### 4.1 Frontend Indítása

```bash
cd frontend
npm start
```

**Várj amíg elindul:**
```
Application bundle generation complete.
Watch mode enabled. Watching for file changes...
➜  Local:   http://localhost:4200/
```

### 4.2 Böngészőben Tesztelés

1. **Nyisd meg:** http://localhost:4200

2. **Ellenőrizd a Dashboard-ot:**
   - ✅ Service Status Cards láthatóak
   - ✅ Online Services száma frissül
   - ✅ Running Strategies látható
   - ✅ Total Profit számok

3. **Teszteld a Service Control-t:**
   - Kattints egy service "Stop" gombjára
   - Várd meg, amíg a státusz változik
   - Kattints "Start" gombjára
   - Ellenőrizd, hogy visszaáll

4. **Teszteld az Auto-Refresh-t:**
   - Indíts/állíts le egy service-t manuálisan (terminálban)
   - 5 másodperc múlva a Dashboard automatikusan frissül

### 4.3 Frontend Build Tesztelése

```bash
cd frontend
npm run build
```

**Siker esetén:**
```
Application bundle generation complete. [4.426 seconds]

Output location: /home/tipcsy/Metatrade.Python.Framework.2/frontend/dist/frontend
```

### 4.4 Production Build Futtatása

```bash
# Build után
cd dist/frontend
python3 -m http.server 8080

# Böngészőben: http://localhost:8080
```

---

## 5. API Tesztelés curl-lel

### 5.1 Data Service

```bash
# Symbols lekérdezése
curl http://localhost:5001/symbols | jq

# OHLC adatok (ha van adat)
curl "http://localhost:5001/ohlc?symbol=EURUSD&timeframe=H1&limit=100" | jq
```

### 5.2 Strategy Service

```bash
# Stratégiák listázása
curl http://localhost:5004/strategies | jq

# Stratégia indítása (példa)
curl -X POST http://localhost:5004/strategy/start \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MA_Cross_Test",
    "strategy_type": "MA_CROSSOVER",
    "symbol": "EURUSD",
    "timeframe": "H1",
    "parameters": {}
  }' | jq
```

### 5.3 Pattern Service

```bash
# Mintázatok lekérdezése
curl http://localhost:5003/patterns | jq

# Pattern scan indítása
curl -X POST http://localhost:5003/scan/start \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "timeframe": "H1"
  }' | jq
```

### 5.4 AI Service

```bash
# Modellek listázása
curl http://localhost:5005/models | jq

# Előrejelzés (ha van betanított modell)
curl -X POST http://localhost:5005/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_123",
    "symbol": "EURUSD",
    "timeframe": "H1"
  }' | jq
```

---

## 6. End-to-End Teszt

### 6.1 Teljes Workflow Teszt

```bash
#!/bin/bash
# E2E test script

echo "🧪 End-to-End Test"

# 1. Ellenőrizd, hogy minden service fut
echo "Step 1: Health checks..."
./test_all_services.sh

# 2. Backtest indítása
echo "Step 2: Starting backtest..."
BACKTEST_ID=$(curl -s -X POST http://localhost:5006/backtest/start \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_type": "MA_CROSSOVER",
    "symbol": "EURUSD",
    "timeframe": "H1",
    "from_time": 1704067200,
    "to_time": 1706745600,
    "initial_balance": 10000.0,
    "parameters": {"fast_period": 10, "slow_period": 30}
  }' | jq -r '.data.backtest_id')

echo "Backtest ID: $BACKTEST_ID"

# 3. Várj a befejezésre
echo "Step 3: Waiting for completion..."
sleep 5

# 4. Eredmények lekérdezése
echo "Step 4: Getting results..."
curl -s http://localhost:5006/backtest/$BACKTEST_ID | jq '.data.performance'

echo "✓ E2E Test Complete!"
```

---

## 7. Troubleshooting

### 7.1 Service Nem Indul

**Probléma:** Service nem válaszol a health check-re

**Megoldás:**
```bash
# Ellenőrizd a log-ot
tail -f services/backtesting-service/backtesting-service.log

# Ellenőrizd, hogy a port szabad-e
lsof -i :5006

# Indítsd újra a service-t
pkill -f "backtesting-service/main.py"
cd services/backtesting-service && python3 main.py
```

### 7.2 Frontend Nem Indul

**Probléma:** npm start hibát dob

**Megoldás:**
```bash
# Töröld a node_modules-t és újra telepítsd
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

### 7.3 Port Foglalt

**Probléma:** "Address already in use"

**Megoldás:**
```bash
# Találd meg, mi használja a portot
lsof -i :5006

# Öld meg a folyamatot
kill -9 <PID>

# Vagy használd a stop scriptet
./stop_all_services.sh
```

### 7.4 Backtest Nem Fut

**Probléma:** Backtest "failed" státuszba kerül

**Ellenőrzés:**
```bash
# Nézd meg a részletes hibát
curl http://localhost:5006/backtest/$BACKTEST_ID | jq '.data.error_message'

# Ellenőrizd a Data Service-t
curl http://localhost:5001/health

# Ellenőrizd, hogy van-e adat
curl "http://localhost:5001/ohlc?symbol=EURUSD&timeframe=H1&limit=1" | jq
```

### 7.5 Frontend API Hívások Sikertelenül

**Probléma:** Dashboard nem tölt be adatokat

**Ellenőrzés:**
1. Nyisd meg a böngésző Developer Tools-t (F12)
2. Nézd meg a Console-t hibákért
3. Nézd meg a Network tab-ot
4. Ellenőrizd az environment.ts-ben a helyes portokat

**Javítás:**
```bash
# Ellenőrizd az environment fájlt
cat frontend/src/environments/environment.ts

# Backend API-nak futnia kell
curl http://localhost:5000/health
```

---

## 📊 Test Checklist

Használd ezt a checklistet a teljes rendszer teszteléséhez:

### Backend Services
- [ ] Backend API (5000) - Health check OK
- [ ] Data Service (5001) - Health check OK
- [ ] MT5 Service (5002) - Health check OK
- [ ] Pattern Service (5003) - Health check OK
- [ ] Strategy Service (5004) - Health check OK
- [ ] AI Service (5005) - Health check OK
- [ ] Backtesting Service (5006) - Health check OK

### Backtesting Service Funkciók
- [ ] `/strategies` endpoint működik
- [ ] `/metrics/definitions` endpoint működik
- [ ] Backtest indítható
- [ ] Backtest státusz lekérdezhető
- [ ] Backtest eredmények lekérdezhetők
- [ ] Backtest listázható
- [ ] Backtest törölhető

### Frontend
- [ ] npm install sikeres
- [ ] npm start sikeres
- [ ] Dashboard betölt
- [ ] Service status-ok láthatóak
- [ ] Service control gombok működnek
- [ ] Auto-refresh működik (5s)
- [ ] npm build sikeres
- [ ] Production build futtatható

### End-to-End
- [ ] Összes service egyszerre indítható
- [ ] Python test suite lefut hibátlanul
- [ ] Frontend kommunikál a backend-del
- [ ] Backtest eredmények megjelennek

---

## 🚀 Gyors Parancsok Összefoglalója

```bash
# Service management
./start_all_services.sh           # Összes service indítása
./stop_all_services.sh            # Összes service leállítása
./test_all_services.sh            # Health check minden service-re

# Testing
python3 test_backtesting_service.py   # Backtesting teljes teszt
./test_all_services.sh                # Gyors health check

# Frontend
cd frontend && npm start              # Dev server
cd frontend && npm run build          # Production build

# Logs
tail -f services/backtesting-service/backtesting-service.log
```

---

**Dokumentum vége**
