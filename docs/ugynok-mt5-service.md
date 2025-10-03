# MT5 Connection Service Ügynök Leírás

**Verzió:** 1.0.0
**Létrehozva:** 2025. október 3.
**Service Port:** 5002

---

## 1. Ügynök Áttekintés

### 1.1 Ügynök Neve
**MT5 Connection Service Ügynök**

### 1.2 Fő Felelősségek

Az MT5 Connection Service Ügynök a **kizárólagos kapcsolati pont** a MetaTrader 5 Terminal és a rendszer között. Minden MT5 kommunikáció ezen a service-en keresztül történik.

### 1.3 Service Típus
- **MT5 Gateway**
- **Trading Service**
- **Data Provider**

---

## 2. Részletes Felelősségek

### 2.1 MT5 Kapcsolat Kezelés

**Működés:**

1. **Kapcsolódás**
   - `MetaTrader5.initialize()` hívása
   - MT5 Terminal-nak futnia kell
   - Kapcsolat ellenőrzés (`terminal_info()`)
   - Ha sikertelen → újrapróbálkozás 5 másodpercenként

2. **Kapcsolat Fenntartás**
   - Periodikus kapcsolat ellenőrzés (30 másodpercenként)
   - Ha elveszett → automatikus újracsatlakozás
   - Értesítés Backend API-nak

3. **Graceful Shutdown**
   - Service leállításkor `MetaTrader5.shutdown()` hívása

### 2.2 Adatlekérés MT5-ből

**Tick Adatok:**
- `copy_ticks_range(symbol, from, to, flags)`
- Visszaad: timestamp, bid, ask, last, volume, flags

**OHLC Adatok:**
- `copy_rates_range(symbol, timeframe, from, to)`
- Visszaad: timestamp, open, high, low, close, tick_volume, spread, real_volume

**Symbol Információk:**
- `symbol_info(symbol)` - Symbol részletek
- `symbol_info_tick(symbol)` - Aktuális tick

**Account Információk:**
- `account_info()` - Balance, equity, margin stb.

### 2.3 Kereskedési Műveletek

**Pozíció Nyitás:**
```python
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": "EURUSD",
    "volume": 0.1,
    "type": mt5.ORDER_TYPE_BUY,  # vagy SELL
    "price": market_price,
    "sl": stop_loss,
    "tp": take_profit,
    "deviation": 10,
    "magic": 123456,
    "comment": "Strategy ABC",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}
result = mt5.order_send(request)
```

**Pozíció Zárás:**
```python
position = mt5.positions_get(ticket=ticket_id)[0]
close_request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": position.symbol,
    "volume": position.volume,
    "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
    "position": ticket_id,
    "price": market_price,
    "deviation": 10,
    "magic": position.magic,
    "comment": "Close position"
}
result = mt5.order_send(close_request)
```

**Pozíció Módosítás (SL/TP):**
```python
modify_request = {
    "action": mt5.TRADE_ACTION_SLTP,
    "position": ticket_id,
    "sl": new_sl,
    "tp": new_tp
}
result = mt5.order_send(modify_request)
```

### 2.4 Pozíció Lekérés

**Összes Pozíció:**
- `positions_get()` - Minden nyitott pozíció
- `positions_get(symbol="EURUSD")` - Symbol szűrés
- `positions_get(ticket=12345)` - Egy pozíció

**Pending Order-ek:**
- `orders_get()` - Pending order-ek

**History:**
- `history_deals_get(from, to)` - Üzlet történet
- `history_orders_get(from, to)` - Order történet

---

## 3. REST API Endpointok

### 3.1 Kapcsolat

#### GET /health
```json
{
  "status": "healthy",
  "mt5_connected": true,
  "terminal_info": {
    "company": "MetaQuotes Software Corp.",
    "name": "MetaTrader 5",
    "build": 3850
  }
}
```

#### POST /connect
Kapcsolódás MT5-höz

#### POST /disconnect
Kapcsolat bontás

### 3.2 Adatlekérés

#### GET /ticks/{symbol}
**Paraméterek:** from, to (Unix timestamp ms)

#### GET /rates/{symbol}/{timeframe}
**Paraméterek:** from, to
**Timeframe:** M1, M5, M15, M30, H1, H4, D1, W1, MN1

#### GET /symbol-info/{symbol}
Symbol információk (point, digits, spread stb.)

#### GET /account
Account információk

### 3.3 Kereskedés

#### POST /positions/open
**Body:**
```json
{
  "symbol": "EURUSD",
  "type": "buy",
  "volume": 0.1,
  "sl": 1.10000,
  "tp": 1.11000,
  "comment": "My trade"
}
```

**Válasz:**
```json
{
  "success": true,
  "ticket": 123456789,
  "order": 123456788,
  "volume": 0.1,
  "price": 1.10523,
  "retcode": 10009,
  "retcode_message": "Request completed"
}
```

#### POST /positions/{ticket}/close
Pozíció zárás

#### PUT /positions/{ticket}
**Body:**
```json
{
  "sl": 1.10100,
  "tp": 1.11500
}
```

#### GET /positions
Nyitott pozíciók listája

#### GET /positions/{ticket}
Egy pozíció részletei

---

## 4. Implementációs Útmutató

### 4.1 Projekt Struktúra

```
mt5-service/
├── main.py
├── config.py
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── connection.py
│   │   ├── data.py
│   │   └── trading.py
│   ├── core/
│   │   ├── mt5_manager.py      # MT5 kapcsolat kezelés
│   │   ├── data_provider.py    # Adatlekérés
│   │   └── trade_executor.py   # Kereskedési műveletek
│   └── models/
│       ├── trade_request.py
│       └── position.py
└── logs/
```

### 4.2 MT5 Manager

**Felelősség:** MT5 kapcsolat kezelés

**Főbb metódusok:**
- `connect()` - Kapcsolódás MT5-höz
- `disconnect()` - Kapcsolat bontás
- `is_connected()` - Kapcsolat ellenőrzés
- `reconnect()` - Újracsatlakozás
- `get_terminal_info()` - Terminal info

### 4.3 Data Provider

**Felelősség:** Adatlekérés MT5-ből

**Főbb metódusok:**
- `get_ticks(symbol, from, to)` - Tick adatok
- `get_rates(symbol, tf, from, to)` - OHLC adatok
- `get_symbol_info(symbol)` - Symbol info
- `get_account_info()` - Account info

### 4.4 Trade Executor

**Felelősség:** Kereskedési műveletek

**Főbb metódusok:**
- `open_position(request)` - Pozíció nyitás
- `close_position(ticket)` - Pozíció zárás
- `modify_position(ticket, sl, tp)` - Pozíció módosítás
- `get_positions()` - Pozíciók lekérés

---

## 5. Hibakezelés

### 5.1 MT5 Nem Fut
**Kezelés:** Service nem tud elindulni, health check fail

### 5.2 Kapcsolat Elveszett
**Kezelés:** Automatikus újracsatlakozás, értesítés Backend API-nak

### 5.3 Trade Hiba
**Kezelés:**
- Retcode ellenőrzés
- Ha 10009 (success) → OK
- Ha más → hiba üzenet visszaadása

---

## 6. Biztonsági Megjegyzések

**Risk Management:**
- Max pozíció méret ellenőrzés
- Max nyitott pozíciók száma
- Max napi veszteség check

---

## 7. Tesztelés

### 7.1 Unit Tesztek
- MT5 kapcsolat mock
- Trade request validáció

### 7.2 Integration Tesztek
- Valós MT5 kapcsolat (demo account)
- Pozíció nyitás/zárás teszt

---

**Dokumentum vége**
