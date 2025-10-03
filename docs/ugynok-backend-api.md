# Backend API Service Ügynök Leírás

**Verzió:** 1.0.0
**Létrehozva:** 2025. október 3.
**Service Port:** 5000

---

## 1. Ügynök Áttekintés

### 1.1 Ügynök Neve
**Backend API Service Ügynök**

### 1.2 Fő Felelősségek

A Backend API Service Ügynök a rendszer **központi koordinátora és fő belépési pontja**. Felelős a frontend kiszolgálásáért, valamint az összes mikroszolgáltatás életciklusának kezeléséért és monitorozásáért.

### 1.3 Service Típus
- **Főszolgáltatás** (Orchestrator)
- **REST API Gateway**
- **Service Manager**
- **WebSocket Hub**

---

## 2. Részletes Felelősségek

### 2.1 Frontend API Kiszolgálás

**Mit csinál:**
- Biztosítja az összes REST API endpointot a frontend számára
- Aggregálja az adatokat több service-ből
- Egységes válaszformátumot szolgáltat
- Hibakezelést végez és felhasználóbarát üzeneteket ad

**Konkrét feladatok:**
1. Fogadja a frontend kéréseket
2. Validálja a bemeneti paramétereket
3. Továbbítja a kéréseket a megfelelő service-nek
4. Összegyűjti a válaszokat
5. Egységes formátumban visszaküldi a frontend-nek

**Példa flow:**
```
Frontend: GET /api/ticks/EURUSD?from=2025-01-01&to=2025-01-31
  ↓
Backend API: Validálja a paramétereket
  ↓
Backend API: Hívja a Data Service-t
  ↓
Data Service: Lekéri az adatokat az adatbázisból
  ↓
Backend API: Formázza a választ
  ↓
Frontend: Megkapja a tick adatokat JSON-ban
```

### 2.2 Service Orchestration (Vezénylés)

**Mit csinál:**
- Kezeli az összes mikroszolgáltatás életciklusát
- Indítja és leállítja a service-eket
- Monitorozza az állapotukat
- Újraindítja őket ha szükséges

**Konkrét feladatok:**

1. **Service Indítás**
   - Olvassa a service konfigurációt (melyik service, melyik port, milyen parancs)
   - Elindítja a service folyamatot (subprocess)
   - Vár amíg a service elérhető lesz (health check)
   - Naplózza az indítás eredményét

2. **Service Leállítás**
   - Küld egy leállítási kérést a service-nek
   - Vár a graceful shutdown-ra
   - Ha nem áll le → kényszerített leállítás
   - Naplózza a leállítás eredményét

3. **Health Monitoring**
   - 5 másodpercenként ellenőrzi minden service-t
   - Küld egy GET kérést a `/health` endpointra
   - Ha válaszol → Service online
   - Ha nem válaszol → Service offline
   - Ha offline → megpróbálja újraindítani

4. **Service Discovery**
   - Nyilvántartja az összes service címét (host + port)
   - Biztosítja, hogy a service-ek megtalálják egymást
   - Frissíti a címeket ha változnak

**Service Konfigurációs Példa:**
```
services:
  - name: data-service
    port: 5001
    command: python data-service/main.py
    auto_start: true
    restart_on_failure: true

  - name: mt5-service
    port: 5002
    command: python mt5-service/main.py
    auto_start: true
    restart_on_failure: true
```

### 2.3 WebSocket/SSE Hub

**Mit csinál:**
- Kezeli a real-time kommunikációt a frontend felé
- Fogadja az eseményeket más service-ektől
- Továbbítja az eseményeket a frontend-nek

**Esemény típusok:**

1. **Tick események**
   - Új tick érkezett (symbol, bid, ask, timestamp)
   - Forrás: Data Service

2. **Strategy jelzések**
   - Stratégia jelet adott (Buy/Sell signal)
   - Forrás: Strategy Service

3. **Position események**
   - Új pozíció nyitva
   - Pozíció módosítva
   - Pozíció zárva
   - Forrás: MT5 Service

4. **Service státusz események**
   - Service online lett
   - Service offline lett
   - Service hiba
   - Forrás: Backend API (saját monitoring)

5. **Rendszer események**
   - Gap fill elindult
   - Gap fill befejeződött
   - Backtest elindult
   - Backtest befejeződött

**WebSocket Üzenet Formátum:**
```json
{
  "type": "tick",
  "timestamp": "2025-10-03T12:34:56Z",
  "data": {
    "symbol": "EURUSD",
    "bid": 1.10523,
    "ask": 1.10525,
    "timestamp": 1696337696000
  }
}
```

### 2.4 Beállítások Kezelése

**Mit csinál:**
- Tárolja a felhasználói beállításokat
- Biztosítja a beállítások elérését más service-eknek
- Kezeli a beállítások mentését/betöltését

**Beállítás típusok:**
1. **Symbol kiválasztás** - Melyik instrumentumokat monitorozzuk
2. **Timeframe beállítások** - Melyik timeframe-eket használjuk
3. **Gap fill beállítások** - Automatikus gap fill engedélyezve?
4. **Strategy beállítások** - Aktív stratégiák
5. **Általános beállítások** - Log szint, értesítések stb.

**Adatbázis:** `setup.db`

---

## 3. REST API Endpointok

### 3.1 Service Management

#### GET /api/services/status
**Leírás:** Visszaadja az összes service állapotát

**Válasz példa:**
```json
{
  "success": true,
  "data": [
    {
      "name": "data-service",
      "status": "online",
      "port": 5001,
      "uptime": 3600,
      "last_check": "2025-10-03T12:34:56Z"
    },
    {
      "name": "mt5-service",
      "status": "offline",
      "port": 5002,
      "error": "Connection refused"
    }
  ]
}
```

#### POST /api/services/{name}/start
**Leírás:** Elindít egy service-t

**Válasz:**
```json
{
  "success": true,
  "message": "Service 'data-service' sikeresen elindítva"
}
```

#### POST /api/services/{name}/stop
**Leírás:** Leállít egy service-t

#### POST /api/services/{name}/restart
**Leírás:** Újraindít egy service-t

### 3.2 Data Endpoints

#### GET /api/ticks/{symbol}
**Paraméterek:**
- `from`: Kezdő dátum (YYYY-MM-DD)
- `to`: Befejező dátum (YYYY-MM-DD)
- `limit`: Max eredmény (opcionális)

**Működés:** Továbbítja a kérést a Data Service-nek

#### GET /api/ohlc/{symbol}/{timeframe}
**Paraméterek:**
- `from`: Kezdő dátum
- `to`: Befejező dátum

**Működés:** Továbbítja a kérést a Data Service-nek

### 3.3 Strategy Endpoints

#### GET /api/strategies
**Leírás:** Stratégiák listája

**Működés:**
1. Hívja a Strategy Service-t
2. Lekéri az összes stratégiát
3. Visszaküldi a listát

#### POST /api/strategies
**Leírás:** Új stratégia létrehozása

**Body példa:**
```json
{
  "name": "EMA Crossover",
  "type": "python",
  "code": "...",
  "symbols": ["EURUSD", "GBPUSD"],
  "timeframes": ["M15"]
}
```

**Működés:**
1. Validálja a stratégia adatokat
2. Továbbítja a Strategy Service-nek
3. Visszaküldi az új stratégia ID-ját

#### POST /api/strategies/{id}/start
**Leírás:** Stratégia indítása

**Működés:**
1. Ellenőrzi hogy a Strategy Service online-e
2. Továbbítja az indítási kérést
3. Visszajelzi a sikert

### 3.4 Pattern Endpoints

#### GET /api/patterns
**Leírás:** Pattern-ek listája

#### POST /api/patterns/scan
**Leírás:** Pattern keresés indítása

### 3.5 Settings Endpoints

#### GET /api/settings
**Leírás:** Felhasználói beállítások lekérése

**Működés:**
1. Lekéri a `setup.db` adatbázisból
2. Visszaküldi JSON-ban

#### PUT /api/settings
**Leírás:** Beállítások mentése

**Működés:**
1. Validálja a beállításokat
2. Menti a `setup.db` adatbázisba
3. Értesíti az érintett service-eket (pl. Data Service új symbol lista)

---

## 4. Service Monitoring Mechanizmus

### 4.1 Health Check Loop

**Működés:**

1. **Induláskor:**
   - Betölti a service konfigurációt
   - Elindítja az `auto_start: true` service-eket
   - Létrehoz egy background thread-et a monitorozáshoz

2. **Monitoring ciklus (5 másodpercenként):**
```
LOOP:
  FOR EACH service IN services:
    TRY:
      response = HTTP GET http://localhost:{port}/health (timeout: 2s)
      IF response.status == 200:
        service.status = "online"
        service.last_check = NOW
        service.error = null
      ELSE:
        service.status = "error"
        service.error = "Unexpected status: {response.status}"
    CATCH timeout:
      service.status = "offline"
      service.error = "Timeout"
      IF service.restart_on_failure == true:
        CALL start_service(service.name)
    CATCH connection_error:
      service.status = "offline"
      service.error = "Connection refused"
      IF service.restart_on_failure == true:
        CALL start_service(service.name)

  SLEEP 5 seconds
```

3. **Service indítás:**
```
start_service(name):
  IF service már fut:
    RETURN "Already running"

  config = get_service_config(name)
  process = subprocess.Popen(config.command)

  WAIT max 30 seconds:
    IF health_check(name) == OK:
      RETURN "Service started"

  RETURN "Service failed to start"
```

### 4.2 WebSocket Értesítések

**Service státusz változáskor:**
```
IF service.status CHANGED:
  websocket.send({
    "type": "service_status",
    "data": {
      "name": service.name,
      "status": service.status,
      "error": service.error
    }
  })
```

**Frontend megjelenítés:**
- Zöld pötty: Service online
- Piros pötty: Service offline
- Sárga pötty: Service indítás alatt
- Tooltip: Hiba üzenet ha van

---

## 5. Adatbázis

### 5.1 Adatbázis Fájl
**Fájl név:** `data/setup.db`

### 5.2 Táblák

#### 5.2.1 settings
**Leírás:** Általános beállítások

| Mező | Típus | Leírás |
|------|-------|--------|
| id | INTEGER PRIMARY KEY | Egyedi azonosító |
| key | TEXT UNIQUE | Beállítás kulcs |
| value | TEXT | Beállítás érték (JSON) |
| updated_at | TIMESTAMP | Módosítás időpontja |

**Példa adatok:**
```
id | key               | value                     | updated_at
---+-------------------+---------------------------+-------------------
1  | selected_symbols  | ["EURUSD","GBPUSD"]       | 2025-10-03 12:00
2  | timeframes        | ["M1","M5","M15","H1"]    | 2025-10-03 12:00
3  | auto_gap_fill     | true                       | 2025-10-03 12:00
```

#### 5.2.2 service_configs
**Leírás:** Service konfigurációk

| Mező | Típus | Leírás |
|------|-------|--------|
| id | INTEGER PRIMARY KEY | Egyedi azonosító |
| name | TEXT UNIQUE | Service neve |
| port | INTEGER | Port szám |
| command | TEXT | Indítási parancs |
| auto_start | INTEGER | Automatikus indítás (0/1) |
| restart_on_failure | INTEGER | Újraindítás hiba esetén (0/1) |

---

## 6. Implementációs Útmutató

### 6.1 Projekt Struktúra

```
backend-api/
├── main.py                 # Fő belépési pont
├── config.py               # Konfigurációk
├── requirements.txt        # Python függőségek
├── app/
│   ├── __init__.py
│   ├── api/                # REST API routes
│   │   ├── __init__.py
│   │   ├── services.py     # Service management endpoints
│   │   ├── data.py         # Data endpoints
│   │   ├── strategies.py   # Strategy endpoints
│   │   ├── patterns.py     # Pattern endpoints
│   │   └── settings.py     # Settings endpoints
│   ├── core/               # Core logika
│   │   ├── __init__.py
│   │   ├── service_manager.py   # Service orchestration
│   │   ├── health_monitor.py    # Health check loop
│   │   └── websocket_hub.py     # WebSocket kezelés
│   ├── models/             # Adatmodellek
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── settings.py
│   └── database/           # Adatbázis kezelés
│       ├── __init__.py
│       └── setup_db.py
└── logs/                   # Log fájlok
```

### 6.2 Technológia Választás

**Framework:** FastAPI (ajánlott)
- Gyors
- Automatikus API dokumentáció (Swagger)
- Type hints támogatás
- Async support

**WebSocket:** FastAPI beépített WebSocket support

**Adatbázis:** SQLite + SQLAlchemy (opcionális ORM)

**Dependency Injection:** FastAPI beépített DI

### 6.3 Főbb Modulok

#### 6.3.1 Service Manager
**Felelősség:** Service életciklus kezelés

**Főbb metódusok:**
- `load_service_configs()` - Konfiguráció betöltés
- `start_service(name)` - Service indítás
- `stop_service(name)` - Service leállítás
- `restart_service(name)` - Service újraindítás
- `get_service_status(name)` - Service állapot lekérés

#### 6.3.2 Health Monitor
**Felelősség:** Service health ellenőrzés

**Főbb metódusok:**
- `start_monitoring()` - Monitoring indítás
- `stop_monitoring()` - Monitoring leállítás
- `check_health(name)` - Egy service ellenőrzés
- `check_all_services()` - Minden service ellenőrzés

#### 6.3.3 WebSocket Hub
**Felelősség:** Real-time kommunikáció

**Főbb metódusok:**
- `connect(websocket)` - Kliens csatlakozás
- `disconnect(websocket)` - Kliens lecsatlakozás
- `broadcast(event)` - Üzenet küldés mindenkinek
- `send_to_client(client_id, event)` - Üzenet küldés egy kliensnek

---

## 7. Hibakezelés

### 7.1 Service Nem Elérhető

**Probléma:** Egy service offline

**Kezelés:**
1. Backend API detektálja (health check fail)
2. Megpróbálja újraindítani (ha engedélyezve)
3. Ha nem sikerül → értesíti a frontend-et
4. Frontend jelez a felhasználónak
5. API kérés esetén → fallback válasz vagy hiba üzenet

**Példa válasz:**
```json
{
  "success": false,
  "error": "Data Service jelenleg nem elérhető. Kérlek próbáld újra később.",
  "service": "data-service",
  "status": "offline"
}
```

### 7.2 Service Válaszidő Túllépés

**Probléma:** Service lassan válaszol (timeout)

**Kezelés:**
1. Timeout beállítás minden service hívásra (pl. 10s)
2. Ha timeout → hiba válasz frontend-nek
3. Backend API naplózza
4. Health monitor észleli → service restart

### 7.3 WebSocket Kapcsolat Megszakad

**Probléma:** Kliens elveszíti a WebSocket kapcsolatot

**Kezelés:**
1. Frontend automatikus újracsatlakozás
2. Backend API újra elküldi a legutóbbi állapotot
3. Naplózza a kapcsolat megszakadást

---

## 8. Teljesítmény Követelmények

### 8.1 Válaszidők

- Health check: < 100ms
- API endpoint: < 500ms (aggregált adat) vagy < 1s (nagy adat)
- WebSocket üzenet: < 50ms

### 8.2 Egyidejű Kérések

- Minimum 100 egyidejű HTTP kérés
- Minimum 50 egyidejű WebSocket kapcsolat

### 8.3 Erőforrás Használat

- CPU: < 5% idle állapotban
- Memória: < 200 MB

---

## 9. Biztonsági Megjegyzések

**Jelenlegi verzióban nincs biztonsági védelem** (lokális használat)

**Későbbi fejlesztés:**
- JWT autentikáció
- API key service-ek között
- HTTPS
- Rate limiting

---

## 10. Tesztelés

### 10.1 Unit Tesztek

- Service Manager metódusok
- Health Monitor logika
- API endpoint validáció

### 10.2 Integration Tesztek

- Service indítás/leállítás
- Health check működés
- API → Service kommunikáció

### 10.3 E2E Tesztek

- Frontend → Backend → Service → DB → válasz
- WebSocket üzenetek
- Service failure scenario

---

## 11. Deployment

### 11.1 Fejlesztési Környezet

```bash
cd backend-api
python -m venv venv
source venv/bin/activate  # Linux/Mac
# vagy
venv\Scripts\activate  # Windows

pip install -r requirements.txt
python main.py
```

### 11.2 Production Környezet

**systemd service (Linux):**
```
[Unit]
Description=Backend API Service
After=network.target

[Service]
User=trading
WorkingDirectory=/opt/mt5-platform/backend-api
ExecStart=/opt/mt5-platform/backend-api/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

**Dokumentum vége**

*Ez az ügynök leírás szolgál útmutatóként a Backend API Service implementálásához.*
