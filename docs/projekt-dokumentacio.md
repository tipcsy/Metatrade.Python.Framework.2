# MetaTrader 5 Trading Platform 2.0 - Projekt Dokumentáció

**Verzió:** 2.0.0
**Dátum:** 2025. október 3.
**Állapot:** Tervezés

---

## 1. Projekt Áttekintés

### 1.1 Jelenlegi Helyzet

A jelenlegi MT5 Trading Platform egy **monolitikus architektúrájú** alkalmazás, amely az alábbi problémákkal küzd:

- **Teljesítmény problémák**: 10% CPU használat idle állapotban, lassú GUI válaszidők
- **Túlzott komplexitás**: Sok felesleges háttérfolyamat fut egyszerre
- **Nehéz karbantarthatóság**: Minden funkció egy kódbázisban keveredik
- **Skálázási nehézségek**: Új funkciók hozzáadása bonyolult
- **Felelősségek keveredése**: GUI, adatgyűjtés, MT5 kommunikáció, stratégiák egy helyen

### 1.2 Új Megközelítés: Mikroszolgáltatás Architektúra

A 2.0 verzió **tiszta, egyszerű, moduláris** rendszert valósít meg, ahol:

- Minden felelősségi kör **külön service-ben** él
- Service-ek **REST API-n** kommunikálnak egymással
- **Egyszerű deployment**: külön Python folyamatok, külön portok
- **Skálázható**: bármelyik service külön fejleszthető/frissíthető
- **Karbantartható**: tiszta felelősségi körök

### 1.3 Fő Célok

1. **Teljesítmény**: Minimális CPU/memória használat idle állapotban (~0%)
2. **Egyszerűség**: Nincs felesleges complexity (Docker, Kubernetes, message queue)
3. **Gyors válaszidők**: Frontend mindig responsív
4. **Modularitás**: Service-ek függetlenül fejleszthetők
5. **Tiszta architektúra**: Minden service egy felelősség

---

## 2. Rendszer Architektúra

### 2.1 Architektúra Áttekintés

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Angular)                        │
│                     http://localhost:4200                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ HTTP/REST + WebSocket/SSE
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              BACKEND API SERVICE (Orchestrator)                  │
│                     http://localhost:5000                        │
│  - Frontend API endpoints                                        │
│  - Service orchestration (start/stop/monitor)                   │
│  - Service health check                                          │
└─────────────────────────────────────────────────────────────────┘
          │            │            │            │            │            │
          │ REST       │ REST       │ REST       │ REST       │ REST       │ REST
          ▼            ▼            ▼            ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Data Service │ │ MT5 Service  │ │Pattern Svc   │ │Strategy Svc  │ │  AI Service  │ │Backtest Svc  │
│  Port: 5001  │ │  Port: 5002  │ │  Port: 5003  │ │  Port: 5004  │ │  Port: 5005  │ │  Port: 5006  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
       │                │                                    │                                │
       │ DB             │ MT5                                │ DB                             │ DB (read)
       ▼                ▼                                    ▼                                ▼
┌──────────────┐ ┌──────────────┐                    ┌──────────────┐            ┌──────────────┐
│   SQLite     │ │ MetaTrader 5 │                    │ TensorFlow   │            │   SQLite     │
│  Databases   │ │   Terminal   │                    │    Models    │            │  (History)   │
└──────────────┘ └──────────────┘                    └──────────────┘            └──────────────┘
```

### 2.2 Service-ek Áttekintése

| Service | Port | Felelősség | Függőségek |
|---------|------|------------|------------|
| **Backend API** | 5000 | Frontend kiszolgálás, orchestration | Minden service |
| **Data Service** | 5001 | Adatgyűjtés, gap fill, completeness | MT5 Service |
| **MT5 Service** | 5002 | MT5 kommunikáció, kereskedés | MetaTrader 5 |
| **Pattern Service** | 5003 | Indikátorok, pattern felismerés | Data Service |
| **Strategy Service** | 5004 | Stratégiák, pozíciókezelés, live trading | Pattern, Data, MT5 |
| **AI Service** | 5005 | Idősor elemzés, előrejelzés | Data, Strategy |
| **Backtesting Service** | 5006 | Historikus backtest, szimulált idő | Data Service |

### 2.3 Kommunikációs Flow Példák

#### Példa 1: Tick adat megjelenítése a UI-on
```
1. Data Service gyűjti a tick-et MT5-ből (OnFly)
2. Data Service menti DB-be
3. Data Service WebSocket üzenetet küld Backend API-nak
4. Backend API továbbítja WebSocket-en a Frontend-nek
5. Frontend frissíti a megjelenítést
```

#### Példa 2: Stratégia futtatás
```
1. Frontend: Stratégia indítása (POST /api/strategies/{id}/start)
2. Backend API: Továbbítja Strategy Service-nek
3. Strategy Service: Lekéri pattern adatokat Pattern Service-től
4. Pattern Service: Számítja az indikátorokat (EMA, RSI stb.)
5. Strategy Service: Kiértékeli a szabályokat
6. Strategy Service: Ha jel → pozíció nyitás MT5 Service-en keresztül
```

#### Példa 3: Gap fill induláskor
```
1. Backend API Service elindul
2. Backend API: Ellenőrzi Data Service állapotát (health check)
3. Ha Data Service offline → Backend API elindítja
4. Data Service: Automatikus gap fill futtatása
5. Backend API: Jelzi Frontend-nek a folyamat állapotát
```

---

## 3. Service Részletes Leírások

### 3.1 Backend API Service (Főszolgáltatás)

**Port:** 5000
**Technológia:** Python + Flask vagy FastAPI
**Adatbázis:** SQLite (setup.db - felhasználói beállítások)

#### Felelősségek:

1. **Frontend API**
   - REST endpoints a UI számára
   - Felhasználói beállítások mentése/lekérése
   - Adatok aggregálása több service-ből

2. **Service Orchestration**
   - Service-ek indítása/leállítása
   - Health check minden service-re (5 másodpercenként)
   - Service státusz monitoring
   - Automatikus újraindítás ha service leáll

3. **WebSocket/SSE Hub**
   - Real-time adatok továbbítása frontend-nek
   - Tick események
   - Stratégia jelzések
   - Rendszer események

#### REST API Endpointok (Példák):

**Service Management:**
- `GET /api/services/status` - Összes service állapota
- `POST /api/services/{name}/start` - Service indítása
- `POST /api/services/{name}/stop` - Service leállítása
- `POST /api/services/{name}/restart` - Service újraindítása

**Data:**
- `GET /api/ticks/{symbol}?from={}&to={}` - Tick adatok lekérése
- `GET /api/ticks/{symbol}/last` - Az utolsó Tick adato lekérése
- `GET /api/ohlc/{symbol}/{timeframe}?from={}&to={}` - OHLC adatok
- `GET /api/ohlc/{symbol}/{timeframe}/lastclosed` - OHLC adatból elkéri az utolsó zártat


**Strategies:**
- `GET /api/strategies` - Stratégiák listája
- `POST /api/strategies` - Új stratégia létrehozása
- `PUT /api/strategies/{id}` - Stratégia módosítása
- `POST /api/strategies/{id}/start` - Stratégia indítása
- `POST /api/strategies/{id}/stop` - Stratégia leállítása

**Patterns:**
- `GET /api/patterns` - Pattern-ek listája
- `POST /api/patterns/scan` - Pattern keresés indítása

**Settings:**
- `GET /api/settings` - Beállítások lekérése
- `PUT /api/settings` - Beállítások mentése

**WebSocket:**
- `WS /ws/ticks` - Real-time tick adatok
- `WS /ws/signals` - Stratégia jelzések
- `WS /ws/events` - Rendszer események

#### Service Monitor Működése:

```
1. Backend API Service elindul
2. Betölti a service konfigurációt (melyik service, melyik port)
3. 5 másodpercenként ellenőrzi minden service-t:
   - GET http://localhost:{port}/health
   - Ha válaszol → Online
   - Ha nem válaszol → Offline
4. Ha service offline:
   - Backend API megpróbálja elindítani (subprocess.Popen)
   - Frontend-et értesíti (WebSocket)
5. Frontend-en jelzés:
   - Zöld: Service online
   - Piros: Service offline
   - Sárga: Service indítás alatt
```

---

### 3.2 Data Service

**Port:** 5001
**Technológia:** Python
**Adatbázis:** SQLite (ticks, OHLC, completeness táblák)

#### Felelősségek:

1. **Gap Fill**
   - Induláskor automatikus futtatás
   - Hiányzó tick/OHLC adatok pótlása
   - Completeness analízis

2. **Előzmény Letöltés**
   - Manuális letöltés (Frontend-ről indítva)
   - Dátum tartomány letöltése
   - Progress reporting

3. **OnFly Adatgyűjtés**
   - Real-time tick gyűjtés MT5-ből (100ms)
   - Batch DB írás (100-1000 tick)
   - Real-time OHLC candle monitoring

4. **Adatbázis Kezelés**
   - Tick táblák (symbol-alapú particionálás)
   - OHLC táblák (symbol + timeframe)
   - Completeness táblák
   - Monitoring táblák

#### REST API Endpointok:

- `GET /health` - Service health check
- `POST /gap-fill` - Gap fill indítása (manuális)
- `POST /download-history` - Előzmény letöltés
- `GET /download-status` - Letöltés állapot
- `GET /statistics` - Adatgyűjtés statisztika
- `POST /start-collection` - OnFly gyűjtés indítása
- `POST /stop-collection` - OnFly gyűjtés leállítása

#### Automatikus Gap Fill:

```
1. Data Service elindul
2. Ellenőrzi a kiválasztott symbol-ok adatbázisait
3. Meghatározza az utolsó mentett tick/OHLC időpontot
4. Letölti a hiányzó adatokat MT5 Service-en keresztül
5. Jelenti az előrehaladást Backend API-nak (WebSocket)
6. Ha kész → OnFly gyűjtés indul
```

---

### 3.3 MT5 Connection Service

**Port:** 5002
**Technológia:** Python + MetaTrader5 library
**Adatbázis:** Nincs (stateless)

#### Felelősségek:

1. **MT5 Kapcsolat Kezelés**
   - Kapcsolódás MetaTrader 5 Terminal-hoz
   - Kapcsolat fenntartás
   - Újracsatlakozás hiba esetén

2. **Adatlekérés**
   - Tick adatok lekérése (`copy_ticks_range`)
   - OHLC adatok lekérése (`copy_rates_range`)
   - Symbol információk
   - Account információk

3. **Kereskedés**
   - Pozíció nyitás
   - Pozíció zárás
   - Módosítás (stop loss, take profit)
   - Pending order kezelés

4. **MT5 Állapot**
   - Account balance
   - Equity
   - Margin
   - Open positions

#### REST API Endpointok:

- `GET /health` - Service + MT5 kapcsolat állapot
- `POST /connect` - MT5 kapcsolat létrehozása
- `POST /disconnect` - MT5 kapcsolat bontása
- `GET /account` - Account információk
- `GET /ticks/{symbol}?from={}&to={}` - Tick adatok
- `GET /rates/{symbol}/{timeframe}?from={}&to={}` - OHLC adatok
- `GET /positions` - Nyitott pozíciók
- `POST /positions/open` - Pozíció nyitás
- `POST /positions/{ticket}/close` - Pozíció zárás
- `PUT /positions/{ticket}` - Pozíció módosítás

#### MT5 Kapcsolat Működése:

```
1. Service elindul
2. MetaTrader5.initialize() hívása
3. Kapcsolat ellenőrzés (terminal_info())
4. Ha sikeres → állapot: Connected
5. Ha sikertelen → újrapróbálkozás 5 másodpercenként
6. Backend API folyamatosan ellenőrzi (health check)
```

---

### 3.4 Pattern & Indicator Service

**Port:** 5003
**Technológia:** Python + pandas + numpy
**Adatbázis:** SQLite (pattern_definitions.db)

#### Felelősségek:

1. **Technikai Indikátorok**
   - Mozgóátlagok (SMA, EMA)
   - Oszcillátorok (RSI, Stochastic, MACD)
   - Trendindikátorok (ADX, Aroon)
   - Volatility (ATR, Bollinger Bands)

2. **Pattern Felismerés**
   - Candlestick pattern-ek (Doji, Hammer, Engulfing stb.)
   - Chart pattern-ek (Head & Shoulders, Triangle, Channel)
   - Custom pattern-ek

3. **Pattern Futtatás**
   - Real-time pattern scanning minden symbol/timeframe-re
   - Jelzés küldése ha pattern megjelenik

4. **Pattern Kezelés**
   - Pattern definíció betöltése Python fájlból
   - Pattern lista
   - Pattern engedélyezés/tiltás

#### REST API Endpointok:

- `GET /health` - Service állapot
- `GET /patterns` - Összes pattern listája
- `POST /patterns` - Új pattern feltöltése (Python kód)
- `GET /patterns/{id}` - Pattern részletei
- `PUT /patterns/{id}` - Pattern módosítása
- `DELETE /patterns/{id}` - Pattern törlése
- `POST /patterns/{id}/enable` - Pattern engedélyezése
- `POST /patterns/{id}/disable` - Pattern tiltása
- `POST /scan` - Pattern keresés indítása
- `GET /indicators/{symbol}/{timeframe}` - Indikátor értékek

#### Pattern Definíció Formátum (Python fájl):

```
Egy Python fájl tartalmaz egy pattern osztályt:
- Név, leírás
- Szükséges indikátorok
- detect() metódus → True/False
- get_signal() metódus → Buy/Sell/None
```

**Későbbi fejlesztés:** MQL4/5 fájlok beolvasása és Python-ra fordítása

---

### 3.5 Strategy Service

**Port:** 5004
**Technológia:** Python + pandas
**Adatbázis:** SQLite (strategies.db, backtest_results.db)

#### Felelősségek:

1. **Stratégia Futtatás**
   - Pattern-ek láncolása (pl: "EMA crossover ÉS RSI < 30")
   - Real-time kiértékelés
   - Paper trading (demo)
   - Live trading

2. **Backtesting**
   - Historikus adat alapú teszt
   - Teljesítmény metrikák (profit, drawdown, win rate)
   - Grafikus eredmények generálása

3. **Stratégia Kezelés**
   - Stratégia definíció Python fájlból vagy UI drag-and-drop
   - Stratégia mentés/betöltés
   - Stratégia módosítás

4. **Pozíció Menedzsment**
   - Stop loss / Take profit kezelés
   - Trailing stop
   - Részleges zárás

#### REST API Endpointok:

- `GET /health` - Service állapot
- `GET /strategies` - Stratégiák listája
- `POST /strategies` - Új stratégia létrehozása
- `GET /strategies/{id}` - Stratégia részletei
- `PUT /strategies/{id}` - Stratégia módosítása
- `DELETE /strategies/{id}` - Stratégia törlése
- `POST /strategies/{id}/start` - Stratégia indítása (live/paper)
- `POST /strategies/{id}/stop` - Stratégia leállítása
- `POST /strategies/{id}/backtest` - Backtest futtatása
- `GET /strategies/{id}/performance` - Teljesítmény adatok
- `GET /backtests` - Backtest eredmények listája

#### Stratégia Definíció:

**Opció 1: Python fájl**
```
Egy Python fájl tartalmaz egy stratégia osztályt:
- Név, leírás
- Használt pattern-ek/indikátorok
- on_tick() metódus
- Entry logika
- Exit logika
```

**Opció 2: UI Drag-and-Drop**
```
Frontend-en vizuálisan összeépített láncolat:
- Blokkok: Pattern, Indikátor, Feltétel, Akció
- JSON formátumban mentve
- Strategy Service futásidőben értelmezi
```

---

### 3.6 AI Service

**Port:** 5005
**Technológia:** Python + TensorFlow
**Adatbázis:** SQLite (model_metadata.db)

#### Felelősségek:

1. **Idősor Elemzés**
   - LSTM/GRU modellek idősor előrejelzésre
   - Régi adatokból jövőbeli ár előrejelzés

2. **Stratégia Optimalizáció**
   - Stratégia paraméterek optimalizálása
   - Backtesting eredmények elemzése

3. **Model Kezelés**
   - Model training
   - Model mentés/betöltés
   - Model verziókezelés

4. **Inference**
   - Real-time előrejelzés
   - Confidence score

#### REST API Endpointok:

- `GET /health` - Service állapot
- `GET /models` - Modellek listája
- `POST /models/train` - Model tanítás
- `GET /models/{id}` - Model részletei
- `POST /models/{id}/predict` - Előrejelzés
- `GET /models/{id}/performance` - Model teljesítmény

**Megjegyzés:** Az AI Service a későbbi fázisokban kerül implementálásra.

---

### 3.7 Backtesting Service

**Port:** 5006
**Technológia:** Python + pandas
**Adatbázis:** SQLite (backtest_results.db - írás/olvasás, tick/ohlc DB-k - read only)

#### Felelősségek:

1. **Szimulált Idő (Time Machine)**
   - Historikus adatok "visszajátszása" szimulált időben
   - Event-driven backtest (bar-by-bar vagy tick-by-tick)
   - No look-ahead bias

2. **Virtuális Pozíció Kezelés**
   - Virtuális pozíciók (nem éri el az MT5-öt)
   - SL/TP/Trailing Stop szimuláció
   - Spread és commission figyelembevétele

3. **Teljesítmény Metrikák**
   - Total Profit, Win Rate, Profit Factor
   - Max Drawdown, Sharpe Ratio
   - Trade log, Equity curve generálás

4. **Párhuzamos Backtesting**
   - Parameter sweep (több paraméter kombináció tesztelése)
   - Walk-forward analysis

**FONTOS:** A Backtesting Service **külön service**, mert:
- Erőforrás szeparáció (CPU-igényes, ne lassítsa a live kereskedést)
- Párhuzamos futtatás (több backtest egyszerre)
- Tiszta architektúra (külön felelősségi kör)

#### REST API Endpointok:

- `GET /health` - Service állapot
- `POST /backtest/start` - Backtest indítás
- `POST /backtest/batch` - Batch backtest (parameter sweep)
- `GET /backtest/{id}/status` - Backtest állapot
- `GET /backtest/{id}/results` - Backtest eredmények
- `POST /backtest/{id}/stop` - Backtest leállítás
- `DELETE /backtest/{id}` - Backtest törlés
- `GET /backtest/list` - Összes backtest listája

#### Backtest Flow:

```
1. Frontend: Backtest indítás kérés
2. Backtesting Service: Betölti a strategiát
3. Backtesting Service: Betölti historikus OHLC/Tick adatokat (Data Service DB-ből)
4. Időgép: 2024-01-04-től 2024-12-31-ig "szalad"
5. Minden bar-nál:
   - Indikátorok számítása
   - Stratégia logika futtatása
   - Pozíció nyitás/zárás (virtuális)
   - Teljesítmény számítás
6. Végén: Összesített metrikák, trade log, equity curve
7. Eredmény visszaadása Frontend-nek
```

---

## 4. Adatbázis Struktúra

### 4.1 Adatbázis Fájlok

**Közös adatbázisok (database/ mappa):**

```
database/
├── 2025/
│   ├── EURUSD_ticks_01.db           # Symbol-alapú tick particionálás havonta új db
│   ├── EURUSD_ticks_02.db
│   ├── EURUSD_ohlc.db               # Symbol-alapú OHLC
│   ├── GBPUSD_ticks_2025_01.db
│   ├── GBPUSD_ohlc.db
│   └── completeness.db   # Completeness tracking
```

**Service-specifikus adatbázisok:**

```
database/
├── setup.db                  # Backend API - Felhasználói beállítások
├── pattern_definitions.db    # Pattern Service - Pattern definíciók
├── strategies.db             # Strategy Service - Stratégiák
├── backtest_results.db       # Strategy Service - Backtest eredmények
└── model_metadata.db         # AI Service - ML model metadata
```

### 4.2 Tick Adatbázis Struktúra

**Fájl formátum:** `{SYMBOL}_ticks_{MONTH}.db`

**Tábla:** `ticks`

| Mező | Típus | Leírás |
|------|-------|--------|
| id | INTEGER PRIMARY KEY | Egyedi azonosító |
| symbol | TEXT | Instrumentum neve |
| timestamp | INTEGER | Unix timestamp (milliszekundum) |
| date_readable| TEXT | Csak a dátumot tartalmazza: 2025-09-03 |
| bid | REAL | Bid ár |
| ask | REAL | Ask ár |
| last | REAL | Utolsó ár |
| volume | INTEGER | Volumen |
| flags | INTEGER | MT5 flags |

**Index:** `CREATE INDEX idx_timestamp ON ticks(timestamp)`

### 4.3 OHLC Adatbázis Struktúra

**Fájl formátum:** `{SYMBOL}_ohlc.db`

**Tábla:** `ohlc_data`

| Mező | Típus | Leírás |
|------|-------|--------|
| id | INTEGER PRIMARY KEY | Egyedi azonosító |
| symbol | TEXT | Instrumentum neve |
| timeframe | TEXT | Timeframe (M1, M5, H1 stb.) |
| timestamp | INTEGER | Bar kezdési idő (Unix timestamp (milliszekundum)) |
| open | REAL | Nyitó ár |
| high | REAL | Legmagasabb ár |
| low | REAL | Legalacsonyabb ár |
| close | REAL | Záró ár |
| tick_volume | INTEGER | Tick volumen |
| spread | INTEGER | Spread |
| real_volume | INTEGER | Valós volumen |
| is_closed | INTEGER | Bar lezárva (0/1) |

**Index:**
- `CREATE INDEX idx_symbol_timeframe_timestamp ON ohlc_data(symbol, timeframe, timestamp)`

### 4.4 Completeness Adatbázis

**Fájl:** `completeness.db`

**Táblák:**

1. **tick_data_completeness**

| Mező | Típus | Leírás |
|------|-------|--------|
| id | INTEGER PRIMARY KEY | Egyedi azonosító |
| symbol | TEXT | Instrumentum |
| date_readable | TEXT | Dátum (YYYY-MM-DD) |
| status | TEXT | COMPLETE / PARTIAL / EMPTY |
| record_count | INTEGER | Tick-ek száma |
| first_timestamp | INTEGER | Első tick ideje (Unix timestamp (milliszekundum)) |
| last_timestamp | INTEGER | Utolsó tick ideje (Unix timestamp (milliszekundum)) |
| last_analyzed | TIMESTAMP | Utolsó elemzés időpontja |

2. **ohlc_data_completeness**

| Mező | Típus | Leírás |
|------|-------|--------|
| id | INTEGER PRIMARY KEY | Egyedi azonosító |
| symbol | TEXT | Instrumentum |
| timeframe | TEXT | Timeframe |
| date_readable | TEXT | Dátum (YYYY-MM-DD) |
| status | TEXT | COMPLETE / PARTIAL / EMPTY |
| record_count | INTEGER | Bárok száma |
| expected_records | INTEGER | Elvárt bárok száma |
| last_analyzed | TIMESTAMP | Utolsó elemzés időpontja |

### 4.5 Strategy Adatbázis

**Fájl:** `strategies.db`

**Tábla:** `strategies`

| Mező | Típus | Leírás |
|------|-------|--------|
| id | INTEGER PRIMARY KEY | Stratégia ID |
| name | TEXT | Stratégia neve |
| description | TEXT | Leírás |
| type | TEXT | python / drag_drop |
| code | TEXT | Python kód vagy JSON |
| symbols | TEXT | Symbol lista (JSON) |
| timeframes | TEXT | Timeframe lista (JSON) |
| status | TEXT | ACTIVE / INACTIVE |
| mode | TEXT | LIVE / PAPER / BACKTEST |
| created_at | TIMESTAMP | Létrehozás időpontja |
| updated_at | TIMESTAMP | Módosítás időpontja |

---

## 5. Implementációs Lépések

### 5.1 Fázis 1: Alapinfrastruktúra (2-3 hét)

**Cél:** Service-ek alapvető váza és kommunikáció

**Feladatok:**

1. **Projekt struktúra létrehozása**
   - Új projekt mappa: `mt5-trading-platform-v2/`
   - Service mappák: `backend-api/`, `data-service/`, `mt5-service/` stb.
   - Közös könyvtár: `shared/` (közös utility-k, modellek)

2. **Backend API Service alapok**
   - Flask/FastAPI projekt inicializálása
   - Health check endpoint
   - Service discovery mechanizmus
   - Logging beállítása

3. **Service-ek skeleton létrehozása**
   - Minden service: alapvető flask app + health endpoint
   - Port konfigurálás (5000-5005)
   - Service indítható és válaszol health check-re

4. **Service Orchestration**
   - Backend API tudja indítani/leállítani a service-eket
   - Health monitoring (5 másodpercenként)
   - Service status API endpoint

5. **Adatbázis migráció tervezése**
   - Meglévő SQLite struktúra dokumentálása
   - Migráció script terv

**Tesztpont:** Minden service elindul külön, Backend API látja őket, health check működik

---

### 5.2 Fázis 2: Data Service + MT5 Service (3-4 hét)

**Cél:** Adatgyűjtés és MT5 kommunikáció működik

**Feladatok:**

1. **MT5 Service implementálás**
   - MT5 kapcsolat kezelés
   - Tick/OHLC lekérés endpointok
   - Position management endpointok
   - Újracsatlakozás logika

2. **Data Service - Adatbázis kezelés**
   - Meglévő SQLite struktúra átmásolása
   - Tick storage implementation
   - OHLC storage implementation
   - Completeness tracking

3. **Gap Fill implementálás**
   - Indulási gap fill logika
   - Hiányzó adatok detektálása
   - MT5 Service hívása adatlekéréshez
   - Progress reporting

4. **OnFly gyűjtés**
   - Real-time tick collection (100ms)
   - Batch DB írás
   - OHLC candle monitoring
   - WebSocket üzenetek Backend API-nak

5. **REST API endpointok**
   - Data Service összes endpointja
   - MT5 Service összes endpointja

**Tesztpont:**
- Gap fill fut és tölti a DB-t
- OnFly gyűjtés működik
- Backend API-n keresztül lekérdezhető az adat

---

### 5.3 Fázis 3: Pattern & Indicator Service (2-3 hét)

**Cél:** Technikai elemzés és pattern felismerés

**Feladatok:**

1. **Indikátor library**
   - Mozgóátlagok (SMA, EMA, WMA)
   - Oszcillátorok (RSI, Stochastic, MACD)
   - Trendindikátorok (ADX, Aroon)
   - Volatility (ATR, Bollinger Bands)

2. **Candlestick pattern-ek**
   - Doji, Hammer, Shooting Star
   - Engulfing, Harami
   - Morning/Evening Star

3. **Chart pattern-ek**
   - Support/Resistance
   - Trendlines
   - Head & Shoulders
   - Triangle, Channel

4. **Pattern definition framework**
   - Python fájl betöltés
   - Pattern validation
   - Pattern execution engine

5. **Real-time pattern scanning**
   - Periodikus scan (symbol/timeframe kombinációkra)
   - Jelzés generálás
   - WebSocket notification Backend API-nak

**Tesztpont:**
- Indikátorok számíthatók
- Pattern-ek felismerhetők
- Jelzés érkezik ha pattern megjelenik

---

### 5.4 Fázis 4: Strategy Service + Backtesting (3-4 hét)

**Cél:** Stratégiák futtatása és tesztelése

**Feladatok:**

1. **Stratégia engine**
   - Pattern láncolás mechanizmus
   - Entry/Exit logika kiértékelés
   - Position management

2. **Backtesting framework**
   - Historikus adat betöltés
   - Event-driven backtest
   - Teljesítmény metrikák (profit, drawdown, win rate)
   - Trade log generálás

3. **Paper trading**
   - Virtuális pozíció menedzsment
   - Real-time stratégia futtatás
   - Trade log

4. **Live trading**
   - MT5 Service pozíció nyitás/zárás hívása
   - Risk management (max pozíció, max veszteség)
   - Emergency stop

5. **Stratégia management**
   - Python fájl betöltés
   - Drag-and-drop JSON parser
   - Stratégia lista/módosítás/törlés

**Tesztpont:**
- Backtest fut és teljesítmény metrikákat ad
- Paper trading működik
- Live trading tesztelése kis összeggel

---

### 5.5 Fázis 5: Frontend (Angular) (4-5 hét)

**Cél:** Működő Angular frontend

**Feladatok:**

1. **Angular projekt setup**
   - Angular 17+ projekt
   - Material Design
   - Routing

2. **Dashboard**
   - Service status megjelenítés
   - Account info
   - Nyitott pozíciók

3. **Data Management**
   - Gap fill indítás
   - Előzmény letöltés
   - Progress bar

4. **Pattern Management**
   - Pattern lista
   - Pattern feltöltés
   - Pattern engedélyezés/tiltás

5. **Strategy Management**
   - Stratégia lista
   - Stratégia szerkesztés (kód editor)
   - Drag-and-drop strategy builder (későbbi verzió)
   - Backtest indítás és eredmények
   - Live/Paper toggle

6. **Real-time adatok**
   - WebSocket kapcsolat Backend API-hoz
   - Tick/Signal események megjelenítése
   - Pozíció státusz frissítés

7. **Settings**
   - Symbol kiválasztás
   - Service beállítások
   - Általános beállítások

**Tesztpont:**
- Frontend elérhető böngészőben
- Backend API-hoz csatlakozik
- Minden funkció elérhető és működik

---

### 5.6 Fázis 6: AI Service + Finomhangolás (3-4 hét)

**Cél:** AI képességek és rendszer optimalizáció

**Feladatok:**

1. **AI Service alapok**
   - TensorFlow integráció
   - LSTM/GRU model architektúra
   - Training pipeline

2. **Idősor előrejelzés**
   - Adat előkészítés
   - Model training
   - Inference API

3. **Stratégia optimalizáció**
   - Paraméter grid search
   - Genetikus algoritmus

4. **Integration**
   - Strategy Service AI hívásai
   - AI jel használata stratégiákban

5. **Teljesítmény optimalizáció**
   - Profiling minden service-re
   - Bottleneck-ek azonosítása
   - Cache mechanizmusok

6. **Dokumentáció**
   - Felhasználói kézikönyv
   - API dokumentáció
   - Fejlesztői dokumentáció

**Tesztpont:**
- AI model tud előrejelzést adni
- Teljes rendszer stabil és gyors
- Dokumentáció teljes

---

## 6. Technológiai Stack

### 6.1 Backend (Python)

**Alap framework:**
- **Flask** vagy **FastAPI** - REST API framework
- Flask: Egyszerűbb, jól dokumentált
- FastAPI: Gyorsabb, automatikus API dokumentáció (Swagger)

**Ajánlás:** FastAPI (modern, gyors, type hints)

**Könyvtárak:**
- `MetaTrader5` - MT5 kapcsolat
- `pandas` - Adatelemzés
- `numpy` - Matematikai műveletek
- `SQLAlchemy` (opcionális) - ORM ha kell
- `websockets` vagy `socketio` - Real-time kommunikáció
- `requests` - HTTP kliens (service-ek közötti hívás)
- `pydantic` - Adatvalidáció
- `python-dotenv` - Környezeti változók

### 6.2 Frontend (Angular)

**Framework:**
- Angular 17+ (legújabb verzió)

**UI Library:**
- Angular Material - Material Design komponensek

**Könyvtárak:**
- `rxjs` - Reactive programming (beépített)
- `socket.io-client` - WebSocket kommunikáció
- `chart.js` vagy `lightweight-charts` (ha kell chart)
- `monaco-editor` - Kód szerkesztő (stratégia szerkesztéshez)

### 6.3 Adatbázis

**SQLite 3**
- Meglévő struktúra megtartása
- Fájl alapú, egyszerű
- Nincs külön DB szerver

**Opcionális jövőbeli átállás:**
- PostgreSQL (ha skálázás kell)

### 6.4 AI/ML

**TensorFlow 2.x**
- Idősor modellek (LSTM, GRU)
- Keras API (egyszerűbb)

**Alternatíva:**
- PyTorch (rugalmasabb, kutatási célokra)

### 6.5 Kommunikáció

**REST API:**
- JSON formátum
- HTTP/HTTPS

**Real-time:**
- **WebSocket** (ajánlott - kétirányú)
- vagy **SSE** (Server-Sent Events - egyirányú)

### 6.6 Deployment

**Fejlesztési környezet:**
- Python virtuális környezet (venv)
- Minden service külön terminal ablak

**Későbbi production:**
- Docker (opcionális)
- systemd (Linux service)
- PM2 (Node.js process manager - Python-hoz is használható)

---

## 7. Fejlesztési Irányelvek

### 7.1 Kódolási Szabályok

**Python:**
- PEP 8 stíluskövetés
- Type hints használata
- Docstring minden függvényhez
- Max 80-100 karakter/sor

**Példa:**
```python
def calculate_ema(prices: List[float], period: int) -> List[float]:
    """
    Exponenciális mozgóátlag számítása.

    Args:
        prices: Árfolyamok listája
        period: EMA periódus

    Returns:
        EMA értékek listája
    """
    # Implementáció
    pass
```

**Angular/TypeScript:**
- Angular style guide követése
- Strong typing
- Component/Service szétválasztás
- Reactive programming (RxJS)

### 7.2 REST API Design

**Naming convention:**
- Főnevek használata (pl. `/strategies` nem `/getStrategies`)
- Hierarchia: `/api/strategies/{id}/backtest`
- Query paraméterek: `/api/ticks?symbol=EURUSD&from=2025-01-01`

**HTTP Methods:**
- `GET` - Lekérés
- `POST` - Létrehozás
- `PUT` - Teljes módosítás
- `PATCH` - Részleges módosítás
- `DELETE` - Törlés

**Status kódok:**
- 200 OK - Sikeres lekérés
- 201 Created - Sikeres létrehozás
- 400 Bad Request - Hibás kérés
- 404 Not Found - Nem található
- 500 Internal Server Error - Szerver hiba

**Response formátum:**
```json
{
  "success": true,
  "data": { /* adat */ },
  "message": "Sikeres művelet",
  "timestamp": "2025-10-03T12:34:56Z"
}
```

### 7.3 Hibakezelés

**Try-Catch minden hívásban:**
```python
try:
    result = do_something()
    return {"success": True, "data": result}
except Exception as e:
    logger.error(f"Hiba: {e}")
    return {"success": False, "error": str(e)}
```

**Service-ek közötti hívás:**
- Timeout beállítás (5-10 másodperc)
- Újrapróbálkozás (3x)
- Fallback érték ha service offline

### 7.4 Logging

**Log szintek:**
- `DEBUG` - Részletes debug info
- `INFO` - Normál működés
- `WARNING` - Figyelmeztetés
- `ERROR` - Hiba
- `CRITICAL` - Kritikus hiba

**Log formátum:**
```
[2025-10-03 12:34:56] [INFO] [DataService] Gap fill started for EURUSD
```

**Log fájlok:**
```
logs/
├── backend-api.log
├── data-service.log
├── mt5-service.log
├── pattern-service.log
├── strategy-service.log
└── ai-service.log
```

### 7.5 Tesztelés

**Unit tesztek:**
- Minden üzleti logika függvényhez
- Python: `pytest`
- Angular: `Jasmine + Karma`

**Integration tesztek:**
- Service-ek közötti kommunikáció
- REST API endpointok

**E2E tesztek:**
- Frontend → Backend → Service flow
- Angular: `Cypress` vagy `Playwright`

---

## 8. Service Lifecycle Management

### 8.1 Service Indítás és Leállítás

**Cél:** Egységes és megbízható service indítási/leállítási mechanizmus

#### 8.1.1 Program Induláskor Automatikus Service Indítás

**Működés:**

```
1. Felhasználó elindítja a főprogramot (Backend API Service)
2. Backend API betölti a config.json-t
3. Config alapján eldönti, mely service-eket kell automatikusan indítani:
   - Data Service (auto-start: true, gap fill: true)
   - MT5 Service (auto-start: true)
   - Pattern Service (auto-start: true)
   - Strategy Service (auto-start: false - manuális)
   - Backtesting Service (auto-start: false - on-demand)
   - AI Service (auto-start: false - on-demand)
4. Backend API indítja a service-eket (subprocess.Popen)
5. Health check várja amíg a service-ek online lesznek
6. Ha valamelyik nem indul el → retry 3x, utána error notification
7. Frontend WebSocket-en értesítést kap minden service státusz változásról
```

**Config példa (config.json):**
```json
{
  "services": {
    "data-service": {
      "auto_start": true,
      "port": 5001,
      "path": "services/data-service/main.py",
      "gap_fill_on_start": true
    },
    "mt5-service": {
      "auto_start": true,
      "port": 5002,
      "path": "services/mt5-service/main.py"
    },
    "pattern-service": {
      "auto_start": true,
      "port": 5003,
      "path": "services/pattern-service/main.py"
    },
    "strategy-service": {
      "auto_start": false,
      "port": 5004,
      "path": "services/strategy-service/main.py"
    },
    "backtesting-service": {
      "auto_start": false,
      "port": 5006,
      "path": "services/backtesting-service/main.py"
    },
    "ai-service": {
      "auto_start": false,
      "port": 5005,
      "path": "services/ai-service/main.py"
    }
  }
}
```

#### 8.1.2 Service Orchestration (Backend API)

**Backend API Service Monitor:**

```
Service Monitor Loop (5 másodpercenként):

FOR EACH service IN services:
  # 1. Health check
  response = http_get(f"http://localhost:{service.port}/health", timeout=2s)

  IF response.ok:
    service.status = "ONLINE"
    service.last_heartbeat = NOW
  ELSE:
    service.status = "OFFLINE"

    # 2. Ha offline ÉS auto_restart == true → indítás
    IF service.auto_restart:
      IF service.restart_attempts < 3:
        log(f"Restarting {service.name}...")
        start_service(service)
        service.restart_attempts += 1
      ELSE:
        log(f"Service {service.name} failed after 3 restart attempts")
        notify_user(f"CRITICAL: {service.name} nem indul el")

  # 3. WebSocket notification Frontend-nek
  send_websocket_event({
    "type": "service_status_change",
    "service": service.name,
    "status": service.status
  })
```

**Service indítás (Python subprocess):**

```python
import subprocess
import sys

def start_service(service_config):
    """Service indítása subprocess-ként"""

    # Python path
    python_exe = sys.executable

    # Service main.py elérési útja
    service_path = service_config['path']

    # Environment variables (ha kell)
    env = os.environ.copy()
    env['SERVICE_PORT'] = str(service_config['port'])

    # Subprocess indítás
    process = subprocess.Popen(
        [python_exe, service_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=os.path.dirname(service_path)
    )

    # Process ID tárolása (később kill-hez kell)
    service_config['process_id'] = process.pid
    service_config['process'] = process

    log(f"Service {service_config['name']} started (PID: {process.pid})")

    return process
```

#### 8.1.3 Graceful Shutdown

**Cél:** Service-ek rendezett leállítása (ne szakadjon félbe művelet)

**Shutdown Flow:**

```
1. Felhasználó bezárja a programot (vagy CTRL+C)
2. Backend API fogja a SIGINT/SIGTERM signal-t
3. Backend API küldi a shutdown jelet minden service-nek:
   - POST http://localhost:{port}/shutdown
4. Minden service megkapja a shutdown jelet:
   - Befejezi az aktuális műveleteket (max 10 másodperc)
   - Lezárja a DB kapcsolatokat
   - Bezárja a fájlokat
   - Válaszol: {"status": "shutdown_complete"}
5. Backend API várja amíg minden service rendeseb leáll (max 15 másodperc)
6. Ha valamelyik nem áll le időben → SIGKILL (force kill)
7. Backend API maga is leáll
```

**Service shutdown endpoint (minden service-ben):**

```python
@app.post("/shutdown")
async def shutdown():
    """Graceful shutdown endpoint"""

    logger.info("Shutdown signal received")

    # 1. Jelzés hogy ne fogadjon új kéréseket
    app.state.accepting_requests = False

    # 2. Futó műveletek befejezése
    await finish_pending_operations()

    # 3. DB kapcsolatok lezárása
    close_database_connections()

    # 4. Fájlok lezárása
    close_open_files()

    logger.info("Shutdown complete")

    # 5. Process kilépés (1 másodperc delay)
    threading.Timer(1.0, lambda: os._exit(0)).start()

    return {"status": "shutdown_complete"}
```

### 8.2 Process Management

**Egyszerű megoldás (Built-in):**
- Backend API indítja/állítja le a service-eket
- subprocess.Popen használata
- Process ID tárolása
- Manual restart Frontend-ről

**Komplexebb megoldás (PM2 vagy systemd - később):**
- PM2: Node.js process manager, de Python-hoz is működik
- systemd: Linux service management
- Automatikus restart ha crash
- Log aggregálás

### 8.3 Service Függőségek Kezelése

**Probléma:** Néhány service függ mástól (pl. Strategy Service függ Pattern Service-től)

**Megoldás: Dependency Injection + Health Check**

```
Indítási sorrend:

1. Backend API (mindig első)
2. MT5 Service (független)
3. Data Service (függ MT5 Service-től)
4. Pattern Service (függ Data Service-től)
5. Strategy Service (függ Pattern + Data Service-től)
6. AI Service (függ Data Service-től)
7. Backtesting Service (függ Data Service-től)

Minden service indítás előtt:
  - Ellenőrzés: függőség service online-e?
  - Ha nem → várás max 30 másodperc
  - Ha 30 másodperc után sem online → error
```

---

## 9. Logging és Monitoring Részletesen

### 9.1 Központi Logging Rendszer

**Cél:** Minden service logja egységes formátumban, könnyen kereshető, központi helyen

#### 9.1.1 Log Formátum (Egységesített)

**Minden service ezt a formátumot használja:**

```
[TIMESTAMP] [LEVEL] [SERVICE_NAME] [MODULE] MESSAGE
```

**Példák:**
```
[2025-10-03 14:35:22.123] [INFO] [DataService] [gap_filler] Gap fill started for EURUSD
[2025-10-03 14:35:23.456] [ERROR] [MT5Service] [connection] Failed to connect to MT5: timeout
[2025-10-03 14:35:24.789] [DEBUG] [StrategyService] [position_manager] Trailing stop updated: 1.10500 -> 1.10520
```

**Python logging konfiguráció (minden service-ben):**

```python
import logging
import sys

def setup_logging(service_name: str, log_level: str = "INFO"):
    """Egységes logging setup minden service-hez"""

    # Log formátum
    log_format = "[%(asctime)s] [%(levelname)s] [" + service_name + "] [%(module)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Root logger konfiguráció
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 1. Fájl handler (service-specifikus log fájl)
            logging.FileHandler(f"logs/{service_name.lower()}.log"),
            # 2. Console handler (stdout)
            logging.StreamHandler(sys.stdout),
            # 3. Központi log fájl handler (összes service egy fájlban)
            logging.FileHandler("logs/all-services.log")
        ]
    )

    logger = logging.getLogger(service_name)
    return logger
```

#### 9.1.2 Log Szintek és Használatuk

**DEBUG:**
- Részletes debug információk (változó értékek, flow)
- Csak development-ben vagy hibakereséskor
- Példa: "Indicator EMA_20 calculated: 1.10523"

**INFO:**
- Normál működési események
- Fontos állapotváltozások
- Példa: "Gap fill completed: 1,250,000 ticks downloaded"

**WARNING:**
- Figyelmeztetések (nem kritikus)
- Váratlan de kezelt helyzetek
- Példa: "Data Service response slow: 3.5 seconds"

**ERROR:**
- Hibák (műveletek nem sikerültek)
- Kivételek, exception-ök
- Példa: "Failed to save ticks to database: connection timeout"

**CRITICAL:**
- Kritikus hibák (service nem tud működni)
- Azonnali beavatkozás szükséges
- Példa: "MT5 connection lost, cannot continue trading"

#### 9.1.3 Log Fájlok Struktúra

```
logs/
├── all-services.log              # Összes service közös log
├── backend-api.log               # Backend API service log
├── data-service.log              # Data Service log
├── mt5-service.log               # MT5 Service log
├── pattern-service.log           # Pattern Service log
├── strategy-service.log          # Strategy Service log
├── backtesting-service.log       # Backtesting Service log
├── ai-service.log                # AI Service log
├── archived/                     # Régi log-ok archívum
│   ├── 2025-09/
│   │   ├── data-service-2025-09-01.log.gz
│   │   └── data-service-2025-09-02.log.gz
│   └── 2025-10/
```

#### 9.1.4 Log Rotation

**Cél:** Log fájlok ne növekedjenek végtelenül

**Rotating File Handler:**

```python
from logging.handlers import RotatingFileHandler

# Max 10 MB fájl méret, max 5 backup
file_handler = RotatingFileHandler(
    f"logs/{service_name.lower()}.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5
)
```

**TimedRotatingFileHandler (naponta új fájl):**

```python
from logging.handlers import TimedRotatingFileHandler

file_handler = TimedRotatingFileHandler(
    f"logs/{service_name.lower()}.log",
    when="midnight",
    interval=1,
    backupCount=30  # 30 napig őrizzük
)
```

### 9.2 Monitoring és Metrics

**Cél:** Rendszer teljesítményének folyamatos figyelése

#### 9.2.1 Service Metrics (Minden service gyűjti)

**Alap Metrikák:**
- Uptime (mennyi ideje fut)
- Request count (hány kérés érkezett)
- Response time (átlagos válaszidő)
- Error rate (hibaarány %)
- CPU használat (%)
- Memória használat (MB)

**Metric endpoint (minden service-ben):**

```python
@app.get("/metrics")
def get_metrics():
    """Service metrikák"""
    return {
        "service": "data-service",
        "uptime_seconds": get_uptime(),
        "request_count": metrics.total_requests,
        "avg_response_time_ms": metrics.avg_response_time,
        "error_rate": metrics.error_count / metrics.total_requests,
        "cpu_percent": psutil.cpu_percent(),
        "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "active_connections": len(active_connections),
        "timestamp": datetime.now().isoformat()
    }
```

#### 9.2.2 Custom Metrics (Service-specifikus)

**Data Service:**
- Tick collection rate (tick/s)
- Database write speed (records/s)
- Gap fill progress (%)

**MT5 Service:**
- MT5 connection status (connected/disconnected)
- MT5 response time (ms)
- Open positions count

**Strategy Service:**
- Active strategies count
- Open positions count
- Today's P/L

**Backtesting Service:**
- Running backtests count
- Queue length
- Average backtest duration

#### 9.2.3 Monitoring Dashboard (Frontend)

**Real-time Monitoring Megjelenítés:**

```
┌─────────────────────────────────────────────────────────┐
│                  SYSTEM MONITORING                      │
├─────────────────────────────────────────────────────────┤
│ Data Service:                                           │
│   Uptime: 3h 24m                                        │
│   CPU: 5.2%   Memory: 145 MB                           │
│   Tick Rate: 1,250 ticks/s                             │
│   Avg Response: 45ms                                    │
│   Error Rate: 0.02%                                     │
├─────────────────────────────────────────────────────────┤
│ MT5 Service:                                            │
│   Uptime: 3h 24m                                        │
│   CPU: 2.1%   Memory: 78 MB                            │
│   MT5 Status: ● CONNECTED                              │
│   Avg Response: 12ms                                    │
│   Open Positions: 3                                     │
├─────────────────────────────────────────────────────────┤
│ Strategy Service:                                       │
│   Uptime: 2h 15m                                        │
│   CPU: 3.8%   Memory: 112 MB                           │
│   Active Strategies: 2                                  │
│   Today's P/L: +$125.50                                │
├─────────────────────────────────────────────────────────┤
│ [View Detailed Metrics] [View Logs] [Export CSV]       │
└─────────────────────────────────────────────────────────┘
```

#### 9.2.4 Alerting és Notifications

**Alert Szabályok:**

```python
# Alert rules konfig
alert_rules = [
    {
        "name": "High CPU Usage",
        "condition": "cpu_percent > 80",
        "severity": "WARNING",
        "notification": "email"
    },
    {
        "name": "Service Offline",
        "condition": "service.status == 'OFFLINE'",
        "severity": "CRITICAL",
        "notification": "email + popup"
    },
    {
        "name": "High Error Rate",
        "condition": "error_rate > 0.05",
        "severity": "ERROR",
        "notification": "email"
    },
    {
        "name": "MT5 Disconnected",
        "condition": "mt5_status == 'DISCONNECTED'",
        "severity": "CRITICAL",
        "notification": "email + popup + sound"
    }
]
```

**Notification Típusok:**
- **Popup** (Frontend): Azonnal megjelenik a UI-on
- **Email**: Email értesítés (ha be van állítva)
- **Sound**: Hangjelzés (Frontend)
- **Desktop Notification**: OS-szintű notification (Windows toast)

### 9.3 Log Viewing és Searching (Frontend)

**Log Viewer Component:**

```
┌─────────────────────────────────────────────────────────┐
│                     LOG VIEWER                          │
├─────────────────────────────────────────────────────────┤
│ Service: [All Services ▼]                              │
│ Level: [All ▼]                                          │
│ Search: [__________________] 🔍                         │
│ Time Range: [Last 1 hour ▼]                            │
├─────────────────────────────────────────────────────────┤
│ [2025-10-03 14:35:22] [INFO] [DataService] Gap fill... │
│ [2025-10-03 14:35:23] [ERROR] [MT5Service] Failed...   │
│ [2025-10-03 14:35:24] [DEBUG] [StrategyService] Trail..│
│ ...                                                     │
├─────────────────────────────────────────────────────────┤
│ [Export] [Clear] [Auto-refresh: ON]                    │
└─────────────────────────────────────────────────────────┘
```

**Funkciók:**
- Real-time log streaming (WebSocket)
- Filter service-re
- Filter log level-re
- Text search
- Time range filter
- Export to file

---

## 10. Error Recovery és Resilience

### 10.1 Automatikus Service Restart

**Cél:** Ha egy service crash-el, automatikusan újraindul

**Backend API Monitoring + Auto-restart:**

```
Service Monitor Loop:
  FOR EACH service IN services:
    IF service.status == "OFFLINE" AND service.auto_restart == true:
      IF service.restart_attempts < MAX_RESTART_ATTEMPTS (3):
        # Exponential backoff
        wait_time = 2 ^ restart_attempts  # 2s, 4s, 8s
        sleep(wait_time)

        log(f"Attempting to restart {service.name} (attempt {restart_attempts + 1}/3)")
        start_service(service)
        service.restart_attempts += 1

      ELSE:
        log(f"CRITICAL: {service.name} failed after 3 restart attempts")
        send_critical_alert(service.name)
        service.auto_restart = false  # Ne próbálkozzon tovább
```

### 10.2 Circuit Breaker Pattern

**Cél:** Ha egy service nem válaszol, ne próbálkozzunk folyamatosan (ne terhelje le tovább)

**Működés:**

```
Circuit States:
  - CLOSED: Normál működés, kérések átmennek
  - OPEN: Service nem elérhető, kérések azonnal hibát adnak (fail-fast)
  - HALF_OPEN: Teszt állapot, 1 kérés megy át tesztelésre

Circuit Breaker Logic:

IF circuit == CLOSED:
  try:
    response = call_service()
    IF response.ok:
      success_count++
      return response
    ELSE:
      failure_count++
      IF failure_count > THRESHOLD (5):
        circuit = OPEN
        start_timeout_timer(30 seconds)
  catch Exception:
    failure_count++
    IF failure_count > THRESHOLD:
      circuit = OPEN

ELSE IF circuit == OPEN:
  # Fail-fast, ne hívjuk a service-t
  return {"error": "Service unavailable (circuit open)"}

  # Ha lejár a timeout (30s)
  IF timeout_expired:
    circuit = HALF_OPEN

ELSE IF circuit == HALF_OPEN:
  try:
    response = call_service()
    IF response.ok:
      circuit = CLOSED  # Service újra működik
      failure_count = 0
    ELSE:
      circuit = OPEN  # Még mindig nem működik
  catch Exception:
    circuit = OPEN
```

**Python példa:**

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"
        self.next_attempt_time = None

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() < self.next_attempt_time:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = "HALF_OPEN"

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e

    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def on_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.next_attempt_time = time.time() + self.timeout
```

### 10.3 Retry Politika

**Cél:** Service hívások újrapróbálása hiba esetén

**Exponential Backoff + Jitter:**

```python
import time
import random

def retry_with_backoff(func, max_retries=3, base_delay=1):
    """
    Retry with exponential backoff + jitter
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                # Utolsó próbálkozás sikertelen
                raise e

            # Exponential backoff: 1s, 2s, 4s, 8s, ...
            delay = base_delay * (2 ** attempt)

            # Jitter: +/- 20% random variáció
            jitter = delay * 0.2 * (random.random() - 0.5)
            total_delay = delay + jitter

            log(f"Retry {attempt + 1}/{max_retries} after {total_delay:.2f}s")
            time.sleep(total_delay)
```

**Használat:**

```python
# MT5 Service hívása retry-val
def call_mt5_service():
    response = requests.get("http://localhost:5002/health", timeout=2)
    if not response.ok:
        raise Exception("MT5 Service not responding")
    return response.json()

try:
    result = retry_with_backoff(call_mt5_service, max_retries=3)
except Exception as e:
    logger.error(f"MT5 Service unavailable after 3 retries: {e}")
```

### 10.4 Fallback Mechanizmusok

**Cél:** Ha egy service nem elérhető, használjunk alternatív adatot vagy működést

**Példák:**

**1. Cached Data (Pattern Service):**
```python
def get_indicators(symbol, timeframe):
    try:
        # Próbáljuk live-ból lekérni
        response = requests.get(f"http://localhost:5003/indicators/{symbol}/{timeframe}")
        if response.ok:
            data = response.json()
            cache.set(f"indicators_{symbol}_{timeframe}", data, ttl=60)
            return data
        else:
            raise Exception("Pattern Service unavailable")
    except Exception:
        # Fallback: cached data
        cached_data = cache.get(f"indicators_{symbol}_{timeframe}")
        if cached_data:
            logger.warning("Using cached indicator data (Pattern Service offline)")
            return cached_data
        else:
            # Nincs cache sem → default értékek
            return get_default_indicators()
```

**2. Degraded Mode (Strategy Service):**
```python
def run_strategy(strategy_id):
    try:
        # Próbáljuk pattern jelzésekkel futtatni
        pattern_signals = get_pattern_signals(strategy.symbol)
        return execute_strategy_with_patterns(strategy, pattern_signals)
    except Exception:
        # Fallback: Degraded mode - csak price action alapján
        logger.warning("Running strategy in DEGRADED mode (no patterns)")
        return execute_strategy_price_action_only(strategy)
```

### 10.5 Data Integrity Recovery

**Cél:** Félig kész műveletek visszaállítása

**Transaction Log (Write-Ahead Log):**

```python
# Data Service - Tick mentés
def save_ticks_batch(ticks):
    transaction_id = generate_transaction_id()

    # 1. Log the intent (Write-Ahead Log)
    transaction_log.write({
        "transaction_id": transaction_id,
        "operation": "save_ticks",
        "ticks_count": len(ticks),
        "status": "IN_PROGRESS",
        "timestamp": datetime.now()
    })

    try:
        # 2. Tényleges művelet
        db.execute_many("INSERT INTO ticks VALUES (?)", ticks)
        db.commit()

        # 3. Commit a log-ban
        transaction_log.update(transaction_id, {"status": "COMPLETED"})

    except Exception as e:
        # 4. Rollback
        db.rollback()
        transaction_log.update(transaction_id, {"status": "FAILED"})
        raise e
```

**Recovery on Startup:**

```python
def recover_failed_transactions():
    """Service induláskor futtatandó recovery"""

    failed_transactions = transaction_log.get_all(status="IN_PROGRESS")

    for tx in failed_transactions:
        log(f"Recovering failed transaction: {tx.transaction_id}")

        if tx.operation == "save_ticks":
            # Töröljük a részlegesen mentett adatokat
            db.execute(f"DELETE FROM ticks WHERE transaction_id = ?", tx.transaction_id)

        transaction_log.update(tx.transaction_id, {"status": "RECOVERED"})
```

### 10.6 Health Check és Dependency Monitoring

**Service Health Endpoint:**

```python
@app.get("/health")
def health_check():
    """Részletes health check"""

    # Check DB connection
    db_healthy = check_database_connection()

    # Check dependencies
    dependencies = {
        "mt5_service": check_dependency("http://localhost:5002/health"),
        "data_service": check_dependency("http://localhost:5001/health")
    }

    all_healthy = db_healthy and all(dependencies.values())

    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "service": "pattern-service",
        "database": "connected" if db_healthy else "disconnected",
        "dependencies": dependencies,
        "timestamp": datetime.now().isoformat()
    }, 200 if all_healthy else 503
```

---

## 11. Későbbi Fejlesztések

### 8.1 MQL4/5 Fordító

**Cél:** MT4/MT5 indicator/EA fájlok Python-ra fordítása

**Módszer:**
1. MQL fájl parsing (AST)
2. MQL szintaxis → Python szintaxis mapping
3. MT5 függvények → pandas/numpy ekvivalensek
4. Automatikus pattern/strategy generálás

**Előny:** Meglévő MQL stratégiák újrahasznosítása

### 8.2 Docker Konténerizáció

**Cél:** Egyszerű deployment

**Konténerek:**
- `backend-api` konténer
- `data-service` konténer
- `mt5-service` konténer (Windows konténer MT5-höz)
- `pattern-service` konténer
- `strategy-service` konténer
- `ai-service` konténer
- `frontend` konténer (Nginx + Angular)

**Docker Compose:**
- Egy paranccsal indul minden
- Service discovery beépítve
- Volume mount adatbázisokhoz

### 8.3 Biztonsági Elemek

**Ha távoli elérés kell:**

1. **Autentikáció**
   - JWT token alapú
   - Login/logout
   - Token refresh

2. **Autorizáció**
   - Role-based access (Admin, Trader, Viewer)
   - Endpoint védelem

3. **HTTPS**
   - SSL/TLS tanúsítvány
   - Minden kommunikáció titkosított

4. **API Key**
   - Service-ek közötti kommunikáció védelem
   - Header: `X-API-Key: {secret}`

### 8.4 Cloud Deployment

**Ha VPS/Cloud szerver kell:**

1. **Infrastructure:**
   - VPS bérlés (AWS, Azure, DigitalOcean)
   - Domain név
   - SSL tanúsítvány (Let's Encrypt)

2. **Deployment stratégia:**
   - CI/CD pipeline (GitHub Actions)
   - Automatikus deploy git push után
   - Blue-green deployment

3. **Monitoring:**
   - Uptime monitoring (UptimeRobot)
   - Error tracking (Sentry)
   - Performance monitoring (New Relic)

### 8.5 Multi-user Support

**Ha több felhasználó kell:**

1. **User management**
   - Regisztráció/Login
   - User adatbázis tábla
   - Session management

2. **Izolált adatok**
   - Felhasználónként külön stratégiák
   - Felhasználónként külön beállítások
   - Trade history elkülönítve

3. **Limit-ek**
   - Max stratégia/user
   - Rate limiting API-n

### 8.6 Advanced Features

1. **Telegram/Discord bot**
   - Trade jelzések küldése
   - Parancsok (start/stop stratégia)
   - Account balance riport

2. **Email notifications**
   - Trade alert
   - Kritikus hiba értesítés
   - Napi/heti riport

3. **Mobile app**
   - React Native / Flutter
   - Pozíció monitoring
   - Push notifications

4. **Advanced backtesting**
   - Multi-symbol backtesting
   - Walk-forward optimization
   - Monte Carlo szimukáció

5. **Social trading**
   - Stratégia megosztás
   - Copy trading
   - Leaderboard

---

## 9. Összefoglalás

### 9.1 Projekt Előnyei

✅ **Modularitás** - Minden funkció külön service-ben
✅ **Skálázhatóság** - Service-ek egymástól függetlenül skálázhatók
✅ **Karbantarthatóság** - Tiszta felelősségi körök
✅ **Teljesítmény** - Optimalizált, minimális CPU használat
✅ **Rugalmasság** - Új service-ek könnyen hozzáadhatók
✅ **Egyszerűség** - Nincs felesleges complexity

### 9.2 Várható Fejlesztési Idő

- **Fázis 1**: 2-3 hét
- **Fázis 2**: 3-4 hét
- **Fázis 3**: 2-3 hét
- **Fázis 4**: 3-4 hét
- **Fázis 5**: 4-5 hét
- **Fázis 6**: 3-4 hét

**Összesen: ~4-5 hónap** (egy fejlesztő, teljes munkaidő)

### 9.3 Sikerkritériumok

1. ✅ Minden service önállóan fut és figyelhető
2. ✅ Gap fill automatikusan fut induláskor
3. ✅ OnFly gyűjtés stabil, 0% CPU idle-ban
4. ✅ Pattern-ek real-time detektálhatók
5. ✅ Stratégiák backtestje pontos eredményt ad
6. ✅ Paper trading működik
7. ✅ Frontend responsív és real-time
8. ✅ Teljes rendszer stabil 24/7

### 9.4 Következő Lépések

1. ✅ **Ügynök leírások elkészítése** (külön dokumentumok)
2. ✅ **Projekt struktúra létrehozása**
3. ✅ **Fázis 1 indítása** (Backend API Service)

---

**Dokumentum vége**

*Készítette: AI Assistant*
*Dátum: 2025. október 3.*
