# Data Service Ügynök Leírás

**Verzió:** 1.0.0
**Létrehozva:** 2025. október 3.
**Service Port:** 5001

---

## 1. Ügynök Áttekintés

### 1.1 Ügynök Neve
**Data Service Ügynök**

### 1.2 Fő Felelősségek

A Data Service Ügynök felelős a **teljes adatgyűjtési és adatkezelési folyamatért**. Ez a service biztosítja, hogy minden tick és OHLC adat elérhető legyen az adatbázisban, és folyamatosan frissítve legyen.

### 1.3 Service Típus
- **Data Collection Service**
- **Database Manager**
- **Gap Filler**

---

## 2. Részletes Felelősségek

### 2.1 Gap Fill (Hiányzó Adatok Pótlása)

**Cél:** Biztosítani, hogy az adatbázisban ne legyenek hiányos időszakok

**Működés:**

1. **Induláskor Automatikus Gap Fill**
   - Service elindul
   - Ellenőrzi a kiválasztott symbol-ok adatbázisait
   - Meghatározza az utolsó mentett adat időpontját
   - Összehasonlítja a jelenlegi időponttal
   - Letölti a hiányzó időszak adatait MT5 Service-en keresztül

2. **Hiány Detektálás Algoritmusa**

**FONTOS:** Letöltés előtt **MINDIG ellenőrizzük a completeness táblát**, hogy ne töltsünk le duplikáltan!

```
FOR EACH symbol IN selected_symbols:
  # Tick adatbázis ellenőrzés
  last_tick_timestamp = get_last_tick(symbol)
  current_timestamp = NOW

  gap_duration = current_timestamp - last_tick_timestamp

  IF gap_duration > 1 hour:
    # Van hiány, de ellenőrizzük a completeness-t
    date_range = generate_date_range(last_tick_timestamp, current_timestamp)

    FOR EACH day IN date_range:
      completeness_status = check_completeness(symbol, day, "tick")

      IF completeness_status == "COMPLETE":
        # Ez a nap már teljes, kihagyás
        CONTINUE
      ELSE:
        # PARTIAL vagy EMPTY, letöltés szükséges
        download_ticks(symbol, day)
        update_completeness(symbol, day, "COMPLETE")

  # OHLC adatbázis ellenőrzés
  FOR EACH timeframe IN timeframes:
    last_bar_timestamp = get_last_bar(symbol, timeframe)
    expected_bar_count = calculate_expected_bars(last_bar_timestamp, NOW, timeframe)
    actual_bar_count = count_bars(symbol, timeframe, last_bar_timestamp, NOW)

    IF actual_bar_count < expected_bar_count:
      # Van hiányzó bar, de ellenőrizzük a completeness-t
      date_range = generate_date_range(last_bar_timestamp, NOW)

      FOR EACH day IN date_range:
        completeness_status = check_completeness(symbol, day, "ohlc", timeframe)

        IF completeness_status == "COMPLETE":
          # Ez a nap már teljes, kihagyás
          CONTINUE
        ELSE:
          # PARTIAL vagy EMPTY, letöltés szükséges
          download_bars(symbol, timeframe, day)
          update_completeness(symbol, day, "COMPLETE")
```

3. **Letöltési Folyamat**
   - Dátum tartomány felosztása kisebb batch-ekre (pl. 1 napos)
   - Batch-enként letöltés MT5 Service-től
   - Adatok validálása
   - Adatbázisba mentés
   - Completeness státusz frissítés

4. **Progress Reporting**
   - Minden batch letöltése után státusz frissítés
   - WebSocket üzenet küldése Backend API-nak
   - Frontend látja a folyamatot (progress bar)

### 2.2 Előzmény Letöltés (Manuális)

**Cél:** Felhasználó által kért időszak letöltése

**Működés:**

1. **Kérés fogadása Backend API-tól**
   - POST /download-history
   - Paraméterek: symbols, start_date, end_date, data_types (tick/ohlc)

2. **Validáció**
   - Dátum tartomány helyes?
   - Symbol-ok léteznek?
   - Adattípusok helyesek?

3. **Letöltés Ütemezés**
   - Egyidőben max 1 letöltés futhat
   - Ha már fut letöltés → várakozási sor
   - Priority: manuális előzmény > automatikus gap fill

4. **Batch Letöltés**
   - Nagyobb időszak felosztása napokra/hetekre
   - Batch-enként letöltés
   - Progress reporting
   - Hiba esetén újrapróbálkozás

5. **Eredmény Összesítés**
   - Letöltött tick-ek száma
   - Letöltött bárok száma
   - Hibák listája (ha volt)
   - Teljes futási idő

### 2.3 OnFly Adatgyűjtés (Real-time)

**Cél:** Folyamatos real-time tick és OHLC gyűjtés

**Működés:**

#### Tick Gyűjtés
1. **Collection Loop**
```
LOOP (every 100 milliseconds):
  FOR EACH symbol IN monitored_symbols:
    tick = request_tick_from_mt5_service(symbol)

    IF tick válid:
      buffer.add(tick)

    IF buffer.size >= BATCH_SIZE (pl. 100):
      save_ticks_to_db(buffer)
      buffer.clear()
      send_websocket_event("new_ticks", buffer)
```

2. **Batch Database Write**
   - Gyűjtés batch-ekben (100-1000 tick)
   - Egyetlen INSERT művelettel mentés (gyorsabb)
   - Transaction használat

3. **Real-time Notification**
   - Minden batch mentése után WebSocket üzenet
   - Backend API továbbítja a frontend-nek
   - Frontend frissíti a megjelenítést

#### OHLC Candle Monitoring
1. **Candle Completion Detection**
```
FOR EACH symbol IN monitored_symbols:
  FOR EACH timeframe IN timeframes:
    current_bar_start = calculate_bar_start(NOW, timeframe)
    last_saved_bar = get_last_bar(symbol, timeframe)

    IF current_bar_start > last_saved_bar.timestamp:
      # Új bar indult, az előző lezárult
      download_completed_bar(symbol, timeframe, last_saved_bar.timestamp)
```

2. **Bar Letöltés**
   - Csak a lezárt bar-okat tölti le
   - A jelenleg formálódó bar NEM kerül mentésre
   - `is_closed = 1` flag beállítás

3. **Notification**
   - WebSocket üzenet új bar-ról
   - Pattern Service tudja, hogy új adat érkezett
   - Strategy Service frissíti a pozícióit

### 2.4 Adatbázis Kezelés

**Felelősség:** Symbol-alapú particionált adatbázisok kezelése

#### Tick Adatbázisok
**Formátum:** `database/{YEAR}/{SYMBOL}_ticks_{YEAR}_{MONTH}.db`

**Példa:** `database/2025/EURUSD_ticks_2025_01.db`

**Létrehozás:**
- Automatikus ha nem létezik
- Havi particionálás (egy hónap = egy fájl)
- Index létrehozás (timestamp)

**Adatok mentése:**
- Batch INSERT (gyors)
- Timestamp index használat (gyors lekérdezés)

#### OHLC Adatbázisok
**Formátum:** `database/{YEAR}/{SYMBOL}_ohlc.db`

**Példa:** `database/2025/EURUSD_ohlc.db`

**Táblák:**
- Egy tábla: `ohlc_data`
- Oszlop: timeframe (M1, M5, H1 stb.)
- Index: (symbol, timeframe, timestamp)

#### Completeness Adatbázis
**Fájl:** `database/{YEAR}/completeness_monitoring.db`

**Táblák:**
- `tick_data_completeness` - Tick adatok completeness státusz
- `ohlc_data_completeness` - OHLC adatok completeness státusz

**Completeness Frissítés:**
- Minden gap fill után
- Minden manuális letöltés után
- Naponta egyszer (scheduled job)

### 2.5 Completeness Analízis

**Cél:** Adatbázis minőség ellenőrzés ÉS duplikált letöltés megakadályozása

**FONTOS:** A Completeness-nek **kettős feladata** van:
1. **Minőségbiztosítás**: Ellenőrzi, hogy az adatok teljesek-e
2. **Duplikáció megakadályozás**: Ha egy nap már COMPLETED, akkor azt kihagyja újbóli letöltésnél

**Működés:**

1. **Naponta Futó Analízis**
   - Éjszaka fut (pl. 02:00)
   - Ellenőrzi az előző nap adatait
   - Frissíti a completeness státuszokat

2. **Analízis Lépések**
```
FOR EACH symbol IN symbols:
  FOR EACH day IN last_7_days:
    tick_count = count_ticks(symbol, day)
    expected_tick_count = estimate_expected_ticks(symbol, day)

    IF tick_count == 0:
      status = EMPTY
    ELSE IF tick_count < expected_tick_count * 0.8:
      status = PARTIAL
    ELSE:
      status = COMPLETE

    save_completeness_status(symbol, day, status, tick_count)
```

3. **Completeness Ellenőrzés Letöltésnél**

Amikor előzmény letöltést vagy gap fill-t végzünk, **MINDEN nap esetén ellenőrizzük a completeness státuszt**:

```
FOR EACH day IN date_range:
  completeness_status = check_completeness(symbol, day, data_type)

  IF completeness_status == "COMPLETE":
    # Ez a nap már teljes, KIHAGYÁS
    log("Nap kihagyva: {day} - már COMPLETE")
    SKIP to next day

  ELSE IF completeness_status == "PARTIAL":
    # Részleges adat van, újra letöltés szükséges
    log("Nap újra letöltése: {day} - PARTIAL")
    download_data(symbol, day)
    update_completeness(symbol, day, "COMPLETE")

  ELSE IF completeness_status == "EMPTY":
    # Nincs adat, letöltés szükséges
    log("Nap letöltése: {day} - EMPTY")
    download_data(symbol, day)
    update_completeness(symbol, day, "COMPLETE")
```

**Példa Eset:**
- Felhasználó kéri: "Töltsd le a 2024-es évet EURUSD-re"
- Service ellenőrzi minden napra (365 nap) a completeness táblát:
  - 2024-01-01 → COMPLETE → **KIHAGYÁS**
  - 2024-01-02 → COMPLETE → **KIHAGYÁS**
  - 2024-01-03 → PARTIAL → **ÚJRA LETÖLTÉS**
  - 2024-01-04 → EMPTY → **LETÖLTÉS**
  - ...
- Eredmény: Csak azok a napok kerülnek letöltésre, amelyek PARTIAL vagy EMPTY állapotban vannak

4. **Hiányosságok Riportálása**
   - Ha EMPTY vagy PARTIAL napot talál
   - Értesíti a Backend API-t
   - Frontend jelzi a felhasználónak
   - Opcionális: automatikus gap fill indítás

---

## 3. REST API Endpointok

### 3.1 Health Check

#### GET /health
**Leírás:** Service állapot ellenőrzés

**Válasz:**
```json
{
  "status": "healthy",
  "service": "data-service",
  "uptime": 3600,
  "database": {
    "connected": true,
    "ticks_db_count": 5,
    "ohlc_db_count": 5
  },
  "collection": {
    "active": true,
    "monitored_symbols": ["EURUSD", "GBPUSD"],
    "ticks_per_second": 15.2
  }
}
```

### 3.2 Gap Fill

#### POST /gap-fill
**Leírás:** Manuális gap fill indítás

**Body:**
```json
{
  "symbols": ["EURUSD", "GBPUSD"],
  "force": false
}
```

**Válasz:**
```json
{
  "success": true,
  "job_id": "gap-fill-12345",
  "message": "Gap fill elindítva 2 symbol-ra"
}
```

#### GET /gap-fill/status/{job_id}
**Leírás:** Gap fill folyamat állapot

**Válasz:**
```json
{
  "job_id": "gap-fill-12345",
  "status": "running",
  "progress": 45,
  "symbols_completed": 1,
  "symbols_total": 2,
  "ticks_downloaded": 125000,
  "bars_downloaded": 5000
}
```

### 3.3 Előzmény Letöltés

#### POST /download-history
**Leírás:** Előzmény adatok letöltése

**Body:**
```json
{
  "symbols": ["EURUSD"],
  "start_date": "2025-01-01",
  "end_date": "2025-01-31",
  "data_types": ["tick", "ohlc"]
}
```

**Válasz:**
```json
{
  "success": true,
  "job_id": "download-12345",
  "estimated_duration": 300
}
```

#### GET /download-history/status/{job_id}
**Leírás:** Letöltés állapot

### 3.4 OnFly Gyűjtés

#### POST /start-collection
**Leírás:** Real-time gyűjtés indítása

**Body:**
```json
{
  "symbols": ["EURUSD", "GBPUSD"],
  "data_types": ["tick", "ohlc"]
}
```

#### POST /stop-collection
**Leírás:** Real-time gyűjtés leállítása

#### GET /collection/status
**Leírás:** Gyűjtés állapot

**Válasz:**
```json
{
  "active": true,
  "monitored_symbols": ["EURUSD", "GBPUSD"],
  "ticks_collected": 1500000,
  "bars_collected": 50000,
  "collection_rate": 15.2,
  "started_at": "2025-10-03T10:00:00Z",
  "uptime": 7200
}
```

### 3.5 Adatok Lekérés

#### GET /ticks/{symbol}
**Paraméterek:**
- `from`: Kezdő timestamp (Unix ms)
- `to`: Befejező timestamp (Unix ms)
- `limit`: Max eredmény (default: 10000)

**Válasz:**
```json
{
  "success": true,
  "symbol": "EURUSD",
  "count": 1000,
  "data": [
    {
      "timestamp": 1696337696000,
      "bid": 1.10523,
      "ask": 1.10525,
      "last": 1.10524,
      "volume": 100
    }
  ]
}
```

#### GET /ohlc/{symbol}/{timeframe}
**Paraméterek:**
- `from`: Kezdő timestamp
- `to`: Befejező timestamp
- `limit`: Max eredmény

**Válasz:**
```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "M15",
  "count": 100,
  "data": [
    {
      "timestamp": 1696337700000,
      "open": 1.10520,
      "high": 1.10530,
      "low": 1.10515,
      "close": 1.10525,
      "tick_volume": 150,
      "is_closed": 1
    }
  ]
}
```

### 3.6 Statisztikák

#### GET /statistics
**Leírás:** Adatbázis statisztikák

**Válasz:**
```json
{
  "databases": {
    "ticks": [
      {
        "symbol": "EURUSD",
        "year": 2025,
        "month": 1,
        "total_ticks": 2500000,
        "file_size_mb": 150.5,
        "first_tick": "2025-01-01T00:00:00Z",
        "last_tick": "2025-01-31T23:59:59Z"
      }
    ],
    "ohlc": [
      {
        "symbol": "EURUSD",
        "total_bars": 50000,
        "timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"],
        "file_size_mb": 25.3
      }
    ]
  },
  "completeness": {
    "complete_days": 28,
    "partial_days": 2,
    "empty_days": 1
  }
}
```

---

## 4. Implementációs Útmutató

### 4.1 Projekt Struktúra

```
data-service/
├── main.py                 # Fő belépési pont
├── config.py               # Konfigurációk
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── api/                # REST API
│   │   ├── __init__.py
│   │   ├── gap_fill.py
│   │   ├── download.py
│   │   ├── collection.py
│   │   └── data.py
│   ├── core/               # Core logika
│   │   ├── __init__.py
│   │   ├── gap_filler.py           # Gap fill logika
│   │   ├── history_downloader.py   # Előzmény letöltés
│   │   ├── tick_collector.py       # OnFly tick gyűjtés
│   │   ├── bar_monitor.py          # OHLC monitoring
│   │   └── completeness_analyzer.py
│   ├── database/           # Adatbázis kezelés
│   │   ├── __init__.py
│   │   ├── tick_storage.py
│   │   ├── ohlc_storage.py
│   │   └── completeness_storage.py
│   └── utils/              # Segéd funkciók
│       ├── __init__.py
│       └── mt5_client.py   # MT5 Service kliens
└── logs/
```

### 4.2 Főbb Modulok

#### 4.2.1 Gap Filler
**Felelősség:** Hiányzó adatok pótlása

**Főbb metódusok:**
- `check_gaps()` - Hiányok detektálása
- `fill_gaps(symbol, start, end)` - Hiány pótlás
- `schedule_gap_fill()` - Ütemezett gap fill

#### 4.2.2 Tick Collector
**Felelősség:** Real-time tick gyűjtés

**Főbb metódusok:**
- `start_collection(symbols)` - Gyűjtés indítás
- `stop_collection()` - Gyűjtés leállítás
- `collect_tick(symbol)` - Egy tick lekérés
- `save_batch(ticks)` - Batch mentés

#### 4.2.3 Bar Monitor
**Felelősség:** OHLC candle monitoring

**Főbb metódusok:**
- `start_monitoring(symbols, timeframes)` - Monitoring indítás
- `check_new_bars()` - Új bárok ellenőrzés
- `download_completed_bar(symbol, tf)` - Lezárt bar letöltés

#### 4.2.4 Completeness Analyzer
**Felelősség:** Adatminőség ellenőrzés

**Főbb metódusok:**
- `analyze_day(symbol, date)` - Egy nap elemzés
- `analyze_period(symbol, start, end)` - Időszak elemzés
- `update_completeness_status()` - Státusz frissítés

---

## 5. Teljesítmény Optimalizáció

### 5.1 Batch Írás
- Tick-ek gyűjtése bufferben
- 100-1000 tick-enként írás
- Transaction használat

### 5.2 Index Használat
- Timestamp index minden táblán
- Gyors range query-k

### 5.3 Párhuzamos Letöltés
- Multi-threading symbol-onként
- Max 5 párhuzamos letöltés

### 5.4 Cache
- Utolsó N tick cache-elése memóriában
- Gyakori lekérdezések cache-elése

---

## 6. Hibakezelés

### 6.1 MT5 Service Nem Elérhető
- Újrapróbálkozás 3x
- Exponential backoff (1s, 2s, 4s)
- Hiba notification Backend API-nak

### 6.2 Adatbázis Hiba
- Transaction rollback
- Újrapróbálkozás
- Log fájlba írás

### 6.3 Lemez Megtelt
- Ellenőrzés letöltés előtt
- Figyelmeztetés ha < 10% hely
- Leállás ha < 5% hely

---

## 7. Tesztelés

### 7.1 Unit Tesztek
- Gap detektálás logika
- Completeness számítás
- Batch írás működés

### 7.2 Integration Tesztek
- MT5 Service kommunikáció
- Adatbázis írás/olvasás
- WebSocket üzenetek

### 7.3 Performance Tesztek
- 10000 tick/s írás
- 1 millió tick lekérés < 1s
- Párhuzamos letöltések

---

**Dokumentum vége**
