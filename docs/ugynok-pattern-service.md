# Pattern & Indicator Service Ügynök Leírás

**Verzió:** 1.0.0
**Létrehozva:** 2025. október 3.
**Service Port:** 5003

---

## 1. Ügynök Áttekintés

### 1.1 Ügynök Neve
**Pattern & Indicator Service Ügynök**

### 1.2 Fő Felelősségek

A Pattern & Indicator Service Ügynök felelős a **technikai elemzésért**: indikátorok számításáért és chart/candlestick pattern-ek felismeréséért.

### 1.3 Service Típus
- **Technical Analysis Engine**
- **Pattern Recognition**
- **Signal Generator**

---

## 2. Részletes Felelősségek

### 2.1 Technikai Indikátorok

**Támogatott Indikátorok:**

#### Mozgóátlagok
- **SMA** (Simple Moving Average) - Egyszerű mozgóátlag
- **EMA** (Exponential Moving Average) - Exponenciális mozgóátlag
- **WMA** (Weighted Moving Average) - Súlyozott mozgóátlag

#### Oszcillátorok
- **RSI** (Relative Strength Index) - Relatív erősség index
- **Stochastic** - Sztochasztikus oszcillátor
- **MACD** (Moving Average Convergence Divergence) - MACD
- **CCI** (Commodity Channel Index) - Árucsatorna index
- **Williams %R** - Williams százalék R

#### Trend Indikátorok
- **ADX** (Average Directional Index) - Átlagos irányított index
- **Aroon** - Aroon indikátor
- **Parabolic SAR** - Parabolikus SAR

#### Volatility Indikátorok
- **ATR** (Average True Range) - Átlagos valós tartomány
- **Bollinger Bands** - Bollinger szalagok
- **Standard Deviation** - Szórás

#### Volume Indikátorok
- **OBV** (On-Balance Volume) - Egyenlegvolumen
- **MFI** (Money Flow Index) - Pénzáramlási index

**Indikátor Számítás Példa (EMA):**
```
EMA számítás:
1. Első EMA érték = SMA(period)
2. Simítási tényező (multiplier) = 2 / (period + 1)
3. EMA[i] = (Close[i] - EMA[i-1]) * multiplier + EMA[i-1]
```

### 2.2 Candlestick Pattern-ek

**Támogatott Pattern-ek:**

#### Reversal Pattern-ek (Fordulópontok)
- **Doji** - Bizonytalanságot jelez
- **Hammer** - Bullish reversal
- **Hanging Man** - Bearish reversal
- **Shooting Star** - Bearish reversal
- **Inverted Hammer** - Bullish reversal
- **Engulfing** (Bullish/Bearish) - Elnyel minta
- **Harami** (Bullish/Bearish) - Harami minta
- **Morning Star** - Bullish reversal (3 candle)
- **Evening Star** - Bearish reversal (3 candle)
- **Piercing Line** - Bullish reversal
- **Dark Cloud Cover** - Bearish reversal

#### Continuation Pattern-ek (Folytatás)
- **Three White Soldiers** - Bullish folytatás
- **Three Black Crows** - Bearish folytatás
- **Rising/Falling Three Methods**

**Pattern Felismerés Példa (Doji):**
```
Doji felismerés:
1. Body méret = |Open - Close|
2. Wick méret = High - Low
3. Ha Body < Wick * 0.1 → Doji
```

### 2.3 Chart Pattern-ek

**Támogatott Pattern-ek:**

#### Trendvonalak
- **Support** - Támasz vonal
- **Resistance** - Ellenállás vonal
- **Trendline** - Trend vonal (uptrend/downtrend)

#### Formációk
- **Head & Shoulders** - Fej és vállak (bearish)
- **Inverse Head & Shoulders** - Fordított fej és vállak (bullish)
- **Double Top/Bottom** - Dupla csúcs/fenék
- **Triple Top/Bottom** - Tripla csúcs/fenék
- **Triangle** - Háromszög (ascending, descending, symmetrical)
- **Flag** - Zászló
- **Pennant** - Háromszög zászló
- **Wedge** - Ék (rising, falling)
- **Channel** - Csatorna

**Pattern Felismerés Algoritmus:**
```
Support/Resistance vonal detektálás:
1. Lókális minimum/maximum pontok keresése
2. Vízszintes vonal húzása
3. Vonal tesztelése (hány touchpoint)
4. Ha >= 3 touchpoint → valid support/resistance
```

### 2.4 Pattern Definíció Framework

**Pattern Python Fájl Struktúra:**

```python
# patterns/my_custom_pattern.py

class MyCustomPattern:
    """Pattern leírása"""

    def __init__(self):
        self.name = "My Custom Pattern"
        self.description = "Ez egy custom pattern"
        self.required_indicators = ["EMA_20", "RSI"]

    def detect(self, data: pd.DataFrame) -> bool:
        """
        Pattern detektálás

        Args:
            data: OHLC adatok + indikátorok (pandas DataFrame)

        Returns:
            True ha pattern jelen van
        """
        # Pattern logika
        ema_20 = data['EMA_20'].iloc[-1]
        rsi = data['RSI'].iloc[-1]

        if data['close'].iloc[-1] > ema_20 and rsi < 30:
            return True

        return False

    def get_signal(self, data: pd.DataFrame) -> str:
        """
        Jel típusa

        Returns:
            "BUY" vagy "SELL" vagy None
        """
        if self.detect(data):
            return "BUY"
        return None
```

**Pattern Betöltés:**
1. Python fájl beolvasása
2. Osztály dinamikus importálása
3. Validálás (van detect() és get_signal()?)
4. Pattern lista-ba felvétel

**Pattern Mentés Adatbázisba:**
- `pattern_definitions.db` → `patterns` tábla
- Mezők: id, name, description, code (Python kód), enabled, created_at

### 2.5 Real-time Pattern Scanning

**Működés:**

1. **Scan Loop** (1 percenként)
```
LOOP (every 1 minutes):
  FOR EACH symbol IN monitored_symbols:
    FOR EACH timeframe IN timeframes:
      # Lekér adatokat
      data = get_ohlc_data(symbol, timeframe, last_100_bars)

      # Számít indikátorokat
      data = calculate_indicators(data)

      # Fut pattern-eken
      FOR EACH pattern IN enabled_patterns:
        IF pattern.detect(data):
          signal = pattern.get_signal(data)
          send_signal_notification(symbol, timeframe, pattern.name, signal)
```

2. **Signal Notification**
   - WebSocket üzenet Backend API-nak
   - Backend API továbbítja Frontend-nek
   - Frontend jelzés (popup/értesítés)

---

## 3. REST API Endpointok

### 3.1 Health Check

#### GET /health
```json
{
  "status": "healthy",
  "patterns_loaded": 15,
  "indicators_available": 20,
  "scanning": true
}
```

### 3.2 Pattern Management

#### GET /patterns
Pattern-ek listája

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "EMA Crossover",
      "description": "EMA20 átmetszi EMA50-et",
      "enabled": true,
      "required_indicators": ["EMA_20", "EMA_50"]
    }
  ]
}
```

#### POST /patterns
Új pattern feltöltése

**Body:**
```json
{
  "name": "My Pattern",
  "code": "class MyPattern:..."
}
```

#### PUT /patterns/{id}
Pattern módosítása

#### DELETE /patterns/{id}
Pattern törlése

#### POST /patterns/{id}/enable
Pattern engedélyezése

#### POST /patterns/{id}/disable
Pattern tiltása

### 3.3 Scanning

#### POST /scan
Pattern keresés indítása (azonnal)

**Body:**
```json
{
  "symbols": ["EURUSD"],
  "timeframes": ["M15", "H1"]
}
```

**Válasz:**
```json
{
  "success": true,
  "job_id": "scan-12345",
  "estimated_duration": 30
}
```

#### GET /scan/{job_id}/results
Scan eredmények

```json
{
  "job_id": "scan-12345",
  "status": "completed",
  "results": [
    {
      "symbol": "EURUSD",
      "timeframe": "M15",
      "pattern": "EMA Crossover",
      "signal": "BUY",
      "timestamp": "2025-10-03T12:34:56Z"
    }
  ]
}
```

### 3.4 Indicators

#### GET /indicators/{symbol}/{timeframe}
Indikátor értékek lekérése

**Paraméterek:**
- `indicators`: Vesszővel elválasztott lista (pl: "EMA_20,RSI,MACD")
- `limit`: Max bárok száma

**Válasz:**
```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "M15",
  "data": [
    {
      "timestamp": 1696337700000,
      "close": 1.10525,
      "EMA_20": 1.10520,
      "RSI": 45.2,
      "MACD": 0.00015
    }
  ]
}
```

---

## 4. Implementációs Útmutató

### 4.1 Projekt Struktúra

```
pattern-service/
├── main.py
├── config.py
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── patterns.py
│   │   └── indicators.py
│   ├── core/
│   │   ├── indicator_engine.py    # Indikátor számítások
│   │   ├── pattern_detector.py    # Pattern felismerés
│   │   └── pattern_scanner.py     # Real-time scanning
│   ├── indicators/                # Indikátor implementációk
│   │   ├── __init__.py
│   │   ├── moving_averages.py
│   │   ├── oscillators.py
│   │   └── trend.py
│   ├── patterns/                  # Beépített pattern-ek
│   │   ├── __init__.py
│   │   ├── candlestick.py
│   │   └── chart.py
│   └── database/
│       └── pattern_db.py
└── logs/
```

### 4.2 Indicator Engine

**Könyvtár:** pandas + numpy + ta-lib (opcionális)

**Indikátor Számítás:**
```python
def calculate_ema(data: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
    """
    EMA számítás

    Args:
        data: OHLC adatok
        period: EMA periódus
        column: Melyik oszlopon számítson

    Returns:
        EMA értékek
    """
    return data[column].ewm(span=period, adjust=False).mean()
```

### 4.3 Pattern Detector

**Pattern Validáció:**
```python
def validate_pattern(pattern_code: str) -> Tuple[bool, str]:
    """
    Pattern kód validálása

    Returns:
        (valid, error_message)
    """
    try:
        # Futtasd a kódot sandboxban
        exec(pattern_code, {})
        # Ellenőrizd hogy van detect() és get_signal()
        return True, ""
    except Exception as e:
        return False, str(e)
```

---

## 5. Teljesítmény

**Indikátor Számítás:** Gyors (pandas vektorizált műveletek)
**Pattern Scan:** 100 symbol × 6 timeframe = ~5-10 másodperc

---

## 6. Tesztelés

### 6.1 Unit Tesztek
- Indikátor számítás pontosság
- Pattern felismerés helyesség

### 6.2 Backtesting
- Pattern jelzések historikus adatokon
- Pontosság mérése

---

**Dokumentum vége**
