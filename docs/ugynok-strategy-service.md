# Strategy Service Ügynök Leírás

**Verzió:** 1.0.0
**Létrehozva:** 2025. október 3.
**Service Port:** 5004

---

## 1. Ügynök Áttekintés

### 1.1 Ügynök Neve
**Strategy Service Ügynök**

### 1.2 Fő Felelősségek

A Strategy Service Ügynök felelős a **kereskedési stratégiák futtatásáért, backtesting-ért és pozíció menedzsmentért**.

### 1.3 Service Típus
- **Strategy Execution Engine**
- **Backtesting Engine**
- **Position Manager**

---

## 2. Részletes Felelősségek

### 2.1 Stratégia Futtatás

**Stratégia Típusok:**

1. **Python kód alapú**
   - Egy Python fájl = egy stratégia
   - Osztály definíció szükséges
   - Metódusok: `on_tick()`, `on_bar()`, entry/exit logika

2. **Drag-and-Drop alapú**
   - Frontend-en vizuálisan építhető
   - Blokkok: Indikátor, Pattern, Feltétel, Akció
   - JSON formátumban mentve

**Stratégia Láncolás (Pattern + Indikátor):**
```
IF (Pattern "EMA Crossover" észlelve)
  AND (RSI < 30)
  AND (MACD > 0)
THEN
  Nyiss BUY pozíciót
```

**Futtatási Módok:**
1. **Paper Trading** - Virtuális kereskedés (demo)
2. **Live Trading** - Valós kereskedés MT5-ben
3. **Backtest** - Historikus adat alapú teszt

### 2.2 Backtesting

**Cél:** Stratégia tesztelése historikus adatokon

**Backtest Folyamat:**
```
1. Betöltöd a historikus OHLC adatokat (Data Service-ből)
2. Végigmész minden bár-on (time loop)
3. Minden bár-nál:
   - Számítsd az indikátorokat
   - Futtasd a stratégia logikát
   - Ha entry signal → nyiss virtuális pozíciót
   - Ha exit signal → zárd virtuális pozíciót
   - Számítsd a profit/loss-t
4. Végén:
   - Összesítsd a teljesítményt
   - Generálj metrikákat
```

**Teljesítmény Metrikák:**
- **Total Profit/Loss** - Összes nyereség/veszteség
- **Win Rate** - Nyerő ügyletek aránya
- **Profit Factor** - Nyereség / Veszteség arány
- **Max Drawdown** - Maximális tőke visszaesés
- **Sharpe Ratio** - Kockázattal korrigált hozam
- **Average Trade** - Átlagos ügylet eredmény
- **Trade Count** - Ügyletek száma

**Backtest Eredmény:**
```json
{
  "strategy_id": 1,
  "symbol": "EURUSD",
  "timeframe": "M15",
  "period": {
    "start": "2025-01-01",
    "end": "2025-10-03"
  },
  "metrics": {
    "total_trades": 150,
    "winning_trades": 90,
    "losing_trades": 60,
    "win_rate": 60.0,
    "total_profit": 5000.0,
    "total_loss": -2000.0,
    "net_profit": 3000.0,
    "profit_factor": 2.5,
    "max_drawdown": -800.0,
    "sharpe_ratio": 1.5
  },
  "trades": [...]
}
```

### 2.3 Stratégia Kezelés

**Stratégia Definíció (Python):**
```python
# strategies/ema_crossover.py

class EMACrossoverStrategy:
    """EMA Crossover Stratégia"""

    def __init__(self):
        self.name = "EMA Crossover"
        self.symbols = ["EURUSD", "GBPUSD"]
        self.timeframe = "M15"
        self.position = None

    def on_bar(self, data: pd.DataFrame):
        """
        Új bar érkezett

        Args:
            data: OHLC adatok + indikátorok
        """
        ema_20 = data['EMA_20'].iloc[-1]
        ema_50 = data['EMA_50'].iloc[-1]
        ema_20_prev = data['EMA_20'].iloc[-2]
        ema_50_prev = data['EMA_50'].iloc[-2]

        # Golden Cross (Bullish)
        if ema_20 > ema_50 and ema_20_prev <= ema_50_prev:
            if not self.position:
                self.open_position("BUY", 0.1)

        # Death Cross (Bearish)
        elif ema_20 < ema_50 and ema_20_prev >= ema_50_prev:
            if self.position and self.position.type == "BUY":
                self.close_position()

    def open_position(self, type, volume):
        """Pozíció nyitás"""
        # MT5 Service hívása
        pass

    def close_position(self):
        """Pozíció zárás"""
        # MT5 Service hívása
        pass
```

**Stratégia Betöltés:**
1. Python fájl beolvasása
2. Osztály dinamikus importálása
3. Validálás
4. Stratégia lista-ba felvétel

**Drag-and-Drop Stratégia (JSON):**
```json
{
  "name": "My Strategy",
  "blocks": [
    {
      "type": "pattern",
      "pattern_id": 1,
      "pattern_name": "EMA Crossover"
    },
    {
      "type": "condition",
      "operator": "AND",
      "left": {"indicator": "RSI", "operator": "<", "value": 30}
    },
    {
      "type": "action",
      "action": "OPEN_BUY",
      "volume": 0.1,
      "sl": 50,
      "tp": 100
    }
  ]
}
```

### 2.4 Pozíció Menedzsment (Komplex)

A Strategy Service egyik **legfontosabb** funkciója a professzionális pozíciókezelés. Ez a rész kritikus a sikeres kereskedéshez.

#### 2.4.1 Risk Management (Kockázatkezelés)

**Max Pozíció Méret:**
- Számítás az account equity alapján
- Példa: Max 2% kockázat per trade
```
account_equity = 10,000 USD
risk_per_trade = 2%  (200 USD)
stop_loss_pips = 50 pips
pip_value = 10 USD/lot (EURUSD standard lot)

lot_size = risk_per_trade / (stop_loss_pips * pip_value)
lot_size = 200 / (50 * 10) = 0.4 lot
```

**Max Nyitott Pozíciók:**
- Egyidejűleg maximum N pozíció lehet nyitva (pl. 5)
- Ellenőrzés pozíció nyitás előtt
- Ha eléri a limitet → nem nyit új pozíciót

**Napi Veszteség Limit:**
- Ha a nap folyamán eléri a max veszteséget (pl. -500 USD) → kereskedés leállítása napra
- Reset másnap 00:00-kor

**Max Drawdown Limit:**
- Ha az equity visszaesés eléri a limitet (pl. -10%) → összes pozíció zárása, kereskedés leállítása

**Correlation Check:**
- Azonos irányba mozgó párok pozícióinak ellenőrzése
- Példa: EURUSD és GBPUSD gyakran együtt mozog → ne nyiss 5 BUY pozíciót egyszerre mindkettőn

#### 2.4.2 Stop Loss Kezelés

**Fix Stop Loss:**
- Egyszerű, fix pip távolság
- Példa: 50 pip SL

**ATR-alapú Stop Loss:**
- Dinamikus SL a volatilitás alapján
- Példa: SL = Entry ± (2 * ATR)

**Percentage-alapú Stop Loss:**
- Account equity % alapján
- Példa: SL olyan távolságban, hogy max 2% veszteség

**Stop Loss to Break Even (SL to BE):**
```
Pozíció nyitva: BUY EURUSD @ 1.1000, SL @ 1.0950, TP @ 1.1100

IF current_price >= entry_price + (take_profit - entry_price) * 0.5:
  # Ha elértük a profit 50%-át
  move_stop_loss_to_break_even()
  # Új SL: 1.1000 (entry ár) vagy kissé felette (pl. +5 pip)
```

**Trailing Stop:**
```
Pozíció nyitva: BUY EURUSD @ 1.1000, SL @ 1.0950

Ár mozog felfelé:
- 1.1020 → SL mozog 1.0970-re (20 pip trailing)
- 1.1050 → SL mozog 1.1000-re (50 pip trailing)
- 1.1080 → SL mozog 1.1030-ra (80 pip trailing)

Ha ár visszafordul és eléri az SL-t → pozíció zárva profit-tal

Trailing distance: fix (pl. 30 pip) vagy ATR-alapú
```

#### 2.4.3 Take Profit Kezelés

**Fix Take Profit:**
- Egyszerű, fix pip távolság
- Példa: 100 pip TP

**Risk/Reward alapú TP:**
- Példa: Ha SL = 50 pip, akkor TP = 100 pip (1:2 RR)

**Részleges Take Profit (Partial Close):**
```
Pozíció: BUY 1.0 lot EURUSD @ 1.1000, TP1 @ 1.1050, TP2 @ 1.1100

TP1 elérésekor (1.1050):
  - Zárja a pozíció 50%-át (0.5 lot)
  - Maradék 0.5 lot fut tovább TP2 felé
  - SL mozog BE-re (1.1000)

TP2 elérésekor (1.1100):
  - Zárja a maradék 50%-ot (0.5 lot)
```

**Trailing TP:**
- TP is mozoghat az árral (kevésbé gyakori)

#### 2.4.4 Pozícióépítés (Position Scaling/Pyramiding)

**Cél:** Egy nyerő pozíció méretének növelése a trend folytatódása esetén

**Scaling In (Pyramiding):**
```
Stratégia: Trend-following, BUY signal EURUSD

Entry 1: BUY 0.1 lot @ 1.1000, SL @ 1.0950
  → Ár emelkedik 1.1030-ra

Entry 2: BUY 0.1 lot @ 1.1030, SL @ 1.0980
  → Ár emelkedik 1.1060-ra

Entry 3: BUY 0.1 lot @ 1.1060, SL @ 1.1010

Végső pozíció: 0.3 lot (3 entry), átlagár: 1.1030
```

**Szabályok:**
- Csak nyerő pozíció esetén nyitható újabb pozíció
- Max 3-5 scaling step
- Minden új entry-nél az összes SL mozog felfelé

**Scaling Out:**
- A pozíció fokozatos zárása
- Lásd: Részleges Take Profit

**Average Down (NEM ajánlott!):**
- Veszteséges pozíció átlagolása (nem javasolt, mert növeli a kockázatot)

#### 2.4.5 Pozíció Monitorozás

**Real-time Monitoring Loop:**
```
LOOP (every tick vagy every N seconds):
  FOR EACH open_position:
    # 1. Ellenőrzi a trailing stop-ot
    check_and_update_trailing_stop(position)

    # 2. Ellenőrzi a SL to BE feltételt
    check_and_move_sl_to_be(position)

    # 3. Ellenőrzi a partial TP-t
    check_and_close_partial_tp(position)

    # 4. Ellenőrzi az emergency exit feltételeket
    check_emergency_exit(position)

    # 5. Ellenőrzi a max time in trade-et
    check_max_time_in_trade(position)
```

**Emergency Exit:**
- Váratlan hírek, piaci események
- Account equity esése kritikus szint alá
- Service leállás esetén (fail-safe)

**Max Time in Trade:**
- Ha egy pozíció túl sokáig nyitva van (pl. 7 nap) → automatikus zárás
- Főleg intraday stratégiákhoz

---

## 3. REST API Endpointok

### 3.1 Stratégiák

#### GET /strategies
Stratégiák listája

#### POST /strategies
Új stratégia létrehozása

**Body:**
```json
{
  "name": "EMA Crossover",
  "type": "python",
  "code": "class EMACrossover:...",
  "symbols": ["EURUSD"],
  "timeframes": ["M15"],
  "mode": "paper"
}
```

#### PUT /strategies/{id}
Stratégia módosítása

#### DELETE /strategies/{id}
Stratégia törlése

#### POST /strategies/{id}/start
Stratégia indítása (paper/live)

#### POST /strategies/{id}/stop
Stratégia leállítása

#### GET /strategies/{id}/performance
Stratégia teljesítmény

### 3.2 Backtesting

#### POST /strategies/{id}/backtest
Backtest futtatása

**Body:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "M15",
  "start_date": "2025-01-01",
  "end_date": "2025-10-03",
  "initial_balance": 10000
}
```

#### GET /backtests
Backtest eredmények listája

#### GET /backtests/{id}
Egy backtest részletei

---

## 4. Implementációs Útmutató

### 4.1 Projekt Struktúra

```
strategy-service/
├── main.py
├── config.py
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── strategies.py
│   │   └── backtests.py
│   ├── core/
│   │   ├── strategy_engine.py     # Stratégia futtatás
│   │   ├── backtest_engine.py     # Backtesting
│   │   └── position_manager.py    # Pozíció kezelés
│   ├── strategies/                # Beépített stratégiák
│   │   └── ema_crossover.py
│   └── database/
│       └── strategy_db.py
└── logs/
```

---

## 5. Tesztelés

### 5.1 Backtest Tesztek
- Stratégia logika helyesség
- Teljesítmény metrikák pontosság

### 5.2 Paper Trading Tesztek
- Virtuális pozíció kezelés
- Real-time működés

---

**Dokumentum vége**
