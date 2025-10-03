# Backtesting Service Ügynök Leírás

**Verzió:** 1.0.0
**Létrehozva:** 2025. október 3.
**Service Port:** 5006

---

## 1. Ügynök Áttekintés

### 1.1 Ügynök Neve
**Backtesting Service Ügynök**

### 1.2 Fő Felelősségek

A Backtesting Service Ügynök felelős a **kereskedési stratégiák historikus adatokon való teszteléséért**. Ez a service egy **"időgép"** - szimulált időben fut, és visszajátssza a történelmi tick/OHLC adatokat, mintha azok valós időben érkeznének.

### 1.3 Service Típus
- **Backtesting Engine**
- **Time Machine Simulator**
- **Performance Analyzer**

### 1.4 Miért Külön Service?

**Indoklás:**
1. **Erőforrás szeparáció**: Backtesting CPU-igényes, ne lassítsa a live kereskedést
2. **Párhuzamos futtatás**: Több backtest futhat egyszerre különböző paraméterekkel
3. **Skálázhatóság**: Több backtesting service indítható külön gépeken
4. **Tiszta architektúra**: Külön felelősségi kör

---

## 2. Részletes Felelősségek

### 2.1 Szimulált Idő (Time Machine)

**Cél:** Történelmi adatok "visszajátszása" valós időként

**Működés:**

```
Backtest Setup:
  start_date = 2024-01-04
  end_date = 2024-12-31
  timeframe = M15 (15 perces)

Time Loop:
  current_simulated_time = start_date

  WHILE current_simulated_time <= end_date:
    # 1. Betöltés az adatbázisból
    bar_data = load_ohlc_bar(symbol, timeframe, current_simulated_time)

    # 2. Stratégia futtatás
    strategy.on_bar(bar_data)

    # 3. Pozíció kezelés
    update_positions(bar_data)

    # 4. Teljesítmény számítás
    update_performance_metrics()

    # 5. Következő időpont
    current_simulated_time += timeframe_duration
```

**Fontos:**
- A backtesting service **NEM éri el az MT5-öt**
- Minden adat az SQLite adatbázisból jön
- A stratégiák úgy gondolják, hogy "valós időben" futnak

### 2.2 Event-Driven Backtesting

**Mi az Event-Driven Backtest?**
- A backtest minden egyes tick/bar eseményre reagál
- Nem "jövőbe látás" (no look-ahead bias)
- Reálisan szimulálja a valós kereskedést

**Bar-by-Bar Backtest:**
```
FOR EACH bar IN historical_data:
  # Csak a már lezárt bar-ok adatai elérhetők
  available_data = bars[0:current_index]

  # Indikátorok számítása (csak múltbéli adatokból)
  ema_20 = calculate_ema(available_data, 20)
  rsi = calculate_rsi(available_data, 14)

  # Stratégia logika futtatása
  signal = strategy.evaluate(ema_20, rsi, available_data)

  # Ha van signal, pozíció nyitás/zárás
  IF signal == "BUY":
    open_position("BUY", entry_price=bar.close)
  ELIF signal == "SELL" and position_open:
    close_position(exit_price=bar.close)

  # Következő bar
  current_index += 1
```

**Tick-by-Tick Backtest (Opcionális, részletesebb):**
- Minden egyes tick-et visszajátszik
- Pontosabb fill price szimuláció
- Lassabb, de reálisabb

### 2.3 Pozíció Szimuláció

**Virtuális Pozíció Kezelés:**
- A backtest **NEM nyit valós pozíciókat** az MT5-ben
- Minden pozíció virtuális (memóriában)
- Követi a valós pozíció szabályokat (SL, TP, trailing stop)

**Pozíció Megnyitás:**
```python
class VirtualPosition:
    def __init__(self, type, entry_price, volume, sl, tp, timestamp):
        self.type = type  # BUY / SELL
        self.entry_price = entry_price
        self.volume = volume
        self.stop_loss = sl
        self.take_profit = tp
        self.open_timestamp = timestamp
        self.close_timestamp = None
        self.exit_price = None
        self.profit = 0.0
        self.status = "OPEN"

# Pozíció nyitás backtestben
def open_backtest_position(signal, current_bar):
    position = VirtualPosition(
        type=signal.type,
        entry_price=current_bar.close,  # Entry az aktuális bar záróárán
        volume=0.1,
        sl=calculate_stop_loss(signal),
        tp=calculate_take_profit(signal),
        timestamp=current_bar.timestamp
    )
    backtest_positions.append(position)
```

**Pozíció Zárás:**
```python
def check_and_close_positions(current_bar):
    for position in open_positions:
        # 1. Ellenőrzi SL-t
        if position.type == "BUY" and current_bar.low <= position.stop_loss:
            close_position(position, position.stop_loss, "SL")

        # 2. Ellenőrzi TP-t
        elif position.type == "BUY" and current_bar.high >= position.take_profit:
            close_position(position, position.take_profit, "TP")

        # 3. Trailing stop
        elif trailing_stop_enabled:
            update_trailing_stop(position, current_bar)
```

**Spread és Commission:**
- Backtestben figyelembe kell venni a spread-et és a jutalékot
- Entry price: bid (SELL) vagy ask (BUY)
- Spread: fix vagy dinamikus (historikus spread adatok)
- Commission: bróker jutalék per lot

```python
# Spread hozzáadása
def calculate_entry_price(signal_type, bar_close, spread_pips):
    if signal_type == "BUY":
        entry_price = bar_close + spread_pips * point_value
    else:  # SELL
        entry_price = bar_close - spread_pips * point_value
    return entry_price

# Profit számítás jutalékkal
profit = (exit_price - entry_price) * volume * contract_size
profit -= commission_per_lot * volume
```

### 2.4 Teljesítmény Számítás

**Backtest végén összesített metrikák:**

#### 2.4.1 Alap Metrikák

**Total Trades (Összes Ügylet):**
- Hány pozíció nyílt/zárult

**Winning Trades / Losing Trades:**
- Nyerő és vesztes ügyletek száma

**Win Rate (Nyerési Arány):**
```
win_rate = (winning_trades / total_trades) * 100
```

**Total Profit / Total Loss:**
- Összes nyereség és veszteség összege

**Net Profit:**
```
net_profit = total_profit - total_loss
```

**Profit Factor:**
```
profit_factor = total_profit / abs(total_loss)
```
- Jó érték: > 1.5
- Rossz érték: < 1.0

#### 2.4.2 Kockázati Metrikák

**Max Drawdown (Maximális Visszaesés):**
```
Equity csúcs: 12,000 USD
Equity völgy: 9,500 USD

Max Drawdown = (12,000 - 9,500) / 12,000 = 20.83%
```

**Max Consecutive Losses:**
- Egymást követő vesztes ügyletek max száma

**Max Consecutive Wins:**
- Egymást követő nyerő ügyletek max száma

**Average Win / Average Loss:**
```
average_win = total_profit / winning_trades
average_loss = total_loss / losing_trades
```

**Risk/Reward Ratio:**
```
risk_reward = average_win / abs(average_loss)
```

#### 2.4.3 Statisztikai Metrikák

**Sharpe Ratio:**
```
sharpe_ratio = (average_return - risk_free_rate) / std_deviation_of_returns
```
- Jó érték: > 1.0
- Kiváló érték: > 2.0

**Sortino Ratio:**
- Hasonló a Sharpe-hoz, de csak a downside volatilitást veszi figyelembe

**Calmar Ratio:**
```
calmar_ratio = annual_return / max_drawdown
```

**Recovery Factor:**
```
recovery_factor = net_profit / max_drawdown
```

**Expectancy (Várható érték per trade):**
```
expectancy = (win_rate * average_win) - ((1 - win_rate) * abs(average_loss))
```

#### 2.4.4 Trade Analízis

**Longest Winning/Losing Streak:**
- Leghosszabb nyerő/vesztes sorozat

**Average Trade Duration:**
- Átlagos pozíció tartási idő

**Average MAE (Maximum Adverse Excursion):**
- Átlagos maximális veszteség a pozíció alatt

**Average MFE (Maximum Favorable Excursion):**
- Átlagos maximális nyereség a pozíció alatt

### 2.5 Backtest Eredmény Generálás

**Trade Log:**
```json
{
  "trade_id": 1,
  "symbol": "EURUSD",
  "type": "BUY",
  "entry_time": "2024-03-15 10:00:00",
  "entry_price": 1.10500,
  "exit_time": "2024-03-15 14:30:00",
  "exit_price": 1.10700,
  "volume": 0.1,
  "profit": 20.0,
  "exit_reason": "TP",
  "duration_minutes": 270
}
```

**Equity Curve (Tőke Görbe):**
- Minden trade után az equity változása
- Grafikusan ábrázolható (Frontend)

```json
{
  "timestamp": "2024-03-15 14:30:00",
  "equity": 10020.0,
  "balance": 10020.0,
  "drawdown": 0.0
}
```

**Backtest Summary:**
```json
{
  "backtest_id": "bt-12345",
  "strategy": "EMA Crossover",
  "symbol": "EURUSD",
  "timeframe": "M15",
  "period": {
    "start": "2024-01-04",
    "end": "2024-12-31"
  },
  "initial_balance": 10000.0,
  "final_balance": 13500.0,
  "net_profit": 3500.0,
  "total_trades": 250,
  "winning_trades": 145,
  "losing_trades": 105,
  "win_rate": 58.0,
  "profit_factor": 1.8,
  "max_drawdown": -800.0,
  "max_drawdown_percent": 7.2,
  "sharpe_ratio": 1.45,
  "average_trade": 14.0,
  "largest_win": 150.0,
  "largest_loss": -80.0,
  "average_win": 35.0,
  "average_loss": -20.0,
  "max_consecutive_wins": 8,
  "max_consecutive_losses": 5,
  "average_trade_duration_minutes": 180
}
```

### 2.6 Párhuzamos Backtesting

**Parameter Sweep (Paraméter Optimalizáció):**

Több backtest futtatása különböző paraméterekkel párhuzamosan:

```
Stratégia: EMA Crossover
Paraméterek:
  - EMA Fast: 10, 15, 20
  - EMA Slow: 40, 50, 60
  - SL pips: 30, 50, 70

Kombináció: 3 * 3 * 3 = 27 backtest

Minden backtest külön folyamatban/szálon fut
Végén: legjobb paraméter kombináció kiválasztása
```

**Walk-Forward Analysis:**
- Időszak felosztása train/test részekre
- Optimalizáció train perióduson
- Validáció test perióduson
- Rollover és ismétlés

---

## 3. REST API Endpointok

### 3.1 Health Check

#### GET /health
**Leírás:** Service állapot ellenőrzés

**Válasz:**
```json
{
  "status": "healthy",
  "service": "backtesting-service",
  "running_backtests": 2,
  "queued_backtests": 5
}
```

### 3.2 Backtest Indítás

#### POST /backtest/start
**Leírás:** Új backtest futtatása

**Body:**
```json
{
  "strategy_id": 1,
  "symbol": "EURUSD",
  "timeframe": "M15",
  "start_date": "2024-01-04",
  "end_date": "2024-12-31",
  "initial_balance": 10000,
  "parameters": {
    "ema_fast": 20,
    "ema_slow": 50,
    "sl_pips": 50,
    "tp_pips": 100
  }
}
```

**Válasz:**
```json
{
  "success": true,
  "backtest_id": "bt-12345",
  "status": "queued",
  "estimated_duration": 120
}
```

#### POST /backtest/batch
**Leírás:** Több backtest indítása párhuzamosan (parameter sweep)

**Body:**
```json
{
  "strategy_id": 1,
  "symbol": "EURUSD",
  "timeframe": "M15",
  "start_date": "2024-01-04",
  "end_date": "2024-12-31",
  "initial_balance": 10000,
  "parameter_grid": {
    "ema_fast": [10, 15, 20],
    "ema_slow": [40, 50, 60],
    "sl_pips": [30, 50]
  }
}
```

**Válasz:**
```json
{
  "success": true,
  "batch_id": "batch-456",
  "total_backtests": 18,
  "status": "queued"
}
```

### 3.3 Backtest Állapot

#### GET /backtest/{backtest_id}/status
**Leírás:** Backtest futás állapot

**Válasz:**
```json
{
  "backtest_id": "bt-12345",
  "status": "running",
  "progress": 45,
  "current_date": "2024-06-15",
  "trades_executed": 120,
  "current_balance": 11500.0,
  "estimated_remaining_time": 60
}
```

#### GET /backtest/{backtest_id}/results
**Leírás:** Backtest eredmények lekérése

**Válasz:**
```json
{
  "backtest_id": "bt-12345",
  "status": "completed",
  "summary": {
    "net_profit": 3500.0,
    "win_rate": 58.0,
    "profit_factor": 1.8,
    "max_drawdown": -800.0,
    "...": "..."
  },
  "trades": [...],
  "equity_curve": [...]
}
```

### 3.4 Backtest Kezelés

#### POST /backtest/{backtest_id}/stop
**Leírás:** Futó backtest leállítása

#### DELETE /backtest/{backtest_id}
**Leírás:** Backtest eredmények törlése

#### GET /backtest/list
**Leírás:** Összes backtest listája

**Query params:**
- `status`: queued, running, completed, failed
- `strategy_id`: filter stratégia szerint
- `limit`: max eredmény (default 50)

---

## 4. Implementációs Útmutató

### 4.1 Projekt Struktúra

```
backtesting-service/
├── main.py
├── config.py
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── backtest.py
│   │   └── results.py
│   ├── core/
│   │   ├── backtest_engine.py       # Fő backtesting logika
│   │   ├── time_machine.py          # Szimulált idő kezelés
│   │   ├── position_simulator.py    # Virtuális pozíció kezelés
│   │   ├── performance_calculator.py # Teljesítmény metrikák
│   │   └── parameter_optimizer.py   # Parameter sweep
│   ├── strategies/
│   │   └── strategy_loader.py       # Stratégia betöltés
│   ├── database/
│   │   ├── data_loader.py           # Historikus adatok betöltése
│   │   └── backtest_storage.py      # Backtest eredmény tárolás
│   └── utils/
│       └── metrics.py
└── logs/
```

### 4.2 Főbb Modulok

#### 4.2.1 Backtest Engine
**Felelősség:** Backtest futtatás koordinálása

**Főbb metódusok:**
- `run_backtest(strategy, params)` - Backtest indítás
- `process_bar(bar_data)` - Egy bar feldolgozása
- `finalize_backtest()` - Backtest befejezése, eredmény generálás

#### 4.2.2 Time Machine
**Felelősség:** Szimulált idő kezelés

**Főbb metódusok:**
- `initialize(start_date, end_date)` - Idő inicializálás
- `next_bar()` - Következő bar lekérése
- `get_current_time()` - Aktuális szimulált idő

#### 4.2.3 Position Simulator
**Felelősség:** Virtuális pozíció kezelés

**Főbb metódusok:**
- `open_position(type, price, volume)` - Pozíció nyitás
- `close_position(position_id, price)` - Pozíció zárás
- `update_positions(bar_data)` - Pozíciók frissítése (SL/TP check)

#### 4.2.4 Performance Calculator
**Felelősség:** Teljesítmény metrikák számítása

**Főbb metódusok:**
- `calculate_metrics(trades)` - Összes metrika számítás
- `calculate_sharpe_ratio(returns)` - Sharpe ratio
- `calculate_drawdown(equity_curve)` - Drawdown
- `generate_equity_curve(trades)` - Equity görbe generálás

---

## 5. Teljesítmény Optimalizáció

### 5.1 Adatbázis Lekérdezés

**Batch Betöltés:**
- Ne bar-by-bar olvassa az adatbázisból
- Egyszerre töltse be a teljes időszakot memóriába (ha elfér)
- Használjon indexeket (timestamp)

### 5.2 Párhuzamosítás

**Multi-processing:**
- Több backtest párhuzamos futtatása külön process-ekben
- CPU core-ok kihasználása

**Threading:**
- Könnyű párhuzamosítás egyszerűbb esetekre

### 5.3 Caching

**Indicator Cache:**
- Indikátorok értékeinek cache-elése
- Ha ugyanazt a paramétert többször használjuk

---

## 6. Hibakezelés

### 6.1 Hiányzó Adatok
- Ha nincs elég historikus adat → figyelmeztetés
- Indikátor számításhoz szükséges minimum periódus ellenőrzés

### 6.2 Stratégia Hiba
- Try-catch minden stratégia hívás körül
- Ha hiba van → backtest leállítása, részleges eredmény mentése

### 6.3 Memória Limit
- Nagy timeframe + hosszú időszak → sok memória
- Chunked processing ha szükséges

---

## 7. Tesztelés

### 7.1 Unit Tesztek
- Teljesítmény metrikák pontosság
- Pozíció szimuláció helyesség

### 7.2 Integration Tesztek
- Adatbázis adatok betöltése
- Stratégia futtatás

### 7.3 Benchmark Tesztek
- Backtest sebesség mérés
- Memória használat mérés

---

**Dokumentum vége**
