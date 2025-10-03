# Frontend Ügynök Leírás

**Verzió:** 1.0.0
**Létrehozva:** 2025. október 3.
**Port:** 4200 (Angular default)

---

## 1. Ügynök Áttekintés

### 1.1 Ügynök Neve
**Frontend Ügynök (Angular SPA)**

### 1.2 Fő Felelősségek

A Frontend Ügynök felelős a **felhasználói felület megjelenítéséért és a backend szolgáltatások kezeléséért**. Ez egy Angular-alapú Single Page Application (SPA), amely a Backend API Service-en keresztül kommunikál az összes többi service-szel.

### 1.3 Alkalmazás Típus
- **Single Page Application (SPA)**
- **Real-time Dashboard**
- **Service Management UI**

### 1.4 Technológia

**Framework:** Angular 17+
**UI Library:** Angular Material
**Chart Library:** Opcionális (lightweight-charts vagy chart.js)
**Real-time:** WebSocket vagy SSE

---

## 2. Részletes Felelősségek

### 2.1 Dashboard (Főoldal)

**Cél:** Áttekintés az egész rendszerről egy helyen

**Komponensek:**

#### 2.1.1 Service Status Panel
**Megjelenítés:**
```
┌─────────────────────────────────────────────┐
│          SERVICE ÁLLAPOT                    │
├─────────────────────────────────────────────┤
│ ● Backend API        [ONLINE]   CPU: 2%    │
│ ● Data Service       [ONLINE]   CPU: 5%    │
│ ● MT5 Service        [ONLINE]   Connected  │
│ ● Pattern Service    [ONLINE]   Active: 5  │
│ ● Strategy Service   [ONLINE]   Running: 2 │
│ ● Backtesting Svc    [IDLE]     Queue: 0   │
│ ● AI Service         [ONLINE]   Models: 3  │
├─────────────────────────────────────────────┤
│ [Start All] [Stop All] [Restart All]       │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Service státusz megjelenítése (Online/Offline/Error)
- Service indítás/leállítás gomb
- CPU/Memória használat (ha elérhető)
- Health check 5 másodpercenként (automatic refresh)

#### 2.1.2 Account Info Panel
**Megjelenítés:**
```
┌─────────────────────────────────────────────┐
│          MT5 ACCOUNT INFORMÁCIÓ             │
├─────────────────────────────────────────────┤
│ Account Number:  12345678                   │
│ Balance:         $10,500.00                 │
│ Equity:          $10,750.00                 │
│ Margin:          $2,150.00                  │
│ Free Margin:     $8,600.00                  │
│ Margin Level:    500.0%                     │
├─────────────────────────────────────────────┤
│ Open Positions:  3                          │
│ Pending Orders:  1                          │
│ Today's P/L:     +$250.00 (+2.38%)          │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Real-time frissítés (WebSocket)
- Equity graph (mini chart, opcionális)

#### 2.1.3 Nyitott Pozíciók Listája
**Táblázat:**
| Ticket | Symbol | Type | Volume | Entry | Current | S/L | T/P | Profit |
|--------|--------|------|--------|-------|---------|-----|-----|--------|
| 123456 | EURUSD | BUY  | 0.10   | 1.1050| 1.1065  | 1.1000 | 1.1150 | +$15.00 |
| 123457 | GBPUSD | SELL | 0.05   | 1.2600| 1.2590  | 1.2650 | 1.2550 | +$5.00  |

**Funkciók:**
- Pozíció részletek
- Close gomb (azonnali zárás)
- Modify gomb (SL/TP módosítás)
- Profit szín kódolás (zöld=nyereség, piros=veszteség)

#### 2.1.4 Recent Signals (Legutóbbi Jelzések)
**Lista:**
```
┌─────────────────────────────────────────────┐
│          LEGUTÓBBI JELZÉSEK                 │
├─────────────────────────────────────────────┤
│ 14:35 | EURUSD M15 | EMA Crossover | BUY   │
│ 14:20 | GBPUSD H1  | RSI Oversold  | BUY   │
│ 13:45 | USDJPY M15 | MACD Signal   | SELL  │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Pattern/Strategy jelzések megjelenítése
- Real-time frissítés
- Részletek nézet (kattintásra)

### 2.2 Data Management (Adatkezelés)

**Cél:** Tick/OHLC adatok letöltése, gap fill kezelése

#### 2.2.1 Symbol Selection
**Megjelenítés:**
```
┌─────────────────────────────────────────────┐
│          SYMBOL KIVÁLASZTÁS                 │
├─────────────────────────────────────────────┤
│ [x] EURUSD                                  │
│ [x] GBPUSD                                  │
│ [ ] USDJPY                                  │
│ [ ] AUDUSD                                  │
│ ...                                         │
├─────────────────────────────────────────────┤
│ [Select All] [Deselect All] [Save]         │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Checkbox lista az összes elérhető symbol-ra
- Kiválasztott symbol-ok mentése (Backend API-nak küldi)

#### 2.2.2 Gap Fill
**Megjelenítés:**
```
┌─────────────────────────────────────────────┐
│          GAP FILL                           │
├─────────────────────────────────────────────┤
│ Kiválasztott Symbolok: EURUSD, GBPUSD      │
│ Utolsó Gap Fill: 2025-10-03 10:00:00       │
│ Státusz: Idle                               │
├─────────────────────────────────────────────┤
│ [Run Gap Fill Now]                          │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Manuális gap fill indítás
- Progress bar (amikor fut)
- Log megjelenítés (real-time)

#### 2.2.3 Historical Data Download (Előzmény Letöltés)
**Form:**
```
┌─────────────────────────────────────────────┐
│          ELŐZMÉNY LETÖLTÉS                  │
├─────────────────────────────────────────────┤
│ Symbol:      [EURUSD ▼]                     │
│ Start Date:  [2024-01-01] 📅                │
│ End Date:    [2024-12-31] 📅                │
│ Data Types:  [x] Tick  [x] OHLC             │
├─────────────────────────────────────────────┤
│ [Download]                                  │
└─────────────────────────────────────────────┘
```

**Progress Dialog:**
```
┌─────────────────────────────────────────────┐
│          LETÖLTÉS FOLYAMATBAN               │
├─────────────────────────────────────────────┤
│ Symbol: EURUSD                              │
│ Progress: ████████████░░░░░░  65%          │
│ Downloaded: 1,500,000 ticks                 │
│ Remaining: ~45 seconds                      │
├─────────────────────────────────────────────┤
│ [Cancel]                                    │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Dátum picker
- Multi-symbol letöltés (batch)
- Progress tracking (WebSocket)
- Cancel lehetőség

#### 2.2.4 Data Statistics (Adatbázis Statisztika)
**Táblázat:**
| Symbol | Year | Month | Ticks | OHLC Bars | File Size | Completeness |
|--------|------|-------|-------|-----------|-----------|--------------|
| EURUSD | 2025 | 01    | 2.5M  | 50K       | 150 MB    | 100% ✓       |
| EURUSD | 2025 | 02    | 2.3M  | 48K       | 140 MB    | 98%  ⚠      |
| GBPUSD | 2025 | 01    | 2.1M  | 50K       | 130 MB    | 100% ✓       |

**Funkciók:**
- Completeness színkódolás
- Kattintásra: részletes napi bontás

### 2.3 Pattern Management (Pattern Kezelés)

**Cél:** Pattern-ek feltöltése, engedélyezése, letiltása

#### 2.3.1 Pattern List
**Táblázat:**
| ID | Name          | Type       | Symbols       | Timeframes | Status   | Actions |
|----|---------------|------------|---------------|------------|----------|---------|
| 1  | EMA Crossover | Indicator  | EURUSD, GBPUSD| M15, H1    | ✓ Active | [Edit] [Disable] [Delete] |
| 2  | RSI Oversold  | Indicator  | All           | M15        | ✓ Active | [Edit] [Disable] [Delete] |
| 3  | Doji Pattern  | Candlestick| EURUSD        | M15        | ✗ Inactive| [Edit] [Enable] [Delete] |

**Funkciók:**
- Pattern lista megjelenítés
- Enable/Disable toggle
- Edit gomb → részletes nézet
- Delete gomb (megerősítéssel)

#### 2.3.2 Upload New Pattern
**Form:**
```
┌─────────────────────────────────────────────┐
│          ÚJ PATTERN FELTÖLTÉS               │
├─────────────────────────────────────────────┤
│ Pattern Name: [________________]            │
│ Description:  [________________]            │
│ Python File:  [Choose File] pattern.py     │
├─────────────────────────────────────────────┤
│ [Upload] [Cancel]                           │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Python fájl feltöltés
- Validálás (szintaxis ellenőrzés Backend-en)
- Sikeres feltöltés után megjelenik a listában

#### 2.3.3 Pattern Details/Edit
**Megjelenítés:**
```
┌─────────────────────────────────────────────┐
│          PATTERN RÉSZLETEK                  │
├─────────────────────────────────────────────┤
│ Name: EMA Crossover                         │
│ Type: Indicator                             │
│ Symbols: [EURUSD ▼] [GBPUSD ▼] [Add]       │
│ Timeframes: [M15 ▼] [H1 ▼] [Add]           │
│ Status: [x] Active                          │
├─────────────────────────────────────────────┤
│ Code Preview:                               │
│ ┌─────────────────────────────────────────┐ │
│ │ class EMACrossover:                     │ │
│ │   def detect(self, data):               │ │
│ │     ...                                 │ │
│ └─────────────────────────────────────────┘ │
├─────────────────────────────────────────────┤
│ [Save] [Cancel]                             │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Pattern beállítások szerkesztése
- Kód előnézet (read-only vagy szerkeszthető - Monaco Editor)
- Symbol/Timeframe lista kezelés

### 2.4 Strategy Management (Stratégia Kezelés)

**Cél:** Stratégiák létrehozása, futtatása, teljesítmény figyelése

#### 2.4.1 Strategy List
**Táblázat:**
| ID | Name            | Mode  | Symbols | Status  | P/L Today | Total P/L | Actions |
|----|-----------------|-------|---------|---------|-----------|-----------|---------|
| 1  | Trend Following | Live  | EURUSD  | ✓ Running| +$50.00  | +$1,200   | [Stop] [Edit] [Backtest] |
| 2  | Scalping        | Paper | GBPUSD  | ✓ Running| +$15.00  | +$300     | [Stop] [Edit] [Go Live] |
| 3  | Swing Trading   | Off   | USDJPY  | ✗ Stopped| $0.00    | -$50      | [Start] [Edit] [Delete] |

**Funkciók:**
- Stratégia lista megjelenítés
- Mode jelzés (Live/Paper/Off)
- Start/Stop gomb
- P/L színkódolás
- Edit/Delete gomb

#### 2.4.2 Create New Strategy

**Opció 1: Python Kód alapú**
```
┌─────────────────────────────────────────────┐
│          ÚJ STRATÉGIA LÉTREHOZÁSA           │
├─────────────────────────────────────────────┤
│ Strategy Name: [________________]           │
│ Type: ● Python Code  ○ Drag & Drop         │
│ Python File: [Choose File] strategy.py     │
├─────────────────────────────────────────────┤
│ [Create] [Cancel]                           │
└─────────────────────────────────────────────┘
```

**Opció 2: Drag & Drop Builder (Későbbi fejlesztés)**
```
┌─────────────────────────────────────────────┐
│          STRATEGY BUILDER                   │
├─────────────────────────────────────────────┤
│ Blocks:                     Canvas:         │
│ ┌─────────────┐            ┌──────────────┐│
│ │ Indicators  │            │ [EMA Cross]  ││
│ │ - EMA       │            │      ↓       ││
│ │ - RSI       │            │    [AND]     ││
│ │ - MACD      │            │      ↓       ││
│ ├─────────────┤            │  [RSI < 30]  ││
│ │ Conditions  │            │      ↓       ││
│ │ - AND       │            │ [OPEN BUY]   ││
│ │ - OR        │            └──────────────┘│
│ ├─────────────┤                             │
│ │ Actions     │                             │
│ │ - OPEN BUY  │                             │
│ │ - CLOSE     │                             │
│ └─────────────┘                             │
├─────────────────────────────────────────────┤
│ [Save] [Test] [Cancel]                      │
└─────────────────────────────────────────────┘
```

#### 2.4.3 Strategy Details/Settings
**Form:**
```
┌─────────────────────────────────────────────┐
│          STRATÉGIA BEÁLLÍTÁSOK              │
├─────────────────────────────────────────────┤
│ Name: Trend Following                       │
│ Symbols: [EURUSD ▼] [Add]                   │
│ Timeframe: [M15 ▼]                          │
│ Mode: ● Live  ○ Paper  ○ Off                │
├─────────────────────────────────────────────┤
│ Risk Management:                            │
│   Max Position Size: [0.10] lot            │
│   Max Open Positions: [3]                   │
│   Daily Loss Limit: [$500]                  │
│   SL pips: [50]                             │
│   TP pips: [100]                            │
│   [x] Trailing Stop (30 pips)              │
│   [x] SL to BE at 50% profit               │
├─────────────────────────────────────────────┤
│ [Save] [Cancel]                             │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Stratégia paraméterek szerkesztése
- Risk management beállítások
- Mode váltás (Live/Paper/Off)

#### 2.4.4 Backtest
**Form:**
```
┌─────────────────────────────────────────────┐
│          BACKTEST INDÍTÁS                   │
├─────────────────────────────────────────────┤
│ Strategy: Trend Following                   │
│ Symbol: [EURUSD ▼]                          │
│ Timeframe: [M15 ▼]                          │
│ Start Date: [2024-01-01] 📅                 │
│ End Date: [2024-12-31] 📅                   │
│ Initial Balance: [$10,000]                  │
├─────────────────────────────────────────────┤
│ [Run Backtest]                              │
└─────────────────────────────────────────────┘
```

**Progress:**
```
┌─────────────────────────────────────────────┐
│          BACKTEST FUTÁS                     │
├─────────────────────────────────────────────┤
│ Progress: ████████████░░░░░░  60%          │
│ Current Date: 2024-08-15                    │
│ Trades Executed: 120                        │
│ Current Balance: $11,500                    │
├─────────────────────────────────────────────┤
│ [Cancel]                                    │
└─────────────────────────────────────────────┘
```

**Results:**
```
┌─────────────────────────────────────────────┐
│          BACKTEST EREDMÉNYEK                │
├─────────────────────────────────────────────┤
│ Net Profit: $3,500 (35%)                    │
│ Total Trades: 250                           │
│ Win Rate: 58%                               │
│ Profit Factor: 1.8                          │
│ Max Drawdown: -$800 (-7.2%)                 │
│ Sharpe Ratio: 1.45                          │
├─────────────────────────────────────────────┤
│ Equity Curve:                               │
│ [Mini Line Chart]                           │
├─────────────────────────────────────────────┤
│ Trade Log: (250 trades)                     │
│ [View Details] [Export CSV]                 │
├─────────────────────────────────────────────┤
│ [Save Results] [Close]                      │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Backtest paraméterek beállítása
- Futás követés (progress bar)
- Eredmények megjelenítése
- Trade log letöltése

### 2.5 AI Management (AI Kezelés)

**Cél:** AI modellek kezelése, training, inference

#### 2.5.1 Model List
**Táblázat:**
| ID | Name              | Symbol | TF  | Type | Accuracy | Trained    | Status | Actions |
|----|-------------------|--------|-----|------|----------|------------|--------|---------|
| 1  | EURUSD M15 LSTM   | EURUSD | M15 | LSTM | 67%      | 2025-09-01 | Active | [Details] [Deactivate] |
| 2  | GBPUSD H1 GRU     | GBPUSD | H1  | GRU  | 62%      | 2025-08-15 | Inactive| [Details] [Activate] |

**Funkciók:**
- Model lista megjelenítés
- Activate/Deactivate toggle
- Details gomb → részletes nézet

#### 2.5.2 Train New Model
**Form:**
```
┌─────────────────────────────────────────────┐
│          AI MODEL TRAINING                  │
├─────────────────────────────────────────────┤
│ Symbol: [EURUSD ▼]                          │
│ Timeframe: [M15 ▼]                          │
│ Model Type: [LSTM ▼]                        │
│ Start Date: [2023-01-01] 📅                 │
│ End Date: [2024-12-31] 📅                   │
├─────────────────────────────────────────────┤
│ Parameters:                                 │
│   Sequence Length: [60]                     │
│   LSTM Units: [50]                          │
│   Epochs: [50]                              │
│   Batch Size: [32]                          │
├─────────────────────────────────────────────┤
│ [Start Training]                            │
└─────────────────────────────────────────────┘
```

**Progress:**
```
┌─────────────────────────────────────────────┐
│          TRAINING FOLYAMATBAN               │
├─────────────────────────────────────────────┤
│ Epoch: 30/50 (60%)                          │
│ Train Loss: 0.0023                          │
│ Val Loss: 0.0031                            │
│ Time Remaining: ~15 minutes                 │
├─────────────────────────────────────────────┤
│ Loss Graph:                                 │
│ [Mini Line Chart]                           │
├─────────────────────────────────────────────┤
│ [Cancel]                                    │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Training paraméterek beállítása
- Futás követés (epoch progress)
- Loss graph (real-time)

#### 2.5.3 AI Prediction View
**Megjelenítés:**
```
┌─────────────────────────────────────────────┐
│          AI ELŐREJELZÉS                     │
├─────────────────────────────────────────────┤
│ Model: EURUSD M15 LSTM v2.0                 │
│ Current Price: 1.10500                      │
│ Predicted Price: 1.10550                    │
│ Direction: ▲ UP                             │
│ Confidence: 82%                             │
│ Timestamp: 2025-10-03 14:30:00              │
├─────────────────────────────────────────────┤
│ [Refresh Prediction]                        │
└─────────────────────────────────────────────┘
```

**Funkciók:**
- Aktív model előrejelzése
- Confidence vizualizálás (progress bar)
- Manual refresh gomb

### 2.6 Settings (Beállítások)

**Cél:** Globális beállítások kezelése

#### 2.6.1 General Settings
```
┌─────────────────────────────────────────────┐
│          ÁLTALÁNOS BEÁLLÍTÁSOK              │
├─────────────────────────────────────────────┤
│ MT5 Terminal Path:                          │
│ [C:\Program Files\...\terminal64.exe]      │
│                                             │
│ MT5 Account:                                │
│ [12345678]                                  │
│                                             │
│ Database Path:                              │
│ [C:\Trading\database\]                      │
├─────────────────────────────────────────────┤
│ [Save] [Reset to Default]                  │
└─────────────────────────────────────────────┘
```

#### 2.6.2 Service Settings
```
┌─────────────────────────────────────────────┐
│          SERVICE BEÁLLÍTÁSOK                │
├─────────────────────────────────────────────┤
│ Auto-start Services on Launch:             │
│ [x] Backend API                             │
│ [x] Data Service (with Gap Fill)           │
│ [x] MT5 Service                             │
│ [x] Pattern Service                         │
│ [ ] Backtesting Service                     │
│ [ ] AI Service                              │
├─────────────────────────────────────────────┤
│ [Save]                                      │
└─────────────────────────────────────────────┘
```

#### 2.6.3 Notification Settings
```
┌─────────────────────────────────────────────┐
│          ÉRTESÍTÉSI BEÁLLÍTÁSOK             │
├─────────────────────────────────────────────┤
│ Notifications:                              │
│ [x] Strategy Signals                        │
│ [x] Pattern Detections                      │
│ [x] Service Errors                          │
│ [x] Gap Fill Completed                      │
│ [ ] Every Trade (might be noisy)           │
├─────────────────────────────────────────────┤
│ Sound Alerts:                               │
│ [x] Enable Sound                            │
│ Volume: [████████░░] 80%                    │
├─────────────────────────────────────────────┤
│ [Save]                                      │
└─────────────────────────────────────────────┘
```

---

## 3. Kommunikáció Backend-del

### 3.1 HTTP REST API

**Angular Service példa:**
```typescript
// services/backend-api.service.ts
@Injectable()
export class BackendApiService {
  private baseUrl = 'http://localhost:5000/api';

  constructor(private http: HttpClient) {}

  // Service status
  getServicesStatus(): Observable<ServiceStatus[]> {
    return this.http.get<ServiceStatus[]>(`${this.baseUrl}/services/status`);
  }

  // Start service
  startService(serviceName: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/services/${serviceName}/start`, {});
  }

  // Get strategies
  getStrategies(): Observable<Strategy[]> {
    return this.http.get<Strategy[]>(`${this.baseUrl}/strategies`);
  }

  // Start backtest
  startBacktest(params: BacktestParams): Observable<BacktestJob> {
    return this.http.post<BacktestJob>(`${this.baseUrl}/strategies/${params.strategyId}/backtest`, params);
  }
}
```

### 3.2 WebSocket (Real-time)

**Angular WebSocket Service példa:**
```typescript
// services/websocket.service.ts
@Injectable()
export class WebSocketService {
  private socket: WebSocket;
  public messages$: Subject<any> = new Subject();

  connect() {
    this.socket = new WebSocket('ws://localhost:5000/ws/events');

    this.socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.messages$.next(data);
    };
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
    }
  }
}
```

**Komponens használat:**
```typescript
// components/dashboard.component.ts
export class DashboardComponent implements OnInit {
  constructor(private wsService: WebSocketService) {}

  ngOnInit() {
    this.wsService.messages$.subscribe((message) => {
      if (message.type === 'new_tick') {
        this.updateTickData(message.data);
      } else if (message.type === 'signal') {
        this.showSignalNotification(message.data);
      } else if (message.type === 'service_status_change') {
        this.updateServiceStatus(message.data);
      }
    });
  }
}
```

---

## 4. Implementációs Útmutató

### 4.1 Projekt Struktúra

```
frontend/
├── src/
│   ├── app/
│   │   ├── components/
│   │   │   ├── dashboard/
│   │   │   ├── data-management/
│   │   │   ├── pattern-management/
│   │   │   ├── strategy-management/
│   │   │   ├── ai-management/
│   │   │   └── settings/
│   │   ├── services/
│   │   │   ├── backend-api.service.ts
│   │   │   ├── websocket.service.ts
│   │   │   └── notification.service.ts
│   │   ├── models/
│   │   │   ├── service-status.model.ts
│   │   │   ├── strategy.model.ts
│   │   │   └── backtest.model.ts
│   │   ├── shared/
│   │   │   ├── header/
│   │   │   ├── sidebar/
│   │   │   └── notification/
│   │   ├── app-routing.module.ts
│   │   └── app.component.ts
│   ├── assets/
│   ├── environments/
│   └── index.html
├── angular.json
├── package.json
└── tsconfig.json
```

### 4.2 Routing

```typescript
// app-routing.module.ts
const routes: Routes = [
  { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'data', component: DataManagementComponent },
  { path: 'patterns', component: PatternManagementComponent },
  { path: 'strategies', component: StrategyManagementComponent },
  { path: 'strategies/:id', component: StrategyDetailsComponent },
  { path: 'ai', component: AiManagementComponent },
  { path: 'settings', component: SettingsComponent },
];
```

### 4.3 State Management

**Egyszerű megoldás (Services):**
- Minden major feature-hez egy service
- Service tárolja a state-et
- Komponensek subscribe-olnak

**Komplexebb megoldás (NgRx - opcionális):**
- Redux pattern Angular-ban
- Centralizált state management
- Actions, Reducers, Selectors

---

## 5. UI/UX Irányelvek

### 5.1 Design Principles

**Egyszerűség:**
- Minimal design
- Tiszta layout
- Nem túlzsúfolt

**Konzisztencia:**
- Egységes színek, font, méret
- Angular Material komponensek

**Responsiveness:**
- Gyors betöltés
- Minimal lag
- Real-time frissítés

### 5.2 Színek

**Alapszínek:**
- Primary: Kék (#2196F3)
- Accent: Zöld (#4CAF50)
- Warn: Piros (#F44336)

**Státusz színek:**
- Online: Zöld
- Offline: Piros
- Warning: Sárga
- Idle: Szürke

**P/L színek:**
- Profit: Zöld
- Loss: Piros
- Break-even: Szürke

### 5.3 Iconok

**Material Icons használata:**
- Service status: `circle` (filled)
- Start: `play_arrow`
- Stop: `stop`
- Edit: `edit`
- Delete: `delete`
- Settings: `settings`

---

## 6. Tesztelés

### 6.1 Unit Tesztek (Jasmine + Karma)

**Példa:**
```typescript
// dashboard.component.spec.ts
describe('DashboardComponent', () => {
  let component: DashboardComponent;
  let fixture: ComponentFixture<DashboardComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [DashboardComponent],
      imports: [HttpClientTestingModule],
    });
    fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should load services on init', () => {
    component.ngOnInit();
    expect(component.services.length).toBeGreaterThan(0);
  });
});
```

### 6.2 E2E Tesztek (Cypress)

**Példa:**
```javascript
// cypress/e2e/dashboard.cy.js
describe('Dashboard', () => {
  it('should display services', () => {
    cy.visit('/dashboard');
    cy.get('.service-status').should('have.length.greaterThan', 0);
  });

  it('should start service', () => {
    cy.visit('/dashboard');
    cy.contains('Start').first().click();
    cy.contains('Online').should('exist');
  });
});
```

---

## 7. Build & Deployment

### 7.1 Development

```bash
# Install dependencies
npm install

# Start dev server
ng serve

# App runs on http://localhost:4200
```

### 7.2 Production Build

```bash
# Build for production
ng build --configuration production

# Output: dist/frontend/
```

### 7.3 Deployment

**Opciók:**

1. **Helyi deployment:**
   - Builded fájlok a `dist/` mappából
   - Egyszerű file server (pl. Python `http.server`)

2. **Nginx:**
   - Static fájlok serving
   - Reverse proxy Backend API-hoz

3. **Docker (opcionális):**
   - Frontend konténer (Nginx + Angular build)

---

**Dokumentum vége**
