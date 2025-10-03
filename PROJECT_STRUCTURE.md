# Project Structure - MT5 Trading Platform 2.0

## Complete Directory Tree

```
Metatrade.Python.Framework.2/
│
├── 📄 README.md                          # Main project documentation
├── 📄 QUICK_START.md                     # Quick start guide for developers
├── 📄 PHASE1_IMPLEMENTATION_REPORT.md    # Phase 1 implementation report
├── 📄 PROJECT_STRUCTURE.md               # This file
├── 📄 config.json                        # Central configuration file
├── 📄 .env.example                       # Environment variables template
│
├── 🔧 test_services.py                   # Service health check test script
├── 🔧 start_all_services.sh              # Start all services script
├── 🔧 stop_all_services.sh               # Stop all services script
│
├── 📁 docs/                              # Documentation folder
│   ├── projekt-dokumentacio.md           # Full project documentation
│   ├── kiegeszites-config-realtime-performance.md
│   ├── ugynok-backend-api.md             # Backend API agent docs
│   ├── ugynok-data-service.md            # Data Service agent docs
│   ├── ugynok-mt5-service.md             # MT5 Service agent docs
│   ├── ugynok-pattern-service.md         # Pattern Service agent docs
│   ├── ugynok-strategy-service.md        # Strategy Service agent docs
│   ├── ugynok-ai-service.md              # AI Service agent docs
│   ├── ugynok-backtesting-service.md     # Backtesting Service agent docs
│   └── ugynok-frontend.md                # Frontend agent docs
│
├── 📁 services/                          # All microservices
│   │
│   ├── 📁 backend-api/                   # Backend API Service (Port 5000)
│   │   ├── main.py                       # Service entry point
│   │   ├── requirements.txt              # Python dependencies
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── api/
│   │       │   ├── __init__.py
│   │       │   └── routes.py             # API endpoints
│   │       ├── core/
│   │       │   ├── __init__.py
│   │       │   └── service.py            # ServiceOrchestrator
│   │       └── models/
│   │           └── __init__.py
│   │
│   ├── 📁 data-service/                  # Data Service (Port 5001)
│   │   ├── main.py                       # Service entry point
│   │   ├── requirements.txt              # Python dependencies
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── api/
│   │       │   ├── __init__.py
│   │       │   └── routes.py             # API endpoints
│   │       ├── core/
│   │       │   ├── __init__.py
│   │       │   └── service.py            # DataCollector
│   │       └── models/
│   │           └── __init__.py
│   │
│   ├── 📁 mt5-service/                   # MT5 Service (Port 5002)
│   │   ├── main.py                       # Service entry point
│   │   ├── requirements.txt              # Python dependencies
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── api/
│   │       │   ├── __init__.py
│   │       │   └── routes.py             # API endpoints
│   │       ├── core/
│   │       │   ├── __init__.py
│   │       │   └── service.py            # MT5Connector
│   │       └── models/
│   │           └── __init__.py
│   │
│   ├── 📁 pattern-service/               # Pattern Service (Port 5003)
│   │   ├── main.py                       # Service entry point
│   │   ├── requirements.txt              # Python dependencies
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── api/
│   │       │   ├── __init__.py
│   │       │   └── routes.py             # API endpoints
│   │       ├── core/
│   │       │   ├── __init__.py
│   │       │   └── service.py            # PatternAnalyzer
│   │       └── models/
│   │           └── __init__.py
│   │
│   ├── 📁 strategy-service/              # Strategy Service (Port 5004)
│   │   ├── main.py                       # Service entry point
│   │   ├── requirements.txt              # Python dependencies
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── api/
│   │       │   ├── __init__.py
│   │       │   └── routes.py             # API endpoints
│   │       ├── core/
│   │       │   ├── __init__.py
│   │       │   └── service.py            # StrategyEngine
│   │       └── models/
│   │           └── __init__.py
│   │
│   ├── 📁 ai-service/                    # AI Service (Port 5005)
│   │   ├── main.py                       # Service entry point
│   │   ├── requirements.txt              # Python dependencies
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── api/
│   │       │   ├── __init__.py
│   │       │   └── routes.py             # API endpoints
│   │       ├── core/
│   │       │   ├── __init__.py
│   │       │   └── service.py            # AIModelManager
│   │       └── models/
│   │           └── __init__.py
│   │
│   └── 📁 backtesting-service/           # Backtesting Service (Port 5006)
│       ├── main.py                       # Service entry point
│       ├── requirements.txt              # Python dependencies
│       └── app/
│           ├── __init__.py
│           ├── api/
│           │   ├── __init__.py
│           │   └── routes.py             # API endpoints
│           ├── core/
│           │   ├── __init__.py
│           │   └── service.py            # BacktestEngine
│           └── models/
│               └── __init__.py
│
├── 📁 shared/                            # Shared utilities and models
│   ├── __init__.py
│   ├── models/
│   │   └── __init__.py                   # ServiceStatus, TickData, OHLCData
│   ├── utils/
│   │   └── __init__.py                   # setup_logging(), format_response()
│   └── config/
│       └── __init__.py                   # load_config(), get_service_config()
│
├── 📁 database/                          # Database files (will be created)
│   └── (SQLite databases will be here)
│
└── 📁 logs/                              # Log files (will be created)
    └── (Service logs will be here)
```

## File Count Summary

### Services
- **7 Services** created
- **49 Python files** in services/
- **7 requirements.txt** files

### Shared Components
- **4 Python files** in shared/
- Models, Utils, Config packages

### Documentation & Scripts
- **3 Main documentation files** (README, QUICK_START, PHASE1_REPORT)
- **1 Configuration file** (config.json)
- **1 Environment template** (.env.example)
- **3 Utility scripts** (test, start, stop)

### Total Project Files
- **~70+ files** created in Phase 1

## Service Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Angular)                        │
│                     http://localhost:4200                        │
│                         (Phase 5)                                │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ HTTP/REST + WebSocket
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              BACKEND API SERVICE (Orchestrator)                  │
│                     http://localhost:5000                        │
│  - Frontend API endpoints                                        │
│  - Service orchestration (start/stop/monitor)                   │
│  - WebSocket hub for real-time events                           │
└─────────────────────────────────────────────────────────────────┘
          │            │            │            │            │            │
          │ REST       │ REST       │ REST       │ REST       │ REST       │ REST
          ▼            ▼            ▼            ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Data Service │ │ MT5 Service  │ │Pattern Svc   │ │Strategy Svc  │ │  AI Service  │ │Backtest Svc  │
│  Port: 5001  │ │  Port: 5002  │ │  Port: 5003  │ │  Port: 5004  │ │  Port: 5005  │ │  Port: 5006  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## Technology Stack

### Backend Services
- **Framework:** FastAPI 0.104.1
- **Server:** Uvicorn 0.24.0
- **Python:** 3.11.9

### Libraries by Service
- **Backend API:** FastAPI, websockets, pydantic
- **Data Service:** FastAPI, pandas, numpy, sqlalchemy
- **MT5 Service:** FastAPI, MetaTrader5
- **Pattern Service:** FastAPI, pandas, numpy, ta-lib
- **Strategy Service:** FastAPI, pandas, numpy
- **AI Service:** FastAPI, tensorflow, scikit-learn
- **Backtesting Service:** FastAPI, pandas, numpy

### Database
- **SQLite 3** for all data storage
- Partitioned by symbol/month for ticks
- Partitioned by symbol for OHLC

## Configuration

### config.json Structure
```json
{
  "system": {...},           // Environment, log level
  "services": {
    "backend-api": {...},    // Port 5000
    "data-service": {...},   // Port 5001
    "mt5-service": {...},    // Port 5002
    "pattern-service": {...},// Port 5003
    "strategy-service": {...},// Port 5004
    "ai-service": {...},     // Port 5005
    "backtesting-service": {...} // Port 5006
  },
  "database": {...},         // SQLite config
  "trading": {...},          // Symbols, timeframes
  "monitoring": {...}        // Health check interval
}
```

## Next Steps

See **PHASE1_IMPLEMENTATION_REPORT.md** for detailed Phase 2 recommendations.

---

**Structure Last Updated:** October 3, 2025
**Phase:** Phase 1 Complete ✅
