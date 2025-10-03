# MT5 Trading Platform 2.0 - Phase 1 Complete

## Project Overview

This is Phase 1 of the MT5 Trading Platform 2.0 - a microservices-based trading platform with clean architecture.

## Project Structure

```
.
├── services/                    # All microservices
│   ├── backend-api/            # Port 5000 - Main orchestrator
│   ├── data-service/           # Port 5001 - Data collection & storage
│   ├── mt5-service/            # Port 5002 - MT5 connection
│   ├── pattern-service/        # Port 5003 - Technical indicators & patterns
│   ├── strategy-service/       # Port 5004 - Strategy execution
│   ├── ai-service/             # Port 5005 - AI/ML models
│   └── backtesting-service/    # Port 5006 - Backtesting engine
├── shared/                      # Shared utilities and models
│   ├── models/                 # Common data models
│   ├── utils/                  # Common utilities
│   └── config/                 # Configuration helpers
├── database/                    # Database files location
├── logs/                        # Log files location
├── config.json                  # Central configuration file
└── .env.example                 # Environment variables template

```

## Phase 1 Implementation Status

✅ **Completed:**
- Project structure created
- All 7 microservices skeleton implemented
- Health check endpoints for all services
- Shared utilities and models
- Configuration management (config.json)
- Environment variables template (.env.example)
- Logging infrastructure setup

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Backend API | 5000 | Main orchestrator and frontend API gateway |
| Data Service | 5001 | Data collection, storage, and gap fill |
| MT5 Service | 5002 | MetaTrader 5 connection and trading |
| Pattern Service | 5003 | Technical indicators and pattern recognition |
| Strategy Service | 5004 | Strategy execution and position management |
| AI Service | 5005 | AI/ML model training and inference |
| Backtesting Service | 5006 | Historical backtesting engine |

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Activate existing virtual environment
source .venv/bin/activate
```

### 2. Install Dependencies

Each service has its own requirements.txt. Install them as needed:

```bash
# Backend API
cd services/backend-api
pip install -r requirements.txt

# Data Service
cd ../data-service
pip install -r requirements.txt

# MT5 Service
cd ../mt5-service
pip install -r requirements.txt

# Pattern Service
cd ../pattern-service
pip install -r requirements.txt

# Strategy Service
cd ../strategy-service
pip install -r requirements.txt

# AI Service
cd ../ai-service
pip install -r requirements.txt

# Backtesting Service
cd ../backtesting-service
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy .env.example to .env and fill in your values
cp .env.example .env
# Edit .env with your MT5 credentials and other settings
```

### 4. Start Services

Each service can be started individually:

```bash
# Backend API (Main orchestrator)
cd services/backend-api
python main.py

# Data Service
cd services/data-service
python main.py

# MT5 Service
cd services/mt5-service
python main.py

# And so on for other services...
```

## Testing Services

### Health Check All Services

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

### Test Script

A test script is provided to check all services at once:

```bash
python test_services.py
```

## Configuration

Main configuration is in `config.json`:

- **System settings**: Environment, log level, data directory
- **Service settings**: Auto-start, auto-restart, ports, paths
- **Database settings**: Type, partitioning strategy
- **Trading settings**: Symbols, timeframes
- **Monitoring settings**: Health check interval, metrics

## Logging

All services log to:
- Individual log files: `logs/{service-name}.log`
- Console output (stdout)

Log format:
```
[YYYY-MM-DD HH:MM:SS] [LEVEL] [SERVICE_NAME] [MODULE] MESSAGE
```

## Next Steps (Phase 2)

1. **Service Orchestration** (Backend API)
   - Implement service start/stop/restart
   - Health monitoring loop
   - Auto-restart on failure
   - WebSocket hub for real-time events

2. **MT5 Service Implementation**
   - MT5 connection management
   - Tick/OHLC data fetching
   - Position management

3. **Data Service Implementation**
   - Gap fill logic
   - OnFly data collection
   - Database operations

## Documentation

Full project documentation is in the `docs/` folder:
- `projekt-dokumentacio.md` - Main project documentation
- `kiegeszites-config-realtime-performance.md` - Configuration and performance details
- `ugynok-*.md` - Service-specific agent documentation

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints
- Document all public methods
- Keep functions small and focused

### API Design
- Use REST conventions
- Return consistent JSON responses
- Handle errors gracefully
- Log all operations

### Error Handling
- Use try-except blocks
- Return meaningful error messages in Hungarian
- Log errors with context

## Architecture Principles

1. **Modularity**: Each service has a single responsibility
2. **Scalability**: Services can be scaled independently
3. **Maintainability**: Clean separation of concerns
4. **Performance**: Optimized for low CPU/memory usage
5. **Resilience**: Auto-restart, circuit breakers, retry logic

## License

Private project - All rights reserved

## Contact

For questions or issues, refer to the project documentation in `docs/` folder.
