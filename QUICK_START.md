# Quick Start Guide
## MT5 Trading Platform 2.0

This is a quick reference for developers to get started with the platform.

---

## 🚀 Quick Setup (5 minutes)

### 1. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 2. Install Dependencies for All Services
```bash
# One-liner to install all dependencies
for service in services/*/; do
    echo "Installing $service..."
    pip install -r "${service}requirements.txt"
done
```

### 3. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your MT5 credentials
nano .env
```

### 4. Start Services
```bash
# Option 1: Start all at once
./start_all_services.sh

# Option 2: Start individually
cd services/backend-api && python main.py &
cd services/data-service && python main.py &
cd services/mt5-service && python main.py &
cd services/pattern-service && python main.py &
```

### 5. Test Services
```bash
python test_services.py
```

---

## 📁 Project Structure

```
services/               # All microservices
  ├── backend-api/      # Port 5000 - Main API
  ├── data-service/     # Port 5001 - Data management
  ├── mt5-service/      # Port 5002 - MT5 connection
  ├── pattern-service/  # Port 5003 - Patterns
  ├── strategy-service/ # Port 5004 - Strategies
  ├── ai-service/       # Port 5005 - AI/ML
  └── backtesting-service/ # Port 5006 - Backtests

shared/                 # Shared utilities
  ├── models/          # Common models
  ├── utils/           # Utilities
  └── config/          # Config helpers

database/              # SQLite databases
logs/                  # Log files
config.json            # Configuration
.env                   # Environment vars (create from .env.example)
```

---

## 🔌 Service Endpoints

### Backend API (Port 5000)
```bash
GET  /health              # Health check
GET  /api/status          # System status
```

### Data Service (Port 5001)
```bash
GET  /health              # Health check
POST /gap-fill            # Start gap fill
GET  /statistics          # Stats
```

### MT5 Service (Port 5002)
```bash
GET  /health              # Health + MT5 status
POST /connect             # Connect to MT5
GET  /account             # Account info
```

### Pattern Service (Port 5003)
```bash
GET  /health                        # Health check
GET  /patterns                      # List patterns
GET  /indicators/{symbol}/{tf}      # Get indicators
```

### Strategy Service (Port 5004)
```bash
GET  /health                    # Health check
GET  /strategies                # List strategies
POST /strategies/{id}/start     # Start strategy
```

### AI Service (Port 5005)
```bash
GET  /health                    # Health check
GET  /models                    # List models
POST /models/{id}/predict       # Predict
```

### Backtesting Service (Port 5006)
```bash
GET  /health                    # Health check
POST /backtest/start            # Start backtest
GET  /backtest/{id}/status      # Status
```

---

## 🛠️ Common Commands

### Start/Stop Services
```bash
# Start all
./start_all_services.sh

# Stop all
./stop_all_services.sh

# Start one service
cd services/backend-api
python main.py
```

### Test Services
```bash
# Test all health endpoints
python test_services.py

# Test one service
curl http://localhost:5000/health
```

### View Logs
```bash
# Real-time logs for a service
tail -f logs/backend-api.log

# Console logs (when using start_all_services.sh)
tail -f logs/backend-api.console.log

# All logs
tail -f logs/*.log
```

### Check Running Services
```bash
# See all running Python services
ps aux | grep 'python.*services'

# See specific ports
lsof -i :5000  # Backend API
lsof -i :5001  # Data Service
# etc...
```

---

## 🏗️ Development Workflow

### Adding New Endpoint

1. **Define route** in `app/api/routes.py`:
```python
@router.get("/my-endpoint")
async def my_endpoint():
    return {"success": True, "data": "Hello"}
```

2. **Add business logic** in `app/core/service.py`:
```python
class MyService:
    def do_something(self):
        # Your logic here
        pass
```

3. **Test**:
```bash
curl http://localhost:5000/my-endpoint
```

### Adding New Service

1. Copy structure from existing service
2. Update `SERVICE_NAME` and `SERVICE_PORT` in `main.py`
3. Add to `config.json` services section
4. Implement your logic

### Using Shared Components

```python
# Import shared models
from shared.models import TickData, OHLCData

# Import shared utils
from shared.utils import format_response, get_timestamp_ms

# Import config
from shared.config import load_config, get_service_config
```

---

## 🐛 Troubleshooting

### Service Won't Start
```bash
# Check if port is already in use
lsof -i :5000

# Kill process on port
kill $(lsof -t -i:5000)

# Check dependencies
cd services/backend-api
pip install -r requirements.txt
```

### Import Errors
```bash
# Make sure shared package is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/tipcsy/Metatrade.Python.Framework.2"

# Or run from project root
cd /home/tipcsy/Metatrade.Python.Framework.2
python services/backend-api/main.py
```

### Health Check Fails
```bash
# Check if service is running
ps aux | grep 'python.*backend-api'

# Check logs
tail -f logs/backend-api.log

# Try manual start
cd services/backend-api
python main.py
```

---

## 📊 Monitoring

### Service Health
```bash
# All services
python test_services.py

# One service
curl http://localhost:5000/health
```

### Logs
```bash
# Service logs
tail -f logs/backend-api.log

# All logs
tail -f logs/*.log

# Search logs
grep ERROR logs/*.log
```

### Metrics (Future)
- CPU usage per service
- Memory usage per service
- Request count and latency
- Error rates

---

## 📚 Documentation

- **README.md** - Project overview
- **PHASE1_IMPLEMENTATION_REPORT.md** - Implementation details
- **docs/projekt-dokumentacio.md** - Full architecture documentation
- **docs/ugynok-*.md** - Service-specific agent documentation
- **config.json** - Configuration reference

---

## 🎯 Next Phase

**Phase 2 Focus:**
1. Service Orchestration (Backend API)
2. MT5 Connection (MT5 Service)
3. Data Collection (Data Service)
4. WebSocket Hub (Backend API)

---

## 💡 Tips

1. **Always activate virtual environment** before running services
2. **Check logs** when something doesn't work
3. **Use test_services.py** to verify all services
4. **Start Backend API first**, then other services
5. **Keep config.json updated** when adding services

---

**Happy Coding! 🚀**
