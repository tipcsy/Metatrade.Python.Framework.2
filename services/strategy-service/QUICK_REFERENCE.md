# Strategy Service - Quick Reference Guide

## Service Information
- **Port**: 5003
- **Base URL**: http://localhost:5003
- **API Docs**: http://localhost:5003/docs
- **Status**: Running in Mock Mode (no MT5 required)

## Quick Start

### 1. Start the Service
```bash
cd /home/tipcsy/Metatrade.Python.Framework.2/services/strategy-service
python main.py
```

### 2. Test the Service
```bash
python test_strategy_service.py
```

## Common Operations

### Create Strategy
```bash
# MA Crossover
curl -X POST http://localhost:5003/strategies \
  -H "Content-Type: application/json" \
  -d '{"strategy_type":"MA_CROSSOVER","symbol":"EURUSD","timeframe":"H1","parameters":{"fast_period":10,"slow_period":20,"stop_loss_pips":50,"take_profit_pips":100}}'

# RSI Strategy
curl -X POST http://localhost:5003/strategies \
  -H "Content-Type: application/json" \
  -d '{"strategy_type":"RSI","symbol":"GBPUSD","timeframe":"M15","parameters":{"rsi_period":14,"oversold_level":30,"overbought_level":70,"stop_loss_pips":40,"take_profit_pips":80}}'
```

### Strategy Controls
```bash
# List all strategies
curl http://localhost:5003/strategies

# Get strategy details
curl http://localhost:5003/strategies/STRAT_0001

# Start strategy
curl -X POST http://localhost:5003/strategies/STRAT_0001/start

# Pause strategy
curl -X POST http://localhost:5003/strategies/STRAT_0001/pause

# Resume strategy
curl -X POST http://localhost:5003/strategies/STRAT_0001/resume

# Stop strategy
curl -X POST http://localhost:5003/strategies/STRAT_0001/stop

# Delete strategy
curl -X DELETE http://localhost:5003/strategies/STRAT_0001
```

### Position Management
```bash
# List all positions
curl http://localhost:5003/positions

# List open positions only
curl "http://localhost:5003/positions?status=open"

# List positions for specific symbol
curl "http://localhost:5003/positions?symbol=EURUSD"

# Get position details
curl http://localhost:5003/positions/POS_000001

# Close specific position
curl -X POST http://localhost:5003/positions/close \
  -H "Content-Type: application/json" \
  -d '{"position_id":"POS_000001","close_price":1.1050}'

# Close all positions
curl -X POST http://localhost:5003/positions/close-all

# Close positions for specific strategy
curl -X POST "http://localhost:5003/positions/close-all?strategy_id=STRAT_0001"
```

### Statistics & Monitoring
```bash
# Position statistics
curl http://localhost:5003/positions/statistics

# Strategy statistics
curl http://localhost:5003/strategies/statistics

# Risk status
curl http://localhost:5003/risk/status

# Update balance
curl -X POST "http://localhost:5003/risk/update-balance?new_balance=12000.0"

# Reset daily loss
curl -X POST http://localhost:5003/risk/reset-daily-loss
```

## Strategy Types & Parameters

### MA_CROSSOVER
```json
{
  "strategy_type": "MA_CROSSOVER",
  "symbol": "EURUSD",
  "timeframe": "H1",
  "parameters": {
    "fast_period": 10,      // Fast MA period
    "slow_period": 20,      // Slow MA period
    "stop_loss_pips": 50,   // SL in pips
    "take_profit_pips": 100 // TP in pips
  }
}
```

### RSI
```json
{
  "strategy_type": "RSI",
  "symbol": "GBPUSD",
  "timeframe": "M15",
  "parameters": {
    "rsi_period": 14,         // RSI calculation period
    "oversold_level": 30,     // Buy signal threshold
    "overbought_level": 70,   // Sell signal threshold
    "stop_loss_pips": 40,     // SL in pips
    "take_profit_pips": 80    // TP in pips
  }
}
```

## Default Risk Settings
- **Account Balance**: $10,000
- **Max Risk per Trade**: 2%
- **Max Positions**: 5
- **Max Daily Loss**: 5%
- **Max Drawdown**: 20%

## Common Timeframes
- M1, M5, M15, M30 (Minutes)
- H1, H4 (Hours)
- D1 (Daily)
- W1 (Weekly)
- MN1 (Monthly)

## Common Currency Pairs
- EURUSD, GBPUSD, USDJPY
- AUDUSD, USDCAD, USDCHF
- NZDUSD, EURJPY, GBPJPY

## Response Codes
- **200**: Success
- **400**: Bad request (invalid parameters)
- **404**: Not found (strategy/position doesn't exist)
- **500**: Server error

## Logs
Service logs are written to:
```
/home/tipcsy/Metatrade.Python.Framework.2/logs/strategy-service.log
```

## Files Created

### Core Components
- `/home/tipcsy/Metatrade.Python.Framework.2/services/strategy-service/app/core/position_manager.py`
- `/home/tipcsy/Metatrade.Python.Framework.2/services/strategy-service/app/core/risk_manager.py`
- `/home/tipcsy/Metatrade.Python.Framework.2/services/strategy-service/app/core/strategy_engine.py`

### API & Models
- `/home/tipcsy/Metatrade.Python.Framework.2/services/strategy-service/app/api/routes.py`
- `/home/tipcsy/Metatrade.Python.Framework.2/services/strategy-service/app/models/__init__.py`

### Main & Tests
- `/home/tipcsy/Metatrade.Python.Framework.2/services/strategy-service/main.py`
- `/home/tipcsy/Metatrade.Python.Framework.2/services/strategy-service/test_strategy_service.py`

## Troubleshooting

### Service won't start
```bash
# Check if port is already in use
lsof -ti:5003

# Kill existing process
lsof -ti:5003 | xargs kill -9

# Restart service
python main.py
```

### Can't connect to service
```bash
# Check if service is running
curl http://localhost:5003/health

# Check logs
tail -f ../../logs/strategy-service.log
```

### Strategy not executing trades
- Strategy must be in RUNNING status (not STOPPED or PAUSED)
- Check risk limits haven't been exceeded
- In mock mode, you need to manually feed price data
- Use test_strategy_service.py to verify functionality
