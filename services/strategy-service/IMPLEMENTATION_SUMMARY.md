# Strategy Service - Implementation Summary

## Overview
The Strategy Service is a comprehensive trading strategy execution and position management service for the MT5 trading platform. It runs on **port 5003** and operates in **mock mode** (no MT5 connection required for testing).

## Architecture

### Core Components

#### 1. Position Manager (`app/core/position_manager.py`)
- **Purpose**: Manages all trading positions (open and closed)
- **Key Features**:
  - Position creation with automatic ID generation
  - Stop Loss (SL) and Take Profit (TP) tracking
  - Automatic position closure when SL/TP is hit
  - Position exposure calculation by symbol
  - Profit/loss tracking
  - Position statistics (win rate, total profit, etc.)

#### 2. Risk Manager (`app/core/risk_manager.py`)
- **Purpose**: Handles risk calculations and position sizing
- **Key Features**:
  - Dynamic position sizing based on account equity and risk percentage
  - Maximum positions limit enforcement
  - Daily loss limit tracking
  - Drawdown monitoring
  - Stop loss and take profit validation
  - Risk/reward ratio calculation
  - Account balance tracking

**Default Risk Parameters**:
- Account Balance: $10,000
- Max Risk per Trade: 2%
- Max Positions: 5
- Max Daily Loss: 5%
- Max Drawdown: 20%

#### 3. Strategy Engine (`app/core/strategy_engine.py`)
- **Purpose**: Manages strategy lifecycle and execution
- **Key Features**:
  - Strategy creation, start, stop, pause, resume, delete
  - Two built-in strategy templates
  - Strategy statistics tracking
  - Automatic position management integration

### Built-in Strategy Templates

#### 1. Moving Average Crossover (`MA_CROSSOVER`)
- **Logic**: Trades based on fast and slow moving average crossovers
- **Parameters**:
  - `fast_period`: Fast MA period (default: 10)
  - `slow_period`: Slow MA period (default: 20)
  - `stop_loss_pips`: Stop loss in pips (default: 50)
  - `take_profit_pips`: Take profit in pips (default: 100)
- **Signals**:
  - BUY when fast MA crosses above slow MA
  - SELL when fast MA crosses below slow MA

#### 2. RSI Strategy (`RSI`)
- **Logic**: Trades based on RSI overbought/oversold levels
- **Parameters**:
  - `rsi_period`: RSI calculation period (default: 14)
  - `oversold_level`: Oversold threshold (default: 30)
  - `overbought_level`: Overbought threshold (default: 70)
  - `stop_loss_pips`: Stop loss in pips (default: 50)
  - `take_profit_pips`: Take profit in pips (default: 100)
- **Signals**:
  - BUY when RSI < oversold level
  - SELL when RSI > overbought level

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /docs` - Swagger UI documentation

### Strategy Management
- `GET /strategies` - List all strategies
- `POST /strategies` - Create a new strategy
- `GET /strategies/statistics` - Get strategy engine statistics
- `GET /strategies/{strategy_id}` - Get specific strategy details
- `POST /strategies/{strategy_id}/start` - Start a strategy
- `POST /strategies/{strategy_id}/stop` - Stop a strategy
- `POST /strategies/{strategy_id}/pause` - Pause a strategy
- `POST /strategies/{strategy_id}/resume` - Resume a paused strategy
- `DELETE /strategies/{strategy_id}` - Delete a strategy

### Position Management
- `GET /positions` - List all positions (with filters: status, symbol, strategy_id)
- `GET /positions/statistics` - Get position statistics
- `GET /positions/{position_id}` - Get specific position details
- `POST /positions/close` - Close a specific position
- `POST /positions/close-all` - Close all open positions

### Risk Management
- `GET /risk/status` - Get current risk status
- `POST /risk/reset-daily-loss` - Reset daily loss counter
- `POST /risk/update-balance` - Update account balance

## Data Models

### StrategyCreateRequest
```json
{
  "strategy_type": "MA_CROSSOVER" | "RSI",
  "symbol": "EURUSD",
  "timeframe": "H1",
  "parameters": {
    "fast_period": 10,
    "slow_period": 20,
    "stop_loss_pips": 50,
    "take_profit_pips": 100
  }
}
```

### StrategyResponse
```json
{
  "strategy_id": "STRAT_0001",
  "name": "MA Crossover",
  "symbol": "EURUSD",
  "timeframe": "H1",
  "status": "RUNNING" | "STOPPED" | "PAUSED" | "ERROR",
  "created_at": "2025-10-04T08:42:52.782639",
  "started_at": "2025-10-04T08:42:52.787485",
  "stopped_at": null,
  "signals_count": 0,
  "trades_count": 0,
  "parameters": {...}
}
```

### PositionResponse
```json
{
  "position_id": "POS_000001",
  "symbol": "EURUSD",
  "type": "BUY" | "SELL",
  "volume": 0.1,
  "open_price": 1.1000,
  "close_price": 1.1050,
  "stop_loss": 1.0950,
  "take_profit": 1.1100,
  "strategy_id": "STRAT_0001",
  "status": "OPEN" | "CLOSED" | "PENDING",
  "open_time": "2025-10-04T08:42:52.782639",
  "close_time": "2025-10-04T08:43:52.782639",
  "profit": 50.0,
  "commission": 0.0,
  "swap": 0.0
}
```

## Usage Examples

### Create and Start a Strategy

```bash
# Create MA Crossover strategy
curl -X POST http://localhost:5003/strategies \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_type": "MA_CROSSOVER",
    "symbol": "EURUSD",
    "timeframe": "H1",
    "parameters": {
      "fast_period": 10,
      "slow_period": 20,
      "stop_loss_pips": 50,
      "take_profit_pips": 100
    }
  }'

# Start the strategy
curl -X POST http://localhost:5003/strategies/STRAT_0001/start
```

### Create RSI Strategy

```bash
curl -X POST http://localhost:5003/strategies \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_type": "RSI",
    "symbol": "GBPUSD",
    "timeframe": "M15",
    "parameters": {
      "rsi_period": 14,
      "oversold_level": 30,
      "overbought_level": 70,
      "stop_loss_pips": 40,
      "take_profit_pips": 80
    }
  }'
```

### Monitor Risk Status

```bash
# Get current risk status
curl http://localhost:5003/risk/status

# Update account balance
curl -X POST "http://localhost:5003/risk/update-balance?new_balance=12000.0"
```

### View Positions and Statistics

```bash
# Get all open positions
curl "http://localhost:5003/positions?status=open"

# Get position statistics
curl http://localhost:5003/positions/statistics

# Get strategy statistics
curl http://localhost:5003/strategies/statistics
```

## Testing

A comprehensive test suite is provided in `test_strategy_service.py`:

```bash
cd /home/tipcsy/Metatrade.Python.Framework.2/services/strategy-service
python test_strategy_service.py
```

The test suite covers:
- Health check
- Strategy creation (MA Crossover and RSI)
- Strategy lifecycle (start, stop, pause, resume, delete)
- Position management
- Risk status monitoring
- Statistics endpoints

## File Structure

```
strategy-service/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # All API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── position_manager.py    # Position tracking
│   │   ├── risk_manager.py        # Risk calculations
│   │   └── strategy_engine.py     # Strategy execution
│   └── models/
│       └── __init__.py             # Pydantic models
├── main.py                         # Service entry point
├── requirements.txt                # Dependencies
├── test_strategy_service.py        # Test suite
└── IMPLEMENTATION_SUMMARY.md       # This file
```

## Key Features

### 1. Risk-First Approach
- All trading decisions go through risk manager
- Position sizing based on account equity and risk percentage
- Automatic enforcement of risk limits
- Daily loss and drawdown tracking

### 2. Flexible Strategy System
- Easy to add new strategy templates
- Base strategy class for custom implementations
- Support for multiple strategies running simultaneously
- Per-strategy position tracking

### 3. Comprehensive Position Management
- Automatic SL/TP monitoring
- Position exposure tracking
- Win rate and profit statistics
- Trade-by-trade history

### 4. Mock Mode Operation
- Works without MT5 connection for testing
- Allows full API testing
- Easy integration testing

### 5. Production-Ready Logging
- Detailed logging of all operations
- Separate log file for the service
- Request/response tracking

## Important Notes

1. **Port**: The service runs on port **5003** (not 5004 as initially specified in main.py - this has been corrected)

2. **Mock Mode**: The service currently runs in mock mode without actual MT5 connection. This allows testing all functionality without broker integration.

3. **Position Calculations**:
   - Simplified pip value calculation ($10 per pip per lot)
   - Assumes standard forex pairs (4-digit pricing with 5th decimal)
   - Real implementation would need broker-specific pip value calculations

4. **Risk Limits**: Can be disabled via `risk_manager.disable_risk_limits()` but this is NOT recommended for production

5. **Strategy State**: Currently in-memory only. For production, implement persistence layer for:
   - Active strategies
   - Open positions
   - Account balance
   - Daily loss tracking

## Future Enhancements

1. **Integration with MT5 Service**:
   - Real-time price feed integration
   - Actual order execution
   - Position synchronization with MT5

2. **Data Service Integration**:
   - Historical data for backtesting
   - Real-time bar updates
   - Indicator calculations

3. **Additional Strategy Templates**:
   - Bollinger Bands
   - MACD
   - Fibonacci retracement
   - Custom drag-and-drop builder

4. **Enhanced Risk Management**:
   - Correlation analysis between positions
   - Portfolio-level risk limits
   - Time-based risk limits (no trading during news)

5. **Backtesting Engine**:
   - Historical strategy testing
   - Performance metrics
   - Optimization tools

6. **Persistence Layer**:
   - Database integration for strategy state
   - Position history
   - Trade journal

## Dependencies

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0
requests==2.31.0
pandas==2.1.1
numpy==1.26.0
```

## Running the Service

```bash
# Navigate to service directory
cd /home/tipcsy/Metatrade.Python.Framework.2/services/strategy-service

# Start the service
python main.py

# Service will be available at http://localhost:5003
# API documentation at http://localhost:5003/docs
```

## Conclusion

The Strategy Service provides a robust foundation for algorithmic trading with comprehensive risk management, flexible strategy templates, and production-ready error handling. It's designed to integrate seamlessly with the MT5 Service and Data Service while maintaining independent testability through mock mode.
