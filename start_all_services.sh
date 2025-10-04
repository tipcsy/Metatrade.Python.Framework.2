#!/bin/bash

# Start all MT5 Trading System services
echo "🚀 Starting MT5 Trading System Services..."

# Start services using Python in background
for service in backend-api data-service mt5-service pattern-service strategy-service ai-service backtesting-service; do
    echo "Starting $service..."
    cd "services/$service"
    nohup python3 main.py > "$service.log" 2>&1 &
    echo $! > "$service.pid"
    cd - > /dev/null
done

echo "✓ All services started"
echo ""
echo "Service URLs:"
echo "  Backend API:      http://localhost:5000"
echo "  Data Service:     http://localhost:5001"
echo "  MT5 Service:      http://localhost:5002"
echo "  Pattern Service:  http://localhost:5003"
echo "  Strategy Service: http://localhost:5004"
echo "  AI Service:       http://localhost:5005"
echo "  Backtesting:      http://localhost:5006"
