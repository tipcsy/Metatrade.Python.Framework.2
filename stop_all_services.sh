#!/bin/bash

# Stop all MT5 Trading System services
echo "🛑 Stopping MT5 Trading System Services..."

# Kill all Python service processes
pkill -f "services/.*/main.py"

# Clean up PID files
for service in backend-api data-service mt5-service pattern-service strategy-service ai-service backtesting-service; do
    rm -f "services/$service/$service.pid"
done

echo "✓ All services stopped"
