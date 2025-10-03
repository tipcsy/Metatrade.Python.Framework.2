#!/bin/bash
# MT5 Trading Platform 2.0 - Start All Services
# This script starts all microservices in the background

echo "============================================================"
echo "MT5 Trading Platform 2.0 - Starting All Services"
echo "============================================================"
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Function to start a service
start_service() {
    local service_name=$1
    local service_port=$2
    local service_path="services/$service_name"

    echo "Starting $service_name on port $service_port..."

    # Change to service directory
    cd "$PROJECT_ROOT/$service_path"

    # Start service in background with nohup
    nohup python main.py > "$PROJECT_ROOT/logs/${service_name}.console.log" 2>&1 &
    local pid=$!

    echo "  ✅ Started $service_name (PID: $pid)"

    # Return to project root
    cd "$PROJECT_ROOT"

    # Wait a bit before starting next service
    sleep 1
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Start all services
echo ""
echo "Starting services..."
echo ""

start_service "backend-api" 5000
start_service "data-service" 5001
start_service "mt5-service" 5002
start_service "pattern-service" 5003
# Uncomment below to auto-start these services
# start_service "strategy-service" 5004
# start_service "ai-service" 5005
# start_service "backtesting-service" 5006

echo ""
echo "============================================================"
echo "All services started!"
echo "============================================================"
echo ""
echo "Service status:"
echo "  Backend API:      http://localhost:5000/health"
echo "  Data Service:     http://localhost:5001/health"
echo "  MT5 Service:      http://localhost:5002/health"
echo "  Pattern Service:  http://localhost:5003/health"
echo ""
echo "To check all services, run: python test_services.py"
echo "To stop all services, run: ./stop_all_services.sh"
echo ""
echo "Console logs are in: logs/*.console.log"
echo "Service logs are in: logs/*.log"
echo ""
echo "============================================================"
