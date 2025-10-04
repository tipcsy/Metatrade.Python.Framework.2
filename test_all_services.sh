#!/bin/bash

# Test all MT5 Trading System services
echo "🧪 Testing All Services..."
echo "================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to test service health
test_service() {
    local name=$1
    local port=$2

    echo -n "Testing $name (port $port)... "

    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null)

    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED (HTTP $response)${NC}"
        return 1
    fi
}

# Test all services
test_service "Backend API" 5000
test_service "Data Service" 5001
test_service "MT5 Service" 5002
test_service "Pattern Service" 5003
test_service "Strategy Service" 5004
test_service "AI Service" 5005
test_service "Backtesting Service" 5006

echo ""
echo "================================"
echo "Health check complete!"
