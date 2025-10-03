#!/bin/bash
# MT5 Trading Platform 2.0 - Stop All Services
# This script stops all running microservices

echo "============================================================"
echo "MT5 Trading Platform 2.0 - Stopping All Services"
echo "============================================================"
echo ""

# Find and kill all Python processes running main.py from services
echo "Stopping all services..."

# Find PIDs of running services
pids=$(ps aux | grep 'python.*services/.*main.py' | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "No running services found."
else
    echo "Found running services with PIDs: $pids"
    echo ""

    for pid in $pids; do
        # Get the command to show which service it is
        cmd=$(ps -p $pid -o args= | head -1)
        echo "Stopping: $cmd (PID: $pid)"
        kill $pid

        # Wait for graceful shutdown
        sleep 1

        # Force kill if still running
        if ps -p $pid > /dev/null 2>&1; then
            echo "  Force killing PID $pid..."
            kill -9 $pid
        fi
    done

    echo ""
    echo "✅ All services stopped."
fi

echo ""
echo "============================================================"
