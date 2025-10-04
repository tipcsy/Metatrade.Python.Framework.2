@echo off
REM Batch Script - Start All MT5 Trading System Services
REM Usage: start_all_services.bat

echo.
echo Starting MT5 Trading System Services...
echo ==========================================
echo.

REM Start Backend API
echo Starting backend-api on port 5000...
cd services\backend-api
start /B python main.py > backend-api.log 2>&1
cd ..\..

REM Start Data Service
echo Starting data-service on port 5001...
cd services\data-service
start /B python main.py > data-service.log 2>&1
cd ..\..

REM Start MT5 Service
echo Starting mt5-service on port 5002...
cd services\mt5-service
start /B python main.py > mt5-service.log 2>&1
cd ..\..

REM Start Pattern Service
echo Starting pattern-service on port 5003...
cd services\pattern-service
start /B python main.py > pattern-service.log 2>&1
cd ..\..

REM Start Strategy Service
echo Starting strategy-service on port 5004...
cd services\strategy-service
start /B python main.py > strategy-service.log 2>&1
cd ..\..

REM Start AI Service
echo Starting ai-service on port 5005...
cd services\ai-service
start /B python main.py > ai-service.log 2>&1
cd ..\..

REM Start Backtesting Service
echo Starting backtesting-service on port 5006...
cd services\backtesting-service
start /B python main.py > backtesting-service.log 2>&1
cd ..\..

echo.
echo ==========================================
echo All services started successfully!
echo.
echo Service URLs:
echo   Backend API:      http://localhost:5000
echo   Data Service:     http://localhost:5001
echo   MT5 Service:      http://localhost:5002
echo   Pattern Service:  http://localhost:5003
echo   Strategy Service: http://localhost:5004
echo   AI Service:       http://localhost:5005
echo   Backtesting:      http://localhost:5006
echo.
echo To stop all services, run: stop_all_services.bat
echo.
pause
