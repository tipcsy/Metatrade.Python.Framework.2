@echo off
REM Batch Script - Test All MT5 Trading System Services
REM Usage: test_all_services.bat

echo.
echo Testing All Services...
echo ================================
echo.

REM Requires curl to be installed (included in Windows 10+)

echo Testing Backend API (port 5000)...
curl -s http://localhost:5000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Backend API
) else (
    echo [FAILED] Backend API
)

echo Testing Data Service (port 5001)...
curl -s http://localhost:5001/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Data Service
) else (
    echo [FAILED] Data Service
)

echo Testing MT5 Service (port 5002)...
curl -s http://localhost:5002/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] MT5 Service
) else (
    echo [FAILED] MT5 Service
)

echo Testing Pattern Service (port 5003)...
curl -s http://localhost:5003/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Pattern Service
) else (
    echo [FAILED] Pattern Service
)

echo Testing Strategy Service (port 5004)...
curl -s http://localhost:5004/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Strategy Service
) else (
    echo [FAILED] Strategy Service
)

echo Testing AI Service (port 5005)...
curl -s http://localhost:5005/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] AI Service
) else (
    echo [FAILED] AI Service
)

echo Testing Backtesting Service (port 5006)...
curl -s http://localhost:5006/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Backtesting Service
) else (
    echo [FAILED] Backtesting Service
)

echo.
echo ================================
echo Health check complete!
echo.
pause
