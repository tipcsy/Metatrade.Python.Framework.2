@echo off
REM Batch Script - Stop All MT5 Trading System Services
REM Usage: stop_all_services.bat

echo.
echo Stopping MT5 Trading System Services...
echo ==========================================
echo.

REM Kill all Python processes running main.py
echo Stopping all service processes...

taskkill /F /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq main.py*" 2>nul

if %ERRORLEVEL% EQU 0 (
    echo Services stopped successfully
) else (
    echo No running service processes found
)

echo.
echo ==========================================
echo All services stopped
echo.
pause
