# PowerShell Script - Stop All MT5 Trading System Services
# Usage: .\stop_all_services.ps1

Write-Host "🛑 Stopping MT5 Trading System Services..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$services = @(
    @{Name="backend-api"; Path="services/backend-api"},
    @{Name="data-service"; Path="services/data-service"},
    @{Name="mt5-service"; Path="services/mt5-service"},
    @{Name="pattern-service"; Path="services/pattern-service"},
    @{Name="strategy-service"; Path="services/strategy-service"},
    @{Name="ai-service"; Path="services/ai-service"},
    @{Name="backtesting-service"; Path="services/backtesting-service"}
)

foreach ($service in $services) {
    Write-Host "Stopping $($service.Name)..." -ForegroundColor Yellow

    Push-Location $service.Path

    $pidFile = "$($service.Name).pid"

    if (Test-Path $pidFile) {
        $pid = Get-Content $pidFile

        try {
            Stop-Process -Id $pid -Force -ErrorAction Stop
            Write-Host "✓ $($service.Name) stopped (PID: $pid)" -ForegroundColor Green
        } catch {
            Write-Host "✗ $($service.Name) process not running" -ForegroundColor Red
        }

        Remove-Item $pidFile -Force
    } else {
        Write-Host "✗ No PID file found for $($service.Name)" -ForegroundColor Red
    }

    Pop-Location
}

# Kill any remaining Python processes running services
Write-Host ""
Write-Host "Cleaning up any remaining service processes..." -ForegroundColor Yellow

$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*services*main.py*"
}

if ($pythonProcesses) {
    $pythonProcesses | Stop-Process -Force
    Write-Host "✓ Cleaned up remaining processes" -ForegroundColor Green
} else {
    Write-Host "No additional processes found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "All services stopped" -ForegroundColor Green
