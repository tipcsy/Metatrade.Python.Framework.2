# PowerShell Script - Start All MT5 Trading System Services
# Usage: .\start_all_services.ps1

Write-Host "🚀 Starting MT5 Trading System Services..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$services = @(
    @{Name="backend-api"; Port=5000; Path="services/backend-api"},
    @{Name="data-service"; Port=5001; Path="services/data-service"},
    @{Name="mt5-service"; Port=5002; Path="services/mt5-service"},
    @{Name="pattern-service"; Port=5003; Path="services/pattern-service"},
    @{Name="strategy-service"; Port=5004; Path="services/strategy-service"},
    @{Name="ai-service"; Port=5005; Path="services/ai-service"},
    @{Name="backtesting-service"; Port=5006; Path="services/backtesting-service"}
)

foreach ($service in $services) {
    Write-Host "Starting $($service.Name) on port $($service.Port)..." -ForegroundColor Yellow

    Push-Location $service.Path

    # Start Python process in background
    $process = Start-Process -FilePath "python" -ArgumentList "main.py" -NoNewWindow -PassThru

    # Save PID
    $process.Id | Out-File "$($service.Name).pid"

    Write-Host "✓ $($service.Name) started (PID: $($process.Id))" -ForegroundColor Green

    Pop-Location
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "All services started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Service URLs:"
Write-Host "  Backend API:      http://localhost:5000"
Write-Host "  Data Service:     http://localhost:5001"
Write-Host "  MT5 Service:      http://localhost:5002"
Write-Host "  Pattern Service:  http://localhost:5003"
Write-Host "  Strategy Service: http://localhost:5004"
Write-Host "  AI Service:       http://localhost:5005"
Write-Host "  Backtesting:      http://localhost:5006"
Write-Host ""
Write-Host "To stop all services, run: .\stop_all_services.ps1" -ForegroundColor Yellow
