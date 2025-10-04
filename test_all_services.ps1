# PowerShell Script - Test All MT5 Trading System Services
# Usage: .\test_all_services.ps1

Write-Host "🧪 Testing All Services..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

$services = @(
    @{Name="Backend API"; Port=5000},
    @{Name="Data Service"; Port=5001},
    @{Name="MT5 Service"; Port=5002},
    @{Name="Pattern Service"; Port=5003},
    @{Name="Strategy Service"; Port=5004},
    @{Name="AI Service"; Port=5005},
    @{Name="Backtesting Service"; Port=5006}
)

$passed = 0
$failed = 0

foreach ($service in $services) {
    Write-Host -NoNewline "Testing $($service.Name) (port $($service.Port))... "

    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$($service.Port)/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop

        if ($response.StatusCode -eq 200) {
            Write-Host "✓ OK" -ForegroundColor Green
            $passed++
        } else {
            Write-Host "✗ FAILED (HTTP $($response.StatusCode))" -ForegroundColor Red
            $failed++
        }
    } catch {
        Write-Host "✗ FAILED (Not responding)" -ForegroundColor Red
        $failed++
    }
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Health check complete!" -ForegroundColor Cyan
Write-Host "Passed: $passed" -ForegroundColor Green
Write-Host "Failed: $failed" -ForegroundColor Red
