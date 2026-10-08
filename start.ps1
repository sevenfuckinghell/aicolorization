# ColorRevive PowerShell Launcher
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  ColorRevive AI Colorization Launcher" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

Write-Host "Starting FastAPI Backend on http://localhost:8001..." -ForegroundColor Green
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot\colorrevive\backend'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001"

Write-Host "Starting Next.js Frontend on http://localhost:3000..." -ForegroundColor Green
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot\colorrevive\frontend'; npm run dev"

Write-Host ""
Write-Host "Both servers are launching!" -ForegroundColor Yellow
Write-Host "- Web UI:  http://localhost:3000" -ForegroundColor White
Write-Host "- Backend: http://localhost:8001" -ForegroundColor White
Write-Host "- Docs:    http://localhost:8001/docs" -ForegroundColor White
Write-Host "==============================================" -ForegroundColor Cyan
