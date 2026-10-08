@echo off
title ColorRevive Launcher
echo ==============================================
echo   ColorRevive AI Colorization Launcher
echo ==============================================

echo Starting FastAPI Backend on http://localhost:8001...
start "ColorRevive Backend" cmd /k "cd /d %~dp0colorrevive\backend && .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001"

echo Starting Next.js Frontend on http://localhost:3000...
start "ColorRevive Frontend" cmd /k "cd /d %~dp0colorrevive\frontend && npm run dev"

echo.
echo Both servers have been launched!
echo - Web UI:  http://localhost:3000
echo - Backend: http://localhost:8001
echo - Docs:    http://localhost:8001/docs
echo ==============================================
