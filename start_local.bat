@echo off
echo =======================================================
echo  CERDS v2 — Cyberbullying Detection + LangChain AI
echo =======================================================

echo.
echo [1/4] Starting MongoDB via Docker...
docker-compose up -d mongo
timeout /t 2 /nobreak >nul

echo.
echo [2/4] Starting ML Service (FastAPI + LangChain)...
start "ML Service" cmd /k "cd /d %~dp0ml-service && call venv\Scripts\activate && uvicorn main:app --reload --port 8000"

echo.
echo [3/4] Starting Backend API (Node.js)...
start "Backend Service" cmd /k "cd /d %~dp0backend && npm run dev"

echo.
echo [4/4] Waiting 8s before starting Frontend (let ML service warm up)...
timeout /t 8 /nobreak >nul

echo Starting Frontend UI (React/Vite)...
start "Frontend UI" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo =======================================================
echo  All services launched in separate windows!
echo.
echo   Frontend  : http://localhost:5173
echo   Backend   : http://localhost:5000
echo   ML Service: http://localhost:8000/docs
echo.
echo   LangChain health check (wait ~15s after start):
echo   http://localhost:8000/moderation/health
echo =======================================================
echo.
pause
