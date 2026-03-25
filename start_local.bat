@echo off
echo =======================================================
echo Starting Cyberbullying Detection App Locally
echo =======================================================

echo Starting MongoDB via Docker...
docker-compose up -d mongo

echo Starting ML Service (Python/FastAPI)...
start "ML Service" cmd /k "cd ml-service && uvicorn main:app --reload --port 8000"

echo Starting Backend API (Node.js)...
start "Backend Service" cmd /k "cd backend && npm run dev"

echo Waiting 5 seconds before starting Frontend...
timeout /t 5 /nobreak >nul

echo Starting Frontend UI (React/Vite)...
start "Frontend Service" cmd /k "cd frontend && npm run dev"

echo =======================================================
echo All services launched in separate windows!
echo - Frontend: http://localhost:5173
echo - Backend: http://localhost:5000
echo - ML Service: http://localhost:8000
echo =======================================================
