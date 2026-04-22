@echo off
setlocal

set "ROOT=%~dp0"

echo ==========================================
echo AI Learning App - Clean Start
echo ==========================================
echo.
echo [1/3] Stopping stale processes on ports 3000 and 4000...

for %%P in (3000 4000) do (
	for /f "tokens=5" %%A in ('netstat -aon ^| findstr /R /C:":%%P .*LISTENING"') do (
		taskkill /PID %%A /F >nul 2>&1
	)
)

echo [2/3] Starting backend on http://localhost:4000 ...
start "Backend" cmd /k "cd /d "%ROOT%backend" && npm start"
timeout /t 2 /nobreak > nul

echo [3/3] Starting frontend on http://localhost:3000 ...
start "Frontend" cmd /k "cd /d "%ROOT%frontend" && npm run dev -- --host 127.0.0.1 --port 3000 --strictPort"
timeout /t 2 /nobreak > nul

echo.
echo Started successfully.
echo Backend : http://localhost:4000
echo Frontend: http://localhost:3000
echo.
start http://localhost:3000
