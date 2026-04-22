@echo off
setlocal

echo ==========================================
echo AI Learning App - Stop Servers
echo ==========================================
echo.

set "FOUND=0"
for %%P in (3000 4000) do (
  for /f "tokens=5" %%A in ('netstat -aon ^| findstr /R /C:":%%P .*LISTENING"') do (
    set "FOUND=1"
    echo Stopping PID %%A on port %%P ...
    taskkill /PID %%A /F >nul 2>&1
  )
)

if "%FOUND%"=="0" (
  echo No running backend/frontend listeners found.
) else (
  echo Done. Ports 3000 and 4000 are now free.
)

echo.
