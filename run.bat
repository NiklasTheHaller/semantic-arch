@echo off
setlocal
echo ============================================================
echo   Semantic Architecture Workbench (Windows)
echo ============================================================
cd /d "%~dp0"

REM Terminate any lingering previous instance to prevent port 3030 collision
taskkill /F /IM earlybird-arch.exe >nul 2>&1

REM If qdrant is not running, attempt to start it via docker-compose if docker is installed
where docker >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Ensuring Qdrant vector database is running...
    docker compose up -d >nul 2>&1
)

echo Starting Workbench in Release Mode on http://localhost:3030 ...
cd earlybird-arch
cargo run --release -- serve --port 3030

pause
