@echo off
echo ========================================================
echo Starting SmartNews FastAPI Backend Server...
echo ========================================================
set VENV_PY=%~dp0..\venv\Scripts\python.exe

if not exist "%VENV_PY%" (
    echo Virtual environment not found at %VENV_PY%
    echo Please run scripts\setup_all.bat first.
    exit /b 1
)

cd /d "%~dp0..\backend"
"%VENV_PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
