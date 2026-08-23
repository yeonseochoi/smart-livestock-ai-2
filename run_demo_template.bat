@echo off
setlocal
cd /d "%~dp0"
set "LLM_PROVIDER=template"

echo Starting safe template mode without an API call...
if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python 3.12 is required. Install it from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    py -3.12 -m venv .venv
)

".venv\Scripts\python.exe" -c "import pandas, numpy, openpyxl, sklearn, matplotlib, folium, requests, xgboost" >nul 2>nul
if errorlevel 1 ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8765'"
".venv\Scripts\python.exe" demo\server.py --port 8765

if errorlevel 1 (
    echo Server failed to start. Check whether port 8765 is already in use.
    pause
)
endlocal
