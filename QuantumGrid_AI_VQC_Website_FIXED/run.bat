@echo off
title QuantumGrid AI
cd /d "%~dp0"

echo ==========================================
echo       QUANTUMGRID AI
echo  Electricity Demand Forecasting
echo ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Install Python 3.10 or 3.11 and try again.
    pause
    exit /b 1
)

echo Installing required packages...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Package installation failed.
    pause
    exit /b 1
)

echo.
echo Starting QuantumGrid AI...
echo Keep this window OPEN while using the website.
echo.
start "" http://127.0.0.1:5000
python app.py

pause
