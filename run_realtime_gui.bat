@echo off
:: ==========================================
:: Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
:: Original by RVC Project
:: File: run_realtime_gui.bat | Version: 1.0
:: Location: Root directory (A:\Thai_RVC_WebUI_v1\)
:: Function: Sets up the working directory and launches realtime_gui.py
:: ==========================================

echo [INFO] Preparing the environment for Thai RVC WebUI...
set "PYTHONPATH=%~dp0"
cd /d "%~dp0"

echo [INFO] Starting Realtime GUI via runtime\python.exe...

if exist "runtime\python.exe" (
    "runtime\python.exe" realtime_gui.py
) else (
    echo [ERROR] runtime\python.exe was not found in the current directory.
    echo [ERROR] Please ensure this batch file is placed in the main folder.
    echo [INFO] Attempting to use system python as a fallback...
    python realtime_gui.py
)

echo.
pause