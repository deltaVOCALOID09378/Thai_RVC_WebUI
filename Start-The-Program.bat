@echo off
setlocal enabledelayedexpansion
title DELTA SYNTH - High-Performance Studio Launcher
color 0A

:: ==============================================================================
:: File Name   : Start-The-Program.bat
:: Version     : 1.8
:: Location    : Root Directory
:: Function    : Launch Thai RVC Studio Environment with Continuous Drag-and-Drop
:: Credit      : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
::               Original by Patiphat Wongyai (Delta) & RVC Project
:: ==============================================================================

cd /d "%~dp0"

set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=0"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "DELTA_PORT=9378"

if "%NUMBER_OF_PROCESSORS%"=="" (
    set "OMP_NUM_THREADS=4"
    set "MKL_NUM_THREADS=4"
) else (
    if %NUMBER_OF_PROCESSORS% GEQ 8 (
        set "OMP_NUM_THREADS=8"
        set "MKL_NUM_THREADS=8"
    ) else (
        set "OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
        set "MKL_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
    )
)

set "PATH=%~dp0;%~dp0runtime;%~dp0runtime\Scripts;%PATH%"
set "PYTHON_CMD=%~dp0runtime\python.exe"

if not exist "!PYTHON_CMD!" (
    echo [WARNING] Internal python runtime not found at !PYTHON_CMD!
    echo [INFO] Attempting to use global Python environment...
    set "PYTHON_CMD=python"
)

:LOOP
cls
echo ============================================================
echo   DELTA SYNTH - High-Performance Execution Engine
echo ============================================================
echo [INFO] Active CPU Threads: %OMP_NUM_THREADS% ^| Port: %DELTA_PORT%
echo [INFO] Ready for processing tasks...

set "TARGET_DIR=%~1"
if not "!TARGET_DIR!"=="" (
    echo [INFO] Drag-and-drop target detected: "!TARGET_DIR!"
    echo [INFO] Starting engine with targeted directory parameter...
    "!PYTHON_CMD!" Program.py --pycmd "!PYTHON_CMD!" --port !DELTA_PORT! --target "!TARGET_DIR!"
) else (
    echo [INFO] No directory dropped. Initializing default studio interface...
    echo [INFO] Starting Web Server on port !DELTA_PORT!...
    "!PYTHON_CMD!" Program.py --pycmd "!PYTHON_CMD!" --port !DELTA_PORT!
)

echo.
echo ============================================================
echo [INFO] System task terminated or process closed cleanly.
echo ============================================================

shift
if not "%~1"=="" (
    echo [INFO] Next queue item detected: "%~1"
    echo [INFO] Press any key to continue processing the next item...
    pause > nul
    goto LOOP
)

echo [INFO] All operations finished. Press any key to exit.
pause > nul
exit /b