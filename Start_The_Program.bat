@echo off
setlocal EnableExtensions
chcp 65001 >nul
title DELTA SYNTH - Thai RVC Studio Unified Launcher (Port 9378)

REM ============================================================================
REM File Name   : Start-The-Program.bat
REM Version     : 1.9 (PR #3 CodeRabbit Anchored Path & Continuous Loop Edition)
REM Description : Main portable launcher for DELTA RVC Studio (Program.py).
REM               Supports drag-and-drop dataset folders and automatic restart.
REM Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
REM ============================================================================

cd /d "%~dp0"

set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PY_EXE=%~dp0runtime\python.exe"
set "MAIN_SCRIPT=%~dp0Program.py"

if not exist "%PY_EXE%" (
    echo [WARN] Portable runtime not found at "%PY_EXE%". Falling back to system python...
    set "PY_EXE=python"
)

if not exist "%MAIN_SCRIPT%" (
    echo [ERROR] Core studio script not found at "%MAIN_SCRIPT%".
    pause
    exit /b 1
)

:LOOP
echo ============================================================================
echo   DELTA SYNTH - THAI RVC STUDIO ^& DIFFSINGER UNIFIED ENGINE
echo   WebUI Port : 9378 ^| Runtime : "%PY_EXE%"
echo ============================================================================

if not "%~1"=="" (
    echo [INFO] Drag-and-drop path detected: "%~1"
    if exist "%~dp0dataset_auto_cleaner.py" (
        "%PY_EXE%" "%~dp0dataset_auto_cleaner.py" "%~1"
    )
)

"%PY_EXE%" "%MAIN_SCRIPT%" --pycmd "%PY_EXE%" --port 9378 %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo [INFO] Studio process exited with code %EXIT_CODE%.
echo Press Ctrl+C to close this window, or press any key to restart the Studio...
pause >nul
goto :LOOP
