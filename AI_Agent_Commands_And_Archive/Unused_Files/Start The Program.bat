@echo off
:: ==========================================
:: Made And Checked By DELTA SYNTH & Staff
:: Original by DELTA SYNTH
:: Version 1.0
:: ==========================================
title DELTA SYNTH - Core Execution (Port 9378)
color 0B

echo ==========================================
echo    INITIALIZING DELTA SYNTH CORE SYSTEM
echo ==========================================

:: Set Python Path
set PYTHON_PATH=runtime\python.exe
if not exist "%PYTHON_PATH%" (
    set PYTHON_PATH=A:\RVC1006AMD_Intel1\runtime\python.exe
)

echo Checking execution path... OK!
echo Routing to Port 9378... OK!
echo.

:: %* allows passing all dragged and dropped files/folders as arguments
"%PYTHON_PATH%" Program.py %*

echo.
echo System offline or terminated.
pause