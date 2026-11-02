@echo off
:: ==========================================
:: Made And Checked By DELTA SYNTH & Staff
:: Original by DELTA SYNTH
:: Version 1.0
:: ==========================================
title DELTA SYNTH - Installation and Verification Tool
color 0A

echo ==========================================
echo    DELTA SYNTH INSTALLATION MODULE
echo ==========================================
echo.
echo [10%%] Initializing environment variables...

:: Set Python Path based on DELTA SYNTH Standards
set PYTHON_PATH=runtime\python.exe
if not exist "%PYTHON_PATH%" (
    set PYTHON_PATH=A:\RVC1006AMD_Intel1\runtime\python.exe
)

echo [30%%] Verifying Python runtime at: %PYTHON_PATH%
if not exist "%PYTHON_PATH%" (
    echo [ERROR] Python runtime not found! Please check your directory.
    pause
    exit /b
)

echo [50%%] Checking for Drag and Drop inputs...
if "%~1" neq "" (
    echo [60%%] Target file detected: %~1
    echo [80%%] Installing target package...
    "%PYTHON_PATH%" -m pip install "%~1"
    goto :FINISH
)

echo [60%%] Verifying requirements.txt...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt is missing from the directory!
    pause
    exit /b
)

echo [80%%] Installing dependencies from requirements.txt...
"%PYTHON_PATH%" -m pip install -r requirements.txt

:FINISH
echo.
echo [100%%] Installation and Verification Completed Successfully!
echo ==========================================
pause