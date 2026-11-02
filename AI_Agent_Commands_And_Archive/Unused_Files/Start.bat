@echo off
REM ==============================================================================
REM File Name      : Start-The-Program.bat
REM Version        : 4.7 (Master Convenience Launcher)
REM Location       : Root Directory (A:\Thai_RVC_WebUI_v1\Start-The-Program.bat)
REM Function       : Launches DELTA RVC Studio x DiffSinger Unified WebUI, supports
REM                  continuous drag-and-drop model folder/file routing by version,
REM                  and executes DiffSinger 1-Click Pipeline via runtime\python.exe.
REM Credit         : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
REM                  Original by Patiphat Wongyai (Delta), RVC Project & OpenVPI DiffSinger
REM ==============================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"
title DELTA SYNTH Unified Studio v5.4 - Model Router and AI Vocal Engine
color 0A

REM Prioritize runtime\python.exe as the default Python interpreter
set "PYTHON_EXE=%~dp0runtime\python.exe"
if not exist "%PYTHON_EXE%" (
    if exist "%~dp0.venv\Scripts\python.exe" (
        set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    ) else if exist "%~dp0venv\Scripts\python.exe" (
        set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    ) else (
        set "PYTHON_EXE=python"
    )
)

echo.
echo [INFO] Starting DELTA RVC x DiffSinger Unified WebUI on Port 9378...
    "%PYTHON_EXE%" "%~dp0Program.py" --pycmd "%PYTHON_EXE%" --port 9378
pause
