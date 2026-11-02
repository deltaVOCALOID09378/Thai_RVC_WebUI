@echo off
setlocal EnableDelayedExpansion
:: ==============================================================================
:: File Name      : Start The Program.bat
:: Version        : 5.5 (Master Unified Studio Launcher)
:: Location       : Root Directory
:: Function       : Launches DELTA SYNTH RVC Studio x DiffSinger Unified WebUI,
::                  supports continuous drag-and-drop routing, DirectML mode, 
::                  Real-time GUI, and DiffSinger v1.2 Trainer GUI.
:: Credit         : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
::                  Original by DELTA SYNTH, RVC Project & OpenVPI DiffSinger
:: ==============================================================================

cd /d "%~dp0"
title DELTA SYNTH Unified Studio v5.5 - Model Router and AI Vocal Engine
color 0B

:: Prioritize runtime\python.exe as the default Python interpreter
set "PYTHON_EXE=%~dp0runtime\python.exe"
if not exist "%PYTHON_EXE%" (
    if exist "%~dp0.venv\Scripts\python.exe" (
        set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    ) else if exist "%~dp0venv\Scripts\python.exe" (
        set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    ) else (
        set "PYTHON_EXE=A:\RVC1006AMD_Intel1\runtime\python.exe"
    )
)

:: Ensure Fallback if absolutely not found
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

:: If a folder or model file is dragged and dropped onto this batch file, route it immediately
if not "%~1"=="" (
    set "TARGET_INPUT=%~1"
    set "VERSION_PROFILE=AUTO"
    goto :ROUTE_PROCESS
)

:MAIN_MENU
cls
echo ==============================================================================
echo   DELTA SYNTH UNIFIED STUDIO v5.5 (RVC v1/v2 x OPENUTAU x DIFFSINGER 2.5.1)
echo   Made And Checked By DELTA SYNTH ^& Staff
echo ==============================================================================
echo   Active Python Runtime : "%PYTHON_EXE%"
echo ==============================================================================
echo.
echo   [1] Launch Unified WebUI Studio (Standard GPU / CPU Mode - Port 9378)
echo   [2] Launch Unified WebUI Studio (DirectML Mode for AMD / Intel GPU)
echo   [3] Launch Real-Time Voice Changer GUI (realtime_gui.py)
echo   [4] Smart Model Path Router (Auto-Detect Version from File/Folder Path)
echo   [5] Force Route Model Files to RVC v1 or RVC v2 Structure
echo   [6] Force Route Model Files to DiffSinger 2.5.1 Checkpoints
echo   [7] Run DiffSinger Standalone Trainer GUI (v1.2)
echo   [8] Exit Program
echo.
set "USER_OPT="
set /p "USER_OPT=Select an option (1-8) or drag and drop a model folder here: "

if not defined USER_OPT goto :MAIN_MENU
set "CLEAN_OPT=%USER_OPT:"=%"

if "%CLEAN_OPT%"=="1" (
    echo.
    echo [INFO] Starting DELTA SYNTH RVC x DiffSinger Unified WebUI on Port 9378...
    "%PYTHON_EXE%" "%~dp0Program.py" --pycmd "%PYTHON_EXE%" --port 9378
    pause
    goto :MAIN_MENU
)
if "%CLEAN_OPT%"=="2" (
    echo.
    echo [INFO] Starting Unified WebUI in DirectML Mode on Port 9378...
    "%PYTHON_EXE%" "%~dp0Program.py" --pycmd "%PYTHON_EXE%" --port 9378 --dml
    pause
    goto :MAIN_MENU
)
if "%CLEAN_OPT%"=="3" (
    echo.
    echo [INFO] Starting Real-Time Voice Conversion GUI...
    "%PYTHON_EXE%" "%~dp0realtime_gui.py"
    pause
    goto :MAIN_MENU
)
if "%CLEAN_OPT%"=="4" (
    set "VERSION_PROFILE=AUTO"
    goto :ASK_TARGET_PATH
)
if "%CLEAN_OPT%"=="5" (
    echo.
    set "VER_SEL=RVC_V2"
    set /p "VER_SEL=Enter RVC Version Code (RVC_V1 / RVC_V2 / PRETRAINED_V1 / PRETRAINED_V2) [Default: RVC_V2]: "
    set "VERSION_PROFILE=!VER_SEL!"
    goto :ASK_TARGET_PATH
)
if "%CLEAN_OPT%"=="6" (
    echo.
    set "DS_SEL=DIFFSINGER_ACOUSTIC"
    set /p "DS_SEL=Enter DiffSinger Profile (DIFFSINGER_ACOUSTIC / DIFFSINGER_VARIANCE / DIFFSINGER_VOCODER): "
    set "VERSION_PROFILE=!DS_SEL!"
    goto :ASK_TARGET_PATH
)
if "%CLEAN_OPT%"=="7" (
    echo.
    :: Updated to target the newly upgraded DiffSinger_Trainer_v1.2.py
    set "DS_TRAINER_SCRIPT=%~dp0DiffSinger_Trainer_v1.2.py"
    
    if exist "!DS_TRAINER_SCRIPT!" (
        echo [INFO] Launching DiffSinger Standalone Trainer GUI (v1.2)...
        "%PYTHON_EXE%" "!DS_TRAINER_SCRIPT!"
    ) else (
        echo [WARN] DiffSinger_Trainer_v1.2.py not found in root. Falling back to Unified WebUI...
        "%PYTHON_EXE%" "%~dp0Program.py" --pycmd "%PYTHON_EXE%" --port 9378
    )
    pause
    goto :MAIN_MENU
)
if "%CLEAN_OPT%"=="8" (
    echo [INFO] Terminating System. Goodbye.
    exit /b 0
)

:: If the user dragged and dropped a folder or file directly into the menu prompt
if exist "%CLEAN_OPT%" (
    set "TARGET_INPUT=%CLEAN_OPT%"
    set "VERSION_PROFILE=AUTO"
    goto :ROUTE_PROCESS
)

echo.
echo [ERROR] Invalid option or target path does not exist. Please try again.
pause
goto :MAIN_MENU

:ASK_TARGET_PATH
echo.
set "INPUT_PATH="
set /p "INPUT_PATH=Drag and drop the model file/folder here or enter the full path: "
if not defined INPUT_PATH goto :MAIN_MENU
set "TARGET_INPUT=%INPUT_PATH:"=%"

:ROUTE_PROCESS
echo.
echo ==============================================================================
echo [INFO] Executing Smart Model Router...
echo [INFO] Target Path     : "%TARGET_INPUT%"
echo [INFO] Version Profile : "%VERSION_PROFILE%"
echo ==============================================================================
"%PYTHON_EXE%" "%~dp0Program.py" --route_model "%TARGET_INPUT%" --version_profile "%VERSION_PROFILE%"
echo.
set "NEXT_ITEM="
set /p "NEXT_ITEM=Drag another folder/file here to continue immediately, or press Enter for Main Menu: "
if not defined NEXT_ITEM goto :MAIN_MENU
set "TARGET_INPUT=%NEXT_ITEM:"=%"
goto :ROUTE_PROCESS