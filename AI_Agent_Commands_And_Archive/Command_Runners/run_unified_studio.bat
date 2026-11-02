@echo off
REM ==============================================================================
REM File Name      : run_unified_studio.bat
REM Version        : v5.1
REM Location       : Root Directory of the Project (Same folder as runtime folder)
REM Role/Function  : Continuous drag-and-drop folder processor, version-based
REM                  model path organizer, and GUI launcher using runtime\python.exe
REM Credits        : Made And Checked By DELTA SYNTH & All Agentic AI
REM                  Original by RVC Project & OpenUtau / DiffSinger Community
REM ==============================================================================

setlocal EnableDelayedExpansion
title Unified Vocal Studio v5.1 - Smart Model Path and Vocal Engine

REM Set default Python executable path to runtime\python.exe first
set "PYTHON_EXE=%~dp0runtime\python.exe"
set "CORE_SCRIPT=%~dp0unified_vocal_studio_core.py"

if not exist "%PYTHON_EXE%" (
    echo [INFO] Local runtime\python.exe was not found. Falling back to system Python.
    set "PYTHON_EXE=python"
)

REM Check if a folder or file was dragged and dropped onto the batch file
if not "%~1"=="" (
    set "TARGET_PATH=%~1"
    set "SELECTED_VERSION=Auto"
    goto :PROCESS_TARGET
)

:MAIN_MENU
cls
echo ==============================================================================
echo   UNIFIED VOCAL STUDIO v5.1 - MODEL PATH ORGANIZER AND VOCAL TUNING SYSTEM
echo   Made And Checked By DELTA SYNTH ^& All Agentic AI
echo ==============================================================================
echo.
echo   [1] Enter or Drag-and-Drop Model Folder/File Path (Auto-Detect Version)
echo   [2] Route Model Files Specifically for RVC v1
echo   [3] Route Model Files Specifically for RVC v2
echo   [4] Route Model Files Specifically for DiffSinger v1 / v2
echo   [5] Launch Bilingual Graphical User Interface (GUI Mode)
echo   [6] Exit Program
echo.
set "USER_CHOICE="
set /p "USER_CHOICE=Select an option (1-6) or drag a folder here and press Enter: "

if not defined USER_CHOICE goto :MAIN_MENU

REM Remove surrounding quotes if the user dragged a folder directly into the prompt
set "CLEAN_INPUT=%USER_CHOICE:"=%"

if "%CLEAN_INPUT%"=="1" (
    set "SELECTED_VERSION=Auto"
    goto :ASK_PATH
)
if "%CLEAN_INPUT%"=="2" (
    set "SELECTED_VERSION=RVC_v1"
    goto :ASK_PATH
)
if "%CLEAN_INPUT%"=="3" (
    set "SELECTED_VERSION=RVC_v2"
    goto :ASK_PATH
)
if "%CLEAN_INPUT%"=="4" (
    set "SELECTED_VERSION=DiffSinger_v2"
    goto :ASK_PATH
)
if "%CLEAN_INPUT%"=="5" (
    echo.
    echo [INFO] Launching Unified Vocal Studio GUI...
    "%PYTHON_EXE%" "%CORE_SCRIPT%"
    goto :MAIN_MENU
)
if "%CLEAN_INPUT%"=="6" (
    goto :EOF
)

REM If the user pasted or dragged a path directly into the main menu prompt
if exist "%CLEAN_INPUT%" (
    set "TARGET_PATH=%CLEAN_INPUT%"
    set "SELECTED_VERSION=Auto"
    goto :PROCESS_TARGET
)

echo.
echo [WARNING] Invalid option or path does not exist. Please try again.
pause
goto :MAIN_MENU

:ASK_PATH
echo.
set "INPUT_PATH="
set /p "INPUT_PATH=Drag and drop the target model folder/file here or type the full path: "
if not defined INPUT_PATH goto :MAIN_MENU
set "TARGET_PATH=%INPUT_PATH:"=%"

:PROCESS_TARGET
echo.
echo [INFO] Processing target path : "%TARGET_PATH%"
echo [INFO] Target version profile : "%SELECTED_VERSION%"
echo.
"%PYTHON_EXE%" "%CORE_SCRIPT%" "%TARGET_PATH%" "%SELECTED_VERSION%"
echo.
echo [SUCCESS] Operation completed. Ready for the next task.
echo.
set "NEXT_PATH="
set /p "NEXT_PATH=Drag another folder here to continue immediately, or press Enter for Main Menu: "
if not defined NEXT_PATH goto :MAIN_MENU
set "TARGET_PATH=%NEXT_PATH:"=%"
goto :PROCESS_TARGET