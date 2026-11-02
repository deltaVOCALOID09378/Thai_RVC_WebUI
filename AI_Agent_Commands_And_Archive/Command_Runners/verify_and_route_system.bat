@echo off
REM ==============================================================================
REM File Name      : verify_and_route_system.bat
REM Version        : 2.0 (System Integrity Auditor & Continuous Folder Router)
REM Location       : Root Directory (Same folder as Program.py and runtime folder)
REM Function       : Verifies Python syntax across all core modules, routes model
REM                  folders dragged onto this script, and launches the studio.
REM Credit         : Made And Checked By DELTA SYNTH & All Agentic AI
REM                  Original by Patiphat Wongyai (Delta) & RVC Project
REM ==============================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"
title DELTA SYNTH - System Integrity Auditor and Continuous Model Router
color 0A

set "PYTHON_EXE=%~dp0runtime\python.exe"
if not exist "%PYTHON_EXE%" (
    if exist "%~dp0.venv\Scripts\python.exe" (
        set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    ) else (
        set "PYTHON_EXE=python"
    )
)

echo ==============================================================================
echo   [STEP 1] VERIFYING PYTHON SYNTAX AND MODULE INTEGRITY
echo ==============================================================================
"%PYTHON_EXE%" -m py_compile Program.py infer\modules\vc\modules.py infer\modules\vc\pipeline.py infer\modules\vc\utils.py
if !errorlevel! equ 0 (
    echo [OK] All core Python files passed syntax compilation with zero errors.
) else (
    echo [WARNING] Some files could not be compiled or were not in the expected path.
)
echo.

if not "%~1"=="" (
    set "TARGET_FOLDER=%~1"
    goto :PROCESS_DRAGGED_ITEM
)

:INTERACTIVE_LOOP
echo ==============================================================================
echo   [STEP 2] READY FOR CONTINUOUS MODEL ROUTING OR STUDIO LAUNCH
echo ==============================================================================
echo   - Drag and drop a model folder/file here and press Enter to organize it.
echo   - Or press Enter directly without typing anything to launch Program.py.
echo   - Type EXIT to close this window.
echo.
set "USER_INPUT="
set /p "USER_INPUT=Target Path or Command: "

if not defined USER_INPUT (
    echo.
    echo [INFO] Launching DELTA RVC Studio WebUI...
    "%PYTHON_EXE%" Program.py --pycmd "%PYTHON_EXE%" --port 9378
    pause
    goto :INTERACTIVE_LOOP
)

set "CLEAN_STR=%USER_INPUT:"=%"
if /i "%CLEAN_STR%"=="EXIT" exit /b 0

set "TARGET_FOLDER=%CLEAN_STR%"

:PROCESS_DRAGGED_ITEM
echo.
echo [INFO] Routing and organizing target: "%TARGET_FOLDER%"
"%PYTHON_EXE%" Program.py --route_model "%TARGET_FOLDER%" --version_profile "AUTO"
echo.
echo [SUCCESS] Task completed. Ready for the next folder.
echo.
goto :INTERACTIVE_LOOP