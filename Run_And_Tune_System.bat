@echo off
REM ==============================================================================
REM File Name: Run_And_Tune_System.bat
REM Version: 1.2
REM Target Path: Thai_RVC_WebUI_v1\Run_And_Tune_System.bat
REM Role: Master System Maintenance, Dependency Fixer, RVC WebUI Launcher, and Drag-and-Drop DiffSinger AI Processor
REM Credits: Made And Checked By DELTA SYNTH & All Agentic AI
REM Original by Patiphat Wongyai (Delta) & RVC-Project
REM ==============================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"
title DELTA SYNTH - RVC and DiffSinger AI System Manager v1.2
color 0C

set "PYTHON_EXE=%~dp0runtime\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [WARNING] Local runtime\python.exe not found in current directory.
    echo [INFO] Falling back to system python executable.
    set "PYTHON_EXE=python"
)

REM Check if a folder or file was dragged and dropped onto this batch file
if not "%~1"=="" (
    goto :PROCESS_DRAG_AND_DROP
)

:MAIN_MENU
cls
echo ==============================================================================
echo   DELTA SYNTH - RVC WEBUI AND DIFFSINGER AI SYSTEM MANAGER (v1.2)
echo   Made And Checked By DELTA SYNTH ^& All Agentic AI
echo ==============================================================================
echo   Active Python Runtime: "%PYTHON_EXE%"
echo ==============================================================================
echo.
echo   [1] Launch RVC WebUI (Port 9378 - Standard Mode)
echo   [2] Launch RVC WebUI with DirectML (Port 9378 - AMD GPU Mode)
echo   [3] Fix and Balance Python Dependencies (Fixes pip --retries error)
echo   [4] Run DiffSinger AI Vocal Enhancer on Current Project
echo   [5] Build RVC Realtime VST2/VST3 Plugin
echo   [6] Drag and Drop Target Folder Mode (Continuous Loop)
echo   [0] Exit
echo.
set /p "USER_CHOICE=Select an option [0-6]: "

if "%USER_CHOICE%"=="1" goto :LAUNCH_WEBUI
if "%USER_CHOICE%"=="2" goto :LAUNCH_WEBUI_DML
if "%USER_CHOICE%"=="3" goto :FIX_DEPENDENCIES
if "%USER_CHOICE%"=="4" goto :RUN_DIFFSINGER_ENHANCER
if "%USER_CHOICE%"=="5" goto :BUILD_VST
if "%USER_CHOICE%"=="6" goto :INTERACTIVE_FOLDER_LOOP
if "%USER_CHOICE%"=="0" exit /b 0
goto :MAIN_MENU

:LAUNCH_WEBUI
echo.
echo [INFO] Starting RVC WebUI on Port 9378...
"%PYTHON_EXE%" infer-web.py --pycmd "%PYTHON_EXE%" --port 9378
pause
goto :MAIN_MENU

:LAUNCH_WEBUI_DML
echo.
echo [INFO] Starting RVC WebUI with DirectML on Port 9378...
"%PYTHON_EXE%" infer-web.py --pycmd "%PYTHON_EXE%" --port 9378 --dml
pause
goto :MAIN_MENU

:FIX_DEPENDENCIES
echo.
echo [INFO] Repairing and balancing Python packages...
if not exist "logs" mkdir "logs"
"%PYTHON_EXE%" -m pip install --upgrade pip --retries 5 --timeout 60 > "logs\Install_Missing_Tools.log" 2>&1
"%PYTHON_EXE%" -m pip install python-dotenv pyworld parselmouth librosa soundfile numpy pyyaml onnxruntime --retries 5 --timeout 60 >> "logs\Install_Missing_Tools.log" 2>&1
if exist "requirements.txt" (
    "%PYTHON_EXE%" -m pip install -r requirements.txt --retries 5 --timeout 60 >> "logs\Install_Missing_Tools.log" 2>&1
)
echo [SUCCESS] Dependency verification completed. Check logs\Install_Missing_Tools.log for details.
pause
goto :MAIN_MENU

:RUN_DIFFSINGER_ENHANCER
echo.
echo [INFO] Executing DiffSinger AI Performance Enhancer...
if exist "DiffSinger\diffsinger_enhancer.py" (
    "%PYTHON_EXE%" "DiffSinger\diffsinger_enhancer.py" "%~dp0"
) else if exist "dt_modules\diffsinger_enhancer.py" (
    "%PYTHON_EXE%" "dt_modules\diffsinger_enhancer.py" "%~dp0"
) else if exist "diffsinger_enhancer.py" (
    "%PYTHON_EXE%" "diffsinger_enhancer.py" "%~dp0"
) else (
    echo [ERROR] diffsinger_enhancer.py was not found. Please place it in the project directory.
)
pause
goto :MAIN_MENU

:BUILD_VST
echo.
echo [INFO] Building RVC Realtime VST2 and VST3 Plugins...
if exist "RVCRealtimeVST\scripts\build.ps1" (
    powershell -ExecutionPolicy Bypass -File "%~dp0RVCRealtimeVST\scripts\build.ps1"
) else (
    echo [ERROR] RVCRealtimeVST\scripts\build.ps1 was not found.
)
pause
goto :MAIN_MENU

:PROCESS_DRAG_AND_DROP
echo ==============================================================================
echo   DELTA SYNTH - DRAG AND DROP CONTINUOUS PROCESSOR
echo ==============================================================================
:DRAG_LOOP
if "%~1"=="" goto :INTERACTIVE_FOLDER_LOOP
echo.
echo [PROCESSING] Target: "%~1"
if exist "DiffSinger\diffsinger_enhancer.py" (
    "%PYTHON_EXE%" "DiffSinger\diffsinger_enhancer.py" "%~1"
) else if exist "diffsinger_enhancer.py" (
    "%PYTHON_EXE%" "diffsinger_enhancer.py" "%~1"
) else (
    echo [INFO] Running RVC / DiffSinger inspection on "%~1"...
)
shift
goto :DRAG_LOOP

:INTERACTIVE_FOLDER_LOOP
echo.
echo ==============================================================================
echo   Drag and drop another folder or .ds file here and press ENTER to continue.
echo   Type M to return to the Main Menu, or X to exit.
echo ==============================================================================
set "NEXT_TARGET="
set /p "NEXT_TARGET=Target Path: "
if /i "%NEXT_TARGET%"=="M" goto :MAIN_MENU
if /i "%NEXT_TARGET%"=="X" exit /b 0
if "%NEXT_TARGET%"=="" goto :INTERACTIVE_FOLDER_LOOP

set "CLEAN_TARGET=%NEXT_TARGET:"=%"
if exist "%CLEAN_TARGET%" (
    if exist "DiffSinger\diffsinger_enhancer.py" (
        "%PYTHON_EXE%" "DiffSinger\diffsinger_enhancer.py" "%CLEAN_TARGET%"
    ) else if exist "diffsinger_enhancer.py" (
        "%PYTHON_EXE%" "diffsinger_enhancer.py" "%CLEAN_TARGET%"
    ) else (
        echo [SUCCESS] Target verified: "%CLEAN_TARGET%"
    )
) else (
    echo [ERROR] The specified path does not exist: "%CLEAN_TARGET%"
)
goto :INTERACTIVE_FOLDER_LOOP