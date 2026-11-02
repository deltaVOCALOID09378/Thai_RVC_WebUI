@echo off
REM ============================================================================
REM File Name      : run_config_binarize.bat
REM Version        : 1.1
REM Target Path    : A:\Thai_RVC_WebUI_v1\run_config_binarize.bat
REM Function       : Drag-and-drop folder/config launcher for DiffSinger Binarize
REM Credit         : Made And Checked By DELTA SYNTH & All Agentic AI
REM Original by    : DELTA SYNTH
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

title DELTA SYNTH - DiffSinger Config & Binarize Runner [v1.1]

set "PYTHON_EXE=runtime\python.exe"
set "HELPER_SCRIPT=fix_diffsinger_config.py"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found at: "%CD%\%PYTHON_EXE%"
    pause
    exit /b 1
)

if not exist "%HELPER_SCRIPT%" (
    echo [ERROR] Helper script not found at: "%CD%\%HELPER_SCRIPT%"
    pause
    exit /b 1
)

:PROCESS_ARGS
set "TARGET_ITEM=%~1"

echo.
echo [INFO] Starting DiffSinger Config Resolver and Binarizer...
if "%TARGET_ITEM%"=="" (
    "%PYTHON_EXE%" "%HELPER_SCRIPT%"
) else (
    "%PYTHON_EXE%" "%HELPER_SCRIPT%" "%TARGET_ITEM%"
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Process finished with exit code: %ERRORLEVEL%
) else (
    echo.
    echo [SUCCESS] Binarization finished without errors.
)

shift
if not "%~1"=="" goto PROCESS_ARGS

:PROMPT_NEXT
echo.
echo ============================================================================
echo [READY] Drag and drop another dataset folder or config.yaml file here
echo         and press Enter to continue, or just press Enter to exit.
echo ============================================================================
set "USER_INPUT="
set /p "USER_INPUT=Target Folder/Config: "

if not "%USER_INPUT%"=="" (
    set "USER_INPUT=!USER_INPUT:"=!"
    "%PYTHON_EXE%" "%HELPER_SCRIPT%" "!USER_INPUT!"
    goto PROMPT_NEXT
)

exit /b 0