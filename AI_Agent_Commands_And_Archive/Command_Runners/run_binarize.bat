@echo off
REM ============================================================================
REM File Name      : run_binarize.bat
REM Version        : 1.0
REM Target Path    : A:\Thai_RVC_WebUI_v1\run_binarize.bat
REM Function       : Automatically locate and execute DiffSinger binarize.py
REM Credit         : Made And Checked By DELTA SYNTH & All Agentic AI
REM Original by    : OpenVPI / DiffSinger Team
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

title DiffSinger Binarizer Launcher [v1.0]

set "PYTHON_EXE=runtime\python.exe"
set "SCRIPT_PATH_1=DiffSinger\scripts\binarize.py"
set "SCRIPT_PATH_2=DiffSinger\binarize.py"

echo [INFO] Checking Python environment...
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found at: "%CD%\%PYTHON_EXE%"
    pause
    exit /b 1
)

:PROCESS_LOOP
set "CONFIG_FILE=%~1"

if "%CONFIG_FILE%"=="" (
    echo.
    echo [INFO] You can drag and drop your config.yaml folder or file here.
    set /p "CONFIG_FILE=Enter path to config.yaml (or press Enter to use default): "
)

if exist "%SCRIPT_PATH_1%" (
    set "TARGET_SCRIPT=%SCRIPT_PATH_1%"
) else if exist "%SCRIPT_PATH_2%" (
    set "TARGET_SCRIPT=%SCRIPT_PATH_2%"
) else (
    echo.
    echo [ERROR] Cannot find binarize.py in "%CD%\DiffSinger\scripts" or "%CD%\DiffSinger"
    echo [INFO] Please verify that the DiffSinger repository files are fully extracted.
    pause
    exit /b 2
)

echo [INFO] Using script: "%TARGET_SCRIPT%"

if "%CONFIG_FILE%"=="" (
    "%PYTHON_EXE%" "%TARGET_SCRIPT%"
) else (
    if exist "%CONFIG_FILE%\config.yaml" (
        "%PYTHON_EXE%" "%TARGET_SCRIPT%" --config "%CONFIG_FILE%\config.yaml"
    ) else (
        "%PYTHON_EXE%" "%TARGET_SCRIPT%" --config "%CONFIG_FILE%"
    )
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Process exited with error code: %ERRORLEVEL%
) else (
    echo.
    echo [SUCCESS] Binarization completed successfully.
)

shift
if not "%~1"=="" goto PROCESS_LOOP

echo.
set /p "RETRY=Do you want to process another folder or config? (Y/N): "
if /i "%RETRY%"=="Y" (
    set "CONFIG_FILE="
    goto PROCESS_LOOP
)

pause
exit /b 0