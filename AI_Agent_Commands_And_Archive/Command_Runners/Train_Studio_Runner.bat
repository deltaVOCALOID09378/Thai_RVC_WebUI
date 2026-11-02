@echo off
chcp 65001 >nul
title DELTA SYNTH - RVC Training Studio High-Performance Engine
cd /d "%~dp0"

echo ============================================================
echo DELTA SYNTH - RVC Studio Training Diagnostic ^& Engine
echo Made And Checked By DELTA SYNTH ^& Gemini AI
echo Original by Patiphat Wongyai (Delta) ^& RVC Project
echo ============================================================
echo.

set "DROPPED_PATH=%~1"
if not "%DROPPED_PATH%"=="" (
    echo [INFO] Detected dropped input directory: "%DROPPED_PATH%"
    set "DATASET_DIR=%DROPPED_PATH%"
    set "EXP_NAME=%~nx1"
    echo [INFO] Inferred Experiment Name: "%EXP_NAME%"
) else (
    set /p "EXP_NAME=Enter Experiment Name (Model Name): "
    set /p "DATASET_DIR=Enter Dataset Directory Path (or drag & drop folder here): "
)

if "%EXP_NAME%"=="" (
    echo [ERROR] No experiment name provided. Aborting.
    pause
    exit /b 1
)

echo.
echo [INFO] Running automatic pre-flight verification and dataset auto-cleaning...
if exist "%~dp0runtime\python.exe" (
    "%~dp0runtime\python.exe" "%~dp0infer\modules\train\train_pipeline_master.py" "%EXP_NAME%" "%DATASET_DIR%"
) else (
    python "%~dp0infer\modules\train\train_pipeline_master.py" "%EXP_NAME%" "%DATASET_DIR%"
)

echo.
echo [INFO] Training task finished.
pause
