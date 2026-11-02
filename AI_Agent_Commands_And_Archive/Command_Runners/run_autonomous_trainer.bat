@echo off
REM ==============================================================================
REM File Name      : run_autonomous_trainer.bat
REM Version        : 3.2 (Next-Gen Autonomous Training & Continuous Folder Drop)
REM Location       : Root Directory (Same folder as Program.py and runtime folder)
REM Function       : Allows dragging and dropping a dataset folder to execute the
REM                  complete RVC training pipeline (Preprocess + F0 + HuBERT +
REM                  Train + FAISS Index) continuously using runtime\python.exe.
REM Credit         : Made And Checked By DELTA SYNTH & All Agentic AI
REM                  Original by Patiphat Wongyai (Delta) & RVC-Project
REM ==============================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"
title DELTA SYNTH - Autonomous Training Pipeline Launcher v3.2
color 0A

set "PYTHON_EXE=%~dp0runtime\python.exe"
if not exist "%PYTHON_EXE%" (
    if exist "%~dp0.venv\Scripts\python.exe" (
        set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    ) else (
        set "PYTHON_EXE=python"
    )
)

if not "%~1"=="" (
    set "DATASET_PATH=%~1"
    for %%I in ("%~1") do set "EXP_NAME=%%~nxI"
    set "MODEL_VER=v2"
    goto :START_PIPELINE
)

:PROMPT_LOOP
cls
echo ==============================================================================
echo   DELTA SYNTH - AUTONOMOUS RVC TRAINING ENGINE v3.2
echo   Made And Checked By DELTA SYNTH ^& All Agentic AI
echo ==============================================================================
echo   Active Python Runtime : "%PYTHON_EXE%"
echo ==============================================================================
echo.
set "INPUT_DIR="
set /p "INPUT_DIR=Drag and drop your dataset audio folder here (or type EXIT to quit): "

if not defined INPUT_DIR goto :PROMPT_LOOP
set "CLEAN_DIR=%INPUT_DIR:"=%"
if /i "%CLEAN_DIR%"=="EXIT" exit /b 0

if not exist "%CLEAN_DIR%" (
    echo [ERROR] The specified folder does not exist: "%CLEAN_DIR%"
    pause
    goto :PROMPT_LOOP
)

set "DATASET_PATH=%CLEAN_DIR%"
for %%I in ("%CLEAN_DIR%") do set "DEFAULT_EXP=%%~nxI"

set "EXP_NAME=%DEFAULT_EXP%"
set /p "EXP_NAME=Enter Experiment/Model Name [Press Enter for %DEFAULT_EXP%]: "

set "MODEL_VER=v2"
set /p "MODEL_VER=Select Model Version (v1 / v2) [Press Enter for v2]: "

:START_PIPELINE
echo.
echo [INFO] Starting Autonomous Training Pipeline for "%EXP_NAME%" (%MODEL_VER%)...
"%PYTHON_EXE%" infer\modules\train\train_pipeline_master.py "%EXP_NAME%" "%DATASET_PATH%" "%MODEL_VER%"
echo.
echo [SUCCESS] Pipeline execution finished.
echo.
set "NEXT_DROP="
set /p "NEXT_DROP=Drag another dataset folder here to train the next model, or press Enter for menu: "
if not defined NEXT_DROP goto :PROMPT_LOOP
set "DATASET_PATH=%NEXT_DROP:"=%"
for %%I in ("%DATASET_PATH%") do set "EXP_NAME=%%~nxI"
goto :START_PIPELINE