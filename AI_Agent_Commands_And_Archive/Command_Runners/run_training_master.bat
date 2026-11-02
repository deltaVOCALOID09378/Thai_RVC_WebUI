@echo off
setlocal
chcp 65001 >nul
title DELTA SYNTH - Automated Training Master Engine
echo ========================================================
echo File Name: run_training_master.bat
echo Version: 1.1
echo Location: Root Directory
echo Function: Drag-and-drop Launcher for Training Pipeline
echo Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
echo Original by Patiphat Wongyai (Delta) & RVC-Project
echo ========================================================

if "%~1"=="" (
    echo [WARNING] No dataset folder detected!
    echo Please drag and drop a valid dataset folder directly onto this batch file.
    pause
    exit /b 1
)

set "DATASET_DIR=%~1"
set "ROOT_DIR=%~dp0"
set "PYTHON_EXE=%ROOT_DIR%runtime\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [INFO] Local python environment not found, falling back to system python.
    set "PYTHON_EXE=python"
)

echo [INFO] Starting Master Training Pipeline...
echo [INFO] Target Dataset Path: %DATASET_DIR%
echo --------------------------------------------------------

"%PYTHON_EXE%" "%ROOT_DIR%infer\modules\train\train_pipeline_master.py" "%~nx1" "%DATASET_DIR%"

echo --------------------------------------------------------
echo [SUCCESS] Training process has terminated.
pause