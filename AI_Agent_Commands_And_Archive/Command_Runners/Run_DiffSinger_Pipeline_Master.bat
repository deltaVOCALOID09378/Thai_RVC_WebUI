@echo off
chcp 65001 >nul
REM ==============================================================================
REM Date: 18/10/2026 (B.E. 2569) | 18 October 2026
REM File: Run_DiffSinger_Pipeline_Master.bat | Version: 3.0.0
REM Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
REM Original by Patiphat Wongyai (Delta) & OpenVPI
REM Description: One-Click Master Launcher for DiffSinger 2.5.1 Full-System Pipeline
REM ==============================================================================

title DELTA SYNTH - DiffSinger 2.5.1 Master Pipeline Runner
echo ==============================================================================
echo   DELTA SYNTH - DiffSinger 2.5.1 Master Pipeline Runner (v3.0)
echo   Supports: Autoclean, Phonemes, Binarize, Train, Export ONNX, OpenUtau Package
echo ==============================================================================
echo.

set "PROJECT_ROOT=%~dp0"
set "RUNTIME_PY=%PROJECT_ROOT%runtime\python.exe"

if exist "%RUNTIME_PY%" (
    set "PYTHON_EXE=%RUNTIME_PY%"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python runtime was not found!
        echo Please ensure runtime\python.exe exists or Python is installed in PATH.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

echo [INFO] Python Interpreter: %PYTHON_EXE%
echo.

if "%~1"=="" (
    echo [USAGE] Drag and drop a dataset folder or YAML config onto this script,
    echo         or press Enter to launch the interactive Pipeline menu.
    echo.
    "%PYTHON_EXE%" "%PROJECT_ROOT%tools\diffsinger\diffsinger_pipeline_master.py" --help
) else (
    echo [INFO] Target Input: %~1
    if exist "%~1\raw" (
        echo [INFO] Detected dataset folder with raw/ subfolder. Starting Autoclean & Dict...
        "%PYTHON_EXE%" "%PROJECT_ROOT%tools\diffsinger\diffsinger_pipeline_master.py" autoclean --dataset "%~1"
        "%PYTHON_EXE%" "%PROJECT_ROOT%tools\diffsinger\diffsinger_pipeline_master.py" dict --dataset "%~1"
    ) else if "%~x1"==".yaml" (
        echo [INFO] Detected YAML configuration file. Starting Binarization...
        "%PYTHON_EXE%" "%PROJECT_ROOT%tools\diffsinger\diffsinger_pipeline_master.py" binarize --config "%~1"
    ) else (
        echo [INFO] Processing target folder...
        "%PYTHON_EXE%" "%PROJECT_ROOT%tools\diffsinger\diffsinger_pipeline_master.py" autoclean --dataset "%~1"
    )
)

echo.
echo ==============================================================================
echo   Process completed. Press any key to exit.
echo ==============================================================================
pause >nul
