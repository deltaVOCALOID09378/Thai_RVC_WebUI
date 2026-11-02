@echo off
chcp 65001 >nul
REM Made And Checked By DELTA SYNTH & Gemini AI
title DiffSinger Trainer v1.0
echo ============================================
echo  DiffSinger Trainer v1.0
echo  Made And Checked By DELTA SYNTH ^& Gemini AI
echo ============================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH. Please install Python 3.8+ first.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo [INFO] No folder dropped. Starting without a preset DiffSinger folder.
    python "%~dp0DiffSinger_Trainer_v1.0.py"
) else (
    echo [INFO] DiffSinger folder: %~1
    python "%~dp0DiffSinger_Trainer_v1.0.py" "%~1"
)

echo.
echo [INFO] Trainer closed.
pause
