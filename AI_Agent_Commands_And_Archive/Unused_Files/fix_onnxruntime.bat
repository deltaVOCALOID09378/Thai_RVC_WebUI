@echo off
REM ============================================================================
REM File Name      : fix_onnxruntime.bat
REM Version        : 1.0
REM Target Path    : A:\Thai_RVC_WebUI_v1\fix_onnxruntime.bat
REM Function       : Install missing onnxruntime module for DiffSinger environment
REM Credit         : Made And Checked By DELTA SYNTH & All Agentic AI
REM Original by    : DELTA SYNTH
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

title DELTA SYNTH - ONNXRuntime Auto Installer [v1.0]

set "PYTHON_EXE=runtime\python.exe"

echo [INFO] Checking Python environment...
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python executable not found at: "%CD%\%PYTHON_EXE%"
    pause
    exit /b 1
)

echo [INFO] Upgrading pip to the latest version...
"%PYTHON_EXE%" -m pip install --upgrade pip

echo [INFO] Installing required onnxruntime and onnx packages...
"%PYTHON_EXE%" -m pip install onnxruntime onnx --upgrade

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Installation failed. Please check your internet connection.
) else (
    echo.
    echo [SUCCESS] Dependencies have been successfully installed.
    echo [INFO] You can now close this window and run the Binarize process again.
)

echo.
pause
exit /b 0