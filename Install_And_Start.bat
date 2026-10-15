@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
TITLE DELTA SYNTH - RVC Auto Installer
color 0E

rem ============================================================================
rem DELTA SYNTH - RVC Auto Installer
rem Made And Checked By DELTA SYNTH & Gemini AI
rem Original by Patiphat Wongyai (Delta) & RVC Project
rem ============================================================================

cd /d "%~dp0"

rem Define core paths
set "RUNTIME_DIR=%~dp0runtime"
set "PY_EXEC=%RUNTIME_DIR%\python.exe"
set "REQ_FILE=%~dp0requirements.txt"

echo ===============================================================
echo  🛠️  DELTA SYNTH - RVC Environment Setup Initialization
echo ===============================================================
echo.

rem Pre-check for runtime environment
if not exist "!PY_EXEC!" (
    echo ❌ [ERROR] Python runtime not found at: !RUNTIME_DIR!
    echo Please ensure the 'runtime' folder exists before proceeding.
    echo.
    pause
    exit /b 1
)

echo 🔄 [1/3] Upgrading core Python tools (pip, setuptools, wheel)...
"!PY_EXEC!" -m pip install --upgrade pip setuptools wheel
if !ERRORLEVEL! equ 0 (
    echo ✅ [SUCCESS] Core tools upgraded successfully.
) else (
    echo ⚠️  [WARNING] Minor issues upgrading core tools. Continuing...
)
echo.

echo 📦 [2/3] Installing essential libraries from requirements.txt...
if exist "!REQ_FILE!" (
    "!PY_EXEC!" -m pip install -r "!REQ_FILE!"
    if !ERRORLEVEL! equ 0 (
        echo ✅ [SUCCESS] All requirements installed flawlessly.
    ) else (
        echo ❌ [ERROR] Some packages failed to install. Please review the logs.
    )
) else (
    echo ⚠️  [WARNING] requirements.txt not found. Skipping this step...
)
echo.

echo 🔌 [3/3] Running dependency updates (python-dotenv)...
"!PY_EXEC!" -m pip install python-dotenv
if !ERRORLEVEL! equ 0 (
    echo ✅ [SUCCESS] Additional dependencies installed correctly.
) else (
    echo ❌ [ERROR] Failed to install additional dependencies.
)
echo.

echo ===============================================================
echo  ✨ 100%% Installation Sequence Completed.
echo  🚀 You can now use Start-The-Program.bat to launch the system.
echo ===============================================================
pause
exit /b 0