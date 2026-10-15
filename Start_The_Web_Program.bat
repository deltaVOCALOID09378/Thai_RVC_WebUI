@echo off
TITLE 🎙️ DELTA SYNTH - RVC Studio Launcher
color 0A
cd /d "%~dp0"

echo ==========================================
echo 🌟 Made And Checked By DELTA SYNTH ^& All Agentic AI
echo 🌟 Original by RVC-Project
echo ==========================================
echo 🚀 Starting AI Voice Conversion System...

:: Drag and Drop Folder Support
set "TARGET_DIR=%~1"
if not "%TARGET_DIR%"=="" (
    echo 📂 Target workspace loaded: %TARGET_DIR%
)

echo 🌐 Launching Web Server on Port 9378...
"runtime\python.exe" Program.py --pycmd "runtime\python.exe" --port 9378

echo.
echo 🛑 Server stopped or encountered an issue.
pause