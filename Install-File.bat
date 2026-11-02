@echo off
setlocal EnableExtensions DisableDelayedExpansion

REM ==============================================================================
REM Date: 2026-09-29
REM File: Install-File.bat
REM Version: 1.4
REM Made And Checked By DELTA SYNTH & Gemini AI
REM Original by RVC Project
REM Path: .\Install-File.bat
REM Description: Interactive installer with custom package download, batch installation from txt/md, and system updates (Boosted by UV).
REM ==============================================================================

set "SCRIPT_DIR=%~dp0"
set "TARGET_DIR=%~1"
if not defined TARGET_DIR set "TARGET_DIR=%SCRIPT_DIR%"
if not exist "%TARGET_DIR%\." if exist "%TARGET_DIR%" set "TARGET_DIR=%~dp1"

pushd "%TARGET_DIR%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Cannot open the target directory: "%TARGET_DIR%"
    pause
    exit /b 1
)

:: Support both runtime and venv
set "PYTHON="
if exist "runtime\python.exe" (
    set "PYTHON=%CD%\runtime\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON=%CD%\venv\Scripts\python.exe"
)

if not defined PYTHON (
    echo [ERROR] Embedded Python was not found!
    echo [INFO] Place this installer beside the runtime or venv folder.
    goto :end_script
)

set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONUTF8=1"

:: Determine if UV is available
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] UV not found. Installing UV via pip...
    "%PYTHON%" -m pip install uv
)
where uv >nul 2>nul
if %errorlevel% equ 0 (
    set "PIP_CMD=uv pip"
) else (
    set "PIP_CMD="%PYTHON%" -m pip"
)

:menu_loop
cls
echo =======================================================
echo     DELTA SYNTH - Environment ^& Package Manager (UV Mode)
echo =======================================================
echo [1] Download a specific package by name
echo [2] Install packages from all .txt and .md files
echo [3] Check installed packages and their versions
echo [4] Update all installed packages to latest versions
echo [5] Update Python pip
echo [6] Install Torch, PyTorch, DirectML, Torchaudio, and ROCm
echo [0] Exit
echo =======================================================
set /p "choice=Select an option (0-6): "

if "%choice%"=="1" goto :opt_specific
if "%choice%"=="2" goto :opt_files
if "%choice%"=="3" goto :opt_check
if "%choice%"=="4" goto :opt_update_all
if "%choice%"=="5" goto :opt_pip
if "%choice%"=="6" goto :opt_core_ml
if "%choice%"=="0" goto :end_script
goto :menu_loop

:opt_specific
echo.
set /p "pkg_name=Enter the exact package name to install: "
if not "%pkg_name%"=="" (
    echo [INFO] Installing %pkg_name% using %PIP_CMD%...
    %PIP_CMD% install %pkg_name%
)
pause
goto :menu_loop

:opt_files
echo.
echo [INFO] Installing from all .txt files...
for %%f in (*.txt) do (
    echo [INFO] Reading %%f...
    %PIP_CMD% install -r "%%f"
)
echo [INFO] Scanning .md files for pip install commands...
echo import os, re, subprocess > temp_md_parser.py
echo pkgs = set() >> temp_md_parser.py
echo for f in os.listdir('.'): >> temp_md_parser.py
echo     if f.endswith('.md'): >> temp_md_parser.py
echo         try: >> temp_md_parser.py
echo             content = open(f, encoding='utf-8').read() >> temp_md_parser.py
echo             for match in re.findall(r'pip install ([a-zA-Z0-9\-\_\=\.\>\, ]+)', content): >> temp_md_parser.py
echo                 for p in match.split(): pkgs.add(p) >> temp_md_parser.py
echo         except: pass >> temp_md_parser.py
echo for p in pkgs: >> temp_md_parser.py
echo     if p and not p.startswith('-'): subprocess.call(r'%PIP_CMD% install ' + p, shell=True) >> temp_md_parser.py
"%PYTHON%" temp_md_parser.py
del temp_md_parser.py
echo [INFO] Installation from text and markdown files completed.
pause
goto :menu_loop

:opt_check
echo.
echo [INFO] Current System Packages and Versions:
"%PYTHON%" -m pip list
pause
goto :menu_loop

:opt_update_all
echo.
echo [INFO] Updating all installed packages to their latest versions...
echo import subprocess > temp_upgrade.py
echo reqs = subprocess.check_output([r'%PYTHON%', '-m', 'pip', 'freeze']).decode('utf-8').splitlines() >> temp_upgrade.py
echo pkgs = [r.split('==')[0] for r in reqs if '==' in r] >> temp_upgrade.py
echo if pkgs: subprocess.call(r'%PIP_CMD% install --upgrade ' + ' '.join(pkgs), shell=True) >> temp_upgrade.py
"%PYTHON%" temp_upgrade.py
del temp_upgrade.py
echo [INFO] All packages updated.
pause
goto :menu_loop

:opt_pip
echo.
echo [INFO] Upgrading pip...
"%PYTHON%" -m pip install --upgrade pip
pause
goto :menu_loop

:opt_core_ml
echo.
echo [INFO] Installing Core ML Frameworks (PyTorch, DirectML, Audio, ROCm components)...
%PIP_CMD% install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
%PIP_CMD% install --upgrade torch-directml
%PIP_CMD% install --upgrade tensorflow-rocm
echo [INFO] Core frameworks installation finished.
pause
goto :menu_loop

:end_script
echo.
echo [SUCCESS] Operation completed. Returning to system...
popd
pause
exit /b 0