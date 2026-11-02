@echo off
SETLOCAL EnableDelayedExpansion

:: ========================================================================
:: Made And Checked By DELTA SYNTH & Gemini AI
:: Original by DELTA SYNTH
:: Version: 1.2
:: ========================================================================

rem reference from https://hituji-ws.com/code/python/python-emb-usage/
SET "DP0=%~dp0"
SET "DP0=%DP0:~0,-1%"

rem --- 1. SET DIRECTORY PATHS ---
SET "LOCALENV_DIR=%DP0%\_local_env"
SET "TMP_DIR=%LOCALENV_DIR%\_tmp"
SET "HOME_DIR=%LOCALENV_DIR%\userprofile"
SET "LOCALAPP_DIR=%LOCALENV_DIR%\localappdata"
SET "ROAMING_DIR=%LOCALENV_DIR%\userroaming"
SET "PYTHON_PATH=%DP0%\runtime"

rem --- 2. AUTO CREATE FOLDERS ---
rem Create necessary folders if they do not exist to prevent runtime errors
if not exist "%TMP_DIR%" mkdir "%TMP_DIR%"
if not exist "%HOME_DIR%" mkdir "%HOME_DIR%"
if not exist "%LOCALAPP_DIR%" mkdir "%LOCALAPP_DIR%"
if not exist "%ROAMING_DIR%" mkdir "%ROAMING_DIR%"

rem --- 3. OVERRIDE WINDOWS ENV ---
SET "TMP=%TMP_DIR%"
SET "TEMP=%TMP_DIR%"
SET "HOME=%HOME_DIR%"
SET "USERPROFILE=%HOME_DIR%"
SET "HOMEPATH=%HOME_DIR%"
SET "LOCALAPPDATA=%LOCALAPP_DIR%"
SET "APPDATA=%ROAMING_DIR%"

rem --- 4. PYTHON ISOLATION ---
rem Clear system Python variables to prevent conflicts
SET PYTHONHOME=
SET PYTHONPATH=
SET "PYTHONEXECUTABLE=%PYTHON_PATH%\python.exe"
SET "PYTHONWEXECUTABLE=%PYTHON_PATH%\pythonw.exe"
SET "PYTHON_EXECUTABLE=%PYTHON_PATH%\python.exe"
SET "PYTHON_BIN_PATH=%PYTHON_PATH%\python.exe"
SET "PYTHON_LIB_PATH=%PYTHON_PATH%\Lib\site-packages"

rem --- 5. PIP & PACKAGE MANAGEMENT ---
rem Move PIP cache directory to the local portable environment
SET "PIP_CACHE_DIR=%LOCALENV_DIR%\pip_cache"
SET "DISTUTILS_USE_SDK=1"

rem --- 6. UPDATE PATH ---
rem Add Python and Scripts to the beginning of the system PATH
SET "PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%DP0%\CMake\Bin;%PATH%"

rem --- 7. DISPLAY STATUS ---
echo ======================================================
echo  PYTHON PORTABLE ENVIRONMENT ACTIVATED
echo ======================================================
echo  Python Path: %PYTHON_PATH%
echo  Home Dir:    %HOME_DIR%
echo ------------------------------------------------------

rem --- 8. EXECUTION LOGIC ---
rem If files or folders are dragged and dropped, pass them to Python
if "%~1"=="" (
    echo [No arguments provided. Starting Python Interactive Shell...]
    "%PYTHON_PATH%\python.exe"
) else (
    echo [Processing input: %*]
    "%PYTHON_PATH%\python.exe" %*
)

pause