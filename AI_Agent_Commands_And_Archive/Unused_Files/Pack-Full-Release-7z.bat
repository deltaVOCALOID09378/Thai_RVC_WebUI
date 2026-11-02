@echo off
setlocal EnableExtensions EnableDelayedExpansion
title DELTA SYNTH - Full Production Release Packager (7-Zip)
color 0B

REM ==============================================================================
REM File Name   : Pack-Full-Release-7z.bat
REM Version     : 1.3
REM Location    : Root Directory
REM Function    : Package the complete Thai RVC WebUI system, runtime, toolsets,
REM               models, and skeleton structures into a self-contained .7z archive.
REM               Excludes personal data, junk files, temp caches, and git metadata.
REM Credit      : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
REM               Original by DELTA SYNTH
REM ==============================================================================

cd /d "%~dp0"

:PROCESS_LOOP
set "SOURCE_DIR=%~1"
if "!SOURCE_DIR!"=="" set "SOURCE_DIR=%~dp0"
if "!SOURCE_DIR:~-1!"=="\" set "SOURCE_DIR=!SOURCE_DIR:~0,-1!"

cls
echo ==============================================================================
echo        DELTA SYNTH - Thai RVC WebUI Full Release Packager (.7z)
echo ==============================================================================
echo [INFO] Source Path     : !SOURCE_DIR!
echo [INFO] Environment     : Standalone / Ready-To-Run Production Distribution
echo.

REM Detect 7-Zip Executable
set "SEVENZIP="
if exist "C:\Program Files\7-Zip\7z.exe" set "SEVENZIP=C:\Program Files\7-Zip\7z.exe"
if "!SEVENZIP!"=="" if exist "C:\Program Files (x86)\7-Zip\7z.exe" set "SEVENZIP=C:\Program Files (x86)\7-Zip\7z.exe"
if "!SEVENZIP!"=="" if exist "%ProgramFiles%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles%\7-Zip\7z.exe"
if "!SEVENZIP!"=="" if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles(x86)%\7-Zip\7z.exe"
if "!SEVENZIP!"=="" (
    where 7z >nul 2>nul
    if !errorlevel! equ 0 set "SEVENZIP=7z"
)

if "!SEVENZIP!"=="" (
    echo [ERROR] 7-Zip executable (7z.exe) not found on this machine!
    echo [ACTION] Please install 7-Zip from https://www.7-zip.org/
    echo.
    pause
    goto ADVANCE_QUEUE
)

REM Prepare Output Directory
set "EXPORT_ROOT=!SOURCE_DIR!\Export File\Full_Release"
if not exist "!EXPORT_ROOT!" mkdir "!EXPORT_ROOT!" >nul 2>&1

REM Ensure Skeleton Folders and Placeholders Exist
if not exist "!SOURCE_DIR!\Output" mkdir "!SOURCE_DIR!\Output" >nul 2>&1
if not exist "!SOURCE_DIR!\Output\.gitkeep" echo. > "!SOURCE_DIR!\Output\.gitkeep"
if not exist "!SOURCE_DIR!\data" mkdir "!SOURCE_DIR!\data" >nul 2>&1
if not exist "!SOURCE_DIR!\data\.gitkeep" echo. > "!SOURCE_DIR!\data\.gitkeep"
if not exist "!SOURCE_DIR!\assets\indices" mkdir "!SOURCE_DIR!\assets\indices" >nul 2>&1
if not exist "!SOURCE_DIR!\assets\indices\.gitkeep" echo. > "!SOURCE_DIR!\assets\indices\.gitkeep"
if not exist "!SOURCE_DIR!\assets\weights" mkdir "!SOURCE_DIR!\assets\weights" >nul 2>&1
if not exist "!SOURCE_DIR!\assets\weights\.gitkeep" echo. > "!SOURCE_DIR!\assets\weights\.gitkeep"
if not exist "!SOURCE_DIR!\logs\mute" mkdir "!SOURCE_DIR!\logs\mute" >nul 2>&1

REM Generate Timestamp for Archive Name
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set "DT=%%I"
if "!DT!"=="" (
    set "ARCHIVE_NAME=Thai_RVC_WebUI_v1_Full_Release.7z"
) else (
    set "YYYY=!DT:~0,4!"
    set "MM=!DT:~4,2!"
    set "DD=!DT:~6,2!"
    set "HH=!DT:~8,2!"
    set "NN=!DT:~10,2!"
    set "ARCHIVE_NAME=Thai_RVC_WebUI_v1_Full_Release_!YYYY!!MM!!DD!_!HH!!NN!.7z"
)

set "ARCHIVE_PATH=!EXPORT_ROOT!\!ARCHIVE_NAME!"
set "TEMP_EXCLUDE=!SOURCE_DIR!\full_release_exclude.tmp"

REM Remove any previously interrupted or existing archive with the exact same name
if exist "!ARCHIVE_PATH!" del /f /q "!ARCHIVE_PATH!" >nul 2>&1

echo [INFO] Destination Path: !ARCHIVE_PATH!
echo [INFO] Compression Mode: 7z LZMA2 (Normal/Optimized Speed)
echo [INFO] Generating Exclusion Filter...

> "!TEMP_EXCLUDE!" echo .git
>> "!TEMP_EXCLUDE!" echo .git\*
>> "!TEMP_EXCLUDE!" echo .github
>> "!TEMP_EXCLUDE!" echo .github\*
>> "!TEMP_EXCLUDE!" echo .claude
>> "!TEMP_EXCLUDE!" echo .claude\*
>> "!TEMP_EXCLUDE!" echo .cursor
>> "!TEMP_EXCLUDE!" echo .cursor\*
>> "!TEMP_EXCLUDE!" echo .hermes
>> "!TEMP_EXCLUDE!" echo .hermes\*
>> "!TEMP_EXCLUDE!" echo .gemini
>> "!TEMP_EXCLUDE!" echo .gemini\*
>> "!TEMP_EXCLUDE!" echo .system_generated
>> "!TEMP_EXCLUDE!" echo .system_generated\*
>> "!TEMP_EXCLUDE!" echo __pycache__
>> "!TEMP_EXCLUDE!" echo __pycache__\*
>> "!TEMP_EXCLUDE!" echo *.pyc
>> "!TEMP_EXCLUDE!" echo *.pyo
>> "!TEMP_EXCLUDE!" echo *.tmp
>> "!TEMP_EXCLUDE!" echo *.temp
>> "!TEMP_EXCLUDE!" echo *.log
>> "!TEMP_EXCLUDE!" echo desktop.ini
>> "!TEMP_EXCLUDE!" echo thumbs.db
>> "!TEMP_EXCLUDE!" echo .DS_Store
>> "!TEMP_EXCLUDE!" echo Export File
>> "!TEMP_EXCLUDE!" echo Export File\*
>> "!TEMP_EXCLUDE!" echo TEMP
>> "!TEMP_EXCLUDE!" echo TEMP\*
>> "!TEMP_EXCLUDE!" echo venv
>> "!TEMP_EXCLUDE!" echo venv\*
>> "!TEMP_EXCLUDE!" echo _local_env
>> "!TEMP_EXCLUDE!" echo _local_env\*
>> "!TEMP_EXCLUDE!" echo data\*
>> "!TEMP_EXCLUDE!" echo DiffSinger_Training\checkpoints\*
>> "!TEMP_EXCLUDE!" echo DiffSinger_Training\data\*
>> "!TEMP_EXCLUDE!" echo DiffSinger_Training\.git\*
>> "!TEMP_EXCLUDE!" echo logs\Andromeda
>> "!TEMP_EXCLUDE!" echo logs\Andromeda\*
>> "!TEMP_EXCLUDE!" echo logs\Ayanami HikaruAI_Soft
>> "!TEMP_EXCLUDE!" echo logs\Ayanami HikaruAI_Soft\*
>> "!TEMP_EXCLUDE!" echo logs\NameAI
>> "!TEMP_EXCLUDE!" echo logs\NameAI\*
>> "!TEMP_EXCLUDE!" echo logs\New_Day
>> "!TEMP_EXCLUDE!" echo logs\New_Day\*
>> "!TEMP_EXCLUDE!" echo logs\Reiya
>> "!TEMP_EXCLUDE!" echo logs\Reiya\*
>> "!TEMP_EXCLUDE!" echo logs\Mairu Maishi_Rock
>> "!TEMP_EXCLUDE!" echo logs\Mairu Maishi_Rock\*
>> "!TEMP_EXCLUDE!" echo full_release_exclude.tmp

echo [PACKING] Executing 7-Zip packaging process... Please wait...
echo.

pushd "!SOURCE_DIR!"
"!SEVENZIP!" a -t7z -mx=5 -ms=on -mmt=on "!ARCHIVE_PATH!" * "-xr@!TEMP_EXCLUDE!"
set "ZIP_EXIT=!errorlevel!"
popd

if exist "!TEMP_EXCLUDE!" del /f /q "!TEMP_EXCLUDE!" >nul 2>&1

echo.
if !ZIP_EXIT! equ 0 (
    echo ==============================================================================
    echo [SUCCESS] Full Production Release packaged successfully!
    echo [LOCATION] !ARCHIVE_PATH!
    for %%F in ("!ARCHIVE_PATH!") do echo [FILE SIZE] %%~zF bytes
    echo ==============================================================================
) else (
    echo ==============================================================================
    echo [ERROR] Packaging finished with exit code !ZIP_EXIT!. Please check log above.
    echo ==============================================================================
)

:ADVANCE_QUEUE
shift
if not "%~1"=="" (
    echo.
    echo [INFO] Next queue item detected: "%~1"
    echo [INFO] Press any key to process next item...
    pause > nul
    goto PROCESS_LOOP
)

echo.
echo [INFO] All operations completed. Press any key to exit.
pause > nul
exit /b 0
