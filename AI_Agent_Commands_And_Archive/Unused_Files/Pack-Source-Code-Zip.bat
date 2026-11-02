@echo off
setlocal EnableExtensions EnableDelayedExpansion
title DELTA SYNTH - Source Code Distribution Packager (.zip)
color 0A

REM ==============================================================================
REM File Name   : Pack-Source-Code-Zip.bat
REM Version     : 1.3
REM Location    : Root Directory
REM Function    : Package clean, lightweight source code, scripts, configurations,
REM               and documentation into a standard .zip archive.
REM               Excludes runtime environments, heavy pretrained models, and junk.
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
echo       DELTA SYNTH - Thai RVC WebUI Clean Source Packager (.zip)
echo ==============================================================================
echo [INFO] Source Path     : !SOURCE_DIR!
echo [INFO] Package Mode    : Pure Source Code ^& Developer Utilities
echo.

REM Detect 7-Zip or PowerShell Fallback
set "SEVENZIP="
if exist "C:\Program Files\7-Zip\7z.exe" set "SEVENZIP=C:\Program Files\7-Zip\7z.exe"
if "!SEVENZIP!"=="" if exist "C:\Program Files (x86)\7-Zip\7z.exe" set "SEVENZIP=C:\Program Files (x86)\7-Zip\7z.exe"
if "!SEVENZIP!"=="" if exist "%ProgramFiles%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles%\7-Zip\7z.exe"
if "!SEVENZIP!"=="" if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles(x86)%\7-Zip\7z.exe"
if "!SEVENZIP!"=="" (
    where 7z >nul 2>nul
    if !errorlevel! equ 0 set "SEVENZIP=7z"
)

REM Prepare Output Directory
set "EXPORT_ROOT=!SOURCE_DIR!\Export File\Source_Release"
if not exist "!EXPORT_ROOT!" mkdir "!EXPORT_ROOT!" >nul 2>&1

REM Ensure Skeleton Placeholders Exist
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
    set "ARCHIVE_NAME=Thai_RVC_WebUI_v1_Source_Code.zip"
) else (
    set "YYYY=!DT:~0,4!"
    set "MM=!DT:~4,2!"
    set "DD=!DT:~6,2!"
    set "HH=!DT:~8,2!"
    set "NN=!DT:~10,2!"
    set "ARCHIVE_NAME=Thai_RVC_WebUI_v1_Source_Code_!YYYY!!MM!!DD!_!HH!!NN!.zip"
)

set "ARCHIVE_PATH=!EXPORT_ROOT!\!ARCHIVE_NAME!"
set "TEMP_EXCLUDE=!SOURCE_DIR!\source_release_exclude.tmp"

REM Remove any previously interrupted or existing archive with the exact same name
if exist "!ARCHIVE_PATH!" del /f /q "!ARCHIVE_PATH!" >nul 2>&1

echo [INFO] Destination Path: !ARCHIVE_PATH!

if not "!SEVENZIP!"=="" (
    echo [INFO] Engine Detected : 7-Zip Executable
    echo [INFO] Excluding Runtime, Large Weights, Cache, Datasets, and Experiments...
    echo.

    > "!TEMP_EXCLUDE!" echo runtime
    >> "!TEMP_EXCLUDE!" echo runtime\*
    >> "!TEMP_EXCLUDE!" echo venv
    >> "!TEMP_EXCLUDE!" echo venv\*
    >> "!TEMP_EXCLUDE!" echo _local_env
    >> "!TEMP_EXCLUDE!" echo _local_env\*
    >> "!TEMP_EXCLUDE!" echo data\*
    >> "!TEMP_EXCLUDE!" echo DiffSinger_Training\checkpoints\*
    >> "!TEMP_EXCLUDE!" echo DiffSinger_Training\data\*
    >> "!TEMP_EXCLUDE!" echo DiffSinger_Training\.git\*
    >> "!TEMP_EXCLUDE!" echo assets\hubert\*.pt
    >> "!TEMP_EXCLUDE!" echo assets\rmvpe\*.pt
    >> "!TEMP_EXCLUDE!" echo assets\rmvpe\*.onnx
    >> "!TEMP_EXCLUDE!" echo assets\pretrained\*.pth
    >> "!TEMP_EXCLUDE!" echo assets\pretrained_v2\*.pth
    >> "!TEMP_EXCLUDE!" echo assets\uvr5_weights\*.pth
    >> "!TEMP_EXCLUDE!" echo assets\uvr5_weights\*.onnx
    >> "!TEMP_EXCLUDE!" echo assets\weights\*.pth
    >> "!TEMP_EXCLUDE!" echo .git
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
    >> "!TEMP_EXCLUDE!" echo Output\*
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
    >> "!TEMP_EXCLUDE!" echo source_release_exclude.tmp

    pushd "!SOURCE_DIR!"
    "!SEVENZIP!" a -tzip -mx=5 -mmt=on "!ARCHIVE_PATH!" * "-xr@!TEMP_EXCLUDE!"
    set "ZIP_EXIT=!errorlevel!"
    popd

    if exist "!TEMP_EXCLUDE!" del /f /q "!TEMP_EXCLUDE!" >nul 2>&1
) else (
    echo [INFO] 7-Zip not detected. Utilizing PowerShell Zip Engine...
    echo.
    powershell -NoProfile -Command ^
        "$src = '!SOURCE_DIR!';" ^
        "$dest = '!ARCHIVE_PATH!';" ^
        "$excludeNames = @('runtime', 'venv', '_local_env', 'TEMP', 'Export File', 'data', '.git', '.github', '.claude', '.cursor', '__pycache__', '.system_generated', '.gemini');" ^
        "$items = Get-ChildItem -Path $src -Exclude $excludeNames | Where-Object { $_.Name -notin $excludeNames };" ^
        "Compress-Archive -Path $items.FullName -DestinationPath $dest -Force;"
    set "ZIP_EXIT=!errorlevel!"
)

echo.
if !ZIP_EXIT! equ 0 (
    echo ==============================================================================
    echo [SUCCESS] Source Code archive created successfully!
    echo [LOCATION] !ARCHIVE_PATH!
    for %%F in ("!ARCHIVE_PATH!") do echo [FILE SIZE] %%~zF bytes
    echo ==============================================================================
) else (
    echo ==============================================================================
    echo [ERROR] Packaging failed with exit code !ZIP_EXIT!.
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
