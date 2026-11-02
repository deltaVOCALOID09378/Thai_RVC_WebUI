@echo off
setlocal EnableExtensions EnableDelayedExpansion
title DELTA SYNTH - Core Assets & Model Health Verifier
color 0A

:: ==============================================================================
:: File Name   : Check-And-Download-Tools.bat
:: Version     : 1.0
:: Location    : Root Directory
:: Function    : Verify and automatically download essential RVC / UVR5 pretrained
::               models, HuBERT feature extractors, and RMVPE pitch models.
:: Credit      : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
::               Original by DELTA SYNTH
:: ==============================================================================

cd /d "%~dp0"

set "TARGET_DIR=%~1"
if "!TARGET_DIR!"=="" set "TARGET_DIR=%~dp0"
cd /d "!TARGET_DIR!"

echo ==============================================================================
echo        DELTA SYNTH - Thai RVC WebUI Core Assets ^& Tools Verifier
echo ==============================================================================
echo [INFO] Working Directory: %CD%
echo.

:: Detect Python Runtime
set "PYTHON_CMD="
if exist "%CD%\runtime\python.exe" (
    set "PYTHON_CMD=%CD%\runtime\python.exe"
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 set "PYTHON_CMD=python"
)

:: Detect Aria2
set "ARIA2_CMD="
if exist "%CD%\aria2\aria2c.exe" set "ARIA2_CMD=%CD%\aria2\aria2c.exe"
if exist "%CD%\aria2c.exe" set "ARIA2_CMD=%CD%\aria2c.exe"
if "!ARIA2_CMD!"=="" (
    where aria2c >nul 2>nul
    if !errorlevel! equ 0 set "ARIA2_CMD=aria2c"
)

:: Verify Directory Structure
for %%D in (
    "assets\hubert"
    "assets\rmvpe"
    "assets\pretrained"
    "assets\pretrained_v2"
    "assets\uvr5_weights"
    "assets\uvr5_weights\onnx_dereverb_By_FoxJoy"
    "assets\weights"
    "assets\indices"
    "logs\mute"
    "Output"
) do (
    if not exist "%%~D" (
        mkdir "%%~D" >nul 2>&1
    )
)

set "MISSING_COUNT=0"
set "HF_BASE=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

echo [1/4] Checking Core Acoustic Models (HuBERT ^& RMVPE)...
call :CHECK_AND_DOWNLOAD "!HF_BASE!/hubert_base.pt" "assets\hubert" "hubert_base.pt"
call :CHECK_AND_DOWNLOAD "!HF_BASE!/rmvpe.pt" "assets\rmvpe" "rmvpe.pt"
call :CHECK_AND_DOWNLOAD "!HF_BASE!/rmvpe.onnx" "assets\rmvpe" "rmvpe.onnx"

echo.
echo [2/4] Checking UVR5 Audio Separation Models...
call :CHECK_AND_DOWNLOAD "!HF_BASE!/uvr5_weights/HP2_all_vocals.pth" "assets\uvr5_weights" "HP2_all_vocals.pth"
call :CHECK_AND_DOWNLOAD "!HF_BASE!/uvr5_weights/HP3_all_vocals.pth" "assets\uvr5_weights" "HP3_all_vocals.pth"
call :CHECK_AND_DOWNLOAD "!HF_BASE!/uvr5_weights/HP5_only_main_vocal.pth" "assets\uvr5_weights" "HP5_only_main_vocal.pth"
call :CHECK_AND_DOWNLOAD "!HF_BASE!/uvr5_weights/VR-DeEchoAggressive.pth" "assets\uvr5_weights" "VR-DeEchoAggressive.pth"
call :CHECK_AND_DOWNLOAD "!HF_BASE!/uvr5_weights/VR-DeEchoDeReverb.pth" "assets\uvr5_weights" "VR-DeEchoDeReverb.pth"
call :CHECK_AND_DOWNLOAD "!HF_BASE!/uvr5_weights/VR-DeEchoNormal.pth" "assets\uvr5_weights" "VR-DeEchoNormal.pth"
call :CHECK_AND_DOWNLOAD "!HF_BASE!/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx" "assets\uvr5_weights\onnx_dereverb_By_FoxJoy" "vocals.onnx"

echo.
echo [3/4] Checking Base Training Models (Pretrained v1)...
for %%M in (
    D32k.pth D40k.pth D48k.pth G32k.pth G40k.pth G48k.pth
    f0D32k.pth f0D40k.pth f0D48k.pth f0G32k.pth f0G40k.pth f0G48k.pth
) do (
    call :CHECK_AND_DOWNLOAD "!HF_BASE!/pretrained/%%M" "assets\pretrained" "%%M"
)

echo.
echo [4/4] Checking Base Training Models (Pretrained v2)...
for %%M in (
    D32k.pth D40k.pth D48k.pth G32k.pth G40k.pth G48k.pth
    f0D32k.pth f0D40k.pth f0D48k.pth f0G32k.pth f0G40k.pth f0G48k.pth
) do (
    call :CHECK_AND_DOWNLOAD "!HF_BASE!/pretrained_v2/%%M" "assets\pretrained_v2" "%%M"
)

echo.
echo ==============================================================================
if !MISSING_COUNT! equ 0 (
    echo [STATUS] ALL REQUIRED ASSETS AND MODELS ARE VERIFIED ^& READY FOR USE.
) else (
    echo [STATUS] Download process completed. Newly acquired assets: !MISSING_COUNT!
)
echo ==============================================================================
echo.
pause
exit /b 0

:: ==============================================================================
:: SUBROUTINE: CHECK_AND_DOWNLOAD
:: %1 = URL, %2 = Destination Directory, %3 = Filename
:: ==============================================================================
:CHECK_AND_DOWNLOAD
set "DL_URL=%~1"
set "DL_DIR=%~2"
set "DL_FILE=%~3"
set "FULL_PATH=%DL_DIR%\%DL_FILE%"

if exist "%FULL_PATH%" (
    echo   [OK] Found: %DL_FILE%
    exit /b 0
)

set /a MISSING_COUNT+=1
echo   [DOWNLOAD] Missing: %DL_FILE% - Fetching payload...

if not exist "%DL_DIR%" mkdir "%DL_DIR%" >nul 2>&1

:: Method 1: Aria2
if not "!ARIA2_CMD!"=="" (
    "!ARIA2_CMD!" --console-log-level=warn -c -x 8 -s 8 -k 1M "%DL_URL%" -d "%DL_DIR%" -o "%DL_FILE%"
    if exist "%FULL_PATH%" (
        echo   [SUCCESS] Downloaded via aria2: %DL_FILE%
        exit /b 0
    )
)

:: Method 2: Python requests / urllib
if not "!PYTHON_CMD!"=="" (
    "!PYTHON_CMD!" -c "import urllib.request, sys; print('Downloading %DL_FILE%...'); urllib.request.urlretrieve(r'%DL_URL%', r'%FULL_PATH%')"
    if exist "%FULL_PATH%" (
        echo   [SUCCESS] Downloaded via Python: %DL_FILE%
        exit /b 0
    )
)

:: Method 3: Built-in Windows curl
where curl >nul 2>nul
if !errorlevel! equ 0 (
    curl -L -k --retry 3 -o "%FULL_PATH%" "%DL_URL%"
    if exist "%FULL_PATH%" (
        echo   [SUCCESS] Downloaded via curl: %DL_FILE%
        exit /b 0
    )
)

:: Method 4: PowerShell Invoke-WebRequest
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('%DL_URL%', '%FULL_PATH%')"
if exist "%FULL_PATH%" (
    echo   [SUCCESS] Downloaded via PowerShell: %DL_FILE%
    exit /b 0
)

echo   [ERROR] Failed to download %DL_FILE%. Please check your internet connection.
exit /b 1
