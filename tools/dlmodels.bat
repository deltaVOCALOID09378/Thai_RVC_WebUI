@echo off
:: ============================================================================
:: Made And Checked By DELTA SYNTH & All Agentic AI
:: Original by 
:: Version: 2.1 (Full Package Integration)
:: ============================================================================

chcp 65001 > nul
setlocal enabledelayedexpansion

:: [MAINTENANCE OPTIONS]
:: Set FORCE_REDOWNLOAD=1 to bypass 'exist' checks and force re-download all files.
set "FORCE_REDOWNLOAD=0"

:: [DYNAMIC TARGET DIRECTORY]
set "TARGET_DIR=%~1"
if "%TARGET_DIR%"=="" set "TARGET_DIR=%~dp0"
cd /d "%TARGET_DIR%"

echo [INFO] Working directory is: %cd%
echo [INFO] Initializing Aria2 requirement check...
echo.

:: [ARIA2 DETECTION LOGIC]
set "ARIA2_CMD="
if exist "aria2\aria2c.exe" set "ARIA2_CMD=aria2\aria2c.exe"
if exist "aria2c.exe" set "ARIA2_CMD=aria2c.exe"
if "%ARIA2_CMD%"=="" (
    where aria2c >nul 2>nul
    if !errorlevel! equ 0 set "ARIA2_CMD=aria2c"
)

if "%ARIA2_CMD%"=="" (
    echo [ERROR] aria2 engine not found.
    echo [ACTION] Please download aria2 from: https://github.com/aria2/aria2/releases/tag/release-1.36.0
    echo [ACTION] Unzip it and put 'aria2c.exe' in this directory or the 'aria2' folder.
    timeout /t 5
    start https://github.com/aria2/aria2/releases/tag/release-1.36.0
    echo.
    pause
    exit /b 1
) else (
    echo [SUCCESS] aria2 engine verified.
    echo.
)

:: [DIRECTORY STRUCTURE VERIFICATION]
echo [INFO] Verifying directory structures...
for %%D in (
    "assets\pretrained"
    "assets\pretrained_v2"
    "assets\uvr5_weights"
    "assets\uvr5_weights\onnx_dereverb_By_FoxJoy"
    "assets\hubert"
    "assets\rmvpe"
) do (
    if not exist "%%~D" (
        echo [ACTION] Creating missing directory: %%~D
        mkdir "%%~D"
    ) else (
        echo [OK] Directory verified: %%~D
    )
)
echo [SUCCESS] Directory structure check completed.
echo.

:: [ASSETS SYNCHRONIZATION PROCESS]
echo [INFO] Starting required files verification...
echo.

:: =======================================================
:: 1. BASE MODELS V1 (Pretrained)
:: =======================================================
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D32k.pth" "assets\pretrained" "D32k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D40k.pth" "assets\pretrained" "D40k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D48k.pth" "assets\pretrained" "D48k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G32k.pth" "assets\pretrained" "G32k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G40k.pth" "assets\pretrained" "G40k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G48k.pth" "assets\pretrained" "G48k.pth"

call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D32k.pth" "assets\pretrained" "f0D32k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D40k.pth" "assets\pretrained" "f0D40k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D48k.pth" "assets\pretrained" "f0D48k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G32k.pth" "assets\pretrained" "f0G32k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G40k.pth" "assets\pretrained" "f0G40k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G48k.pth" "assets\pretrained" "f0G48k.pth"

:: =======================================================
:: 2. BASE MODELS V2 (Pretrained_v2) - FULL SET
:: =======================================================
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D32k.pth" "assets\pretrained_v2" "D32k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth" "assets\pretrained_v2" "D40k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D48k.pth" "assets\pretrained_v2" "D48k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G32k.pth" "assets\pretrained_v2" "G32k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth" "assets\pretrained_v2" "G40k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G48k.pth" "assets\pretrained_v2" "G48k.pth"

call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D32k.pth" "assets\pretrained_v2" "f0D32k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth" "assets\pretrained_v2" "f0D40k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth" "assets\pretrained_v2" "f0D48k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G32k.pth" "assets\pretrained_v2" "f0G32k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth" "assets\pretrained_v2" "f0G40k.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth" "assets\pretrained_v2" "f0G48k.pth"

:: =======================================================
:: 3. UVR5 MODELS (Vocal Separation & Reverb)
:: =======================================================
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2_all_vocals.pth" "assets\uvr5_weights" "HP2_all_vocals.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP3_all_vocals.pth" "assets\uvr5_weights" "HP3_all_vocals.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5_only_main_vocal.pth" "assets\uvr5_weights" "HP5_only_main_vocal.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoAggressive.pth" "assets\uvr5_weights" "VR-DeEchoAggressive.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoDeReverb.pth" "assets\uvr5_weights" "VR-DeEchoDeReverb.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoNormal.pth" "assets\uvr5_weights" "VR-DeEchoNormal.pth"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx" "assets\uvr5_weights\onnx_dereverb_By_FoxJoy" "vocals.onnx"

:: =======================================================
:: 4. CORE EXTRACTION MODELS (HuBERT & RMVPE)
:: =======================================================
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt" "assets\hubert" "hubert_base.pt"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt" "assets\rmvpe" "rmvpe.pt"
call :DOWNLOAD_MODEL "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx" "assets\rmvpe" "rmvpe.onnx"

echo ============================================================================
echo [SUCCESS] Environment files check and synchronization complete.
echo ============================================================================
pause
exit /b 0

:: ============================================================================
:: [SUBROUTINE] DOWNLOAD_MODEL
:: Handles file verification and dynamic downloading via Aria2
:: ============================================================================
:DOWNLOAD_MODEL
set "DL_URL=%~1"
set "DL_DIR=%~2"
set "DL_FILE=%~3"

if "%FORCE_REDOWNLOAD%"=="1" (
    if exist "%DL_DIR%\%DL_FILE%" del /f /q "%DL_DIR%\%DL_FILE%"
)

if exist "%DL_DIR%\%DL_FILE%" (
    echo [OK] Validated: %DL_FILE% in %DL_DIR%
) else (
    echo [WARN] Missing: %DL_FILE%. Initializing payload...
    "%ARIA2_CMD%" --console-log-level=error -c -x 16 -s 16 -k 1M "%DL_URL%" -d "%DL_DIR%" -o "%DL_FILE%"
    
    if exist "%DL_DIR%\%DL_FILE%" (
        echo [SUCCESS] Acquired: %DL_FILE%
    ) else (
        echo [ERROR] Fetch Failed: %DL_FILE%. Please verify network stability.
    )
)
exit /b