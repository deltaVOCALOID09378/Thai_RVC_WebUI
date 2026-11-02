@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 > nul
TITLE DELTA SYNTH - Program Packaging and Distribution Builder
color 0B

:: ==============================================================================
:: File Name   : Package_Program.bat
:: Version     : v2.1
:: Location    : Root Directory
:: Function    : Build Clean Distribution Archive for Thai RVC WebUI (/package_for_program)
:: Credit      : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
::               Original by DELTA SYNTH
:: ==============================================================================

:: Ensure execution from repository root
cd /d "%~dp0"

echo ============================================================
echo   DELTA SYNTH - Program Packaging ^& Distribution Builder
echo ============================================================
echo [INFO] Initializing Packaging Pipeline...
echo.

:: 1. Validate Core Production Assets
echo [STEP 1/4] Validating core production components...
set "CORE_MISSING=0"

if not exist "Program.py" (
    echo [ERROR] Core entry point 'Program.py' is missing!
    set "CORE_MISSING=1"
)
if not exist "Start-The-Program.bat" (
    echo [ERROR] Core launcher 'Start-The-Program.bat' is missing!
    set "CORE_MISSING=1"
)
if not exist "configs\config.py" (
    echo [ERROR] Config module 'configs\config.py' is missing!
    set "CORE_MISSING=1"
)
if not exist "AI_Agent_Commands_And_Archive\README.md" (
    echo [WARNING] Archive folder index is missing.
)

if "!CORE_MISSING!"=="1" (
    echo.
    echo [FATAL] Packaging aborted due to missing critical components.
    pause
    exit /b 1
)
echo [OK] Core components verified.
echo.

:: 2. Pre-Packaging Cleanup (Clean Temporary Files & Bytecode)
echo [STEP 2/4] Sanitizing workspace temporary caches...
if exist "TEMP" (
    echo [INFO] Clearing TEMP directory...
    del /f /q /s "TEMP\*" > nul 2>&1
)

echo [OK] Temporary caches sanitized.
echo.

:: 3. Generate Timestamped Archive Name & Output Directory
set "TIMESTAMP="
for /f %%A in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss" 2^>nul') do (
    set "TIMESTAMP=%%A"
)
if not defined TIMESTAMP (
    set "TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%"
)

set "ARCHIVE_NAME=Thai_RVC_WebUI_v1_Distribution_!TIMESTAMP!.zip"
set "OUTPUT_DIR=%~dp0Published"

if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"
set "TARGET_ZIP=!OUTPUT_DIR!\%ARCHIVE_NAME%"

echo [STEP 3/4] Packaging clean distribution archive...
echo [INFO] Destination: !TARGET_ZIP!

:: Detect 7-Zip Executable
set "SEVENZIP="
if exist "C:\Program Files\7-Zip\7z.exe" set "SEVENZIP=C:\Program Files\7-Zip\7z.exe"
if not defined SEVENZIP if exist "C:\Program Files (x86)\7-Zip\7z.exe" set "SEVENZIP=C:\Program Files (x86)\7-Zip\7z.exe"
if not defined SEVENZIP (
    where 7z >nul 2>nul
    if !errorlevel! equ 0 set "SEVENZIP=7z"
)

if defined SEVENZIP goto :run_sevenzip
goto :run_powershell

:run_sevenzip
echo [INFO] Compression Engine: 7-Zip Executable
echo [INFO] Compressing distribution files...
setlocal DisableDelayedExpansion
"%SEVENZIP%" a -tzip "%TARGET_ZIP%" "%~dp0*" -xr!runtime -xr!venv -xr!_local_env -xr!.git -xr!.github -xr!.claude -xr!.cursor -xr!.hermes -xr!.gemini -xr!.tmp.driveupload -xr!Published -xr!TEMP -xr!logs -xr!weights -xr!pretrained -xr!pretrained_v2 -xr!uvr5_weights -xr!rmvpe -xr!DiffSinger_Training -xr!"Export File" -xr!"For OpenUtau VB Making" -xr!RVCRealtimeVST -xr!__pycache__ -xr!*.pyc -xr!*.zip -xr!*.tmp -xr!*.log -bso0 -bsp0
set "BUILD_STATUS=%errorlevel%"
endlocal & set "BUILD_STATUS=%BUILD_STATUS%"

if !BUILD_STATUS! neq 0 (
    echo [WARNING] 7-Zip encountered error !BUILD_STATUS!. Switching to PowerShell fallback...
    goto :run_powershell
)
goto :verify_archive

:run_powershell
echo [INFO] Compression Engine: PowerShell Engine (Clean Stream Mode)
echo [INFO] Compressing distribution files...
set "FALLBACK_SCRIPT=%~dp0AI_Agent_Commands_And_Archive\Command_Runners\build_package_fallback.ps1"
if not exist "!FALLBACK_SCRIPT!" (
    echo [ERROR] Fallback packaging script missing: !FALLBACK_SCRIPT!
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "!FALLBACK_SCRIPT!" -SourceDir "%~dp0." -TargetZip "!TARGET_ZIP!"
set "BUILD_STATUS=!errorlevel!"
if !BUILD_STATUS! neq 0 (
    echo.
    echo [ERROR] Packaging failed with error code !BUILD_STATUS!.
    pause
    exit /b !BUILD_STATUS!
)
goto :verify_archive

:verify_archive
echo.
echo ============================================================
echo [STEP 4/4] Verification ^& Delivery Status
echo ============================================================
if not exist "!TARGET_ZIP!" (
    echo [ERROR] Generated package file not found!
    pause
    exit /b 1
)
for %%A in ("!TARGET_ZIP!") do (
    set /a "SIZE_MB=%%~zA / 1048576"
    echo [SUCCESS] Package created successfully: %%~nxA
    echo [INFO] Package Path: %%~fA
    echo [INFO] Package Size: !SIZE_MB! MB (%%~zA bytes)
)
echo ============================================================
echo [STATUS] Distribution build process completed successfully.
echo ============================================================
if "%~1"=="/silent" exit /b 0
pause
exit /b 0
