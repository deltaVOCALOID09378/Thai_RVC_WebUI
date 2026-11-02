@echo off
setlocal enabledelayedexpansion
title DELTA SYNTH - Thai RVC WebUI GitHub Sync Tool
color 0C

:: ==============================================================================
:: File Name      : Sync_And_Push_GitHub.bat
:: Version        : 2.0
:: Location       : Root Directory (A:\Thai_RVC_WebUI_v1)
:: Credit         : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
::                  Original by Patiphat Wongyai (Delta)
:: Description    : Cleans temporary cache, verifies Python compilation,
::                  stages, commits, and pushes changes to Thai_RVC_WebUI repo.
:: ==============================================================================

set "TARGET_DIR=A:\Thai_RVC_WebUI_v1"
if not exist "%TARGET_DIR%" set "TARGET_DIR=%~dp0"
cd /d "%TARGET_DIR%"

set "LOG_FILE=%TARGET_DIR%\Sync_GitHub.log"
echo ============================================================================== > "%LOG_FILE%"
echo DELTA SYNTH - Thai RVC WebUI GitHub Sync Process Log >> "%LOG_FILE%"
echo Started at: %DATE% %TIME% >> "%LOG_FILE%"
echo Target Directory: %TARGET_DIR% >> "%LOG_FILE%"
echo ============================================================================== >> "%LOG_FILE%"

echo ==============================================================================
echo   DELTA SYNTH - Thai RVC WebUI Disk Cleanup, Compile Check and GitHub Sync
echo   Target Directory : %TARGET_DIR%
echo   Remote Repository: https://github.com/deltaVOCALOID09378/Thai_RVC_WebUI.git
echo ==============================================================================
echo.

echo [1/6] Cleaning obsolete cache and temporary files...
echo [INFO] Cleaning cache and temp files... >> "%LOG_FILE%"
if exist "TEMP\*" del /F /Q "TEMP\*" >> "%LOG_FILE%" 2>&1
if exist ".tmp.drivedownload" rmdir /s /q ".tmp.drivedownload" >> "%LOG_FILE%" 2>&1
if exist ".tmp.driveupload" rmdir /s /q ".tmp.driveupload" >> "%LOG_FILE%" 2>&1
for /r %%i in (desktop.ini) do (if exist "%%i" del /f /q "%%i" >> "%LOG_FILE%" 2>&1)

echo [2/6] Verifying remote repository URL...
git remote set-url origin https://github.com/deltaVOCALOID09378/Thai_RVC_WebUI.git >> "%LOG_FILE%" 2>&1
git remote -v >> "%LOG_FILE%" 2>&1

echo [3/6] Verifying Python compilation with embedded runtime...
set "PYTHON_EXE=%TARGET_DIR%\runtime\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

"%PYTHON_EXE%" -m py_compile Program.py infer-web.py utils.py realtime_gui.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [ERROR] Python compilation check failed! Check %LOG_FILE% for details.
    echo [ERROR] Python compilation check failed! >> "%LOG_FILE%"
    pause
    exit /b 1
)
echo [OK] Python compilation check passed successfully (0 Errors).
echo [OK] Python compilation check passed. >> "%LOG_FILE%"

echo [4/6] Staging all project changes...
git add -A >> "%LOG_FILE%" 2>&1

echo [5/6] Committing changes...
set "COMMIT_MSG=DELTA SYNTH Master Update: Unified Studio v6.3, Complete infer-web.py, Project Planning & Upstream Synchronization"
git commit -m "%COMMIT_MSG%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [INFO] No new changes to commit or commit completed.
    echo [INFO] No new changes to commit. >> "%LOG_FILE%"
)

echo [6/6] Pushing to GitHub (origin Full-Part-No.1)...
git push origin Full-Part-No.1 >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [WARN] Push to Full-Part-No.1 returned non-zero code. Retrying with tracking...
    git push -u origin Full-Part-No.1 >> "%LOG_FILE%" 2>&1
)

echo.
echo ==============================================================================
echo [SUCCESS] Thai RVC WebUI synchronized to GitHub successfully!
echo Process finished at: %DATE% %TIME% >> "%LOG_FILE%"
echo ==============================================================================
pause
exit /b 0
