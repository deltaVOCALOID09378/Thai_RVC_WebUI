@echo off
:: ========================================================================
:: File Name : Copy_Thai_RVC_WebUI_v1.bat
:: Version   : 1.0
:: Location  : Anywhere (e.g., Desktop or Root Directory)
:: Function  : Copy files from source to destination, maintain directory 
::             structure, overwrite existing files immediately without 
::             renaming, and add completely new files to the destination.
:: Credit    : Made And Checked By DELTA SYNTH & All Agentic AI
::             Original by User
:: ========================================================================

echo =====================================================
echo Starting the File Synchronization Process...
echo =====================================================
echo.

:: Define Source and Destination paths
set "SOURCE_DIR=K:\.shortcut-targets-by-id\1SAAyItVhRHAl5vNrOkokKPs0mO9KJmXE\Thai_RVC_WebUI_v1\infer"
set "DEST_DIR=A:\Thai_RVC_WebUI_v1\infer"

echo Source      : %SOURCE_DIR%
echo Destination : %DEST_DIR%
echo.
echo Processing files. Please wait...
echo.

:: Execute Robocopy with specific parameters to meet all conditions
:: /E   : Copies subdirectories, including empty ones (maintains structure).
:: /IS  : Includes the same files (forces an overwrite of existing identical files).
:: /IT  : Includes tweaked files (forces an overwrite of files with different attributes).
:: /R:1 : Specifies the number of retries on failed copies (1 retry).
:: /W:1 : Specifies the wait time between retries (1 second).

robocopy "%SOURCE_DIR%" "%DEST_DIR%" /E /IS /IT /R:1 /W:1

echo.
echo =====================================================
echo The copy process has been successfully completed.
echo All conditions regarding file replacement have been met.
echo =====================================================
echo.
pause