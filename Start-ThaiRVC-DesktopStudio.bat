@echo off
title Thai RVC Studio (C# Desktop Edition) - DELTA SYNTH
:: ==============================================================================
:: Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
:: Original by Patiphat Wongyai (Delta) & RVC Project
:: Description: Desktop C# UI Launcher with Drag & Drop Support
:: ==============================================================================

cd /d "%~dp0"

echo ===================================================
echo Launching Thai RVC Studio (C# Desktop Application)...
echo ===================================================

set "APP_EXE=ThaiRVCStudio\bin\Debug\net10.0-windows\ThaiRVCStudio.exe"

if exist "%APP_EXE%" (
    if "%~1"=="" (
        start "" "%APP_EXE%"
    ) else (
        echo Passing audio target: "%~1"
        start "" "%APP_EXE%" "%~1"
    )
) else (
    echo Binary not found. Building and running with dotnet SDK...
    dotnet run --project ThaiRVCStudio\ThaiRVCStudio.csproj
)

echo Application launched.
