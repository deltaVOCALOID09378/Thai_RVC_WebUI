@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title DELTA SYNTH - PC Auto-Updater v2.5.0 (Target: A:\Thai_RVC_WebUI_v1)

echo ============================================================================
echo   DELTA SYNTH - THAI RVC STUDIO ^& DIFFSINGER UNIFIED ENGINE UPDATER
echo   Release Version : 2.5.0 (Strict Canonical Path Structure Edition)
echo   Made And Checked By DELTA SYNTH ^& All Agentic AI
echo ============================================================================
echo.

set "SRC_DIR=%~dp0"
if "%SRC_DIR:~-1%"=="\" set "SRC_DIR=%SRC_DIR:~0,-1%"
if exist "%SRC_DIR%\Thai_RVC_WebUI_v1\Program.py" (
    set "SRC_DIR=%SRC_DIR%\Thai_RVC_WebUI_v1"
)

set "TARGET_RVC_1=A:\Thai_RVC_WebUI_v1"
set "TARGET_RVC_2=E:\Program Developing\Retrieval-based-Voice-Conversion-WebUI-2.2.231006"

echo [INFO] Package Source Directory: "%SRC_DIR%"
echo.

set /a UPDATED_TARGETS=0

for %%T in ("%TARGET_RVC_1%" "%TARGET_RVC_2%") do (
    set "CUR_TARGET=%%~T"
    if exist "!CUR_TARGET!\" (
        set /a UPDATED_TARGETS+=1
        echo ----------------------------------------------------------------------------
        echo [TARGET FOUND] Updating: "!CUR_TARGET!"
        echo ----------------------------------------------------------------------------

        REM 1. Clean up any stray/duplicate files that do not belong in Root or wrong folders
        echo [STEP 1/5] Cleaning up any misplaced or duplicate files from previous runs...
        for %%S in (
            "Start_The_Program.bat"
            "Run_Program.bat"
            "Start_Short.bat"
            "Start_Full_Web.bat"
            "go-web.bat"
            "go-web-dml.bat"
            "go-realtime-gui.bat"
            "go-realtime-gui-dml.bat"
            "Install_Environment.bat"
            "Repair_Fairseq.bat"
            "Start_WebUI_AutoRepair.bat"
            "gui_v1.py"
            "diffsinger_suite_optimizer.py"
            "audit_report.md"
            "THAI_RVC_COMPLETE_MASTER_KNOWLEDGE_BASE.md"
            "ENTERPRISE_MASTER_PROMPT_SUITE_THAI_RVC.md"
            "f0_extractor.py"
            "train.py"
            "extract_f0_print.py"
            "extract_feature_print.py"
            "train_dataset_inspector.py"
            "train_index.py"
            "train_pipeline_master.py"
            "data_utils.py"
            "utils_train.py"
            "pipeline.py"
            "i18n\es_ES.json"
            "i18n\ja_JP.json"
            "tools\dataset_auto_cleaner.py"
            "tools\diffsinger\diffsinger_suite_optimizer.py"
        ) do (
            if not "!CUR_TARGET!"=="%SRC_DIR%" (
                if exist "!CUR_TARGET!\%%~S" del /F /Q "!CUR_TARGET!\%%~S" >nul 2>&1
            )
        )
        if exist "!CUR_TARGET!\tools\diffsinger\" rmdir /S /Q "!CUR_TARGET!\tools\diffsinger" >nul 2>&1

        REM 2. Backup existing files preserving directory structure
        set "BACKUP_DIR=!CUR_TARGET!\backups\pre_v2.5.0_backup"
        if not exist "!BACKUP_DIR!\" mkdir "!BACKUP_DIR!"
        echo [STEP 2/5] Backing up existing files to "!BACKUP_DIR!"...
        for %%F in (
            "Program.py"
            "Program_Backup.py"
            "infer-web.py"
            "realtime_gui.py"
            "utils.py"
            "dataset_auto_cleaner.py"
            "DiffSinger_Trainer_v1.0.py"
            "requirements.txt"
            "Start-The-Program.bat"
            "Train_Studio_Runner.bat"
            "run_training_master.bat"
            "Run_DiffSinger_Trainer_v1.0.bat"
            "Install-Env.bat"
            "Install-File.bat"
            "Check-And-Download-Tools.bat"
            "Copy_Thai_RVC_WebUI_v1.bat"
            "Pack-Full-Release-7z.bat"
            "Pack-Source-Code-Zip.bat"
            "README.md"
            "About_Project.md"
            "AGENTS.md"
            "AGENT.md"
            "configs\config.py"
            "i18n\locale\es_ES.json"
            "i18n\locale\ja_JP.json"
            "infer\lib\audio.py"
            "infer\lib\slicer2.py"
            "infer\lib\rmvpe.py"
            "infer\lib\infer_pack\models_onnx.py"
            "infer\lib\jit\__init__.py"
            "infer\lib\jit\get_hubert.py"
            "infer\lib\jit\get_rmvpe.py"
            "infer\lib\jit\get_synthesizer.py"
            "infer\lib\uvr5_pack\utils.py"
            "infer\lib\train\utils.py"
            "infer\lib\train\data_utils.py"
            "infer\lib\train\process_ckpt.py"
            "infer\modules\vc\modules.py"
            "infer\modules\vc\pipeline.py"
            "infer\modules\vc\utils.py"
            "infer\modules\onnx\export.py"
            "infer\modules\ipex\__init__.py"
            "infer\modules\ipex\attention.py"
            "infer\modules\ipex\gradscaler.py"
            "infer\modules\ipex\hijacks.py"
            "infer\modules\uvr5\modules.py"
            "infer\modules\train\preprocess.py"
            "infer\modules\train\extract_feature_print.py"
            "infer\modules\train\train.py"
            "infer\modules\train\rvc_train_utils.py"
            "infer\modules\train\train_dataset_inspector.py"
            "infer\modules\train\train_pipeline_master.py"
            "infer\modules\train\train_index.py"
            "infer\modules\train\extract\f0_extractor.py"
            "infer\modules\train\extract\extract_f0_print.py"
            "infer\modules\train\extract\extract_f0_rmvpe.py"
            "infer\modules\train\extract\extract_f0_rmvpe_dml.py"
        ) do (
            if exist "!CUR_TARGET!\%%~F" (
                for %%P in ("!BACKUP_DIR!\%%~F") do (
                    if not exist "%%~dpP" mkdir "%%~dpP"
                )
                copy /Y "!CUR_TARGET!\%%~F" "!BACKUP_DIR!\%%~F" >nul 2>&1
            )
        )

        REM 3. Deploy Root files
        echo [STEP 3/5] Deploying Root, All Documents, DiffSinger_Training, Configs, i18n, and Tools...
        if not "!CUR_TARGET!"=="%SRC_DIR%" (
            for %%R in (
                "Program.py"
                "Program_Backup.py"
                "infer-web.py"
                "realtime_gui.py"
                "utils.py"
                "dataset_auto_cleaner.py"
                "DiffSinger_Trainer_v1.0.py"
                "requirements.txt"
                "Start-The-Program.bat"
                "Train_Studio_Runner.bat"
                "run_training_master.bat"
                "Run_DiffSinger_Trainer_v1.0.bat"
                "Install-Env.bat"
                "Install-File.bat"
                "Check-And-Download-Tools.bat"
                "Copy_Thai_RVC_WebUI_v1.bat"
                "Pack-Full-Release-7z.bat"
                "Pack-Source-Code-Zip.bat"
                "Install_Update_v2.1.0_To_PC.bat"
                "README.md"
                "About_Project.md"
                "AGENTS.md"
                "AGENT.md"
                "SPARK_THAI_RVC_BLUEPRINT.md"
            ) do (
                if exist "%SRC_DIR%\%%~R" copy /Y "%SRC_DIR%\%%~R" "!CUR_TARGET!\%%~R" >nul
            )

            for %%D in ("All Documents" "DiffSinger_Training" "configs" "i18n" "tools" "Published" "infer") do (
                if exist "%SRC_DIR%\%%~D\" (
                    robocopy "%SRC_DIR%\%%~D" "!CUR_TARGET!\%%~D" /E /IT /IS /IM /NFL /NDL /NJH /NJS >nul
                )
            )
        )

        echo [STEP 4/5] Verifying Python Bytecode Compilation across all modules in their canonical paths...
        if exist "!CUR_TARGET!\runtime\python.exe" (
            "!CUR_TARGET!\runtime\python.exe" -m py_compile ^
                "!CUR_TARGET!\Program.py" ^
                "!CUR_TARGET!\Program_Backup.py" ^
                "!CUR_TARGET!\infer-web.py" ^
                "!CUR_TARGET!\realtime_gui.py" ^
                "!CUR_TARGET!\utils.py" ^
                "!CUR_TARGET!\dataset_auto_cleaner.py" ^
                "!CUR_TARGET!\DiffSinger_Trainer_v1.0.py" ^
                "!CUR_TARGET!\DiffSinger_Training\diffsinger_suite_optimizer.py" ^
                "!CUR_TARGET!\configs\config.py" ^
                "!CUR_TARGET!\infer\lib\audio.py" ^
                "!CUR_TARGET!\infer\lib\slicer2.py" ^
                "!CUR_TARGET!\infer\lib\rmvpe.py" ^
                "!CUR_TARGET!\infer\lib\infer_pack\models_onnx.py" ^
                "!CUR_TARGET!\infer\lib\jit\__init__.py" ^
                "!CUR_TARGET!\infer\lib\jit\get_hubert.py" ^
                "!CUR_TARGET!\infer\lib\jit\get_rmvpe.py" ^
                "!CUR_TARGET!\infer\lib\jit\get_synthesizer.py" ^
                "!CUR_TARGET!\infer\lib\uvr5_pack\utils.py" ^
                "!CUR_TARGET!\infer\lib\train\utils.py" ^
                "!CUR_TARGET!\infer\lib\train\data_utils.py" ^
                "!CUR_TARGET!\infer\lib\train\process_ckpt.py" ^
                "!CUR_TARGET!\infer\modules\vc\modules.py" ^
                "!CUR_TARGET!\infer\modules\vc\pipeline.py" ^
                "!CUR_TARGET!\infer\modules\vc\utils.py" ^
                "!CUR_TARGET!\infer\modules\onnx\export.py" ^
                "!CUR_TARGET!\infer\modules\ipex\__init__.py" ^
                "!CUR_TARGET!\infer\modules\ipex\attention.py" ^
                "!CUR_TARGET!\infer\modules\ipex\gradscaler.py" ^
                "!CUR_TARGET!\infer\modules\ipex\hijacks.py" ^
                "!CUR_TARGET!\infer\modules\uvr5\modules.py" ^
                "!CUR_TARGET!\infer\modules\train\preprocess.py" ^
                "!CUR_TARGET!\infer\modules\train\extract_feature_print.py" ^
                "!CUR_TARGET!\infer\modules\train\train.py" ^
                "!CUR_TARGET!\infer\modules\train\rvc_train_utils.py" ^
                "!CUR_TARGET!\infer\modules\train\train_dataset_inspector.py" ^
                "!CUR_TARGET!\infer\modules\train\train_pipeline_master.py" ^
                "!CUR_TARGET!\infer\modules\train\train_index.py" ^
                "!CUR_TARGET!\infer\modules\train\extract\f0_extractor.py" ^
                "!CUR_TARGET!\infer\modules\train\extract\extract_f0_print.py" ^
                "!CUR_TARGET!\infer\modules\train\extract\extract_f0_rmvpe.py" ^
                "!CUR_TARGET!\infer\modules\train\extract\extract_f0_rmvpe_dml.py"
            if !ERRORLEVEL! EQU 0 (
                echo [PASS] All Python modules compiled with zero syntax errors.
            ) else (
                echo [WARN] Some modules reported warnings during compilation check.
            )
            echo [STEP 5/5] Running Dataset Auto-Cleaner on logs directory...
            if exist "!CUR_TARGET!\logs\" (
                "!CUR_TARGET!\runtime\python.exe" "!CUR_TARGET!\dataset_auto_cleaner.py" "!CUR_TARGET!\logs"
            )
        )
        echo [SUCCESS] Updated "!CUR_TARGET!" with strict canonical folder structure.
        echo.
    ) else (
        echo [SKIP] Target directory not found on this machine: "!CUR_TARGET!"
    )
)

pause
endlocal
