@echo off
chcp 65001 >nul
title DELTA SYNTH - DiffSinger AI Training Studio (Miniconda & Universal Hardware Installer)

set "download_url=https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Windows-x86_64.exe"
set "installer_name=miniconda_installer.exe"
set "install_path=%cd%\miniconda"
set "conda_python=%cd%\miniconda\python.exe"
set "conda_pip=%cd%\miniconda\Scripts\pip.exe"

echo [DELTA SYNTH] กำลังดาวน์โหลดตัวติดตั้ง Miniconda3 (Python 3.10)...
powershell -ExecutionPolicy Bypass -command "Invoke-WebRequest -Uri '%download_url%' -OutFile '%installer_name%'"

echo [DELTA SYNTH] กำลังติดตั้ง Miniconda ลงในโฟลเดอร์โปรเจกต์...
"%installer_name%" /InstallationType=JustMe /RegisterPython=0 /S /D=%install_path%

echo [DELTA SYNTH] กำลังล้างไฟล์ติดตั้งชั่วคราว...
if exist "%installer_name%" del "%installer_name%"

echo [DELTA SYNTH] กำลังติดตั้งไลบรารีส่วนหน้า (GUI Requirements)...
"%conda_pip%" install -r requirements.txt --no-warn-script-location

echo [DELTA SYNTH] กำลังตั้งค่าสภาพแวดล้อมฮาร์ดแวร์ (CUDA / DirectML / ROCm / CPU+GPU Hybrid)...
"%conda_python%" setup_conda_envs.py

echo [สำเร็จ] การติดตั้งเสร็จสมบูรณ์! คุณสามารถเปิดโปรแกรมได้จากไฟล์ run_gui.bat
pause
