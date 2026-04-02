@echo off
chcp 65001 > nul
title ระบบ AI Voice Conversion (RVC) โดย DELTA SYNTH
color 0A

:: บังคับให้เริ่มทำงานในโฟลเดอร์ปัจจุบันเสมอ
cd /d "%~dp0"

echo =======================================================
echo          กำลังตรวจสอบและติดตั้งแพ็กเกจเสริมที่จำเป็น...
echo =======================================================
runtime\python.exe -m pip install python-dotenv

echo =======================================================
echo          กำลังเริ่มต้นระบบ RVC WebUI กรุณารอสักครู่...
echo =======================================================
:: เรียกใช้ Python จากโฟลเดอร์ runtime พร้อมระบุพอร์ต
runtime\python.exe infer-web.py --pycmd runtime\python.exe --port 9378

echo.
echo ระบบทำงานเสร็จสิ้น หรือพบข้อผิดพลาด
pause
