@echo off
:: ตั้งค่าให้หน้าต่าง CMD รองรับภาษาไทย
chcp 65001 >nul
setlocal enabledelayedexpansion

:: บังคับให้เริ่มทำงานในโฟลเดอร์ที่ไฟล์ .bat ตัวนี้วางอยู่เสมอ
cd /d "%~dp0"

color 0A
echo ===============================================================
echo        DELTA SYNTH - RVC Environment Auto Installer
echo ===============================================================
echo.
echo สคริปต์นี้จะทำการติดตั้งเครื่องมือส่วนกลาง (Global) และ 
echo ไลบรารีสำหรับใช้งาน RVC ในพื้นที่โฟลเดอร์ปัจจุบัน (Local)
echo.
pause

echo.
echo ===============================================================
echo [1/3] กำลังติดตั้งเครื่องมือส่วนกลาง (Global Tools)
echo ===============================================================
echo เครื่องมือเหล่านี้จะถูกติดตั้งลงในระบบ Windows เพื่อให้เรียกใช้ได้ทุกที่
echo ตรวจสอบและติดตั้ง: Python 3.10, Git และ FFmpeg...

:: ใช้ winget ในการดาวน์โหลดและติดตั้งเครื่องมือส่วนกลางแบบเงียบๆ
winget install -e --id Python.Python.3.10 --accept-source-agreements --accept-package-agreements
winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements
winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements

echo.
echo ===============================================================
echo [2/3] กำลังสร้างพื้นที่ทำงานส่วนตัว (Local Virtual Environment)
echo ===============================================================
:: สร้างโฟลเดอร์ venv ในตำแหน่งเดียวกับไฟล์ .bat
if not exist "venv" (
    echo กำลังสร้างสภาพแวดล้อมจำลอง (venv)...
    python -m venv venv
    echo [สำเร็จ] สร้างโฟลเดอร์ venv เรียบร้อยแล้ว!
) else (
    echo [ข้าม] พบโฟลเดอร์ venv อยู่แล้วในระบบ
)

echo.
echo ===============================================================
echo [3/3] กำลังดาวน์โหลดและติดตั้งไลบรารีสำหรับ RVC (Dependencies)
echo ===============================================================
:: สลับเข้าไปใช้งานในโหมด venv ของโฟลเดอร์นี้
call venv\Scripts\activate.bat

:: อัปเดต pip ให้เป็นเวอร์ชันล่าสุดก่อนเริ่มงาน
python -m pip install --upgrade pip

:: ติดตั้ง PyTorch (เวอร์ชันมาตรฐาน ถ้าคุณเดลต้าใช้ AMD/Intel IPEX สามารถมาแก้ตรงนี้ได้ครับ)
echo กำลังติดตั้ง PyTorch และ Torchvision...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: ตรวจสอบหาไฟล์ requirements.txt และติดตั้ง
if exist "requirements.txt" (
    echo กำลังติดตั้งไลบรารีพื้นฐานจากไฟล์ requirements.txt...
    pip install -r requirements.txt
) else (
    echo [คำเตือน] ไม่พบไฟล์ requirements.txt ในโฟลเดอร์นี้! 
    echo โปรแกรมอาจทำงานไม่สมบูรณ์ กรุณาตรวจสอบอีกครั้ง
)

echo.
color 0E
echo ===============================================================
echo    การติดตั้งทั้งหมดเสร็จสมบูรณ์! (Installation Completed)
echo ===============================================================
echo.
echo สภาพแวดล้อมพร้อมใช้งานสำหรับโปรเจกต์ DELTA SYNTH แล้วครับ
echo หลังจากนี้ คุณสามารถรันสคริปต์เริ่มทำงานของ RVC ได้เลย!
echo.
pause
exit