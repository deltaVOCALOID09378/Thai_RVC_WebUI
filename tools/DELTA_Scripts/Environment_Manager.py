# -*- coding: utf-8 -*-
# ==============================================================================
# File: Environment_Manager.py
# Path: Environment_Manager.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 2.4
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
# Description: Universal Environment Manager mapped to Central Python 3.14

import os
import sys
import subprocess

class EnvironmentManager:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        # สลับมาใช้ Python ส่วนกลางตามคำสั่ง (Switch to Central Python path)
        self.py_exec = r"C:\Program Files\Python314\python.exe"
        if not os.path.exists(self.py_exec):
            print(f"⚠️ [WARNING] ไม่พบ Python ส่วนกลางที่ (Central Python not found at): {self.py_exec}")
            self.py_exec = sys.executable 

    def run_cmd(self, cmd_list, cwd=None):
        """รันคำสั่งระบบและดักจับข้อผิดพลาด (Execute system command safely)"""
        try:
            subprocess.run(cmd_list, check=True, cwd=cwd)
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ [ข้อผิดพลาด/ERROR] คำสั่งล้มเหลว (Command failed): {e}")
            return False
        except FileNotFoundError:
            print(f"\n❌ [ข้อผิดพลาด/ERROR] ไม่พบเครื่องมือ (Command not found): {cmd_list[0]}")
            return False

    def option_install_universal(self, file_paths):
        print("\n--- กำลังประมวลผลและติดตั้งไฟล์เครื่องมือ (Universal Tool Installation) ---")
        for file_path in file_paths:
            file_path = file_path.strip().strip('"')
            if not os.path.isfile(file_path):
                print(f"❌ [ข้อผิดพลาด/ERROR] ไม่พบไฟล์ (File not found): {file_path}")
                continue
                
            ext = os.path.splitext(file_path)[1].lower()
            file_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)

            print(f"🔄 กำลังประมวลผลไฟล์ (Processing): {file_name}")

            if ext == '.txt':
                print(f"📄 [PIP] กำลังติดตั้งรายการเครื่องมือจาก (Reading dependencies from) '{file_name}'...")
                self.run_cmd([self.py_exec, "-m", "pip", "install", "-r", file_path])
            elif ext == '.json' and file_name.lower() == 'package.json':
                print(f"📦 [NPM] ตรวจพบ package.json กำลังรันกระบวนการติดตั้ง...")
                self.run_cmd(["npm", "install"], cwd=file_dir)
            elif ext == '.whl':
                print(f"⚙️ [WHEEL] กำลังรวบรวมและติดตั้ง Wheel package...")
                self.run_cmd([self.py_exec, "-m", "pip", "install", file_path])
            elif ext == '.py':
                print(f"🐍 [PYTHON] กำลังรันสคริปต์ส่วนเสริม...")
                self.run_cmd([self.py_exec, file_path])
            elif ext == '.bat':
                print(f"🪟 [BATCH] กำลังเชื่อมต่อกระบวนการผ่าน Batch script...")
                self.run_cmd([file_path], cwd=file_dir)
            else:
                print(f"⚠️ [ข้าม/SKIP] ยังไม่รองรับการติดตั้งอัตโนมัติสำหรับนามสกุลนี้ (Unsupported auto-install format): {ext}")

    def run(self):
        if len(sys.argv) > 1:
            dropped_items = sys.argv[1:]
            print(f"📥 ตรวจพบไฟล์เป้าหมาย (Detected dropped items): {len(dropped_items)} รายการ")
            self.option_install_universal(dropped_items)
        else:
            print("⚠️ [แจ้งเตือน/WARNING] กระบวนการนี้รับรองการลากวางไฟล์มาทับ (Drag & Drop) ผ่าน .bat โดยตรงเท่านั้น")

if __name__ == "__main__":
    manager = EnvironmentManager()
    manager.run()