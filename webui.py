# -*- coding: utf-8 -*-
# ==============================================================================
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# Original by RVC-Project
# File: webui.py | Version: 1.0 (Compatibility Gateway)
# Description: รันเกตเวย์รองรับการเรียกใช้งานผ่าน webui.py แบบมาตรฐาน
# ==============================================================================
import os
import sys
import runpy

if __name__ == "__main__":
    now_dir = os.path.dirname(os.path.abspath(__file__))
    program_path = os.path.join(now_dir, "Program.py")
    runpy.run_path(program_path, run_name="__main__")
