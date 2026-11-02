# ==============================================================================
# File: fix_ext.py
# Path: fix_ext.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.0
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
import os

filepath = r'infer\modules\vc\modules.py'
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

target = 'output_path = f"{opt_root}/{os.path.basename(path)}.{format1}"'
replacement = 'output_path = f"{opt_root}/{os.path.splitext(os.path.basename(path))[0]}.{format1}"'

if target in text:
    text = text.replace(target, replacement)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print('Fixed double extension bug in modules.py')
else:
    print('Target not found or already fixed.')
