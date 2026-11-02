# ==============================================================================
# File: clean_cn.py
# Path: clean_cn.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
import sys, re, os

replacements = {
    r'模型信息:%s\\n采样率:%s\\n模型是否输入音高引导:%s\\n版本:%s': r'Model Info: %s\\nSample Rate: %s\\nPitch Guidance: %s\\nVersion: %s',
    r'== i18n\("是"\)': r'in ["Yes", "ใช่", "True", "1", "是"]'
}

for root, _, files in os.walk('infer'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                changed = False
                for k, v in replacements.items():
                    if re.search(k, content):
                        content = re.sub(k, v, content)
                        changed = True
                
                if re.search(r'[\u4e00-\u9fff]+', content):
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if re.search(r'[\u4e00-\u9fff]+', line):
                            # Replace Chinese chars with [EN/TH text]
                            lines[i] = re.sub(r'[\u4e00-\u9fff]+', 'EN_Translation', line)
                    content = '\n'.join(lines)
                    changed = True

                if changed:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f'Cleaned {filepath}')
            except Exception as e:
                print(f'Failed on {filepath}: {e}')
