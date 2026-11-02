import os
import re

def clean_old_headers(content):
    lines = content.split('\n')
    new_lines = []
    skip = False
    for line in lines:
        if line.startswith('# =============================================================================='):
            continue
        if line.startswith('# File:') or line.startswith('# Path:') or line.startswith('# Description:') or line.startswith('# Version:') or line.startswith('# Made And Checked By') or line.startswith('# [ Original by'):
            continue
        new_lines.append(line)
    
    while new_lines and new_lines[0].strip() == "":
        new_lines.pop(0)
        
    return '\n'.join(new_lines)

for root, _, files in os.walk('.'):
    if 'runtime' in root or '.git' in root or 'venv' in root or 'tools' in root or 'logs' in root: continue
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            rel_path = os.path.normpath(filepath)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                version = "1.0"
                v_match = re.search(r'Version:\s*([0-9]+\.[0-9]+)', content)
                if v_match:
                    old_v = float(v_match.group(1))
                    version = f"{old_v + 0.1:.1f}"
                elif "DELTA SYNTH" in content:
                    version = "1.1"
                
                clean_content = clean_old_headers(content)
                
                header = f'''# ==============================================================================
# File: {file}
# Path: {rel_path}
# Description: RVC WebUI Core Module / โมดูลหลัก RVC WebUI
# Version: {version}
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
'''
                
                lines = clean_content.split('\n')
                if lines and (lines[0].startswith('#!') or lines[0].startswith('# -*-')):
                    final_content = lines[0] + '\n' + header + '\n'.join(lines[1:])
                else:
                    final_content = header + clean_content
                    
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(final_content)
                print(f"Updated header for {rel_path} to Version {version}")
                
            except Exception as e:
                print(f"Skipped {rel_path}: {e}")
