# ==============================================================================
# File: add_header.py
# Path: add_header.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
import sys, os

header = """# ==============================================================================
"""

for root, _, files in os.walk('.'):
    if 'runtime' in root or '.git' in root or 'venv' in root: continue
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                    # Insert after the first line if it's a shebang or encoding, or just at the top
                    lines = content.split('\n')
                    if lines and (lines[0].startswith('#!') or lines[0].startswith('# -*-')):
                        new_content = lines[0] + '\n' + header + '\n'.join(lines[1:])
                    else:
                        new_content = header + content
                        
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Added header to {filepath}")
            except Exception as e:
                print(f"Skipped {filepath}: {e}")
