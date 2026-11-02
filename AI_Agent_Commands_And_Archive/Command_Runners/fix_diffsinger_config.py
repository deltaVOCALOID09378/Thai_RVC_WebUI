# ==============================================================================
# File Name      : fix_diffsinger_config.py
# Version        : 1.2
# Target Path    : A:\Thai_RVC_WebUI_v1\fix_diffsinger_config.py
# Function       : Auto-locate config.yaml, recursively resolve and patch 
#                  base_config paths, validate dataset paths, and run binarize.py
# Credit         : Made And Checked By DELTA SYNTH & All Agentic AI
# Original by    : OpenVPI / DiffSinger Team & DELTA SYNTH
# ==============================================================================

import os
import re
import sys
import shutil
import subprocess

def index_all_yaml_files(base_dir):
    """สร้างดัชนีไฟล์ .yaml ทั้งหมดในโปรเจกต์เพื่อใช้จับคู่ base_config อัตโนมัติ / Index all YAML files"""
    yaml_map = {}
    all_configs = []
    for root, dirs, files in os.walk(base_dir):
        if "runtime" in root or ".git" in root or "__pycache__" in root:
            continue
        for f in files:
            if f.endswith((".yaml", ".yml")):
                full_path = os.path.abspath(os.path.join(root, f))
                all_configs.append(full_path)
                # เก็บชื่อไฟล์เป็นคีย์ โดยให้ความสำคัญกับไฟล์ในโฟลเดอร์ configs เป็นหลัก
                if f not in yaml_map or "configs" in root.lower():
                    yaml_map[f] = full_path
    return all_configs, yaml_map

def find_binarize_script(base_dir):
    """ค้นหาตำแหน่งสคริปต์ binarize.py / Locate binarize.py"""
    candidates = [
        os.path.join(base_dir, "DiffSinger", "scripts", "binarize.py"),
        os.path.join(base_dir, "DiffSinger", "binarize.py"),
        os.path.join(base_dir, "scripts", "binarize.py")
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
            
    for root, dirs, files in os.walk(base_dir):
        if "runtime" in root or ".git" in root or "__pycache__" in root:
            continue
        if "binarize.py" in files:
            return os.path.abspath(os.path.join(root, "binarize.py"))
    return None

def resolve_target_yaml_path(raw_ref, current_yaml_dir, diffsinger_root, yaml_map):
    """คำนวณหาพาธจริงของไฟล์ที่ถูกอ้างอิงใน base_config / Resolve actual path for base_config entry"""
    clean_ref = raw_ref.strip().strip("'").strip('"')
    if not clean_ref:
        return None

    # 1. ตรวจสอบกรณีเป็นพาธเต็มอยู่แล้ว
    if os.path.isabs(clean_ref) and os.path.isfile(clean_ref):
        return os.path.abspath(clean_ref)

    # 2. ตรวจสอบแบบอ้างอิงจากโฟลเดอร์เดียวกับไฟล์ config ปัจจุบัน
    cand_rel_current = os.path.abspath(os.path.join(current_yaml_dir, clean_ref))
    if os.path.isfile(cand_rel_current):
        return cand_rel_current

    # 3. ตรวจสอบแบบอ้างอิงจากโฟลเดอร์หลักของ DiffSinger
    cand_rel_ds = os.path.abspath(os.path.join(diffsinger_root, clean_ref))
    if os.path.isfile(cand_rel_ds):
        return cand_rel_ds

    # 4. ตรวจสอบในโฟลเดอร์ DiffSinger/configs โดยตรงด้วยชื่อไฟล์
    basename = os.path.basename(clean_ref)
    cand_configs = os.path.abspath(os.path.join(diffsinger_root, "configs", basename))
    if os.path.isfile(cand_configs):
        return cand_configs

    # 5. ค้นหาจากดัชนีไฟล์ .yaml ทั้งโปรเจกต์
    if basename in yaml_map and os.path.isfile(yaml_map[basename]):
        return yaml_map[basename]

    return None

def patch_base_config_recursively(yaml_path, diffsinger_root, yaml_map, visited=None):
    """
    ตรวจสอบและซ่อมแซมบรรทัด base_config ในไฟล์ YAML และไฟล์แม่แบบที่เชื่อมโยงกันทุกชั้น
    Recursively inspect and repair base_config references across the YAML inheritance chain.
    """
    if visited is None:
        visited = set()

    abs_yaml = os.path.abspath(yaml_path)
    if abs_yaml in visited or not os.path.isfile(abs_yaml):
        return
    visited.add(abs_yaml)

    try:
        with open(abs_yaml, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[WARNING] ไม่สามารถอ่านไฟล์ {abs_yaml}: {e}")
        return

    current_dir = os.path.dirname(abs_yaml)
    modified = False
    new_lines = []
    in_base_config_list = False
    discovered_parents = []

    for line in lines:
        stripped = line.strip()

        # กรณีที่ 1: base_config อยู่ในบรรทัดเดียวกัน เช่น base_config: ./base.yaml หรือ base_config: [a.yaml]
        inline_match = re.match(r"^(\s*base_config\s*:\s*)(.+)$", line)
        if inline_match and not stripped.startswith("#"):
            prefix, val_part = inline_match.groups()
            val_clean = val_part.split("#")[0].strip()

            if val_clean.startswith("[") and val_clean.endswith("]"):
                # รูปแบบรายการในบรรทัดเดียว: [configs/base.yaml]
                items = [x.strip() for x in val_clean[1:-1].split(",") if x.strip()]
                resolved_items = []
                for item in items:
                    target = resolve_target_yaml_path(item, current_dir, diffsinger_root, yaml_map)
                    if target:
                        discovered_parents.append(target)
                        norm_target = target.replace("\\", "/")
                        resolved_items.append(f"'{norm_target}'")
                    else:
                        resolved_items.append(item)
                new_val = "[" + ", ".join(resolved_items) + "]"
                if new_val != val_clean:
                    line = f"{prefix}{new_val}\n"
                    modified = True
            elif val_clean:
                # รูปแบบค่าเดียว: base_config: ./base.yaml
                target = resolve_target_yaml_path(val_clean, current_dir, diffsinger_root, yaml_map)
                if target:
                    discovered_parents.append(target)
                    norm_target = target.replace("\\", "/")
                    if norm_target != val_clean.strip("'\""):
                        line = f"{prefix}'{norm_target}'\n"
                        modified = True
                else:
                    print(f"[WARNING] ไม่พบไฟล์แม่แบบ '{val_clean}' ที่ระบุใน {os.path.basename(abs_yaml)}")
            in_base_config_list = False
            new_lines.append(line)
            continue

        # กรณีที่ 2: หัวข้อ base_config: แบบขึ้นบรรทัดใหม่เป็นรายการ (- item)
        if re.match(r"^\s*base_config\s*:\s*(#.*)?$", line):
            in_base_config_list = True
            new_lines.append(line)
            continue

        if in_base_config_list:
            list_match = re.match(r"^(\s*-\s+)(.+)$", line)
            if list_match:
                prefix, item_part = list_match.groups()
                item_clean = item_part.split("#")[0].strip()
                target = resolve_target_yaml_path(item_clean, current_dir, diffsinger_root, yaml_map)
                if target:
                    discovered_parents.append(target)
                    norm_target = target.replace("\\", "/")
                    if norm_target != item_clean.strip("'\""):
                        line = f"{prefix}'{norm_target}'\n"
                        modified = True
                else:
                    print(f"[WARNING] ไม่พบไฟล์แม่แบบ '{item_clean}' ที่ระบุใน {os.path.basename(abs_yaml)}")
                new_lines.append(line)
                continue
            elif stripped == "" or stripped.startswith("#"):
                new_lines.append(line)
                continue
            else:
                in_base_config_list = False

        new_lines.append(line)

    if modified:
        backup_file = abs_yaml + ".bak"
        if not os.path.exists(backup_file):
            shutil.copy2(abs_yaml, backup_file)
        with open(abs_yaml, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"[PATCHED] อัปเดตพาธ base_config ในไฟล์: {abs_yaml}")
        print(f"[PATCHED] Updated base_config path in: {abs_yaml}")

    # ตรวจสอบไฟล์แม่แบบในลำดับถัดไปแบบ Recursive
    for parent_yaml in discovered_parents:
        patch_base_config_recursively(parent_yaml, diffsinger_root, yaml_map, visited)

def main():
    base_dir = os.path.abspath(os.getcwd())
    target_input = sys.argv[1] if len(sys.argv) > 1 else ""

    print("=" * 78)
    print("  DELTA SYNTH - DiffSinger Config & Base_Config Auto-Resolver [v1.2]")
    print(f"  โฟลเดอร์หลัก (Workspace): {base_dir}")
    print("=" * 78)

    binarize_script = find_binarize_script(base_dir)
    if not binarize_script:
        print("[ERROR] ไม่พบไฟล์ binarize.py ในระบบ / binarize.py not found!")
        sys.exit(2)

    diffsinger_root = os.path.dirname(binarize_script)
    if os.path.basename(diffsinger_root).lower() == "scripts":
        diffsinger_root = os.path.dirname(diffsinger_root)

    all_configs, yaml_map = index_all_yaml_files(base_dir)
    selected_config = None

    if target_input:
        target_input = os.path.abspath(target_input.strip('"'))
        if os.path.isfile(target_input) and target_input.endswith((".yaml", ".yml")):
            selected_config = target_input
        elif os.path.isdir(target_input):
            cand = os.path.join(target_input, "config.yaml")
            if os.path.isfile(cand):
                selected_config = cand
            else:
                sub_configs, _ = index_all_yaml_files(target_input)
                if sub_configs:
                    selected_config = sub_configs[0]

    if not selected_config:
        # กรองเอาเฉพาะไฟล์ที่เป็น config ของโมเดลหรือ dataset ขึ้นมาแสดงเป็นลำดับแรก
        all_configs.sort(key=lambda x: (0 if os.path.basename(x) == "config.yaml" else 1, len(x)))
        if not all_configs:
            print("[ERROR] ไม่พบไฟล์ .yaml ในโปรเจกต์เลย / No .yaml files found!")
            sys.exit(2)

        print("\n[FOUND] พบไฟล์ตั้งค่า (Config Files) ในระบบดังนี้:")
        for idx, cfg in enumerate(all_configs[:15], 1):
            rel = os.path.relpath(cfg, base_dir)
            print(f"  [{idx}] {rel}")

        choice = input("\nกรุณาเลือกหมายเลขไฟล์ที่ต้องการใช้ (กด Enter เพื่อเลือกข้อ [1]): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(all_configs):
            selected_config = all_configs[int(choice) - 1]
        else:
            selected_config = all_configs[0]

    selected_config = os.path.abspath(selected_config)
    print(f"\n[SELECTED] ไฟล์ตั้งค่าเป้าหมาย: {selected_config}")

    # ดำเนินการตรวจสอบและซ่อมแซม base_config ทุกชั้น
    print("[INFO] กำลังตรวจสอบความถูกต้องของบรรทัด base_config / Validating base_config chain...")
    patch_base_config_recursively(selected_config, diffsinger_root, yaml_map)

    python_exe = os.path.join(base_dir, "runtime", "python.exe")
    if not os.path.isfile(python_exe):
        python_exe = sys.executable

    env = os.environ.copy()
    env["PYTHONPATH"] = diffsinger_root + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [python_exe, binarize_script, "--config", selected_config]
    print(f"\n[EXECUTING] กำลังรันคำสั่ง: {' '.join(cmd)}")
    print(f"[WORKDIR]   โฟลเดอร์ทำงาน: {diffsinger_root}\n" + "-" * 78)

    result = subprocess.run(cmd, cwd=diffsinger_root, env=env)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()