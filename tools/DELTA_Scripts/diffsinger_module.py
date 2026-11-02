# -*- coding: utf-8 -*-
# ==============================================================================
# File: diffsinger_module.py
# Path: diffsinger_module.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
# ===============================================================
# Description: DiffSinger Training & OpenUtau Voicebank Packaging
#              module for DELTA SYNTH Studio.
#
# [สำคัญ / IMPORTANT]
# โมดูลนี้เขียนขึ้นให้เรียกใช้สคริปต์มาตรฐานของ openvpi/DiffSinger
# (scripts/binarize.py, scripts/train.py, scripts/export.py)
# ผ่าน subprocess เหมือนวิธีที่ Program.py เรียก RVC training scripts
# อยู่แล้ว หากโครงสร้างโฟลเดอร์ / ชื่อ argument ของ DiffSinger ที่คุณ
# ติดตั้งจริงต่างจากนี้ ให้ปรับค่าใน DS_ROOT และคำสั่งใน build_*_cmd()
# ให้ตรงกับเวอร์ชันที่ใช้งานอยู่
#
# This module assumes the standard openvpi/DiffSinger CLI layout.
# If your local DiffSinger install differs, adjust DS_ROOT and the
# build_*_cmd() helpers below to match your actual scripts.
# ===============================================================

import os
import sys
import shutil
import threading
from time import sleep
from subprocess import Popen

now_dir = os.getcwd()

# ปรับ path นี้ให้ชี้ไปที่โฟลเดอร์ที่ติดตั้ง openvpi/DiffSinger จริง
# Point this to wherever your openvpi/DiffSinger repo actually lives.
DS_ROOT = os.getenv("DIFFSINGER_ROOT", os.path.join(now_dir, "DiffSinger"))
DS_PYTHON_EXE = os.path.join(now_dir, "runtime", "python.exe") if os.path.exists(
    os.path.join(now_dir, "runtime", "python.exe")
) else sys.executable

# --- Global cancel flags ---
CANCEL_DS_BINARIZE = False
CANCEL_DS_VARIANCE = False
CANCEL_DS_ACOUSTIC = False
CANCEL_DS_EXPORT = False


def _if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


# --- Cancel functions ---
def cancel_ds_binarize_func():
    global CANCEL_DS_BINARIZE
    CANCEL_DS_BINARIZE = True
    return "🛑 [ระงับ | CANCELLED] หยุดการเตรียมข้อมูล DiffSinger (Binarize stopped by user)"


def cancel_ds_variance_func():
    global CANCEL_DS_VARIANCE
    CANCEL_DS_VARIANCE = True
    return "🛑 [ระงับ | CANCELLED] หยุดการฝึกสอน Variance Model (Variance training stopped)"


def cancel_ds_acoustic_func():
    global CANCEL_DS_ACOUSTIC
    CANCEL_DS_ACOUSTIC = True
    return "🛑 [ระงับ | CANCELLED] หยุดการฝึกสอน Acoustic Model (Acoustic training stopped)"


def cancel_ds_export_func():
    global CANCEL_DS_EXPORT
    CANCEL_DS_EXPORT = True
    return "🛑 [ระงับ | CANCELLED] หยุดการส่งออกโมเดล ONNX (Export stopped)"


def _run_streaming(cmd, log_file, cancel_flag_getter, header_th, header_en, cwd=None):
    """
    รันคำสั่งแบบ subprocess พร้อม log ลงไฟล์ แล้ว yield เนื้อหา log
    แบบเรียลไทม์ (เหมือนรูปแบบที่ Program.py ใช้กับ RVC training)
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    open(log_file, "w", encoding="utf-8").close()

    full_cmd = f'{cmd} > "{log_file}" 2>&1'
    p = Popen(full_cmd, shell=True, cwd=cwd or now_dir)
    done = [False]
    threading.Thread(target=_if_done, args=(done, p)).start()

    while True:
        if cancel_flag_getter():
            p.kill()
            yield f"🛑 [ระงับ | CANCELLED] {header_th} ถูกยกเลิกโดยผู้ใช้ / {header_en} aborted by user."
            return
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                yield f"⏳ [{header_th} | {header_en}]\n" + f.read()
        sleep(1)
        if done[0]:
            break

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if p.returncode == 0:
        yield f"✅ [เสร็จสิ้น | COMPLETED] {header_th} สำเร็จ\n" + content
    else:
        yield f"❌ [ล้มเหลว | FAILED] {header_th} ล้มเหลว (exit code {p.returncode})\n" + content


# ==========================================
# [1] Preprocess / Binarize
# ==========================================
def ds_binarize(raw_data_dir, ds_config_path, exp_dir1):
    """
    เตรียมข้อมูล (แปลง wav+lab/textgrid เป็น binary dataset) สำหรับ DiffSinger
    ตรงกับ: python scripts/binarize.py --config <config.yaml>
    """
    global CANCEL_DS_BINARIZE
    CANCEL_DS_BINARIZE = False

    if not raw_data_dir or not os.path.exists(raw_data_dir):
        yield "❌ [ข้อผิดพลาด | ERROR] ไม่พบโฟลเดอร์ข้อมูลดิบ (Raw data directory not found)."
        return
    if not ds_config_path or not os.path.exists(ds_config_path):
        yield "❌ [ข้อผิดพลาด | ERROR] ไม่พบไฟล์ config (.yaml) ของ DiffSinger."
        return

    binarize_script = os.path.join(DS_ROOT, "scripts", "binarize.py")
    if not os.path.exists(binarize_script):
        yield (
            "❌ [ข้อผิดพลาด | ERROR] ไม่พบ scripts/binarize.py ใน DIFFSINGER_ROOT "
            f"({DS_ROOT})\nกรุณาตั้งค่า environment variable DIFFSINGER_ROOT "
            "ให้ชี้ไปที่โฟลเดอร์ที่ติดตั้ง openvpi/DiffSinger"
        )
        return

    log_file = os.path.join(now_dir, "logs", exp_dir1, "ds_binarize.log")
    cmd = f'"{DS_PYTHON_EXE}" "{binarize_script}" --config "{ds_config_path}"'

    for chunk in _run_streaming(
        cmd, log_file, lambda: CANCEL_DS_BINARIZE,
        "เตรียมข้อมูล DiffSinger", "DiffSinger Binarize",
        cwd=DS_ROOT,
    ):
        yield chunk


# ==========================================
# [2] Train Variance Model (Pitch/Duration)
# ==========================================
def ds_train_variance(exp_name, ds_config_path, max_steps, resume):
    global CANCEL_DS_VARIANCE
    CANCEL_DS_VARIANCE = False

    if not ds_config_path or not os.path.exists(ds_config_path):
        yield "❌ [ข้อผิดพลาด | ERROR] ไม่พบไฟล์ config (.yaml) สำหรับ Variance Model."
        return

    train_script = os.path.join(DS_ROOT, "scripts", "train.py")
    if not os.path.exists(train_script):
        yield f"❌ [ข้อผิดพลาด | ERROR] ไม่พบ scripts/train.py ใน DIFFSINGER_ROOT ({DS_ROOT})"
        return

    reset_flag = "" if resume else "--reset"
    log_file = os.path.join(now_dir, "logs", exp_name, "ds_train_variance.log")
    cmd = (
        f'"{DS_PYTHON_EXE}" "{train_script}" --config "{ds_config_path}" '
        f'--exp_name "{exp_name}_variance" --max_steps {int(max_steps)} {reset_flag}'
    )

    for chunk in _run_streaming(
        cmd, log_file, lambda: CANCEL_DS_VARIANCE,
        "ฝึกสอน Variance Model (Pitch/Duration)", "Variance Model Training",
        cwd=DS_ROOT,
    ):
        yield chunk


# ==========================================
# [3] Train Acoustic Model (เนื้อเสียง)
# ==========================================
def ds_train_acoustic(exp_name, ds_config_path, max_steps, resume):
    global CANCEL_DS_ACOUSTIC
    CANCEL_DS_ACOUSTIC = False

    if not ds_config_path or not os.path.exists(ds_config_path):
        yield "❌ [ข้อผิดพลาด | ERROR] ไม่พบไฟล์ config (.yaml) สำหรับ Acoustic Model."
        return

    train_script = os.path.join(DS_ROOT, "scripts", "train.py")
    if not os.path.exists(train_script):
        yield f"❌ [ข้อผิดพลาด | ERROR] ไม่พบ scripts/train.py ใน DIFFSINGER_ROOT ({DS_ROOT})"
        return

    reset_flag = "" if resume else "--reset"
    log_file = os.path.join(now_dir, "logs", exp_name, "ds_train_acoustic.log")
    cmd = (
        f'"{DS_PYTHON_EXE}" "{train_script}" --config "{ds_config_path}" '
        f'--exp_name "{exp_name}_acoustic" --max_steps {int(max_steps)} {reset_flag}'
    )

    for chunk in _run_streaming(
        cmd, log_file, lambda: CANCEL_DS_ACOUSTIC,
        "ฝึกสอน Acoustic Model (เนื้อเสียง)", "Acoustic Model Training",
        cwd=DS_ROOT,
    ):
        yield chunk


# ==========================================
# [4] Export เป็น ONNX (สำหรับ OpenUtau)
# ==========================================
def ds_export_onnx(exp_name, stage, out_dir):
    """
    stage: "variance" หรือ "acoustic"
    ตรงกับ: python scripts/export.py <stage> --exp_name <exp> --out <path>.onnx
    """
    global CANCEL_DS_EXPORT
    CANCEL_DS_EXPORT = False

    export_script = os.path.join(DS_ROOT, "scripts", "export.py")
    if not os.path.exists(export_script):
        yield f"❌ [ข้อผิดพลาด | ERROR] ไม่พบ scripts/export.py ใน DIFFSINGER_ROOT ({DS_ROOT})"
        return

    os.makedirs(out_dir, exist_ok=True)
    out_name = "variance.onnx" if stage == "variance" else "acoustic.onnx"
    out_path = os.path.join(out_dir, out_name)

    log_file = os.path.join(now_dir, "logs", exp_name, f"ds_export_{stage}.log")
    cmd = (
        f'"{DS_PYTHON_EXE}" "{export_script}" {stage} '
        f'--exp_name "{exp_name}_{stage}" --out "{out_path}"'
    )

    for chunk in _run_streaming(
        cmd, log_file, lambda: CANCEL_DS_EXPORT,
        f"ส่งออกโมเดล {stage} เป็น ONNX", f"Export {stage} to ONNX",
        cwd=DS_ROOT,
    ):
        yield chunk


# ==========================================
# [5] แพ็คเป็น Voicebank ตามโครงสร้างใน Map.md
# ==========================================
def package_voicebank(
    singer_name,
    language,
    acoustic_onnx_path,
    variance_onnx_path,
    dsdict_path,
    icon_path,
    full_pic_path,
    character_desc,
    output_root,
):
    """
    สร้างโฟลเดอร์ {Singer_Name}_{Language}_DiffSinger/ ตามที่ระบุใน Map.md
    พร้อม character.txt, character.yaml, dsconfig.yaml, dsdict.txt,
    acoustic.onnx, variance.onnx และโฟลเดอร์ Picture/
    """
    if not singer_name:
        return "❌ [ข้อผิดพลาด | ERROR] กรุณาระบุชื่อนักร้อง (Singer name is required)."
    if not acoustic_onnx_path or not os.path.exists(acoustic_onnx_path):
        return "❌ [ข้อผิดพลาด | ERROR] ไม่พบไฟล์ acoustic.onnx"
    if not variance_onnx_path or not os.path.exists(variance_onnx_path):
        return "❌ [ข้อผิดพลาด | ERROR] ไม่พบไฟล์ variance.onnx"

    safe_name = singer_name.strip().replace(" ", "_")
    lang_code = (language or "th").strip()
    pkg_dir = os.path.join(output_root, f"{safe_name}_{lang_code}_DiffSinger")
    pic_dir = os.path.join(pkg_dir, "Picture")
    os.makedirs(pic_dir, exist_ok=True)

    # character.txt
    with open(os.path.join(pkg_dir, "character.txt"), "w", encoding="utf-8") as f:
        f.write(character_desc or f"{singer_name} - DiffSinger Voicebank ({lang_code})\n")
        f.write("Original by Patiphat Wongyai (Delta) & RVC Project\n")

    # character.yaml (สำหรับ OpenUtau UI overview)
    char_yaml = f"""name: {singer_name}
image: Picture/Full.png
avatar: Picture/Icon.png
author: DELTA SYNTH
voice: {lang_code}
singer_type: diffsinger
web: https://patiphatwongyai.wixsite.com/deltasynth
"""
    with open(os.path.join(pkg_dir, "character.yaml"), "w", encoding="utf-8") as f:
        f.write(char_yaml)

    # dsconfig.yaml (สำหรับให้ OpenUtau โหลดโมเดล Acoustic/Variance)
    ds_config_yaml = f"""name: {singer_name}
singer_type: diffsinger
text_lang: {lang_code}
acoustic: acoustic.onnx
acoustic_config: dsconfig.yaml
variance: variance.onnx
dictionary: dsdict.txt
# หมายเหตุ: หากใช้ vocoder แยกต่างหาก (เช่น NSF-HiFiGAN) ให้เพิ่ม path ที่นี่
# vocoder: nsf_hifigan.onnx
"""
    with open(os.path.join(pkg_dir, "dsconfig.yaml"), "w", encoding="utf-8") as f:
        f.write(ds_config_yaml)

    # คัดลอกไฟล์โมเดลและทรัพยากรอื่น ๆ
    shutil.copy2(acoustic_onnx_path, os.path.join(pkg_dir, "acoustic.onnx"))
    shutil.copy2(variance_onnx_path, os.path.join(pkg_dir, "variance.onnx"))

    if dsdict_path and os.path.exists(dsdict_path):
        shutil.copy2(dsdict_path, os.path.join(pkg_dir, "dsdict.txt"))
    else:
        open(os.path.join(pkg_dir, "dsdict.txt"), "w", encoding="utf-8").close()

    if icon_path and os.path.exists(icon_path):
        shutil.copy2(icon_path, os.path.join(pic_dir, "Icon.png"))
    if full_pic_path and os.path.exists(full_pic_path):
        shutil.copy2(full_pic_path, os.path.join(pic_dir, "Full.png"))

    return (
        f"🎉 [เสร็จสมบูรณ์ | COMPLETED] สร้าง Voicebank สำเร็จที่:\n{pkg_dir}\n\n"
        "โครงสร้างไฟล์ตรงตาม Map.md:\n"
        "character.txt, character.yaml, dsconfig.yaml, dsdict.txt,\n"
        "acoustic.onnx, variance.onnx, Picture/Icon.png, Picture/Full.png"
    )
