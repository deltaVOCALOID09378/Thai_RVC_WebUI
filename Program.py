# -*- coding: utf-8 -*-
# ==============================================================================
# Date: 12/10/2026 (B.E. 2569) / 12 October 2026
# File Name: Program.py
# Version: 4.7
# Location: Root Directory
# Function: DELTA RVC Studio x OpenUtau & Diffsinger Unified WebUI Engine
# Credit: Made And Checked By DELTA SYNTH [ ปฎิภัทร วงศ์ใหญ่ ] & All Code Agentic AI Engine
#         Original by Patiphat Wongyai (Delta) & RVC Project
# Description: Unified WebUI with robust dataset validation, double extension handling,
#              dynamic F0 resolution, strict failure gates in train1key (preventing
#              phantom indexing), and robust Data Sanitization to prevent KeyError.
# ==============================================================================

import os
import sys

# Configure UTF-8 encoding for standard streams on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gc
import glob
import shutil
import logging
import traceback
import platform
import pathlib
from pathlib import Path
import json
import subprocess
import argparse
from contextlib import suppress
import time
from time import sleep, strftime
from subprocess import Popen
from random import shuffle
import warnings
import threading
import soundfile as sf
import numpy as np
import torch
import gradio as gr
import faiss
import fairseq
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
now_dir = str(PROJECT_ROOT)
os.chdir(now_dir)
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)
load_dotenv()

# Strict runtime executable selection prioritizing runtime\python.exe
DEFAULT_PYTHON = os.path.join(now_dir, "runtime", "python.exe")
PYTHON_EXE = DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else (sys.executable or "python")

from utils import PoliteLogger, format_error_diagnostic, install_global_exception_handler
install_global_exception_handler()

from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from infer.lib.train.process_ckpt import change_info, extract_small_model, merge, show_info
from i18n.i18n import I18nAuto
from configs.config import Config

ACTIVE_CHILD_PROCESSES = set()
CANCEL_REQUESTED = False

# PyTorch 2.6+ Safe Globals registration for fairseq Dictionary
try:
    from fairseq.data.dictionary import Dictionary
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([Dictionary])
except Exception as e:
    logging.warning(f"Could not register safe globals for fairseq: {e}")

# Smart Logging System matching DELTA SYNTH Standard
class DeltaLogger:
    @staticmethod
    def format_msg(level: str, msg: str) -> str:
        timestamp = strftime("%H:%M:%S")
        return f"[{timestamp}] [{level}] {msg}"
    
    @staticmethod
    def inf(msg: str, to_console: bool = False) -> str: 
        formatted = DeltaLogger.format_msg("INF", msg)
        logger.info(msg)
        if to_console and os.getenv("DELTA_CMD_VERBOSE", "0") == "1":
            print(formatted, flush=True)
        return formatted

    @staticmethod
    def wrn(msg: str, to_console: bool = False) -> str: 
        formatted = DeltaLogger.format_msg("WRN", msg)
        logger.warning(msg)
        if to_console and os.getenv("DELTA_CMD_VERBOSE", "0") == "1":
            print(formatted, flush=True)
        return formatted

    @staticmethod
    def err(msg: str, exc: Exception = None, context: str = "Program") -> str: 
        return PoliteLogger.error(msg=msg, exc=exc, context=context)

    @staticmethod
    def ftl(msg: str, exc: Exception = None, context: str = "Program") -> str: 
        return PoliteLogger.error(msg=msg, exc=exc, context=f"Fatal - {context}")

    @staticmethod
    def model(model_name: str, details: str = "") -> str:
        return PoliteLogger.model(model_name, details)

logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("gradio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets", "weights"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets", "uvr5_weights"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets", "indices"), exist_ok=True)
os.environ["TEMP"] = tmp

def _bootstrap_config():
    original_argv = sys.argv[:]
    filtered_argv = [original_argv[0]]
    i = 1
    accepted_single = {"--colab", "--noparallel", "--noautoopen", "--dml"}
    accepted_pairs = {"--port", "--pycmd", "--target"}
    while i < len(original_argv):
        arg = original_argv[i]
        if arg in accepted_pairs:
            filtered_argv.append(arg)
            if i + 1 < len(original_argv):
                filtered_argv.append(original_argv[i + 1])
                i += 1
        elif arg in accepted_single:
            filtered_argv.append(arg)
        i += 1
    sys.argv = filtered_argv
    try:
        return Config()
    finally:
        sys.argv = original_argv

config = _bootstrap_config()
vc = VC(config)

if getattr(config, "dml", False):
    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res
    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

i18n = I18nAuto()

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if_gpu_ok = True
        gpu_infos.append(f"GPU {i}\t{gpu_name}")
        mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))

current_device = getattr(config, "device", "cpu")

if getattr(config, "dml", False) and current_device != "cpu":
    gpu_info = DeltaLogger.inf("⚡ ทำงานด้วยการ์ดจอ USB 3.0 / แกนประมวลผลชุดนอก (Operating via External USB GPU / DirectML)", to_console=True)
    default_batch_size = 1
elif current_device == "cpu":
    gpu_info = DeltaLogger.inf("🖥️ ทำงานด้วยระบบ CPU เป็นเซ็ตเริ่มต้น (Operating via CPU Default Set)", to_console=True)
    default_batch_size = 1
elif if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = DeltaLogger.inf("⚡ ตรวจพบการ์ดจอภายใน (Internal GPU Detected):\n" + "\n".join(gpu_infos), to_console=True)
    default_batch_size = max(1, min(mem) // 2)
else:
    gpu_info = DeltaLogger.inf("🖥️ ทำงานด้วยระบบ CPU เป็นเซ็ตเริ่มต้น (Operating via CPU Default Set)", to_console=True)
    default_batch_size = 1

gpus = "-".join([str(i) for i in range(ngpu)]) if if_gpu_ok else ""

weight_root = os.getenv("weight_root", os.path.join(now_dir, "assets", "weights"))
weight_uvr5_root = os.getenv("weight_uvr5_root", os.path.join(now_dir, "assets", "uvr5_weights"))
index_root = os.getenv("index_root", os.path.join(now_dir, "logs"))

base_source_dir = os.path.join(now_dir, "For OpenUtau VB Making", "A Voicebank File")
SOURCE_DIRS = {
    "1. EN": os.path.join(base_source_dir, "EN"),
    "2. TH": os.path.join(base_source_dir, "TH"),
    "3. JP": os.path.join(base_source_dir, "JP")
}
VOICEBANK_FORMATS = {
    "EN": "English Arpasing",
    "TH": "Thai VCCV",
    "JP": "JPN VCV"
}

sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}

def sanitize_sr(sr):
    sr_str = str(sr).lower().strip()
    if sr_str in ["32k", "32000", "2"]: return "32k"
    if sr_str in ["40k", "40000", "0"]: return "40k"
    if sr_str in ["48k", "48000", "1"]: return "48k"
    return "40k"

def sanitize_bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, str): return v.lower() in ["true", "1", "yes", "y", "t"]
    if isinstance(v, int): return v == 1
    return bool(v)

def sanitize_version(v):
    v_str = str(v).lower().strip()
    if v_str in ["v1", "v2"]: return v_str
    if v_str == "0": return "v1"
    if v_str == "1": return "v2"
    return "v2"

def sanitize_yes_no(v):
    if isinstance(v, str) and v.lower() in ["yes", "y", "true", "1", "t"]: return "Yes"
    if isinstance(v, bool) and v: return "Yes"
    if isinstance(v, int) and v == 1: return "Yes"
    return "No"

def format_time_delta(seconds):
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"

def clean_stem(filename: str) -> str:
    """Safely extracts base stem stripping compound extensions like .wav.npy, .flac.npy, etc."""
    name = os.path.basename(filename)
    compound_exts = [
        ".wav.npy", ".flac.npy", ".mp3.npy", ".ogg.npy", ".m4a.npy",
        ".npy", ".wav", ".flac", ".mp3", ".ogg", ".m4a"
    ]
    for ext in compound_exts:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    return name

def sanitize_output_stem(filename, target_format):
    return clean_stem(filename)

def browse_folder(current_val=""):
    """Interactive Windows folder selection dialog returning clean forward-slash path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        clean_p = str(current_val or "").strip().strip('"')
        init_dir = clean_p if (clean_p and os.path.isdir(clean_p)) else now_dir
        selected = filedialog.askdirectory(initialdir=init_dir, title="เลือกโฟลเดอร์ (Select Folder)")
        root.destroy()
        if selected:
            return os.path.normpath(selected).replace("\\", "/")
    except Exception as e:
        DeltaLogger.wrn(f"Browse folder notice: {e}", to_console=False)
    return current_val

def browse_file(current_val="", filetypes=None):
    """Interactive Windows file selection dialog returning clean forward-slash path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        clean_p = str(current_val or "").strip().strip('"')
        if clean_p and os.path.isfile(clean_p):
            init_dir = os.path.dirname(clean_p)
        elif clean_p and os.path.isdir(clean_p):
            init_dir = clean_p
        else:
            init_dir = now_dir
            
        selected = filedialog.askopenfilename(
            initialdir=init_dir,
            title="เลือกไฟล์ (Select File)",
            filetypes=filetypes or [("All files", "*.*")]
        )
        root.destroy()
        if selected:
            return os.path.normpath(selected).replace("\\", "/")
    except Exception as e:
        DeltaLogger.wrn(f"Browse file notice: {e}", to_console=False)
    return current_val

SKIP_INDEX_DIRS = {"0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature256", "3_feature768", "mute", "__pycache__", ".git"}

def get_model_names(root=weight_root):
    if not os.path.isdir(root): return []
    return sorted(name for name in os.listdir(root) if name.lower().endswith(".pth"))

def get_index_files(root=index_root):
    search_dirs = [root]
    indices_dir = os.path.join(now_dir, "assets", "indices")
    if os.path.isdir(indices_dir) and indices_dir != root: search_dirs.append(indices_dir)
    results = set()
    for s_dir in search_dirs:
        if not os.path.isdir(s_dir): continue
        for cur_root, dirs, files in os.walk(s_dir, topdown=True):
            dirs[:] = [d for d in dirs if d not in SKIP_INDEX_DIRS]
            for file in files:
                if file.lower().endswith(".index") and "trained" not in file.lower():
                    results.add(os.path.normpath(os.path.join(cur_root, file)).replace("\\", "/"))
    return sorted(list(results))

names = get_model_names()
index_paths = get_index_files()
uvr5_names = [name.replace(".pth", "") for name in os.listdir(weight_uvr5_root) if name.endswith(".pth") or "onnx" in name] if os.path.exists(weight_uvr5_root) else []

def get_available_experiments(root="logs"):
    logs_dir = os.path.join(now_dir, root)
    if not os.path.isdir(logs_dir):
        return []
    exps = []
    for item in os.listdir(logs_dir):
        full = os.path.join(logs_dir, item)
        if os.path.isdir(full) and item not in ("mute", "__pycache__", ".git"):
            exps.append(item)
    return sorted(exps)

def inspect_experiment_status(exp_name: str) -> tuple:
    """Inspects an experiment directory and returns readiness, diagnostic badge, recommended version, and count."""
    if not exp_name:
        return False, "⚠️ กรุณาระบุชื่อการทดลอง (Please specify experiment name)", "v2", 0
    
    exp_dir = os.path.join(now_dir, "logs", exp_name)
    if not os.path.exists(exp_dir):
        return False, f"📁 ไม่พบโฟลเดอร์การทดลอง `logs/{exp_name}` (ยังไม่มีข้อมูลสกัด)", "v2", 0
    
    gt_dir = os.path.join(exp_dir, "0_gt_wavs")
    if not os.path.exists(gt_dir): gt_dir = os.path.join(exp_dir, "gt_wavs")
    
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
    
    feat768_dir = os.path.join(exp_dir, "3_feature768")
    feat256_dir = os.path.join(exp_dir, "3_feature256")
    
    has_f768 = os.path.isdir(feat768_dir) and len([f for f in os.listdir(feat768_dir) if f.lower().endswith(".npy")]) > 0
    has_f256 = os.path.isdir(feat256_dir) and len([f for f in os.listdir(feat256_dir) if f.lower().endswith(".npy")]) > 0
    
    detected_ver = "v2" if has_f768 else ("v1" if has_f256 else "v2")
    active_feat_dir = feat768_dir if detected_ver == "v2" else feat256_dir
    
    valid_audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    gt_map = {clean_stem(f): os.path.join(gt_dir, f) for f in (os.listdir(gt_dir) if os.path.isdir(gt_dir) else []) if Path(f).suffix.lower() in valid_audio_exts}
    feat_map = {clean_stem(f): os.path.join(active_feat_dir, f) for f in (os.listdir(active_feat_dir) if os.path.isdir(active_feat_dir) else []) if f.lower().endswith(".npy")}
    f0_map = {clean_stem(f): os.path.join(f0_dir, f) for f in (os.listdir(f0_dir) if os.path.isdir(f0_dir) else []) if f.lower().endswith(".npy")}
    f0nsf_map = {clean_stem(f): os.path.join(f0nsf_dir, f) for f in (os.listdir(f0nsf_dir) if os.path.isdir(f0nsf_dir) else []) if f.lower().endswith(".npy")}
    
    common_keys = sorted(set(gt_map.keys()) & set(feat_map.keys()) & set(f0_map.keys()) & set(f0nsf_map.keys()))
    common_count = len(common_keys)
    
    if common_count > 0:
        badge = (
            f"🟢 **ชุดข้อมูลพร้อมฝึกสอน 100% (Dataset Fully Prepared)**\n\n"
            f"- **เวอร์ชันที่ตรวจพบ (Detected Version):** `{detected_ver.upper()}` ({'768-dim' if detected_ver == 'v2' else '256-dim'})\n"
            f"- **จำนวนคู่ข้อมูลที่สมบูรณ์ (Matched Pairs):** `{common_count}` ไฟล์\n"
            f"- **โฟลเดอร์ GT (0_gt_wavs):** `{len(gt_map)}` ไฟล์ | **Features:** `{len(feat_map)}` ไฟล์\n"
            f"- **โฟลเดอร์ Pitch (2a_f0 / 2b-f0nsf):** `{len(f0_map)}` / `{len(f0nsf_map)}` ไฟล์\n"
            f"💡 *ระบบสามารถเริ่มการฝึกสอนโมเดล (Step 3a) ได้ทันทีโดยไม่ต้องสกัดซ้ำ*"
        )
        return True, badge, detected_ver, common_count
    else:
        badge = (
            f"🟡 **ชุดข้อมูลยังไม่สมบูรณ์ (Incomplete Extracted Dataset)**\n\n"
            f"- **โฟลเดอร์ GT (0_gt_wavs):** `{len(gt_map)}` ไฟล์\n"
            f"- **Features ({'3_feature768' if detected_ver == 'v2' else '3_feature256'}):** `{len(feat_map)}` ไฟล์\n"
            f"- **ความถี่เสียง (2a_f0):** `{len(f0_map)}` ไฟล์ | **Coarse Pitch (2b-f0nsf):** `{len(f0nsf_map)}` ไฟล์\n"
            f"💡 *กรุณากด 'ประมวลผลข้อมูล' และ 'สกัดคุณลักษณะเสียง' ก่อนเริ่มการฝึกสอน*"
        )
        return False, badge, detected_ver, common_count

def get_unified_name(singer, lang_mode):
    singer = str(singer or "").strip() or "Unknown_Singer"
    lang = lang_mode.split(". ")[-1] if lang_mode else "TH"
    return f"{singer}_{lang}"

def update_text_templates(singer_name, lang_mode):
    unified_name = get_unified_name(singer_name, lang_mode)
    lang_short = lang_mode.split(". ")[-1] if lang_mode else "TH"
    voicebank_format = VOICEBANK_FORMATS.get(lang_short, "Thai VCCV")
    
    readme = f"""Engine Development=DELTA SYNTH
Voicer={unified_name}
Voicer Contact={unified_name}
Web=https://patiphatwongyai.wixsite.com/deltasynth
Contact=https://www.youtube.com/@deltaSYNTH0320
------------------------------------------------------------------------
Terms of Use & Guidelines For English
   You are welcome to create content using our works. Please include the hashtag #DELTASYNTH.
   No unauthorized editing, No AI Slop allowed.
------------------------------------------------------------------------
กฎและเงื่อนไขการใช้งาน ภาษาไทย
1. การใช้งาน: ติดแท็ก #DELTASYNTH ทุกครั้ง
2. การดัดแปลง: ไม่อนุญาตให้แก้ไขโดยไม่ได้รับอนุญาต
3. การปกป้อง: ห้ามนำไปใช้สร้างเนื้อหา AI เกรดต่ำ (AI Slop)
4. การใช้งานเชิงพาณิชย์: ห้ามใช้เชิงพาณิชย์โดยไม่ได้รับอนุญาต
"""
    yaml = f"""engine: UTAU\nname: {unified_name}\nimage: Picture/Icon.png\nportrait: Picture/Full.png\nportrait_opacity: 0.5\nportrait_height: 0\nauthor: DELTA SYNTH\n"""
    txt = f"""name={unified_name} {voicebank_format} v.1.0\nsample=sample.wav\nweb=https://patiphatwongyai.wixsite.com/deltasynth\ncontact= DELTA SYNTH\nYoutube=DELTA SYNTH Official\n"""
    return readme, yaml, txt, unified_name

def clear_vram_safe():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with suppress(Exception): torch.cuda.ipc_collect()
        return DeltaLogger.inf("✅ คืนค่าพื้นที่ VRAM เรียบร้อยแล้ว (VRAM Cleared Successfully)", to_console=False)
    except Exception as exc:
        return DeltaLogger.err(f"❌ คืนค่า VRAM ไม่สำเร็จ: {exc}", to_console=False)

def cancel_all_processes():
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = True
    if hasattr(vc, "is_rendering"): vc.is_rendering = False
    cancelled_count = 0
    for p in list(ACTIVE_CHILD_PROCESSES):
        try:
            if p.poll() is None:
                if platform.system() == "Windows":
                    with suppress(Exception):
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True, timeout=5)
                p.terminate()
                sleep(0.1)
                if p.poll() is None: p.kill()
            cancelled_count += 1
        except Exception as e:
            DeltaLogger.wrn(f"Failed to kill process {p.pid}: {e}", to_console=True)
    ACTIVE_CHILD_PROCESSES.clear()
    clear_vram_safe()
    sep_line = "=" * 52
    return f"{sep_line}\n🛑 ยกเลิกกระบวนการทั้งหมดและคืนทรัพยากรเรียบร้อยแล้ว (Cancelled {cancelled_count} processes, VRAM cleared)\n{sep_line}"

def safe_cancel_process():
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = True
    if hasattr(vc, "is_rendering"): vc.is_rendering = False
    clear_vram_safe()
    sep_line = "=" * 52
    return f"{sep_line}\n⚠️ คำสั่งถูกยกเลิกโดยผู้ใช้งานเรียบร้อยแล้ว (Render Cancelled by User)\n🏁 เวลาสิ้นสุด (End Time): {strftime('%H:%M:%S')}\n{sep_line}"

def save_project(base_dir, unified_name, profile_pic, piano_pic, readme_content, yaml_content, txt_content):
    try:
        project_dir = os.path.join(base_dir, unified_name)
        os.makedirs(os.path.join(project_dir, "Picture"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "All File About The Singer Picture"), exist_ok=True)
        
        if profile_pic: shutil.copy(profile_pic, os.path.join(project_dir, "Picture", "Icon.png"))
        if piano_pic: shutil.copy(piano_pic, os.path.join(project_dir, "Picture", "Full.png"))
        
        with open(os.path.join(project_dir, "readme.txt"), "w", encoding="utf-8") as f: f.write(readme_content)
        with open(os.path.join(project_dir, "character.yaml"), "w", encoding="utf-8") as f: f.write(yaml_content)
        with open(os.path.join(project_dir, "character.txt"), "w", encoding="utf-8") as f: f.write(txt_content)
        with open(os.path.join(project_dir, "oto.ini"), "w", encoding="utf-8") as f: f.write("sample.wav=,,,,,\n")
            
        return "### กระบวนการที่ 1: จัดการรายละเอียดของตัวละคร OpenUtau 🟢", "**สถานะ:** เริ่มต้นโปรเจกต์เสร็จสมบูรณ์", DeltaLogger.inf("✅ บันทึกข้อมูลเรียบร้อยแล้ว (Saved Successfully)", to_console=False), project_dir
    except Exception as e:
        return "### กระบวนการที่ 1: จัดการรายละเอียดของตัวละคร OpenUtau 🔴", f"**สถานะ:** ขัดข้อง (Error) - {str(e)}", DeltaLogger.err(f"❌ เกิดข้อผิดพลาด: {str(e)}", to_console=False), base_dir

def get_vc_wrapper(sid, protect0, protect1):
    if sid:
        DeltaLogger.model(sid, "Active Voice Conversion Model Selected")
    else:
        DeltaLogger.inf("🔄 กำลังคืนค่าและยกเลิกโมเดล (Unloading Model)", to_console=False)
    return vc.get_vc(sid, protect0, protect1)

def single_infer_wrapper(spk_item, input_audio0, vc_transform0, f0_file, f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    start_time_sec = time.time()
    start_timestamp = strftime("%H:%M:%S")
    sep_line = "=" * 52

    clean_input = str(input_audio0 or "").strip().strip('"')
    if not clean_input or not os.path.exists(clean_input):
        err_msg = (
            f"{sep_line}\n"
            f"❌ [ERR] ไม่พบไฟล์เสียงต้นฉบับที่ระบุ (Source Audio Not Found): {clean_input}\n"
            f"⏰ เวลาเริ่มเรนเดอร์: {start_timestamp} | เวลาเรนเดอร์เสร็จแล้ว: {strftime('%H:%M:%S')}\n"
            f"{sep_line}"
        )
        return err_msg, None

    info_log, opt_audio = vc.vc_single(
        spk_item, clean_input, vc_transform0, f0_file, f0method0,
        file_index1, file_index2, index_rate1, filter_radius0,
        resample_sr0, rms_mix_rate0, protect0
    )

    elapsed = format_time_delta(time.time() - start_time_sec)
    end_timestamp = strftime("%H:%M:%S")
    fname = os.path.basename(clean_input)

    final_log = (
        f"{sep_line}\n"
        f"🎙️ ลำดับคิว: 1/1 | ชื่อไฟล์: {fname} | เวลาเริ่มเรนเดอร์: {start_timestamp} | เวลาเรนเดอร์เสร็จแล้ว: {end_timestamp}\n"
        f"สถานะ: {info_log}\n"
        f"{sep_line}"
    )
    clear_vram_safe()
    return final_log, opt_audio

def process_batch_wrapper(sid, base_dir, unified_name, index_path, auto_index, f0_method, format1, language_mode, resample_sr, rms_mix_rate, protect, filter_radius, index_rate, vc_transform, spk_item):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    source_dir = SOURCE_DIRS.get(language_mode, "")
    target_dir = os.path.join(base_dir, unified_name)
    format1 = str(format1 or "wav").lower().strip()
    sep_line = "=" * 52

    start_timestamp = strftime("%H:%M:%S")
    start_time_sec = time.time()

    if not os.path.exists(source_dir):
        msg = (
            f"{sep_line}\n"
            f"❌ [ERR] ไม่พบโฟลเดอร์ต้นฉบับ (Source Directory Not Found): {source_dir}\n"
            f"⏰ เวลาเริ่มเรนเดอร์: {start_timestamp} | เวลาเรนเดอร์เสร็จแล้ว: {strftime('%H:%M:%S')}\n"
            f"{sep_line}"
        )
        yield msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        return

    os.makedirs(target_dir, exist_ok=True)
    oto_path_src = os.path.join(source_dir, "oto.ini")
    oto_path_dst = os.path.join(target_dir, "oto.ini")
    if os.path.exists(oto_path_src) and not os.path.exists(oto_path_dst):
        shutil.copy(oto_path_src, oto_path_dst)

    valid_exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    all_source_files = sorted([
        f for f in os.listdir(source_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    total_source_count = len(all_source_files)

    files_to_render = []
    already_done_count = 0

    for fname in all_source_files:
        src_full = os.path.join(source_dir, fname)
        clean_stem_name = sanitize_output_stem(fname, format1)
        expected_dst = os.path.join(target_dir, f"{clean_stem_name}.{format1}")

        if os.path.exists(expected_dst) and os.path.getsize(expected_dst) > 1024:
            already_done_count += 1
        else:
            files_to_render.append(src_full)

    remaining_count = len(files_to_render)

    init_msg = (
        f"{sep_line}\n"
        f"📊 ตรวจสอบความพร้อมคลังเสียง (Pre-Flight Voicebank Audit)\n"
        f"{sep_line}\n"
        f"📁 โฟลเดอร์ต้นฉบับ: {source_dir}\n"
        f"🎯 โฟลเดอร์ปลายทาง: {target_dir}\n"
        f"📑 ไฟล์ต้นแบบทั้งหมด: {total_source_count} ไฟล์\n"
        f"✅ ไฟล์ที่สานต่อสำเร็จแล้วเดิม (Already Completed): {already_done_count} ไฟล์\n"
        f"⏳ ไฟล์ที่ต้องเรนเดอร์ให้เสร็จสิ้น (Pending Tasks): {remaining_count} ไฟล์\n"
        f"⏰ เวลาเริ่มต้น: {start_timestamp}\n"
        f"{sep_line}"
    )
    yield init_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    if remaining_count == 0:
        elapsed = format_time_delta(time.time() - start_time_sec)
        done_msg = (
            f"{init_msg}\n\n"
            f"{sep_line}\n"
            f"🟢 คลังเสียงนี้สานต่องานเดิมจนเสร็จสมบูรณ์ครบถ้วน 100% แล้ว\n"
            f"⏱️ ระยะเวลาที่ใช้ (Elapsed Time): {elapsed} (MM:SS)\n"
            f"🏁 เวลาเรนเดอร์เสร็จแล้ว: {strftime('%H:%M:%S')}\n"
            f"📦 คุณสามารถกดปุ่มแพ็คไฟล์ .zip ได้ทันที (Ready for ZIP Packing)\n"
            f"{sep_line}"
        )
        yield done_msg, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
        clear_vram_safe()
        return

    selected_index = index_path.strip() if index_path else (auto_index or "")
    generator = vc.vc_multi(
        sid=sid,
        dir_path=source_dir,
        opt_root=target_dir,
        paths=files_to_render,
        f0_up_key=vc_transform,
        f0_method=f0_method,
        file_index=selected_index,
        file_index2="",
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        format1=format1
    )

    log_history = []
    current_done = 0
    last_log = ""
    file_start_time = strftime("%H:%M:%S")

    for i, log in enumerate(generator):
        if CANCEL_REQUESTED:
            elapsed = format_time_delta(time.time() - start_time_sec)
            cancel_msg = (
                f"{sep_line}\n"
                f"🛑 คำสั่งถูกยกเลิกโดยผู้ใช้งาน (Render Cancelled by User)\n"
                f"⏱️ ระยะเวลาที่ใช้ไป: {elapsed} (MM:SS)\n"
                f"🏁 เวลาเรนเดอร์เสร็จแล้ว: {strftime('%H:%M:%S')}\n"
                f"{sep_line}"
            )
            yield cancel_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            clear_vram_safe()
            return

        last_log = str(log)
        current_done += 1
        total_finished = already_done_count + current_done
        pct = (total_finished / total_source_count * 100) if total_source_count > 0 else 100
        elapsed = format_time_delta(time.time() - start_time_sec)
        
        file_end_time = strftime("%H:%M:%S")
        fname = os.path.basename(files_to_render[i])

        status_line = f"▶ ลำดับคิว: {total_finished}/{total_source_count} | ชื่อไฟล์: {fname} | เวลาเริ่มเรนเดอร์: {file_start_time} | เวลาเรนเดอร์เสร็จแล้ว: {file_end_time} | สถานะ: {last_log}"
        log_history.insert(0, status_line)
        if len(log_history) > 12:
            log_history.pop()

        display_text = (
            f"{sep_line}\n"
            f"🟢 สถานะการสานต่อและเรนเดอร์ (Live Render Status) - ความคืบหน้า {pct:.1f}%\n"
            f"⏱️ ใช้เวลาไปแล้ว (Elapsed): {elapsed} (MM:SS)\n"
            f"{sep_line}\n"
            + "\n".join(log_history) + "\n"
            f"{sep_line}"
        )
        yield display_text, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        
        file_start_time = strftime("%H:%M:%S")

    elapsed = format_time_delta(time.time() - start_time_sec)
    end_timestamp = strftime("%H:%M:%S")

    if "ระงับ" in last_log or "ยกเลิก" in last_log or "aborted" in last_log or CANCEL_REQUESTED:
        final_msg = (
            f"{sep_line}\n"
            f"⚠️ กระบวนการถูกหยุดกลางคัน (Render Interrupted / Aborted)\n"
            f"⏱️ ระยะเวลาที่ใช้ไป: {elapsed} (MM:SS)\n"
            f"🏁 เวลาเรนเดอร์เสร็จแล้ว: {end_timestamp}\n"
            f"{sep_line}"
        )
        yield final_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    else:
        final_msg = (
            f"{sep_line}\n"
            f"🟢 เสร็จสมบูรณ์ 100%! สานต่องานและเรนเดอร์คลังเสียงสำเร็จทั้งหมด {total_source_count} ไฟล์\n"
            f"✨ ทำการเรนเดอร์ใหม่รอบนี้: {current_done} ไฟล์ | ข้ามไฟล์เดิมที่เสร็จแล้ว: {already_done_count} ไฟล์\n"
            f"⏰ เวลาเริ่มต้น: {start_timestamp} | เวลาเรนเดอร์เสร็จแล้ว: {end_timestamp}\n"
            f"⏱️ รวมระยะเวลาที่ใช้ไปทั้งหมด: {elapsed} (MM:SS)\n"
            f"📦 คุณสามารถกดปุ่มแพ็คไฟล์ .zip ด้านล่างได้แล้ว\n"
            f"{sep_line}"
        )
        yield final_msg, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

    clear_vram_safe()

def batch_render_tracker(sid, dir_input, opt_input, inputs, vc_transform, f0method, file_index, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, format1):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    start_time_sec = time.time()
    start_timestamp = strftime("%H:%M:%S")
    sep_line = "=" * 52
    format1 = str(format1 or "wav").lower().strip()

    dir_input_clean = str(dir_input or "").strip().strip('"')
    opt_input_clean = str(opt_input or "").strip().strip('"')

    if not os.path.exists(dir_input_clean) and not inputs:
        yield f"{sep_line}\n❌ [ERR] ไม่พบโฟลเดอร์หรือไฟล์ต้นฉบับที่ระบุ (Source Directory / Input Not Found)\n{sep_line}"
        return

    os.makedirs(opt_input_clean, exist_ok=True)
    
    if os.path.exists(dir_input_clean):
        oto_src = os.path.join(dir_input_clean, "oto.ini")
        oto_dst = os.path.join(opt_input_clean, "oto.ini")
        if os.path.exists(oto_src) and not os.path.exists(oto_dst):
            shutil.copy(oto_src, oto_dst)

    valid_exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    source_files = []
    if inputs:
        source_files = [f.name if hasattr(f, "name") else str(f) for f in inputs]
    elif os.path.exists(dir_input_clean):
        source_files = [os.path.join(dir_input_clean, f) for f in os.listdir(dir_input_clean) if os.path.splitext(f)[1].lower() in valid_exts]

    total_count = len(source_files)
    files_to_render = []
    already_done = 0

    for src_path in source_files:
        clean_stem_name = sanitize_output_stem(os.path.basename(src_path), format1)
        expected_dst = os.path.join(opt_input_clean, f"{clean_stem_name}.{format1}")
        if os.path.exists(expected_dst) and os.path.getsize(expected_dst) > 1024:
            already_done += 1
        else:
            files_to_render.append(src_path)

    pending_count = len(files_to_render)

    yield (
        f"{sep_line}\n"
        f"⏳ เริ่มกระบวนการตรวจสอบการสานต่อไฟล์ (Smart Batch Resume Initializing)...\n"
        f"📊 ไฟล์ทั้งหมด: {total_count} | สำเร็จแล้วเดิม (Skip): {already_done} | ค้างทำต่อ (Pending): {pending_count}\n"
        f"⏰ เวลาเริ่มต้น: {start_timestamp}\n"
        f"{sep_line}"
    )

    if pending_count == 0:
        elapsed = format_time_delta(time.time() - start_time_sec)
        yield (
            f"{sep_line}\n"
            f"🎉 ทุกไฟล์ถูกเรนเดอร์สานต่อจนเสร็จสมบูรณ์แล้วทั้งหมด 100% (All Files Already Processed)\n"
            f"⏱️ ระยะเวลาที่ใช้: {elapsed} (MM:SS) | 🏁 เวลาเรนเดอร์เสร็จแล้ว: {strftime('%H:%M:%S')}\n"
            f"{sep_line}"
        )
        clear_vram_safe()
        return

    selected_index = file_index.strip() if file_index else (file_index2 or "")
    generator = vc.vc_multi(
        sid=sid,
        dir_path=dir_input_clean if os.path.exists(dir_input_clean) else "",
        opt_root=opt_input_clean,
        paths=files_to_render,
        f0_up_key=vc_transform,
        f0_method=f0method,
        file_index=selected_index,
        file_index2="",
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        format1=format1
    )

    done_in_loop = 0
    for log in generator:
        if CANCEL_REQUESTED:
            yield f"{sep_line}\n🛑 คำสั่งถูกยกเลิกโดยผู้ใช้งาน (Cancelled)\n{sep_line}"
            clear_vram_safe()
            return
        done_in_loop += 1
        total_finished = already_done + done_in_loop
        pct = (total_finished / total_count * 100) if total_count > 0 else 100
        yield f"🟢 กำลังประมวลผล (Processing): {total_finished}/{total_count} ({pct:.1f}%)\n{log}"

    elapsed = format_time_delta(time.time() - start_time_sec)
    yield f"{sep_line}\n🟢 การเรนเดอร์แบบกลุ่มเสร็จสมบูรณ์เรียบร้อยแล้ว {total_count} ไฟล์ (Elapsed: {elapsed})\n{sep_line}"
    clear_vram_safe()

def change_choices():
    current_names = get_model_names()
    current_indices = get_index_files()
    return {"choices": sorted(current_names), "__type__": "update"}, {"choices": sorted(current_indices), "__type__": "update"}

def clean(): return {"value": "", "__type__": "update"}

def export_onnx(ModelPath, ExportedPath):
    from infer.modules.onnx.export import export_onnx as eo
    eo(ModelPath, ExportedPath)

def if_done(done, p):
    while 1:
        if p.poll() is None: sleep(0.5)
        else: break
    done[0] = True

def if_done_multi(done, ps):
    while 1:
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1: break
    done[0] = True

def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sanitize_sr(sr)
    sr = sr_dict.get(str(sr).lower().strip(), 40000)
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    os.makedirs(exp_path, exist_ok=True)
    log_file = os.path.join(exp_path, "preprocess.log")
    with open(log_file, "w", encoding="utf-8") as f: f.write("")
    
    per = 3.0 if getattr(config, "is_half", False) else 3.7
    cmd = [
        PYTHON_EXE, "infer/modules/train/preprocess.py",
        str(trainset_dir), str(sr), str(n_p), str(exp_path),
        str(getattr(config, "noparallel", False)), str(per)
    ]
    logger.info("Executing preprocess: %s", " ".join(cmd))
    p = Popen(cmd, cwd=now_dir)
    ACTIVE_CHILD_PROCESSES.add(p)
    done = [False]
    threading.Thread(target=if_done, args=(done, p,)).start()
    
    while True:
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                yield f.read()
        sleep(1)
        if done[0]: break
        
    ACTIVE_CHILD_PROCESSES.discard(p)
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            log = f.read()
    else:
        log = "Preprocessing completed."
    yield log

def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    if_f0 = sanitize_bool(if_f0)
    version19 = sanitize_version(version19)
    
    gpus_list = str(gpus).split("-") if gpus else ["0"]
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    os.makedirs(exp_path, exist_ok=True)
    log_file = os.path.join(exp_path, "extract_f0_feature.log")
    with open(log_file, "w", encoding="utf-8") as f: f.write("")

    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = [PYTHON_EXE, "infer/modules/train/extract/extract_f0_print.py", exp_path, str(n_p), str(f0method)]
            logger.info("Executing F0 extraction: %s", " ".join(cmd))
            p = Popen(cmd, cwd=now_dir)
            ACTIVE_CHILD_PROCESSES.add(p)
            done = [False]
            threading.Thread(target=if_done, args=(done, p,)).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe_list = str(gpus_rmvpe).split("-")
                leng = len(gpus_rmvpe_list)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe_list):
                    cmd = [
                        PYTHON_EXE, "infer/modules/train/extract/extract_f0_rmvpe.py",
                        str(leng), str(idx), str(n_g), exp_path, str(getattr(config, "is_half", False))
                    ]
                    p = Popen(cmd, cwd=now_dir)
                    ACTIVE_CHILD_PROCESSES.add(p)
                    ps.append(p)
                done = [False]
                threading.Thread(target=if_done_multi, args=(done, ps,)).start()
            else:
                cmd = [PYTHON_EXE, "infer/modules/train/extract/extract_f0_rmvpe_dml.py", exp_path]
                p = Popen(cmd, cwd=now_dir)
                ACTIVE_CHILD_PROCESSES.add(p)
                p.wait()
                ACTIVE_CHILD_PROCESSES.discard(p)
                done = [True]
                
        while True:
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    yield f.read()
            sleep(1)
            if done[0]: break

    leng = len(gpus_list)
    ps = []
    for idx, n_g in enumerate(gpus_list):
        cmd = [
            PYTHON_EXE, "infer/modules/train/extract_feature_print.py",
            str(getattr(config, "device", "cpu")), str(leng), str(idx), str(n_g),
            exp_path, str(version19)
        ]
        p = Popen(cmd, cwd=now_dir)
        ACTIVE_CHILD_PROCESSES.add(p)
        ps.append(p)
    done = [False]
    threading.Thread(target=if_done_multi, args=(done, ps,)).start()
    
    while True:
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                yield f.read()
        sleep(1)
        if done[0]: break
        
    for p in ps: ACTIVE_CHILD_PROCESSES.discard(p)
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            log = f.read()
    else:
        log = "Feature extraction completed."
    yield log

def get_pretrained_models(path_str, f0_str, sr2):
    g_path = os.path.join(now_dir, "assets", f"pretrained{path_str}", f"{f0_str}G{sr2}.pth")
    d_path = os.path.join(now_dir, "assets", f"pretrained{path_str}", f"{f0_str}D{sr2}.pth")
    g_exist = os.path.isfile(g_path)
    d_exist = os.path.isfile(d_path)
    if not g_exist: logger.warning(f"Not exist: {g_path}")
    if not d_exist: logger.warning(f"Not exist: {d_path}")
    return (g_path if g_exist else "", d_path if d_exist else "")

def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)

def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1": sr2 = "40k"
    to_return_sr2 = ({"choices": ["40k", "48k"], "__type__": "update", "value": sr2} if version19 == "v1" else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2})
    f0_str = "f0" if if_f0_3 else ""
    return (*get_pretrained_models(path_str, f0_str, sr2), to_return_sr2)

def change_f0(if_f0_3, sr2, version19):
    path_str = "" if version19 == "v1" else "_v2"
    return ({"visible": if_f0_3, "__type__": "update"}, {"visible": if_f0_3, "__type__": "update"}, *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2))

def click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19):
    """Executes model training with robust stem cleaning, auto-version resolution, and failure gating."""
    sr2 = sanitize_sr(sr2)
    if_f0_3 = sanitize_bool(if_f0_3)
    version19 = sanitize_version(version19)
    if_save_latest13 = sanitize_yes_no(if_save_latest13)
    if_cache_gpu17 = sanitize_yes_no(if_cache_gpu17)
    if_save_every_weights18 = sanitize_yes_no(if_save_every_weights18)

    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    if not os.path.isdir(gt_wavs_dir):
        gt_wavs_dir = os.path.join(exp_dir, "gt_wavs")
        
    feat768 = os.path.join(exp_dir, "3_feature768")
    feat256 = os.path.join(exp_dir, "3_feature256")
    
    if os.path.isdir(feat768) and len([f for f in os.listdir(feat768) if f.lower().endswith(".npy")]) > 0 and not (os.path.isdir(feat256) and len([f for f in os.listdir(feat256) if f.lower().endswith(".npy")]) > 0):
        actual_ver = "v2"
    elif os.path.isdir(feat256) and len([f for f in os.listdir(feat256) if f.lower().endswith(".npy")]) > 0 and not (os.path.isdir(feat768) and len([f for f in os.listdir(feat768) if f.lower().endswith(".npy")]) > 0):
        actual_ver = "v1"
    else:
        actual_ver = version19
        
    feature_dir = feat768 if actual_ver == "v2" else feat256
    f0_dir = os.path.join(exp_dir, "2a_f0")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")

    valid_audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    gt_map = {clean_stem(f): os.path.join(gt_wavs_dir, f) for f in (os.listdir(gt_wavs_dir) if os.path.isdir(gt_wavs_dir) else []) if Path(f).suffix.lower() in valid_audio_exts}
    feat_map = {clean_stem(f): os.path.join(feature_dir, f) for f in (os.listdir(feature_dir) if os.path.isdir(feature_dir) else []) if f.lower().endswith(".npy")}

    if if_f0_3:
        f0_map = {clean_stem(f): os.path.join(f0_dir, f) for f in (os.listdir(f0_dir) if os.path.isdir(f0_dir) else []) if f.lower().endswith(".npy")}
        f0nsf_map = {clean_stem(f): os.path.join(f0nsf_dir, f) for f in (os.listdir(f0nsf_dir) if os.path.isdir(f0nsf_dir) else []) if f.lower().endswith(".npy")}
        common_stems = sorted(set(gt_map.keys()) & set(feat_map.keys()) & set(f0_map.keys()) & set(f0nsf_map.keys()))
    else:
        f0_map, f0nsf_map = {}, {}
        common_stems = sorted(set(gt_map.keys()) & set(feat_map.keys()))

    if len(common_stems) == 0:
        err_report = (
            f"❌ ไม่พบชุดข้อมูลที่สมบูรณ์สำหรับการฝึกโมเดลใน logs/{exp_dir1}\n"
            f"---------------------------------------------------\n"
            f"📊 ผลการสแกนไดเรกทอรี (Directory Scan Audit):\n"
            f"  - ไฟล์เสียง GT (0_gt_wavs): {len(gt_map)} ไฟล์\n"
            f"  - ฟีเจอร์ HuBERT ({os.path.basename(feature_dir)}): {len(feat_map)} ไฟล์ (เวอร์ชัน: {actual_ver.upper()})\n"
            f"  - ความถี่ F0 (2a_f0): {len(f0_map)} ไฟล์\n"
            f"  - Coarse Pitch (2b-f0nsf): {len(f0nsf_map)} ไฟล์\n"
            f"💡 คำแนะนำ: หากคุณสกัดฟีเจอร์เป็น v2 โปรดตรวจสอบว่ามีโฟลเดอร์ 3_feature768 หรือหากเป็น v1 ให้สกัดลง 3_feature256"
        )
        logger.error(err_report)
        yield err_report
        return False, err_report

    opt = []
    for stem in common_stems:
        wav_p = gt_map[stem].replace("\\", "/")
        fea_p = feat_map[stem].replace("\\", "/")
        if if_f0_3:
            f0_p = f0_map[stem].replace("\\", "/")
            f0nsf_p = f0nsf_map[stem].replace("\\", "/")
            opt.append(f"{wav_p}|{fea_p}|{f0_p}|{f0nsf_p}|{spk_id5}")
        else:
            opt.append(f"{wav_p}|{fea_p}|{spk_id5}")

    fea_dim = 256 if actual_ver == "v1" else 768
    mute_dir = os.path.join(now_dir, "logs", "mute")
    if os.path.isdir(mute_dir):
        mute_wav = os.path.join(mute_dir, "0_gt_wavs", f"mute{sr2}.wav").replace("\\", "/")
        mute_fea = os.path.join(mute_dir, f"3_feature{fea_dim}", "mute.npy").replace("\\", "/")
        mute_f0 = os.path.join(mute_dir, "2a_f0", "mute.wav.npy").replace("\\", "/")
        mute_f0nsf = os.path.join(mute_dir, "2b-f0nsf", "mute.wav.npy").replace("\\", "/")
        if os.path.exists(mute_wav) and os.path.exists(mute_fea):
            for _ in range(2):
                if if_f0_3 and os.path.exists(mute_f0) and os.path.exists(mute_f0nsf):
                    opt.append(f"{mute_wav}|{mute_fea}|{mute_f0}|{mute_f0nsf}|{spk_id5}")
                elif not if_f0_3:
                    opt.append(f"{mute_wav}|{mute_fea}|{spk_id5}")

    shuffle(opt)
    filelist_path = os.path.join(exp_dir, "filelist.txt")
    with open(filelist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(opt))
    logger.info("Generated filelist.txt with %d samples", len(opt))

    config_path = "v1/%s.json" % sr2 if actual_ver == "v1" or sr2 == "40k" else "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config.json_config[config_path], f, ensure_ascii=False, indent=4, sort_keys=True)
            f.write("\n")

    save_latest_flag = 1 if if_save_latest13 == "Yes" else 0
    cache_gpu_flag = 1 if if_cache_gpu17 == "Yes" else 0
    save_every_weights_flag = 1 if if_save_every_weights18 == "Yes" else 0

    train_cmd = [
        PYTHON_EXE, "infer/modules/train/train.py",
        "-e", exp_dir1,
        "-sr", str(sr2),
        "-f0", "1" if if_f0_3 else "0",
        "-bs", str(batch_size12),
        "-te", str(total_epoch11),
        "-se", str(save_epoch10),
        "-l", str(save_latest_flag),
        "-c", str(cache_gpu_flag),
        "-sw", str(save_every_weights_flag),
        "-v", str(actual_ver)
    ]
    if gpus16: train_cmd.extend(["-g", str(gpus16)])
    if pretrained_G14: train_cmd.extend(["-pg", str(pretrained_G14)])
    if pretrained_D15: train_cmd.extend(["-pd", str(pretrained_D15)])

    logger.info("Executing training: %s", " ".join(train_cmd))
    log_file = os.path.join(exp_dir, "train.log")
    open(log_file, "w").close()

    p = Popen(train_cmd, cwd=now_dir)
    ACTIVE_CHILD_PROCESSES.add(p)
    done = [False]
    threading.Thread(target=if_done, args=(done, p,)).start()

    while True:
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                if content: yield content
        sleep(1)
        if done[0]: break

    ACTIVE_CHILD_PROCESSES.discard(p)
    exit_code = p.poll()
    if exit_code != 0:
        err_msg = f"❌ การฝึกสอนโมเดลล้มเหลว (Training Process Terminated with Code: {exit_code})"
        logger.error(err_msg)
        yield err_msg
        return False, err_msg

    succ_msg = "🟢 การฝึกสอนเสร็จสมบูรณ์ (Training finished)"
    yield succ_msg
    return True, succ_msg

def train_index(exp_dir1, version19):
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    
    feat768 = os.path.join(exp_dir, "3_feature768")
    feat256 = os.path.join(exp_dir, "3_feature256")
    
    if os.path.isdir(feat768) and len([f for f in os.listdir(feat768) if f.lower().endswith(".npy")]) > 0:
        actual_ver = "v2"
        feature_dir = feat768
    elif os.path.isdir(feat256) and len([f for f in os.listdir(feat256) if f.lower().endswith(".npy")]) > 0:
        actual_ver = "v1"
        feature_dir = feat256
    else:
        actual_ver = sanitize_version(version19)
        feature_dir = feat768 if actual_ver == "v2" else feat256

    if not os.path.exists(feature_dir):
        yield "❌ กรุณาสกัดคุณลักษณะเสียงก่อน (Please extract features first!)"
        return False, "Feature folder missing"
        
    listdir_res = [f for f in os.listdir(feature_dir) if f.lower().endswith(".npy")]
    if len(listdir_res) == 0:
        yield "❌ ไม่พบไฟล์ฟีเจอร์ในโฟลเดอร์ (No feature files found)"
        return False, "Feature files empty"

    infos, npys = [], []
    for name in sorted(listdir_res):
        try:
            npys.append(np.load(os.path.join(feature_dir, name)))
        except Exception as e:
            logger.warning(f"Could not load {name}: {e}")

    if len(npys) == 0:
        yield "❌ ไม่สามารถโหลดไฟล์ฟีเจอร์ได้ (Failed to load any feature files)"
        return False, "Failed to load features"

    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5:
        infos.append("กำลังปรับลดขนาดด้วย kmeans (Trying kmeans %s shape to 10k centers.)" % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (MiniBatchKMeans(n_clusters=10000, verbose=True, batch_size=256 * config.n_cpu, compute_labels=False, init="random").fit(big_npy).cluster_centers_)
        except Exception:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)
    n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39))
    infos.append(f"สร้างดัชนีด้วย {big_npy.shape[0]} vectors, IVF clusters: {n_ivf}")
    yield "\n".join(infos)

    index = faiss.index_factory(256 if actual_ver == "v1" else 768, f"IVF{n_ivf},Flat")
    infos.append("กำลังฝึกสอนดัชนี (Training Index)...")
    yield "\n".join(infos)

    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(index, os.path.join(exp_dir, f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{actual_ver}.index"))

    infos.append("กำลังเพิ่มข้อมูลลงในดัชนี (Adding Data)...")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])

    added_index_name = f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{actual_ver}.index"
    added_index_log = os.path.join(exp_dir, added_index_name)
    faiss.write_index(index, added_index_log)

    indices_dir = os.path.join(now_dir, "assets", "indices")
    os.makedirs(indices_dir, exist_ok=True)
    added_index_assets = os.path.join(indices_dir, added_index_name)
    try:
        shutil.copyfile(added_index_log, added_index_assets)
    except Exception as e:
        logger.warning(f"Copy index to {added_index_assets} failed: {e}")

    succ_log = f"🟢 สร้างดัชนีเสร็จสมบูรณ์: {added_index_log}"
    infos.append(succ_log)
    yield "\n".join(infos)
    return True, succ_log

def train1key(exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19, gpus_rmvpe):
    """Orchestrates end-to-end training with strict failure gates preventing phantom indexing."""
    # Data Sanitization
    sr2 = sanitize_sr(sr2)
    if_f0_3 = sanitize_bool(if_f0_3)
    version19 = sanitize_version(version19)
    if_save_latest13 = sanitize_yes_no(if_save_latest13)
    if_cache_gpu17 = sanitize_yes_no(if_cache_gpu17)
    if_save_every_weights18 = sanitize_yes_no(if_save_every_weights18)
    
    infos = []
    def get_info_str(strr):
        infos.append(strr)
        if len(infos) > 12: infos.pop(0)
        return "\n".join(infos)

    is_ready, audit_msg, auto_ver, common_count = inspect_experiment_status(exp_dir1)
    
    if is_ready and common_count > 0:
        yield get_info_str(f"📦 ตรวจพบชุดข้อมูลที่สกัดพร้อมแล้วใน logs/{exp_dir1}! เริ่มต้นการฝึกสอนโมเดลทันทีจากไฟล์ที่มีอยู่ (ตรวจพบ {auto_ver.upper()}, {common_count} คู่ข้อมูล)")
        version19 = auto_ver
    else:
        yield get_info_str("step1: กำลังประมวลผลข้อมูล (Processing data)...")
        for log in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7):
            yield get_info_str(log)
            
        yield get_info_str("step2: กำลังสกัดระดับเสียงและคุณลักษณะ (Extracting pitch & features)...")
        for log in extract_f0_feature(gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe):
            yield get_info_str(log)

    yield get_info_str("step3a: กำลังฝึกสอนโมเดล (Training model)...")
    train_success = False
    last_train_log = ""
    for log in click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19):
        if isinstance(log, tuple):
            train_success, last_train_log = log
        else:
            last_train_log = str(log)
            if "การฝึกสอนเสร็จสมบูรณ์" in last_train_log:
                train_success = True
            yield get_info_str(last_train_log)

    if not train_success or "ไม่พบชุดข้อมูลที่สมบูรณ์" in last_train_log or "ล้มเหลว" in last_train_log:
        yield get_info_str(
            "🛑 ระงับกระบวนการสร้างดัชนีเนื่องจากการฝึกสอนโมเดลไม่สำเร็จ (Halted index creation to prevent phantom index)\n"
            "💡 โปรดตรวจสอบข้อผิดพลาดด้านบนเพื่อแก้ไขปัญหาก่อนดำเนินการต่อ"
        )
        return

    yield get_info_str("การฝึกสอนโมเดลเสร็จสิ้น. กำลังสร้างดัชนี (Building Index)...")
    for log in train_index(exp_dir1, version19):
        yield get_info_str(str(log))
        
    yield get_info_str("🟢 เสร็จสิ้นกระบวนการทั้งหมด! (All processes completed!)")

def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")): return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r", encoding="utf-8") as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except Exception:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}

F0GPUVisible = getattr(config, "dml", False) == False
def change_f0_method(f0method8):
    return {"visible": (F0GPUVisible if f0method8 == "rmvpe_gpu" else False), "__type__": "update"}

# DELTA SYNTH Corporate UI Styling
custom_theme_css = """
body, .gradio-container {
    font-family: 'Leelawadee UI', 'Segoe UI', Tahoma, sans-serif !important;
    background-color: #1A1A1A !important;
    color: #F0F0F0 !important;
}
.gr-button-primary {
    background-color: #CC2200 !important;
    color: #FFFFFF !important;
    border: none !important;
}
.gr-button-primary:hover {
    background-color: #FF4422 !important;
}
.gr-button-primary:active {
    background-color: #991100 !important;
}
.gr-box, div.gradio-container > div {
    background-color: #242424 !important;
    border: 1px solid #333333 !important;
}
"""

with gr.Blocks(title="DELTA RVC Studio", css=custom_theme_css) as app:
    gr.Markdown("## 🎙️ DELTA SYNTH - Smart AI Voice Studio (RVC WebUI Engine)")
    gr.Markdown(
        value="ระบบประมวลผลเสียงปัญญาประดิษฐ์ระดับมืออาชีพ รองรับงานแปลงเสียง, UVR5 แยกเสียงร้อง, ฝึกสอนโมเดล RVC & OpenUtau Voicebank<br>"
              "Professional Bilingual Voice Studio Engine by DELTA SYNTH & All Code Agentic AI."
    )
    with gr.Tabs():
        with gr.TabItem("📖 คู่มือการใช้งาน (User Guide & Tips)"):
            gr.Markdown(
                """
                ### 🚀 ยินดีต้อนรับสู่ DELTA SYNTH RVC Studio
                - **การอนุมานโมเดล (Inference):** เลือกโมเดลเสียง `.pth` และไฟล์ดัชนี `.index` เพื่อแปลงเสียงเดี่ยว หรือแปลงทั้งโฟลเดอร์สำหรับทำ Voicebank
                - **ระบบ Smart Batch Resume:** ข้ามไฟล์ที่เรนเดอร์เสร็จแล้วโดยอัตโนมัติ สานต่องานเดิมได้ทันทีโดยไม่เรนเดอร์ทับ
                - **การฝึกสอนโมเดล (Training):** มีระบบตรวจสอบข้อมูลอัตโนมัติ ป้องกันการสร้างดัชนีหลอก (Phantom Indexing) และจับคู่นามสกุลไฟล์ได้อย่างแม่นยำ
                - **OpenUtau & DiffSinger:** ออกแบบโครงสร้างรองรับคลังเสียงภาษาอังกฤษ (Arpasing), ภาษาไทย (VCCV), และภาษาญี่ปุ่น (VCV)
                """
            )

        with gr.TabItem("🎙️ การอนุมานโมเดล (Model Inference)"):
            with gr.Row():
                sid0 = gr.Dropdown(label="เลือกเสียงเป้าหมาย (Select Target Voice)", choices=sorted(names))
                with gr.Column():
                    refresh_button = gr.Button("รีเฟรชข้อมูลเสียงและดัชนี (Refresh Voice & Index Data)", variant="primary")
                    clean_button = gr.Button("คืนค่าหน่วยความจำ VRAM (Clear VRAM)", variant="secondary")
                spk_item = gr.Slider(minimum=0, maximum=2333, step=1, label="หมายเลขผู้พูด (Speaker ID)", value=0, visible=False, interactive=True)
                clean_button.click(fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean")
            
            with gr.Tabs():
                with gr.TabItem("การประมวลผลไฟล์เดี่ยว (Single File Processing)"):
                    with gr.Group():
                        with gr.Row():
                            with gr.Column():
                                vc_transform0 = gr.Number(label="ปรับระดับเสียง (Pitch Shifting: Semitones, +12 Octave Up, -12 Down)", value=0)
                                with gr.Row():
                                    input_audio0 = gr.Textbox(label="เส้นทางไฟล์เสียงต้นฉบับ (Source Audio Path)", placeholder=r"H:\Export to UTAU\Test.wav", value=r"H:\Export to UTAU\Test.wav", scale=8)
                                    btn_browse_audio0 = gr.Button("📄 เลือกไฟล์ (Browse)", scale=2, variant="secondary")
                                with gr.Row():
                                    file_index1 = gr.Textbox(label="เส้นทางไฟล์ดัชนี (Feature Index Path, Leave blank for auto)", placeholder=r"logs\model.index", value="", interactive=True, scale=8)
                                    btn_browse_index1 = gr.Button("📑 เลือกดัชนี (Browse)", scale=2, variant="secondary")
                                file_index2 = gr.Dropdown(label="เลือกไฟล์ดัชนีอัตโนมัติ (Auto-Select Index Path)", choices=sorted(index_paths), interactive=True)
                                f0method0 = gr.Radio(label="อัลกอริทึมวิเคราะห์ระดับเสียง (Pitch Extraction Algorithm)", choices=(["pm", "harvest", "crepe", "rmvpe"] if F0GPUVisible else ["pm", "harvest", "rmvpe"]), value="rmvpe", interactive=True)
                                btn_browse_audio0.click(fn=lambda cur: browse_file(cur, [("Audio Files", "*.wav;*.flac;*.mp3;*.m4a;*.ogg"), ("All files", "*.*")]), inputs=[input_audio0], outputs=[input_audio0])
                                btn_browse_index1.click(fn=lambda cur: browse_file(cur, [("Index Files", "*.index"), ("All files", "*.*")]), inputs=[file_index1], outputs=[file_index1])

                            with gr.Column():
                                resample_sr0 = gr.Slider(minimum=0, maximum=48000, step=1, label="ปรับอัตราสุ่มตัวอย่างซ้ำ (Resampling Rate, 0 = Off)", value=0, interactive=True)
                                rms_mix_rate0 = gr.Slider(minimum=0, maximum=1, label="ความเข้มข้นของการผสานระดับเสียง (Volume Envelope Mix Rate)", value=0.25, interactive=True)
                                protect0 = gr.Slider(minimum=0, maximum=0.5, step=0.01, label="การปกป้องพยัญชนะและเสียงหายใจ (Breath Protection, 0.5 = Off)", value=0.45, interactive=True)
                                filter_radius0 = gr.Slider(minimum=0, maximum=7, step=1, label="รัศมีตัวกรองเสียงรบกวน (Median Filter Radius)", value=1, interactive=True)
                                index_rate1 = gr.Slider(minimum=0, maximum=1, label="สัดส่วนการใช้อิทธิพลดัชนี (Index Influence Ratio)", value=0.75, interactive=True)
                                f0_file = gr.File(label="ไฟล์เส้นโค้งระดับเสียง (F0 Curve File, Optional)", visible=False)
                                
                                refresh_button.click(fn=change_choices, inputs=[], outputs=[sid0, file_index2], api_name="infer_refresh")
                    
                    with gr.Group():
                        with gr.Column():
                            but0 = gr.Button("เริ่มประมวลผล (Start Rendering)", variant="primary")
                            with gr.Row():
                                vc_output1 = gr.Textbox(label="สถานะการดำเนินการ (Operation Log & Status)")
                                vc_output2 = gr.Audio(label="ผลลัพธ์เสียงที่สมบูรณ์แบบ (Output Audio)")
                            
                            but0.click(single_infer_wrapper, [spk_item, input_audio0, vc_transform0, f0_file, f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0], [vc_output1, vc_output2], api_name="infer_convert")

                with gr.TabItem("การประมวลผลแบบชุด (Batch Processing)"):
                    gr.Markdown("นำเข้าโฟลเดอร์ไฟล์เสียง หรือ อัปโหลดแบบหลายไฟล์ พร้อมระบบ Smart Batch Resume ข้ามไฟล์ที่เสร็จแล้วอัตโนมัติ")
                    with gr.Row():
                        with gr.Column():
                            vc_transform1 = gr.Number(label="ปรับระดับเสียง (Pitch Shifting)", value=0)
                            with gr.Row():
                                opt_input = gr.Textbox(label="ไดเรกทอรีส่งออก (Output Directory)", value=r"H:\Render file to UTAU\Singer\Language", scale=8)
                                btn_browse_opt = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
                            with gr.Row():
                                file_index3 = gr.Textbox(label="เส้นทางไฟล์ดัชนี (Feature Index Path)", value="", interactive=True, scale=8)
                                btn_browse_index3 = gr.Button("📑 เลือกดัชนี (Browse)", scale=2, variant="secondary")
                            file_index4 = gr.Dropdown(label="เลือกไฟล์ดัชนีอัตโนมัติ (Auto-Select Index)", choices=sorted(index_paths), interactive=True)
                            f0method1 = gr.Radio(label="อัลกอริทึมวิเคราะห์ระดับเสียง (Pitch Extraction Algorithm)", choices=(["pm", "harvest", "crepe", "rmvpe"] if F0GPUVisible else ["pm", "harvest", "rmvpe"]), value="rmvpe", interactive=True)
                            format1 = gr.Radio(label="รูปแบบไฟล์ปลายทาง (Export Format)", choices=["wav", "flac", "mp3", "m4a"], value="wav", interactive=True)
                            refresh_button.click(fn=lambda: change_choices()[1], inputs=[], outputs=file_index4, api_name="infer_refresh_batch")
                            btn_browse_opt.click(fn=browse_folder, inputs=[opt_input], outputs=[opt_input])
                            btn_browse_index3.click(fn=lambda cur: browse_file(cur, [("Index Files", "*.index"), ("All files", "*.*")]), inputs=[file_index3], outputs=[file_index3])

                        with gr.Column():
                            resample_sr1 = gr.Slider(minimum=0, maximum=48000, step=1, label="ปรับอัตราสุ่มตัวอย่างซ้ำ (Resampling Rate)", value=0, interactive=True)
                            rms_mix_rate1 = gr.Slider(minimum=0, maximum=1, label="ความเข้มข้นของการผสานระดับเสียง (Volume Envelope Mix Rate)", value=0.25, interactive=True)
                            protect1 = gr.Slider(minimum=0, maximum=0.5, step=0.01, label="การปกป้องพยัญชนะและเสียงหายใจ (Breath Protection)", value=0.45, interactive=True)
                            filter_radius1 = gr.Slider(minimum=0, maximum=7, step=1, label="รัศมีตัวกรองเสียงรบกวน (Median Filter Radius)", value=1, interactive=True)
                            index_rate2 = gr.Slider(minimum=0, maximum=1, label="สัดส่วนการใช้อิทธิพลดัชนี (Index Influence Ratio)", value=0.75, interactive=True)

                    with gr.Row():
                        with gr.Column(scale=8):
                            with gr.Row():
                                dir_input = gr.Textbox(label="เส้นทางโฟลเดอร์ต้นฉบับ (Source Directory Path)", value=r"H:\Export to UTAU\EN", placeholder=r"H:\Export to UTAU\EN", scale=8)
                                btn_browse_dir = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
                            btn_browse_dir.click(fn=browse_folder, inputs=[dir_input], outputs=[dir_input])
                        inputs = gr.File(file_count="multiple", label="หรือ อัปโหลดไฟล์เสียงหลายไฟล์ (Or Upload Multiple Files)", scale=4)
                    
                    with gr.Row():
                        but1 = gr.Button("ตรวจสอบสถานะ / เริ่มประมวลผลชุด (Check Status / Start Batch Render)", variant="primary")
                        cancel_btn = gr.Button("🛑 ยกเลิกคำสั่ง (Cancel Render)", variant="secondary")
                        vc_output3 = gr.Textbox(label="ตรวจสอบการทำงานแบบเรียลไทม์ (Real-time Status Tracking)")
                        
                        render_event = but1.click(batch_render_tracker, [sid0, dir_input, opt_input, inputs, vc_transform1, f0method1, file_index3, file_index4, index_rate2, filter_radius1, resample_sr1, rms_mix_rate1, protect1, format1], [vc_output3], api_name="infer_convert_batch")
                        cancel_btn.click(safe_cancel_process, inputs=[], outputs=[vc_output3], cancels=[render_event])

            sid0.change(fn=get_vc_wrapper, inputs=[sid0, protect0, protect1], outputs=[spk_item, protect0, protect1, file_index2, file_index4], api_name="infer_change_voice")

        with gr.TabItem("🎚️ เครื่องมือแยกเสียง (UVR5 Separation & De-Echo)"):
            with gr.Group():
                gr.Markdown("ระบบแยกเสียงร้องและเสียงดนตรี พร้อมเครื่องมือลบเสียงสะท้อน (Vocal/Instrument separation using UVR5).")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            dir_wav_input = gr.Textbox(label="โฟลเดอร์เสียงต้นฉบับ (Input Audio Directory)", placeholder=r"C:\Users\Desktop\todo-songs", scale=8)
                            btn_browse_uvr_in = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
                        btn_browse_uvr_in.click(fn=browse_folder, inputs=[dir_wav_input], outputs=[dir_wav_input])
                        wav_inputs = gr.File(file_count="multiple", label="หรือ อัปโหลดไฟล์เสียง (Or Upload Audio Files)")
                    with gr.Column():
                        model_choose = gr.Dropdown(label="เลือกโมเดล (Select Model)", choices=uvr5_names)
                        agg = gr.Slider(minimum=0, maximum=20, step=1, label="ความดุดันในการสกัดเสียง (Aggressiveness)", value=10, interactive=True, visible=False)
                        with gr.Row():
                            opt_vocal_root = gr.Textbox(label="โฟลเดอร์สำหรับเสียงหลัก (Output Vocal Directory)", value="opt", scale=8)
                            btn_browse_vocal = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
                        with gr.Row():
                            opt_ins_root = gr.Textbox(label="โฟลเดอร์สำหรับเสียงดนตรี (Output Instrument Directory)", value="opt", scale=8)
                            btn_browse_ins = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
                        format0 = gr.Radio(label="รูปแบบไฟล์ (Export Format)", choices=["wav", "flac", "mp3", "m4a"], value="flac", interactive=True)
                        btn_browse_vocal.click(fn=browse_folder, inputs=[opt_vocal_root], outputs=[opt_vocal_root])
                        btn_browse_ins.click(fn=browse_folder, inputs=[opt_ins_root], outputs=[opt_ins_root])
                but2 = gr.Button("เริ่มการแยกเสียง (Start Separation)", variant="primary")
                vc_output4 = gr.Textbox(label="สถานะการทำงาน (Operation Log)")
                but2.click(uvr, [model_choose, dir_wav_input, opt_vocal_root, wav_inputs, opt_ins_root, agg, format0], [vc_output4], api_name="uvr_convert")

        with gr.TabItem("🏋️ การฝึกสอนโมเดล (Training)"):
            gr.Markdown("### ⚙️ ระบบฝึกสอนโมเดล RVC อัจฉริยะ (DELTA Zero-Defect Training Pipeline)")
            
            with gr.Group():
                with gr.Row():
                    existing_exps = get_available_experiments()
                    exp_selector = gr.Dropdown(label="เลือกชุดการทดลองที่มีอยู่เดิมใน logs/ (Select Existing Experiment)", choices=existing_exps, value=existing_exps[0] if existing_exps else "")
                    refresh_exp_btn = gr.Button("🔄 สแกนโฟลเดอร์ logs/ (Scan Experiments)", variant="secondary")
                
                exp_status_card = gr.Markdown("📊 เลือกรหัสการทดลองเพื่อตรวจสอบความพร้อมของข้อมูล")
                
                def on_exp_select(name):
                    is_ok, msg, rec_ver, count = inspect_experiment_status(name)
                    return msg, rec_ver, name
                    
                refresh_exp_btn.click(fn=lambda: gr.update(choices=get_available_experiments()), inputs=[], outputs=[exp_selector])
            
            with gr.Row():
                exp_dir1 = gr.Textbox(label="ชื่อการทดลอง (Experiment Name)", value="NameAI")
                sr2 = gr.Radio(label="อัตราการสุ่มตัวอย่างเป้าหมาย (Target Sample Rate)", choices=["40k", "48k"], value="40k", interactive=True)
                if_f0_3 = gr.Radio(label="ใช้การชี้นำระดับเสียงหรือไม่? (Use Pitch Guidance?)", choices=[True, False], value=True, interactive=True)
                version19 = gr.Radio(label="เวอร์ชันของโมเดล (Model Version)", choices=["v1", "v2"], value="v2", interactive=True)
                np7 = gr.Slider(minimum=1, maximum=config.n_cpu, step=1, label="เธรด CPU ที่ใช้ (CPU Threads Used)", value=max(1, int(config.n_cpu // 1.5)), interactive=True)
            
            exp_selector.change(fn=on_exp_select, inputs=[exp_selector], outputs=[exp_status_card, version19, exp_dir1])

            with gr.Group():
                gr.Markdown("#### ขั้นตอนที่ 1: ประมวลผลและหั่นชุดข้อมูลเสียง (Step 1: Preprocess Dataset)")
                with gr.Row():
                    trainset_dir4 = gr.Textbox(label="โฟลเดอร์ชุดข้อมูลเสียงต้นฉบับ (Dataset Audio Directory)", value=r"H:\Data\Singer...", scale=8)
                    btn_browse_trainset = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
                    spk_id5 = gr.Slider(minimum=0, maximum=4, step=1, label="หมายเลขผู้พูด (Speaker ID)", value=0, interactive=True)
                    btn_browse_trainset.click(fn=browse_folder, inputs=[trainset_dir4], outputs=[trainset_dir4])
                with gr.Row():
                    but_prep = gr.Button("ประมวลผลข้อมูล (Process Data)", variant="primary")
                    info1 = gr.Textbox(label="สถานะ (Status)", value="")
                    but_prep.click(preprocess_dataset, [trainset_dir4, exp_dir1, sr2, np7], [info1], api_name="train_preprocess")
            
            with gr.Group():
                gr.Markdown("#### ขั้นตอนที่ 2: สกัดระดับเสียงและคุณลักษณะเสียง (Step 2: Extract F0 & HuBERT Features)")
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(label="หมายเลข GPU (GPU IDs)", value=gpus, interactive=True, visible=F0GPUVisible)
                        gpu_info9 = gr.Textbox(label="ข้อมูลระบบฮาร์ดแวร์ (Hardware Info)", value=gpu_info, visible=F0GPUVisible)
                    with gr.Column():
                        f0method8 = gr.Radio(label="อัลกอริทึมสกัดระดับเสียง (Pitch Extraction Algorithm)", choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"], value="rmvpe", interactive=True)
                        gpus_rmvpe = gr.Textbox(label="GPU สำหรับ RMVPE (GPUs for RMVPE)", value=f"{gpus}-{gpus}" if gpus else "0-0", interactive=True, visible=F0GPUVisible)
                    but_feat = gr.Button("สกัดคุณลักษณะเสียง (Extract Features)", variant="primary")
                    info2 = gr.Textbox(label="สถานะ (Status)", value="", max_lines=8)
                    f0method8.change(fn=change_f0_method, inputs=[f0method8], outputs=[gpus_rmvpe])
                    but_feat.click(extract_f0_feature, [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe], [info2], api_name="train_extract_f0_feature")
            
            with gr.Group():
                gr.Markdown("#### ขั้นตอนที่ 3: ฝึกสอนโมเดลและสร้างดัชนี (Step 3: Model Training & Feature Indexing)")
                with gr.Row():
                    save_epoch10 = gr.Slider(minimum=1, maximum=50, step=1, label="ความถี่ในการบันทึก / รอบ (Save Frequency)", value=40, interactive=True)
                    total_epoch11 = gr.Slider(minimum=2, maximum=1000, step=1, label="จำนวนรอบทั้งหมด (Total Epochs)", value=200, interactive=True)
                    batch_size12 = gr.Slider(minimum=1, maximum=40, step=1, label="ขนาดแบทช์ต่อ GPU (Batch Size per GPU)", value=default_batch_size, interactive=True)
                    if_save_latest13 = gr.Radio(label="บันทึกเฉพาะโมเดลล่าสุด? (Save only latest?)", choices=["Yes", "No"], value="Yes", interactive=True)
                    if_cache_gpu17 = gr.Radio(label="แคชข้อมูลลง VRAM? (Cache to VRAM?)", choices=["Yes", "No"], value="Yes", interactive=True)
                    if_save_every_weights18 = gr.Radio(label="บันทึกไฟล์น้ำหนักเป็นระยะ? (Save weights periodically?)", choices=["Yes", "No"], value="Yes", interactive=True)
                
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            pretrained_G14 = gr.Textbox(label="โมเดลพื้นฐาน G (Pretrained G)", value="assets/pretrained_v2/f0G40k.pth", interactive=True, scale=8)
                            btn_browse_pretrained_g = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    with gr.Column():
                        with gr.Row():
                            pretrained_D15 = gr.Textbox(label="โมเดลพื้นฐาน D (Pretrained D)", value="assets/pretrained_v2/f0D40k.pth", interactive=True, scale=8)
                            btn_browse_pretrained_d = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    gpus16 = gr.Textbox(label="GPU สำหรับฝึกสอน (GPUs for Training)", value=gpus, interactive=True)
                    
                    btn_browse_pretrained_g.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[pretrained_G14], outputs=[pretrained_G14])
                    btn_browse_pretrained_d.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[pretrained_D15], outputs=[pretrained_D15])
                    sr2.change(change_sr2, [sr2, if_f0_3, version19], [pretrained_G14, pretrained_D15])
                    version19.change(change_version19, [sr2, if_f0_3, version19], [pretrained_G14, pretrained_D15, sr2])
                    if_f0_3.change(change_f0, [if_f0_3, sr2, version19], [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15])
                    
                with gr.Row():
                    but3 = gr.Button("เริ่มฝึกสอนโมเดล (Train Model Only)", variant="secondary")
                    but4 = gr.Button("สร้างดัชนีคุณลักษณะ (Build Feature Index Only)", variant="secondary")
                    but5 = gr.Button("⚡ ฝึกสอนอัตโนมัติครบวงจร (One-Click Training)", variant="primary")
                    
                info3 = gr.Textbox(label="สถานะการทำงาน (Operation Log)", value="", max_lines=12)
                
                but3.click(click_train, [exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19], info3, api_name="train_start")
                but4.click(train_index, [exp_dir1, version19], info3)
                but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19, gpus_rmvpe], info3, api_name="train_start_all")

        with gr.TabItem("📦 การจัดการจุดตรวจสอบ (CKPT Processing)"):
            with gr.Group():
                gr.Markdown("#### การผสานโมเดล (Model Fusion/Merging)")
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():
                            ckpt_a = gr.Textbox(label="เส้นทางโมเดล A (Model A Path)", value="", interactive=True, scale=8)
                            btn_browse_ckpt_a = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    with gr.Column(scale=4):
                        with gr.Row():
                            ckpt_b = gr.Textbox(label="เส้นทางโมเดล B (Model B Path)", value="", interactive=True, scale=8)
                            btn_browse_ckpt_b = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    alpha_a = gr.Slider(minimum=0, maximum=1, label="น้ำหนักโมเดล A (Model A Weight)", value=0.5, interactive=True, scale=2)
                btn_browse_ckpt_a.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[ckpt_a], outputs=[ckpt_a])
                btn_browse_ckpt_b.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[ckpt_b], outputs=[ckpt_b])
                with gr.Row():
                    sr_ = gr.Radio(label="อัตราสุ่มตัวอย่าง (Target Sample Rate)", choices=["40k", "48k"], value="40k", interactive=True)
                    if_f0_ = gr.Radio(label="มีระดับเสียงนำทางหรือไม่? (Pitch Guidance?)", choices=["Yes", "No"], value="Yes", interactive=True)
                    info__ = gr.Textbox(label="ข้อมูลโมเดล (Model Info)", value="", max_lines=4, interactive=True)
                    name_to_save0 = gr.Textbox(label="ชื่อบันทึกไฟล์ (Save Name)", value="", max_lines=1, interactive=True)
                    version_2 = gr.Radio(label="เวอร์ชันโมเดล (Model Version)", choices=["v1", "v2"], value="v2", interactive=True)
                with gr.Row():
                    but6 = gr.Button("ผสานโมเดล (Merge Models)", variant="primary")
                    info4 = gr.Textbox(label="สถานะการทำงาน (Operation Log)", value="", max_lines=4)
                but6.click(merge, [ckpt_a, ckpt_b, alpha_a, sr_, if_f0_, info__, name_to_save0, version_2], info4, api_name="ckpt_merge")

            with gr.Group():
                gr.Markdown("#### แก้ไขข้อมูลโมเดล (Modify Model Information)")
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            ckpt_path0 = gr.Textbox(label="เส้นทางไฟล์โมเดล (Model Path)", value="", interactive=True, scale=8)
                            btn_browse_ckpt0 = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    info_ = gr.Textbox(label="ข้อมูลที่ต้องการปรับปรุง (Info to modify)", value="", max_lines=4, interactive=True, scale=4)
                    name_to_save1 = gr.Textbox(label="ชื่อไฟล์ใหม่ (New File Name)", value="", max_lines=1, interactive=True, scale=3)
                btn_browse_ckpt0.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[ckpt_path0], outputs=[ckpt_path0])
                with gr.Row():
                    but7 = gr.Button("บันทึกการแก้ไข (Modify & Save)", variant="primary")
                    info5 = gr.Textbox(label="สถานะ (Status)", value="", max_lines=4)
                but7.click(change_info, [ckpt_path0, info_, name_to_save1], info5, api_name="ckpt_modify")

            with gr.Group():
                gr.Markdown("#### การดึงโมเดลย่อยเพื่อนำไปใช้งาน (Extract Small Model)")
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            ckpt_path2 = gr.Textbox(label="เส้นทางโมเดลหลัก (Large Model Path)", value=r"logs\NameAI\G_23333.pth", interactive=True, scale=8)
                            btn_browse_ckpt2 = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    save_name = gr.Textbox(label="ชื่อบันทึกไฟล์ (Save Name)", value="Model_Deploy.pth", interactive=True, scale=3)
                    sr__ = gr.Radio(label="อัตราสุ่มตัวอย่าง (Sample Rate)", choices=["32k", "40k", "48k"], value="40k", interactive=True)
                    if_f0__ = gr.Radio(label="ชี้นำระดับเสียง? (Pitch Guidance?)", choices=["1", "0"], value="1", interactive=True)
                    version_1 = gr.Radio(label="เวอร์ชัน (Version)", choices=["v1", "v2"], value="v2", interactive=True)
                btn_browse_ckpt2.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[ckpt_path2], outputs=[ckpt_path2])
                with gr.Row():
                    but9 = gr.Button("ดึงโมเดล (Extract)", variant="primary")
                    info7 = gr.Textbox(label="สถานะ (Status)", value="", max_lines=4)
                    ckpt_path2.change(change_info_, [ckpt_path2], [sr__, if_f0__, version_1])
                but9.click(extract_small_model, [ckpt_path2, save_name, sr__, if_f0__, gr.Textbox(value="", visible=False), version_1], info7, api_name="ckpt_extract")

        with gr.TabItem("🌐 ส่งออก ONNX (ONNX Export)"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        ckpt_dir = gr.Textbox(label="เส้นทางโมเดล RVC (RVC Model Path)", value="", interactive=True, scale=8)
                        btn_browse_onnx_ckpt = gr.Button("📄 เลือกโมเดล (Browse)", scale=2, variant="secondary")
                with gr.Column():
                    with gr.Row():
                        onnx_dir = gr.Textbox(label="เส้นทางส่งออก ONNX (ONNX Output Path)", value="", interactive=True, scale=8)
                        btn_browse_onnx_out = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
            btn_browse_onnx_ckpt.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[ckpt_dir], outputs=[ckpt_dir])
            btn_browse_onnx_out.click(fn=browse_folder, inputs=[onnx_dir], outputs=[onnx_dir])
            with gr.Row():
                infoOnnx = gr.Label(label="สถานะ (Status)")
                butOnnx = gr.Button("ส่งออกโมเดล (Export ONNX Model)", variant="primary")
            butOnnx.click(export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx")

        with gr.TabItem("🎤 OpenUtau Voicebank Studio"):
            with gr.Tabs():
                with gr.TabItem("การตั้งค่าตัวละคร (Character Setup)"):
                    gr.Markdown("### กระบวนการที่ 1: จัดการรายละเอียดของตัวละคร OpenUtau 🟢")
                    with gr.Row():
                        singer_name_in = gr.Textbox(label="ชื่อนักร้อง / ตัวละคร (Character Name)", value="Singer", scale=3)
                        lang_mode_in = gr.Dropdown(label="ภาษาและรูปแบบคลังเสียง (Language & Voicebank Format)", choices=["1. EN", "2. TH", "3. JP"], value="2. TH", scale=3)
                        with gr.Column(scale=6):
                            with gr.Row():
                                base_out_dir = gr.Textbox(label="โฟลเดอร์สำหรับส่งออก (Root Export Directory)", value=r"H:\Render file to UTAU", scale=8)
                                btn_browse_base_out = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
                    btn_browse_base_out.click(fn=browse_folder, inputs=[base_out_dir], outputs=[base_out_dir])
                    with gr.Row():
                        pic_icon = gr.Image(type="filepath", label="ไอคอนตัวละคร (Icon.png)")
                        pic_full = gr.Image(type="filepath", label="ภาพตัวละครเต็มตัว (Full.png)")
                    with gr.Row():
                        readme_txt = gr.Textbox(label="เนื้อหา readme.txt", lines=6)
                        char_yaml = gr.Textbox(label="เนื้อหา character.yaml", lines=6)
                        char_txt = gr.Textbox(label="เนื้อหา character.txt", lines=6)
                    save_proj_btn = gr.Button("💾 บันทึกและสร้างโฟลเดอร์โปรเจกต์ (Initialize Project)", variant="primary")
                    proj_status = gr.Textbox(label="สถานะ (Status)")
                    
                    singer_name_in.change(fn=update_text_templates, inputs=[singer_name_in, lang_mode_in], outputs=[readme_txt, char_yaml, char_txt, gr.Textbox(visible=False)])
                    lang_mode_in.change(fn=update_text_templates, inputs=[singer_name_in, lang_mode_in], outputs=[readme_txt, char_yaml, char_txt, gr.Textbox(visible=False)])
                    save_proj_btn.click(fn=save_project, inputs=[base_out_dir, singer_name_in, pic_icon, pic_full, readme_txt, char_yaml, char_txt], outputs=[gr.Markdown(visible=False), proj_status, gr.Textbox(visible=False), gr.Textbox(visible=False)])

        tab_faq = "❓ คำถามที่พบบ่อย (FAQ)"
        with gr.TabItem(tab_faq):
            try:
                faq_p = os.path.join(now_dir, "docs", "th", "faq_th.md")
                if not os.path.exists(faq_p):
                    faq_p = os.path.join(now_dir, "docs", "en", "faq_en.md")
                if os.path.exists(faq_p):
                    with open(faq_p, "r", encoding="utf-8") as f:
                        gr.Markdown(f.read())
                else:
                    gr.Markdown("### DELTA SYNTH FAQ\n- ระบบทำงานบนพอร์ตเริ่มต้น 9378\n- รองรับโมเดล RVC v1/v2 และ DirectML")
            except Exception as e:
                gr.Markdown(f"Error loading FAQ: {e}")

listen_port = int(getattr(config, "listen_port", 9378))
is_colab = getattr(config, "iscolab", False)
no_auto_open = getattr(config, "noautoopen", False)

if __name__ == "__main__":
    DeltaLogger.inf(f"🚀 กำลังเริ่มต้นระบบ DELTA RVC Studio บนพอร์ต {listen_port}...", to_console=True)
    if is_colab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        try:
            app.queue(concurrency_count=511, max_size=1022).launch(
                server_name="0.0.0.0",
                inbrowser=not no_auto_open,
                server_port=listen_port,
                quiet=True,
            )
        except OSError:
            alt_port = 9379
            DeltaLogger.wrn(f"พอร์ต {listen_port} ถูกใช้งานแล้ว กำลังสลับไปยังพอร์ตสำรอง {alt_port}...", to_console=True)
            app.queue(concurrency_count=511, max_size=1022).launch(
                server_name="0.0.0.0",
                inbrowser=not no_auto_open,
                server_port=alt_port,
                quiet=True,
            )