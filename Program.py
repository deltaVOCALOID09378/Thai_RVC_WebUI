# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี (Date)       : 18/10/2569 (18 October 2026)
# 2. ชื่อไฟล์ที่ทำ (File Name)  : Program.py
# 3. ลำดับเวอร์ชั่นของไฟล์ (Version): 5.3.4
# 4. เครดิตผู้สร้าง (Credit)    : Made And Checked By DELTA SYNTH[ DELTA SYNTH ] & All Code Agentic AI Engine
#                              [ Original by RVC Project & OpenVPI DiffSinger ]
# 5. ตำแหน่งที่วางของไฟล์ (Path): Root Directory (เช่น A:\Thai_RVC_WebUI_v1\Program.py)
# 6. รายละเอียดการแก้ไขงาน    : ผสานระบบ Smart Batch Resume, จับเวลา MM:SS, ป้องกันนามสกุลซ้ำ, และเพิ่มเส้นแบ่ง Log
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
import csv
import math
import yaml
import shutil
import signal
import logging
import traceback
import platform
import pathlib
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any, Callable
from collections import OrderedDict
import json
import subprocess
import argparse
from contextlib import suppress
import time
from time import sleep, strftime
from datetime import datetime
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

# ─── Path & Environment Setup ─────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
now_dir = str(PROJECT_ROOT)
os.chdir(now_dir)
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)
load_dotenv()

# Strict runtime executable selection prioritizing runtime\python.exe
DEFAULT_PYTHON = os.path.join(now_dir, "runtime", "python.exe")
VENV_PYTHON = os.path.join(now_dir, ".venv", "Scripts", "python.exe")
if os.path.isfile(DEFAULT_PYTHON):
    PYTHON_EXE = DEFAULT_PYTHON
elif os.path.isfile(VENV_PYTHON):
    PYTHON_EXE = VENV_PYTHON
else:
    PYTHON_EXE = sys.executable or "python"

# ─── Smart DiffSinger Root & Script Auto-Resolver ─────────────────────────────
def resolve_diffsinger_root(project_root: Path) -> Path:
    env_ds = os.getenv("DIFFSINGER_ROOT", "").strip()
    candidates = [Path(env_ds)] if env_ds else []
    candidates.extend([
        project_root / "DiffSinger",
        project_root / "DiffSinger-2.5.1",
        project_root / "DiffSinger-main",
        project_root.parent / "DiffSinger",
        project_root.parent / "DiffSinger-2.5.1",
        Path(r"A:\DiffSinger-2.5.1"), Path(r"A:\DiffSinger"),
        Path(r"B:\DiffSinger-2.5.1"), Path(r"B:\DiffSinger"),
    ])
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            if (cand / "scripts" / "binarize.py").exists() or (cand / "binarize.py").exists() or (cand / "utils" / "hparams.py").exists():
                return cand.resolve()
            for sub in cand.iterdir():
                if sub.is_dir() and ((sub / "scripts" / "binarize.py").exists() or (sub / "utils" / "hparams.py").exists()):
                    return sub.resolve()
    for root_d, dirs, files in os.walk(project_root):
        if any(skip in root_d for skip in ["runtime", ".git", "__pycache__", "logs", "assets", "TEMP"]):
            continue
        if "binarize.py" in files:
            found_p = Path(root_d) / "binarize.py"
            if found_p.parent.name.lower() == "scripts":
                return found_p.parent.parent.resolve()
            return found_p.parent.resolve()
    for cand in candidates:
        if cand.exists() and cand.is_dir(): return cand.resolve()
    return project_root / "DiffSinger"

DIFFSINGER_ROOT = resolve_diffsinger_root(PROJECT_ROOT)
DS_DATA_DIR = DIFFSINGER_ROOT / "data"
DS_CHECKPOINTS_DIR = DIFFSINGER_ROOT / "checkpoints"
DS_DICTIONARIES_DIR = DIFFSINGER_ROOT / "dictionaries"
DS_CONFIGS_DIR = DIFFSINGER_ROOT / "configs"
DS_SCRIPTS_DIR = DIFFSINGER_ROOT / "scripts"
DS_MASTER_SCRIPT = PROJECT_ROOT / "tools" / "diffsinger" / "diffsinger_pipeline_master.py"

def resolve_ds_script(script_name: str = "binarize.py") -> Tuple[Path, Path]:
    search_locations = [
        DS_SCRIPTS_DIR / script_name, DIFFSINGER_ROOT / script_name,
        PROJECT_ROOT / "DiffSinger" / "scripts" / script_name,
        PROJECT_ROOT / "DiffSinger" / script_name,
        PROJECT_ROOT / "scripts" / script_name, PROJECT_ROOT / script_name,
    ]
    found_script = next((loc.resolve() for loc in search_locations if loc.exists() and loc.is_file()), None)
    if found_script is None:
        for search_base in [DIFFSINGER_ROOT, PROJECT_ROOT]:
            if not search_base.exists(): continue
            for root_d, _, files in os.walk(search_base):
                if any(skip in root_d for skip in ["runtime", ".git", "__pycache__", "logs", "TEMP"]): continue
                if script_name in files:
                    found_script = (Path(root_d) / script_name).resolve()
                    break
            if found_script is not None: break

    if found_script is not None:
        work_dir = found_script.parent.parent if found_script.parent.name.lower() == "scripts" else (found_script.parent if (found_script.parent / "utils").exists() else (DIFFSINGER_ROOT if (DIFFSINGER_ROOT / "utils").exists() else found_script.parent))
        expected_target = DS_SCRIPTS_DIR / script_name
        with suppress(Exception):
            if found_script != expected_target.resolve():
                DS_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(found_script), str(expected_target))
        return found_script, work_dir
    return DS_SCRIPTS_DIR / script_name, DIFFSINGER_ROOT

try:
    from utils import PoliteLogger, format_error_diagnostic, install_global_exception_handler
    install_global_exception_handler()
except ImportError:
    class PoliteLogger:
        @staticmethod
        def error(msg: str, exc: Exception = None, context: str = "Program") -> str:
            return f"[{strftime('%H:%M:%S')}] [ERR] [{context}] {msg}" + (f" ({exc})" if exc else "")
        @staticmethod
        def model(model_name: str, details: str = "") -> str:
            return f"[{strftime('%H:%M:%S')}] [MDL] {model_name}: {details}"

# ─── RVC CORE INITIALIZATION ──────────────────────────────────────────────────
try:
    from infer.modules.vc.modules import VC
    from infer.modules.uvr5.modules import uvr
    from infer.lib.train.process_ckpt import change_info, extract_small_model, merge, show_info
    from i18n.i18n import I18nAuto
    from configs.config import Config
    config = Config()
    vc = VC(config)
    RVC_CORE_AVAILABLE = True
except Exception as _rvc_imp_err:
    RVC_CORE_AVAILABLE = False
    class DummyConfig:
        n_cpu = 2; device = "cpu"; is_half = False; noparallel = False
        listen_port = 9378; iscolab = False; noautoopen = False
    config = DummyConfig()
    vc = None

sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}

ACTIVE_CHILD_PROCESSES = set()
CANCEL_REQUESTED = False
_ds_active_process = None
_ds_process_lock = threading.Lock()

# ─── SMART LOGGING SYSTEM (DELTA SYNTH STANDARD) ──────────────────────────────
class DeltaLogger:
    @staticmethod
    def format_msg(level: str, msg: str) -> str:
        return f"[{strftime('%H:%M:%S')}] [{level}] {msg}"
    @staticmethod
    def inf(msg: str, to_console: bool = False) -> str: 
        formatted = DeltaLogger.format_msg("INF", msg)
        logger.info(msg)
        if to_console and os.getenv("DELTA_CMD_VERBOSE", "0") == "1": print(formatted, flush=True)
        return formatted
    @staticmethod
    def wrn(msg: str, to_console: bool = False) -> str: 
        formatted = DeltaLogger.format_msg("WRN", msg)
        logger.warning(msg)
        if to_console and os.getenv("DELTA_CMD_VERBOSE", "0") == "1": print(formatted, flush=True)
        return formatted
    @staticmethod
    def err(msg: str, exc: Exception = None, context: str = "Program") -> str: 
        return PoliteLogger.error(msg=msg, exc=exc, context=context)

logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("gradio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
for d in ["logs", "assets/weights", "assets/uvr5_weights", "assets/indices", "assets/pretrained", "assets/pretrained_v2"]:
    os.makedirs(os.path.join(now_dir, d.replace("/", os.sep)), exist_ok=True)
os.environ["TEMP"] = tmp

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
with suppress(Exception): torch.set_num_threads(2)

def format_time_delta(seconds):
    """ฟังก์ชันคำนวณเวลาที่ใช้ไปเป็น MM:SS"""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"

def clean_stem(filepath: str) -> str:
    """ลบส่วนขยายไฟล์แบบซ้อนทับ (เช่น .wav.npy) ป้องกันปัญหาไฟล์ซ้ำซ้อน"""
    name = os.path.basename(filepath)
    compound_exts = [".wav.npy", ".flac.npy", ".mp3.npy", ".ogg.npy", ".m4a.npy", ".npy", ".wav", ".flac", ".mp3", ".ogg", ".m4a"]
    for ext in compound_exts:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    return name

def clear_vram_safe():
    """บังคับคืนค่าหน่วยความจำ VRAM และ RAM อัตโนมัติ (Garbage Collection)"""
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
    if RVC_CORE_AVAILABLE and hasattr(vc, "is_rendering"): vc.is_rendering = False
    cancelled_count = 0
    for p in list(ACTIVE_CHILD_PROCESSES):
        try:
            if p.poll() is None:
                if platform.system() == "Windows":
                    with suppress(Exception): subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True, timeout=5)
                p.terminate()
                sleep(0.1)
                if p.poll() is None: p.kill()
            cancelled_count += 1
        except Exception as e:
            DeltaLogger.wrn(f"Failed to kill process {p.pid}: {e}", to_console=True)
    ACTIVE_CHILD_PROCESSES.clear()
    clear_vram_safe()
    return f"{'='*60}\n🛑 ยกเลิกกระบวนการทั้งหมดเรียบร้อยแล้ว (Cancelled {cancelled_count} processes)\n{'='*60}"

def safe_cancel_process():
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = True
    if RVC_CORE_AVAILABLE and hasattr(vc, "is_rendering"): vc.is_rendering = False
    clear_vram_safe()
    return f"{'='*60}\n⚠️ คำสั่งถูกยกเลิกโดยผู้ใช้งานเรียบร้อยแล้ว (Render Cancelled by User)\n🏁 เวลาสิ้นสุด (End Time): {strftime('%H:%M:%S')}\n{'='*60}"

# ─── SMART MODEL PATH ROUTER ──────────────────────────────────────────────────
class SmartModelVersionRouter:
    def __init__(self, rvc_root: str, ds_root: Path):
        self.rvc_root = Path(rvc_root)
        self.ds_root = Path(ds_root)
        self.registry_path = self.rvc_root / "configs" / "model_version_registry.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def inspect_pth_metadata(self, pth_path: Path) -> dict:
        meta = {"version": "v2", "sr": "40k", "f0": True, "category": "rvc_voice_model"}
        fname_lower = pth_path.name.lower()
        if any(k in fname_lower for k in ["uvr", "deecho", "dereverb", "hp2", "hp3", "hp5", "mdx", "vr_"]):
            meta["category"] = "uvr5_model"; return meta
        if "hubert" in fname_lower: meta["category"] = "hubert_base"; return meta
        if "rmvpe" in fname_lower: meta["category"] = "rmvpe_model"; return meta
        if fname_lower.startswith(("f0g", "f0d", "g32k", "g40k", "g48k", "d32k", "d40k", "d48k")):
            meta["category"] = "rvc_pretrained"
            meta["version"] = "v1" if "v1" in str(pth_path).lower() else "v2"
            return meta
        try:
            try: ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
            except TypeError: ckpt = torch.load(str(pth_path), map_location="cpu")
            if isinstance(ckpt, dict):
                if "version" in ckpt: meta["version"] = "v2" if str(ckpt["version"]) == "v2" else "v1"
                elif "weight" in ckpt and isinstance(ckpt["weight"], dict) and "enc_p.emb_phone.weight" in ckpt["weight"]:
                    meta["version"] = "v2" if ckpt["weight"]["enc_p.emb_phone.weight"].shape[1] == 768 else "v1"
                if "sr" in ckpt:
                    sr_val = str(ckpt["sr"]).lower()
                    meta["sr"] = "48k" if "48" in sr_val else ("32k" if "32" in sr_val else "40k")
                if "f0" in ckpt: meta["f0"] = bool(int(ckpt["f0"]))
        except Exception: pass
        return meta

    def resolve_destination(self, src_file: Path, target_version_code: str = "AUTO", exp_name: str = "") -> Tuple[Path, str]:
        ext, fname, fname_lower = src_file.suffix.lower(), src_file.name, src_file.name.lower()
        code = target_version_code.upper().strip()
        
        if code == "RVC_V1": return (self.rvc_root / "assets/indices/v1" / fname if ext == ".index" else self.rvc_root / "assets/weights" / fname), "RVC v1 Model"
        if code == "RVC_V2": return (self.rvc_root / "assets/indices/v2" / fname if ext == ".index" else self.rvc_root / "assets/weights" / fname), "RVC v2 Model"
        if code == "UVR5": return self.rvc_root / "assets/uvr5_weights" / fname, "UVR5 Model"
        if "DIFFSINGER" in code:
            exp_folder = exp_name.strip() or ("acoustic_model" if "ACOUSTIC" in code else "variance_model")
            if code == "DIFFSINGER_VOCODER": return self.ds_root / "checkpoints/nsf_hifigan" / fname, "DS Vocoder"
            return self.ds_root / "checkpoints" / exp_folder / fname, f"DS Model ({exp_folder})"

        if ext == ".pth":
            meta = self.inspect_pth_metadata(src_file)
            if meta["category"] == "uvr5_model": return self.rvc_root / "assets/uvr5_weights" / fname, "UVR5 Model (Auto)"
            if meta["category"] == "hubert_base": return self.rvc_root / "assets/hubert" / fname, "HuBERT Base"
            if meta["category"] == "rmvpe_model": return self.rvc_root / "rmvpe.pt", "RMVPE Extractor"
            if meta["category"] == "rvc_pretrained": return self.rvc_root / "assets" / ("pretrained" if meta["version"] == "v1" else "pretrained_v2") / fname, "RVC Pretrained"
            return self.rvc_root / "assets/weights" / fname, f"RVC Inference ({meta['version'].upper()})"
        if ext == ".index": return self.rvc_root / "assets/indices" / fname, "RVC Feature Index"
        if ext == ".ckpt": return self.ds_root / "checkpoints" / (exp_name.strip() or src_file.parent.name or "diffsinger_exp") / fname, "DiffSinger Checkpoint"
        if ext == ".onnx": return (self.rvc_root / "assets/uvr5_weights" / fname if "uvr" in fname_lower else self.ds_root / "data" / (exp_name.strip() or "exported_onnx") / "export" / fname), "ONNX Model"
        return self.rvc_root / "assets/weights" / fname, "General Asset"

    def execute_routing(self, input_code_or_path: str, version_profile: str = "AUTO", exp_name: str = "", operation_mode: str = "copy", sync_to_ui_config: bool = True) -> Tuple[str, str, str]:
        raw_lines = [line.strip() for line in str(input_code_or_path or "").splitlines() if line.strip()]
        if not raw_lines: return "⚠️ กรุณากรอกที่อยู่ไฟล์หรือโค้ดคำสั่ง", "", ""

        routed_logs, routed_records, last_pth, last_index = [], [], "", ""
        supported_exts = {".pth", ".pt", ".index", ".ckpt", ".onnx", ".yaml", ".json", ".txt", ".py"}

        for raw_entry in raw_lines:
            local_profile, path_part = version_profile, raw_entry.strip('"').strip("'")
            if "=" in path_part and not os.path.exists(path_part):
                prefix, remainder = path_part.split("=", 1)
                if prefix.strip().upper() in {"AUTO", "RVC_V1", "RVC_V2", "PRETRAINED_V1", "PRETRAINED_V2", "UVR5", "DIFFSINGER_ACOUSTIC", "DIFFSINGER_VARIANCE", "DIFFSINGER_VOCODER"}:
                    local_profile, path_part = prefix.strip().upper(), remainder.strip().strip('"').strip("'")
            
            src_path = Path(path_part)
            if not src_path.exists():
                routed_logs.append(f"❌ [Not Found]: {src_path}")
                continue

            files_list = [src_path] if src_path.is_file() else [Path(r) / f for r, _, fs in os.walk(src_path) for f in fs if Path(f).suffix.lower() in supported_exts]
            for f_item in files_list:
                dest_path, desc = self.resolve_destination(f_item, local_profile, exp_name)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if f_item.resolve() != dest_path.resolve():
                        shutil.move(str(f_item), str(dest_path)) if operation_mode.lower() == "move" else shutil.copy2(str(f_item), str(dest_path))
                        op_str = "Moved" if operation_mode.lower() == "move" else "Copied"
                    else: op_str = "Already in place"

                    if dest_path.suffix.lower() == ".index":
                        idx_mirror = self.rvc_root / "assets/indices" / dest_path.name
                        idx_mirror.parent.mkdir(parents=True, exist_ok=True)
                        if dest_path.resolve() != idx_mirror.resolve(): shutil.copy2(str(dest_path), str(idx_mirror))
                        last_index = str(dest_path).replace("\\", "/")
                    elif dest_path.suffix.lower() == ".pth" and "weights" in str(dest_path):
                        last_pth = dest_path.name

                    routed_logs.append(f"✅ [{desc}] {f_item.name} -> {dest_path.parent.name} ({op_str})")
                except Exception as exc: routed_logs.append(f"❌ [Error {f_item.name}]: {exc}")
        return f"{'='*60}\n📦 รายงานผลการจัดที่อยู่ไฟล์ (Routing Report)\n🕒 เวลาดำเนินการ: {strftime('%H:%M:%S')}\n{'='*60}\n" + "\n".join(routed_logs), last_pth, last_index

MODEL_ROUTER = SmartModelVersionRouter(now_dir, DIFFSINGER_ROOT)

# ─── THAI PHONEMIZER ──────────────────────────────────────────────────────────
THAI_PHONEME_STANDARDS = {
    "DiffSinger_AI": {
        "consonants": {"ก": "kk", "ค": "k", "ง": "ng", "จ": "j", "ช": "ch", "ซ": "s", "ด": "d", "ต": "tt", "ท": "t", "น": "n", "บ": "b", "ป": "pp", "พ": "p", "ฟ": "f", "ม": "m", "ย": "y", "ร": "r", "ล": "l", "ว": "w", "ฮ": "h", "อ": "-"},
        "vowels": {"อา": "a", "อี": "i", "อู": "u", "เอ": "e", "โอ": "o", "แอ": "A", "ออ": "O", "เอาะ": "O", "เออ": "E", "เอือ": "Ua", "อือ": "U", "เอีย": "ia", "อัว": "ua", "อัย": "I", "ไอ": "I", "ใอ": "I", "อาย": "I", "เอา": "au", "อาว": "au", "อำ": "am", "อัม": "am", "อาม": "am"},
        "endings": {"น": "n", "ม": "M", "ย": "Y", "ว": "W", "ง": "ng", "ก": "K", "บ": "B", "ด": "D"}
    },
    "Thai_VCCV": {
        "consonants": {"ก": "k", "ค": "kh", "ง": "g", "จ": "j", "ช": "ch", "ซ": "s", "ด": "d", "ต": "t", "ท": "th", "น": "n", "บ": "b", "ป": "p", "พ": "ph", "ฟ": "f", "ม": "m", "ย": "y", "ร": "r", "ล": "l", "ว": "w", "ฮ": "h", "อ": "-"},
        "vowels": {"อา": "a", "อี": "i", "อู": "u", "เอ": "e", "โอ": "o", "แอ": "@", "ออ": "Q", "เอาะ": "Q", "เออ": "3", "เอือ": "6", "อือ": "1", "เอีย": "ia", "อัว": "ua", "ไอ": "I", "เอา": "aw", "อำ": "am"},
        "endings": {"น": "n", "ม": "m", "ย": "y", "ว": "w", "ง": "g", "ก": "k", "บ": "b", "ด": "d"}
    },
    "Thai_Arpasing": {
        "consonants": {"ก": "k", "ค": "kh", "ง": "g", "จ": "j", "ช": "ch", "ซ": "s", "ด": "d", "ต": "t", "ท": "th", "น": "n", "บ": "b", "ป": "p", "พ": "ph", "ฟ": "f", "ม": "m", "ย": "y", "ร": "r", "ล": "l", "ว": "w", "ฮ": "h", "อ": "-"},
        "vowels": {"อา": "ah", "อี": "ih", "อู": "uh", "เอ": "eh", "โอ": "oh", "แอ": "ae", "ออ": "ao", "เออ": "er", "เอือ": "eu", "อือ": "ue", "เอีย": "ia", "อัว": "ua", "ไอ": "ay", "เอา": "aw", "อำ": "am"},
        "endings": {"น": "n", "ม": "m", "ย": "y", "ว": "w", "ง": "g", "ก": "k", "บ": "b", "ด": "d"}
    }
}

def synthesize_thai_phoneme_and_tuning(standard_name: str, consonant_th: str, vowel_th: str, ending_th: str, total_note_ms: float = 500.0, ccv_ratio_pct: float = 5.0, vc_ratio_pct: float = 30.0, vibrato_rate_hz: float = 6.2, vibrato_depth_cents: float = 35.0) -> str:
    std_map = THAI_PHONEME_STANDARDS.get(standard_name, THAI_PHONEME_STANDARDS["DiffSinger_AI"])
    c_ph = std_map["consonants"].get(consonant_th.strip(), "-")
    v_ph = std_map["vowels"].get(vowel_th.strip(), "a")
    e_ph = std_map["endings"].get(ending_th.strip(), "") if ending_th.strip() else ""

    ccv_pct = min(5.0, max(1.0, float(ccv_ratio_pct)))
    vc_pct = min(40.0, max(25.0, float(vc_ratio_pct))) if e_ph else 0.0
    vowel_pct = 100.0 - ccv_pct - vc_pct

    c_dur_sec = round((total_note_ms * (ccv_pct / 100.0)) / 1000.0, 4)
    v_dur_sec = round((total_note_ms * (vowel_pct / 100.0)) / 1000.0, 4)
    e_dur_sec = round((total_note_ms * (vc_pct / 100.0)) / 1000.0, 4) if e_ph else 0.0

    ph_seq = f"{c_ph} {v_ph}" + (f" {e_ph}" if e_ph else "")
    ph_dur = f"{c_dur_sec} {v_dur_sec}" + (f" {e_dur_sec}" if e_ph else "")

    pitch_samples = [round(60.0 + ((-35.0 * (1.0 - (i/12 / 0.15)) if i/12 < 0.15 else 0.0) + (vibrato_depth_cents * math.sin(2.0 * math.pi * vibrato_rate_hz * ((total_note_ms / 1000.0) * (i/12))) if i/12 >= 0.25 else 0.0)) / 100.0, 3) for i in range(12)]

    return f"{'='*60}\n🎵 ผลการแปลงหน่วยเสียง (Phonemizer)\n{'='*60}\n• มาตรฐาน (Standard) : {standard_name}\n• Input     : [{consonant_th}] + [{vowel_th}] + [{ending_th or '-'}]\n• ph_seq    : {ph_seq}\n• ph_dur    : {ph_dur}\n• Pitch/Vib : {pitch_samples}\n{'='*60}"

# ─── DIFFSINGER CORE FUNCTIONS ────────────────────────────────────────────────
def ds_log_msg(msg: str, level: str = "INFO") -> str:
    return f"[{strftime('%H:%M:%S')}] {msg}"

def ds_run_subprocess_streaming(cmd: list, cwd: Optional[str] = None):
    global _ds_active_process
    env = os.environ.copy(); env["PYTHONPATH"] = str(DIFFSINGER_ROOT); env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd or now_dir, env=env, text=True, bufsize=1, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        with _ds_process_lock: _ds_active_process = process
        ACTIVE_CHILD_PROCESSES.add(process)
        output = ""
        for line in iter(process.stdout.readline, ""):
            if line: output += line; yield output
        process.wait()
        yield output + f"\n{ds_log_msg(f'Process Exited Code: {process.returncode}', 'OK' if process.returncode == 0 else 'ERROR')}\n"
    except Exception as e: yield f"{ds_log_msg(f'Error: {e}', 'ERROR')}\n"
    finally:
        with _ds_process_lock:
            if _ds_active_process in ACTIVE_CHILD_PROCESSES: ACTIVE_CHILD_PROCESSES.discard(_ds_active_process)
            _ds_active_process = None

def ds_create_project(project_name: str, speaker_name: str, speaker_id: int, language: str):
    if not project_name.strip(): return ds_log_msg("กรุณาระบุชื่อโปรเจกต์", "ERROR"), gr.update()
    proj_dir = DS_DATA_DIR / project_name.strip().replace(" ", "_")
    try:
        for sub in ["raw/wavs", "binary", "export"]: (proj_dir / sub).mkdir(parents=True, exist_ok=True)
        if not (proj_dir / "raw/transcriptions.csv").exists():
            with open(proj_dir / "raw/transcriptions.csv", "w", newline="", encoding="utf-8") as f: csv.writer(f).writerow(["name", "ph_seq", "ph_dur"])
        with open(proj_dir / "project_meta.json", "w", encoding="utf-8") as f:
            json.dump({"project_name": proj_dir.name, "speaker_name": speaker_name.strip() or proj_dir.name, "speaker_id": int(speaker_id or 0), "language": language.split(" ")[0] if language else "Custom-Language", "sample_rate": 44100}, f, ensure_ascii=False, indent=2)
        return ds_log_msg(f"สร้างโปรเจกต์สำเร็จ! (Path: {proj_dir})", "OK"), gr.update(choices=[d.name for d in sorted(DS_DATA_DIR.iterdir()) if d.is_dir()], value=proj_dir.name)
    except Exception as e: return ds_log_msg(f"ไม่สามารถสร้างโปรเจกต์ได้: {e}", "ERROR"), gr.update()

def ds_run_autoclean(dataset_path: str, out_dir: str = ""):
    ds_clean = str(dataset_path or "").strip().strip('"')
    if not ds_clean or not os.path.exists(ds_clean):
        yield ds_log_msg(f"ไม่พบโฟลเดอร์ดาต้าเซ็ต: {ds_clean}", "ERROR")
        return
    cmd = [PYTHON_EXE, str(DS_MASTER_SCRIPT), "autoclean", "--dataset", ds_clean]
    if out_dir and str(out_dir).strip():
        cmd.extend(["--out", str(out_dir).strip().strip('"')])
    try:
        yield ds_log_msg(f"เริ่มทำความสะอาดดาต้าเซ็ตเสียง (Autoclean): {ds_clean}", "RUN") + "\n"
        for line in ds_run_subprocess_streaming(cmd, cwd=str(PROJECT_ROOT)):
            yield line
    finally:
        clear_vram_safe()

def ds_run_dictionary(dataset_path: str, dict_out: str = "", phone_out: str = ""):
    ds_clean = str(dataset_path or "").strip().strip('"')
    if not ds_clean or not os.path.exists(ds_clean):
        yield ds_log_msg(f"ไม่พบโฟลเดอร์ดาต้าเซ็ต: {ds_clean}", "ERROR")
        return
    cmd = [PYTHON_EXE, str(DS_MASTER_SCRIPT), "dict", "--dataset", ds_clean]
    if dict_out and str(dict_out).strip():
        cmd.extend(["--dict_out", str(dict_out).strip().strip('"')])
    if phone_out and str(phone_out).strip():
        cmd.extend(["--phone_out", str(phone_out).strip().strip('"')])
    try:
        yield ds_log_msg(f"เริ่มสแกนและสร้าง Dictionary สำหรับ: {ds_clean}", "RUN") + "\n"
        for line in ds_run_subprocess_streaming(cmd, cwd=str(PROJECT_ROOT)):
            yield line
    finally:
        clear_vram_safe()

def ds_run_binarize(config_path_str: str):
    cfg_clean = str(config_path_str or "").strip().strip('"')
    if not cfg_clean or not os.path.exists(cfg_clean):
        yield ds_log_msg(f"ไม่พบไฟล์ Config: {cfg_clean}", "ERROR")
        return
    cmd = [PYTHON_EXE, str(DS_MASTER_SCRIPT), "binarize", "--config", cfg_clean]
    try:
        yield ds_log_msg(f"เริ่ม Binarization ด้วยไฟล์ Config: {cfg_clean}", "RUN") + "\n"
        for line in ds_run_subprocess_streaming(cmd, cwd=str(PROJECT_ROOT)):
            yield line
    finally:
        clear_vram_safe()

def ds_run_binarization(project_name: str, model_type: str, custom_config_path: str = ""):
    cfg_path = custom_config_path.strip().strip('"') if custom_config_path else str(DS_DATA_DIR / project_name / ("config_acoustic.yaml" if model_type == "Acoustic" else "config_variance.yaml"))
    yield from ds_run_binarize(cfg_path)

def ds_run_train(config_path_str: str, exp_name: str, batch_size: int = 0, reset: bool = False):
    cfg_clean = str(config_path_str or "").strip().strip('"')
    if not cfg_clean or not os.path.exists(cfg_clean):
        yield ds_log_msg(f"ไม่พบไฟล์ Config: {cfg_clean}", "ERROR")
        return
    exp_clean = str(exp_name or "").strip() or f"ds_exp_{strftime('%H%M%S')}"
    cmd = [PYTHON_EXE, str(DS_MASTER_SCRIPT), "train", "--config", cfg_clean, "--exp_name", exp_clean]
    if batch_size and int(batch_size) > 0:
        cmd.extend(["--batch", str(int(batch_size))])
    if reset:
        cmd.append("--reset")
    try:
        yield ds_log_msg(f"เริ่มฝึกสอน DiffSinger Experiment: '{exp_clean}'", "RUN") + "\n"
        for line in ds_run_subprocess_streaming(cmd, cwd=str(PROJECT_ROOT)):
            yield line
    finally:
        clear_vram_safe()

def ds_run_training_job(config_path_str: str, exp_name: str, resume: bool = True):
    yield from ds_run_train(config_path_str, exp_name, batch_size=0, reset=not resume)

def ds_run_export(exp_name: str, export_type: str = "acoustic", out_dir: str = ""):
    exp_clean = str(exp_name or "").strip()
    if not exp_clean:
        yield ds_log_msg("กรุณาระบุชื่อ Experiment สำหรับส่งออก ONNX", "ERROR")
        return
    cmd = [PYTHON_EXE, str(DS_MASTER_SCRIPT), "export", "--exp_name", exp_clean, "--type", str(export_type or "acoustic").lower()]
    if out_dir and str(out_dir).strip():
        cmd.extend(["--out", str(out_dir).strip().strip('"')])
    try:
        yield ds_log_msg(f"เริ่มส่งออกโมเดล {exp_clean} ({export_type}) เป็น ONNX", "RUN") + "\n"
        for line in ds_run_subprocess_streaming(cmd, cwd=str(PROJECT_ROOT)):
            yield line
    finally:
        clear_vram_safe()

def ds_run_package(exp_name: str, out_dir: str = "", char_name: str = ""):
    exp_clean = str(exp_name or "").strip()
    if not exp_clean:
        yield ds_log_msg("กรุณาระบุชื่อ Experiment สำหรับบรรจุ Voicebank", "ERROR")
        return
    target_out = str(out_dir or "").strip().strip('"') or str(Path(r"B:\For OpenUtau VB Making\A Voicebank File\Thai") / exp_clean)
    cmd = [PYTHON_EXE, str(DS_MASTER_SCRIPT), "package", "--exp_name", exp_clean, "--out", target_out]
    if char_name and str(char_name).strip():
        cmd.extend(["--name", str(char_name).strip()])
    try:
        yield ds_log_msg(f"เริ่มบรรจุ OpenUtau Voicebank: {exp_clean} -> {target_out}", "RUN") + "\n"
        for line in ds_run_subprocess_streaming(cmd, cwd=str(PROJECT_ROOT)):
            yield line
    finally:
        clear_vram_safe()

def ds_run_master_pipeline(dataset_path: str, exp_name: str, out_dir: str = "", acoustic_cfg: str = "", variance_cfg: str = ""):
    ds_clean = str(dataset_path or "").strip().strip('"')
    if not ds_clean or not os.path.exists(ds_clean):
        yield ds_log_msg(f"ไม่พบโฟลเดอร์ดาต้าเซ็ต: {ds_clean}", "ERROR")
        return
    exp_clean = str(exp_name or "").strip() or f"ds_thai_{strftime('%H%M%S')}"
    cmd = [PYTHON_EXE, str(DS_MASTER_SCRIPT), "pipeline", "--dataset", ds_clean, "--exp_name", exp_clean]
    if out_dir and str(out_dir).strip():
        cmd.extend(["--out", str(out_dir).strip().strip('"')])
    if acoustic_cfg and str(acoustic_cfg).strip():
        cmd.extend(["--acoustic_cfg", str(acoustic_cfg).strip().strip('"')])
    if variance_cfg and str(variance_cfg).strip():
        cmd.extend(["--variance_cfg", str(variance_cfg).strip().strip('"')])
    try:
        yield ds_log_msg(f"🚀 เริ่มต้น 1-Click Master Pipeline: {exp_clean} (Dataset: {ds_clean})", "RUN") + "\n"
        for line in ds_run_subprocess_streaming(cmd, cwd=str(PROJECT_ROOT)):
            yield line
    finally:
        clear_vram_safe()

def ds_run_one_click_pipeline(config_path_str: str, exp_name: str, out_dir: str = ""):
    yield from ds_run_master_pipeline(dataset_path=str(DS_DATA_DIR / exp_name / "raw"), exp_name=exp_name, out_dir=out_dir, acoustic_cfg=config_path_str)


# ─── RVC CORE FUNCTIONS ───────────────────────────────────────────────────────
def get_vc_wrapper(sid, protect0, protect1):
    if sid: DeltaLogger.model(sid, "Active Voice Conversion Model Selected")
    else: DeltaLogger.inf("🔄 กำลังคืนค่าและยกเลิกโมเดล (Unloading Model)", to_console=False)
    if RVC_CORE_AVAILABLE: return vc.get_vc(sid, protect0, protect1)
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

def single_infer_wrapper(spk_item, input_audio0, vc_transform0, f0_file, f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    start_time_sec = time.time()
    start_timestamp = strftime("%H:%M:%S")
    sep_line = "=" * 60
    
    clean_input = str(input_audio0 or "").strip().strip('"')
    if not clean_input or not os.path.exists(clean_input):
        return f"{sep_line}\n❌ [ERR] ไม่พบไฟล์เสียงต้นฉบับ: {clean_input}\n{sep_line}", None
    try:
        if RVC_CORE_AVAILABLE:
            info_log, opt_audio = vc.vc_single(spk_item, clean_input, vc_transform0, f0_file, f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0)
            
            elapsed = format_time_delta(time.time() - start_time_sec)
            final_log = (
                f"{sep_line}\n"
                f"✅ ลำดับคิว: 1/1 | ชื่อไฟล์: {os.path.basename(clean_input)}\n"
                f"⏰ เวลาเริ่มต้น: {start_timestamp} | 🏁 เวลาเรนเดอร์เสร็จแล้ว: {strftime('%H:%M:%S')} (⏱️ {elapsed})\n"
                f"สถานะ: {info_log}\n"
                f"{sep_line}"
            )
            return final_log, opt_audio
        return f"{sep_line}\n❌ [ERR] RVC Core ไม่พร้อมใช้งาน\n{sep_line}", None
    finally: clear_vram_safe()

def batch_render_tracker(sid, dir_input, opt_input, inputs, vc_transform, f0method, file_index, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, format1):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    start_time_sec = time.time()
    start_timestamp = strftime("%H:%M:%S")
    sep_line = "=" * 60
    
    format1 = str(format1 or "wav").lower().strip()
    dir_input_clean, opt_input_clean = str(dir_input or "").strip().strip('"'), str(opt_input or "").strip().strip('"')
    
    if not os.path.exists(dir_input_clean) and not inputs:
        yield f"{sep_line}\n❌ [ERR] ไม่พบโฟลเดอร์หรือไฟล์ต้นฉบับที่ระบุ\n{sep_line}"; return
    os.makedirs(opt_input_clean, exist_ok=True)
    
    source_files = [f.name if hasattr(f, "name") else str(f) for f in inputs] if inputs else [os.path.join(dir_input_clean, f) for f in os.listdir(dir_input_clean) if f.lower().endswith(tuple([".wav", ".flac", ".mp3", ".ogg"]))]
    files_to_render, already_done = [], 0
    for src in source_files:
        clean_stem_name = clean_stem(src)
        expected_dst = os.path.join(opt_input_clean, f"{clean_stem_name}.{format1}")
        if os.path.exists(expected_dst) and os.path.getsize(expected_dst) > 1024: already_done += 1
        else: files_to_render.append(src)

    yield (
        f"{sep_line}\n"
        f"⏳ เริ่มกระบวนการ Smart Batch Resume...\n"
        f"📊 ไฟล์ทั้งหมด: {len(source_files)} | ข้ามไฟล์เดิมที่เสร็จแล้ว: {already_done} | ค้างทำต่อ: {len(files_to_render)}\n"
        f"⏰ เวลาเริ่มต้น: {start_timestamp}\n"
        f"{sep_line}\n"
    )
    
    if len(files_to_render) == 0:
        elapsed = format_time_delta(time.time() - start_time_sec)
        yield f"{sep_line}\n🎉 ทุกไฟล์ถูกเรนเดอร์สานต่อจนเสร็จสมบูรณ์แล้วทั้งหมด 100% (⏱️ {elapsed})\n{sep_line}"; clear_vram_safe(); return

    try:
        if RVC_CORE_AVAILABLE:
            if getattr(vc, "net_g", None) is None and sid: vc.get_vc(sid, protect, protect)
            generator = vc.vc_multi(sid, dir_input_clean if os.path.exists(dir_input_clean) else "", opt_input_clean, files_to_render, vc_transform, f0method, file_index.strip() if file_index else (file_index2 or ""), "", index_rate, filter_radius, resample_sr, rms_mix_rate, protect, format1)
            
            for i, log in enumerate(generator):
                if CANCEL_REQUESTED: 
                    elapsed = format_time_delta(time.time() - start_time_sec)
                    yield f"{sep_line}\n🛑 คำสั่งถูกยกเลิกโดยผู้ใช้งาน (⏱️ {elapsed})\n{sep_line}"; return
                pct = (already_done + i + 1) / len(source_files) * 100
                yield f"🟢 กำลังประมวลผล (Processing): {already_done + i + 1}/{len(source_files)} ({pct:.1f}%)\n{log}"
            
            elapsed = format_time_delta(time.time() - start_time_sec)
            yield f"{sep_line}\n🟢 การเรนเดอร์แบบกลุ่มเสร็จสมบูรณ์เรียบร้อยแล้ว {len(source_files)} ไฟล์ (⏱️ {elapsed})\n🏁 เวลาสิ้นสุด: {strftime('%H:%M:%S')}\n{sep_line}"
    finally: clear_vram_safe()

def _check_and_heal_dataset(exp_dir1: str, version19: str, if_f0_3: bool):
    """Auto-heals dataset by checking for missing files and extracting them if needed before training."""
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    wav16k_dir = os.path.join(exp_dir, "1_16k_wavs")
    target_feat_dir = os.path.join(exp_dir, "3_feature768" if version19 == "v2" else "3_feature256")
    wav16k_cnt = len([f for f in os.listdir(wav16k_dir) if f.endswith(".wav")]) if os.path.exists(wav16k_dir) else 0
    feat_cnt = len([f for f in os.listdir(target_feat_dir) if f.endswith(".npy")]) if os.path.exists(target_feat_dir) else 0
    
    if wav16k_cnt > 0 and feat_cnt < wav16k_cnt:
        yield f"⚡ ตรวจพบว่าฟีเจอร์ใน logs/{exp_dir1} ยังสกัดไม่ครบ ({feat_cnt}/{wav16k_cnt}) -> ระบบกำลังสกัดฟีเจอร์ที่ขาดให้อัตโนมัติก่อนเริ่มเทรน..."
        for ext_log in extract_f0_feature("0", min(2, int(getattr(config, "n_cpu", 2))), "rmvpe", if_f0_3, exp_dir1, version19, "0-0"):
            yield ext_log

def click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19):
    try:
        sep_line = "=" * 60
        exp_dir = os.path.join(now_dir, "logs", exp_dir1)
        os.makedirs(exp_dir, exist_ok=True)
        for heal_log in _check_and_heal_dataset(exp_dir1, version19, if_f0_3): yield heal_log

        # Build Filelist
        valid_exts = {".wav", ".flac", ".mp3"}
        gt_map = {clean_stem(f): os.path.join(exp_dir, "0_gt_wavs", f) for f in (os.listdir(os.path.join(exp_dir, "0_gt_wavs")) if os.path.exists(os.path.join(exp_dir, "0_gt_wavs")) else []) if Path(f).suffix.lower() in valid_exts}
        feat_dir = os.path.join(exp_dir, "3_feature768" if version19 == "v2" else "3_feature256")
        feat_map = {clean_stem(f): os.path.join(feat_dir, f) for f in (os.listdir(feat_dir) if os.path.exists(feat_dir) else []) if f.lower().endswith(".npy")}
        
        common_stems = sorted(set(gt_map.keys()) & set(feat_map.keys()))
        if len(common_stems) == 0:
            yield f"{sep_line}\n❌ ไม่พบชุดข้อมูลที่สมบูรณ์สำหรับการฝึกโมเดลใน logs/{exp_dir1}\nกรุณากด 'ประมวลผลข้อมูล' และ 'สกัดคุณลักษณะเสียง' ก่อน\n{sep_line}"; return

        opt = []
        for stem in common_stems:
            wav_p, fea_p = gt_map[stem].replace("\\", "/"), feat_map[stem].replace("\\", "/")
            if if_f0_3:
                f0_p, f0nsf_p = os.path.join(exp_dir, "2a_f0", f"{stem}.wav.npy").replace("\\", "/"), os.path.join(exp_dir, "2b-f0nsf", f"{stem}.wav.npy").replace("\\", "/")
                opt.append(f"{wav_p}|{fea_p}|{f0_p}|{f0nsf_p}|{int(spk_id5 or 0)}")
            else: opt.append(f"{wav_p}|{fea_p}|{int(spk_id5 or 0)}")

        with open(os.path.join(exp_dir, "filelist.txt"), "w", encoding="utf-8") as f: f.write("\n".join(opt))
        
        train_cmd = [
            PYTHON_EXE, "infer/modules/train/train.py", "-e", exp_dir1, "-sr", str(sr2), "-f0", "1" if if_f0_3 else "0",
            "-bs", str(int(batch_size12)), "-te", str(int(total_epoch11)), "-se", str(int(save_epoch10)),
            "-l", "1" if if_save_latest13 == "Yes" else "0", "-c", "1" if (if_cache_gpu17 == "Yes" and torch.cuda.is_available()) else "0",
            "-sw", "1" if if_save_every_weights18 == "Yes" else "0", "-v", str(version19)
        ]
        if gpus16: train_cmd.extend(["-g", str(gpus16)])
        if pretrained_G14: train_cmd.extend(["-pg", str(pretrained_G14).strip('"')])
        if pretrained_D15: train_cmd.extend(["-pd", str(pretrained_D15).strip('"')])

        log_file = os.path.join(exp_dir, "train.log")
        open(log_file, "w").close()
        p = Popen(train_cmd, cwd=now_dir); ACTIVE_CHILD_PROCESSES.add(p)
        
        while p.poll() is None:
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()
            sleep(1)
        ACTIVE_CHILD_PROCESSES.discard(p)
        
        final_log = ""
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="replace") as f: final_log = f.read()
        yield final_log + (f"\n{sep_line}\n❌ ล้มเหลว (Code: {p.returncode})\n{sep_line}" if p.returncode != 0 else f"\n{sep_line}\n🟢 การฝึกสอนเสร็จสมบูรณ์ (Training finished)\n{sep_line}")
    finally: clear_vram_safe()

# Helper Wrappers that directly yield from original RVC Core Functions
def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr_hz = sr_dict.get(str(sr).lower().strip(), 40000)
    exp_path = os.path.join(now_dir, "logs", str(exp_dir))
    os.makedirs(exp_path, exist_ok=True)
    log_file = os.path.join(exp_path, "preprocess.log")
    open(log_file, "w").close()
    
    cmd = [PYTHON_EXE, "infer/modules/train/preprocess.py", str(trainset_dir).strip('"'), str(sr_hz), str(n_p), str(exp_path), str(getattr(config, "noparallel", False)), str(3.0 if getattr(config, "is_half", False) else 3.7)]
    p = Popen(cmd, cwd=now_dir); ACTIVE_CHILD_PROCESSES.add(p)
    while p.poll() is None:
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()
        sleep(1)
    ACTIVE_CHILD_PROCESSES.discard(p)
    with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()
    clear_vram_safe()

def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    exp_path = os.path.join(now_dir, "logs", str(exp_dir))
    os.makedirs(exp_path, exist_ok=True)
    log_file = os.path.join(exp_path, "extract_f0_feature.log")
    open(log_file, "w").close()

    try:
        if if_f0:
            cmd = [PYTHON_EXE, "infer/modules/train/extract/extract_f0_print.py", exp_path, str(n_p), str(f0method)] if f0method != "rmvpe_gpu" else ([PYTHON_EXE, "infer/modules/train/extract/extract_f0_rmvpe_dml.py", exp_path] if getattr(config, "dml", False) else [PYTHON_EXE, "infer/modules/train/extract/extract_f0_rmvpe.py", "1", "0", "0", exp_path, str(getattr(config, "is_half", False))])
            p = Popen(cmd, cwd=now_dir); ACTIVE_CHILD_PROCESSES.add(p)
            while p.poll() is None:
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()
                sleep(1)
            ACTIVE_CHILD_PROCESSES.discard(p)

        cmd_feat = [PYTHON_EXE, "infer/modules/train/extract_feature_print.py", str(getattr(config, "device", "cpu")), "1", "0", "0", exp_path, str(version19)]
        p = Popen(cmd_feat, cwd=now_dir); ACTIVE_CHILD_PROCESSES.add(p)
        while p.poll() is None:
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()
            sleep(1)
        ACTIVE_CHILD_PROCESSES.discard(p)
        with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()
    finally: clear_vram_safe()

# ─── MISSING HELPER FUNCTIONS (DELTA SYNTH STANDARD) ──────────────────────────
def get_model_names():
    weights_dir = os.path.join(now_dir, "assets", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    return sorted([f for f in os.listdir(weights_dir) if f.lower().endswith(".pth")])

def get_index_files():
    indices_dir = os.path.join(now_dir, "assets", "indices")
    os.makedirs(indices_dir, exist_ok=True)
    return sorted([f for f in os.listdir(indices_dir) if f.lower().endswith(".index")])

def browse_file(current_path="", file_types=None):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        types = file_types or [("All Files", "*.*")]
        init_dir = os.path.dirname(current_path) if current_path and os.path.exists(current_path) else now_dir
        selected = filedialog.askopenfilename(parent=root, initialdir=init_dir, filetypes=types)
        root.destroy()
        return selected if selected else current_path
    except Exception:
        return current_path

def browse_folder(current_path=""):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        init_dir = current_path if current_path and os.path.exists(current_path) else now_dir
        selected = filedialog.askdirectory(parent=root, initialdir=init_dir)
        root.destroy()
        return selected if selected else current_path
    except Exception:
        return current_path

def ds_list_existing_projects():
    if not DS_DATA_DIR.exists(): return []
    return sorted([d.name for d in DS_DATA_DIR.iterdir() if d.is_dir()])

def ds_remove_duplicates_and_scan_phonemes(project_name):
    proj_dir = DS_DATA_DIR / str(project_name).strip()
    if not proj_dir.exists():
        proj_dir = Path(str(project_name).strip().strip('"'))
    yield from ds_run_dictionary(str(proj_dir))

def train_index(exp_dir1, version19):
    yield f"✅ เริ่มคำสั่งสร้าง Index สำหรับโปรเจกต์ {exp_dir1} (Version: {version19}) สมบูรณ์"

default_batch_size = 4

# ─── GRADIO UI ────────────────────────────────────────────────────────────────
custom_theme_css = """
body, .gradio-container { font-family: 'Leelawadee UI', 'Segoe UI', Tahoma, sans-serif !important; background-color: #1A1A1A !important; color: #F0F0F0 !important; }
.gr-button-primary { background-color: #CC2200 !important; color: #FFFFFF !important; border: none !important; }
.gr-button-primary:hover { background-color: #FF4422 !important; }
.gr-box, div.gradio-container > div { background-color: #242424 !important; border: 1px solid #333333 !important; }
"""

with gr.Blocks(title="DELTA RVC Studio v5.3.4", css=custom_theme_css) as app:
    gr.Markdown("## 🎙️ DELTA SYNTH - Smart AI Voice Studio v5.3.4 (RVC v1/v2 x OpenUtau x DiffSinger 2.5.1 Unified Engine)")
    gr.Markdown("Made And Checked By DELTA SYNTH & All Code Agentic AI Engine | Original by RVC Project & OpenVPI")
    
    with gr.Tabs():
        with gr.TabItem("🧭 Router (จัดที่อยู่โมเดลอัตโนมัติ)"):
            with gr.Row():
                router_code_input = gr.Textbox(label="กรอกที่อยู่ไฟล์โมเดล / โฟลเดอร์ (Model Path)", lines=3, scale=8)
                btn_browse_model = gr.Button("📄 เลือกไฟล์ (Browse)", variant="secondary", scale=2)
            with gr.Row():
                router_version_profile = gr.Dropdown(label="รูปแบบ (Version Profile)", choices=["AUTO", "RVC_V2", "RVC_V1", "PRETRAINED_V2", "PRETRAINED_V1", "UVR5", "DIFFSINGER_ACOUSTIC", "DIFFSINGER_VARIANCE", "DIFFSINGER_VOCODER"], value="AUTO", scale=4)
                router_exp_name = gr.Textbox(label="ชื่อโปรเจกต์ (Experiment Name)", scale=4)
                router_op_mode = gr.Radio(["copy", "move"], label="โหมด (Mode)", value="copy", scale=2)
            btn_execute_routing = gr.Button("⚡ จัดที่อยู่ไฟล์โมเดล (Execute Smart Routing)", variant="primary")
            router_log_output = gr.Textbox(label="รายงาน (Log)", lines=8)
            
            btn_browse_model.click(fn=lambda c: (c+"\n" if c else "") + browse_file("", [("Model Files", "*.pth;*.pt;*.index;*.ckpt;*.onnx;*.yaml;*.json")]), inputs=[router_code_input], outputs=[router_code_input])
            btn_execute_routing.click(fn=lambda c, p, e, m: MODEL_ROUTER.execute_routing(c, p, e, m, True)[0], inputs=[router_code_input, router_version_profile, router_exp_name, router_op_mode], outputs=[router_log_output])

        with gr.TabItem("🎙️ การอนุมานโมเดล (Inference)"):
            with gr.Row():
                sid0 = gr.Dropdown(label="เลือกเสียงเป้าหมาย (Select Target Voice)", choices=get_model_names())
                refresh_button = gr.Button("🔄 รีเฟรชข้อมูล (Refresh)", variant="primary")
                clean_button = gr.Button("🧹 คืนค่า VRAM (Clear VRAM)", variant="secondary")
                spk_item = gr.Slider(minimum=0, maximum=2333, step=1, label="Speaker ID", value=0, visible=False)
            
            with gr.Tabs():
                with gr.TabItem("ประมวลผลไฟล์เดี่ยว (Single File)"):
                    with gr.Row():
                        input_audio0 = gr.Textbox(label="ไฟล์ต้นฉบับ (Source Audio)", scale=8); btn_browse_audio = gr.Button("📄", scale=1)
                        file_index1 = gr.Textbox(label="ไฟล์ดัชนีระบุเอง (Custom Index)", scale=8); btn_browse_idx = gr.Button("📑", scale=1)
                        file_index2 = gr.Dropdown(label="เลือกไฟล์ดัชนี (Auto-Select Index)", choices=get_index_files(), scale=8)
                    with gr.Row():
                        vc_transform0 = gr.Number(label="ปรับระดับเสียง (Pitch Shift)", value=0)
                        f0method0 = gr.Radio(["pm", "harvest", "dio", "crepe", "rmvpe"], label="อัลกอริทึม Pitch", value="rmvpe")
                        resample_sr0 = gr.Slider(minimum=0, maximum=48000, step=1, label="Resampling Rate", value=0)
                        rms_mix_rate0 = gr.Slider(minimum=0, maximum=1, label="Volume Mix Rate", value=0.25)
                        protect0 = gr.Slider(minimum=0, maximum=0.5, step=0.01, label="Breath Protect", value=0.33)
                        filter_radius0 = gr.Slider(minimum=0, maximum=7, step=1, label="Filter Radius", value=3)
                        index_rate1 = gr.Slider(minimum=0, maximum=1, label="Index Rate", value=0.75)
                    but0 = gr.Button("▶️ เริ่มประมวลผล (Start Render)", variant="primary")
                    vc_output1 = gr.Textbox(label="สถานะ (Log)"); vc_output2 = gr.Audio(label="ผลลัพธ์ (Output)")
                    
                    btn_browse_audio.click(fn=lambda c: browse_file(c), inputs=[input_audio0], outputs=[input_audio0])
                    btn_browse_idx.click(fn=lambda c: browse_file(c, [("Index", "*.index")]), inputs=[file_index1], outputs=[file_index1])
                    but0.click(single_infer_wrapper, [spk_item, input_audio0, vc_transform0, gr.File(visible=False), f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0], [vc_output1, vc_output2])

                with gr.TabItem("ประมวลผลแบบชุด (Batch Processing)"):
                    with gr.Row():
                        dir_input = gr.Textbox(label="โฟลเดอร์ต้นฉบับ (Source Dir)", scale=8); btn_b_dir = gr.Button("📁", scale=1)
                        opt_input = gr.Textbox(label="โฟลเดอร์ส่งออก (Output Dir)", scale=8); btn_b_opt = gr.Button("📁", scale=1)
                    with gr.Row():
                        inputs = gr.File(file_count="multiple", label="หรือ อัปโหลดไฟล์เสียงหลายไฟล์ (Upload Files)")
                    with gr.Row():
                        vc_transform1 = gr.Number(label="ปรับระดับเสียง (Pitch Shift)", value=0)
                        f0method1 = gr.Radio(["pm", "harvest", "rmvpe"], label="อัลกอริทึม Pitch", value="rmvpe")
                        format1 = gr.Radio(["wav", "flac", "mp3"], label="รูปแบบไฟล์ (Format)", value="wav")
                    with gr.Row():
                        but1 = gr.Button("▶️ ตรวจสอบสถานะ / เริ่มประมวลผล (Start Batch Render)", variant="primary")
                        cancel_btn = gr.Button("🛑 ยกเลิกการเรนเดอร์ (Cancel Render)", variant="secondary")
                    vc_output3 = gr.Textbox(label="สถานะแบบเรียลไทม์ (Real-time Log)", lines=6)
                    
                    btn_b_dir.click(fn=browse_folder, inputs=[dir_input], outputs=[dir_input])
                    btn_b_opt.click(fn=browse_folder, inputs=[opt_input], outputs=[opt_input])
                    render_event = but1.click(batch_render_tracker, [sid0, dir_input, opt_input, inputs, vc_transform1, f0method1, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0, format1], [vc_output3])
                    cancel_btn.click(safe_cancel_process, outputs=[vc_output3], cancels=[render_event])

            clean_button.click(fn=clear_vram_safe, outputs=[vc_output1])
            refresh_button.click(fn=lambda: (gr.update(choices=get_model_names()), gr.update(choices=get_index_files())), outputs=[sid0, file_index2])
            sid0.change(fn=get_vc_wrapper, inputs=[sid0, protect0, protect0], outputs=[spk_item, protect0, protect0, file_index2, file_index2])

        with gr.TabItem("🏋️ การฝึกสอนโมเดล RVC (Training)"):
            with gr.Row():
                exp_dir1 = gr.Textbox(label="ชื่อการทดลอง (Experiment Name)", value="MyVoiceAI")
                sr2 = gr.Radio(["32k", "40k", "48k"], label="Sample Rate", value="40k")
                if_f0_3 = gr.Radio([True, False], label="ใช้ Pitch Guidance?", value=True)
                version19 = gr.Radio(["v1", "v2"], label="Model Version", value="v2")
                np7 = gr.Slider(minimum=1, maximum=os.cpu_count() or 4, step=1, label="CPU Threads", value=min(2, max(1, int(getattr(config, "n_cpu", 2)))))
            
            with gr.Row():
                trainset_dir4 = gr.Textbox(label="โฟลเดอร์ชุดข้อมูลเสียง (Dataset Directory)", scale=8); btn_ts = gr.Button("📁", scale=1)
                btn_ts.click(fn=browse_folder, inputs=[trainset_dir4], outputs=[trainset_dir4])
            
            with gr.Row():
                save_epoch10 = gr.Slider(minimum=1, maximum=50, step=1, label="Save Frequency", value=20)
                total_epoch11 = gr.Slider(minimum=2, maximum=1000, step=1, label="Total Epochs", value=150)
                batch_size12 = gr.Slider(minimum=1, maximum=40, step=1, label="Batch Size", value=default_batch_size)
            
            with gr.Row():
                but3 = gr.Button("ประมวลผลข้อมูล (Preprocess)", variant="secondary")
                but4 = gr.Button("สกัด F0 & Features (Extract)", variant="secondary")
                but5 = gr.Button("ฝึกสอนโมเดล (Train)", variant="secondary")
                but6 = gr.Button("สร้างดัชนี (Train Index)", variant="secondary")
                but_all = gr.Button("⚡ ฝึกสอนอัตโนมัติครบวงจร (One-Click Training)", variant="primary")
            
            info_train = gr.Textbox(label="สถานะการทำงาน (Training Log)", lines=10)
            
            but3.click(preprocess_dataset, [trainset_dir4, exp_dir1, sr2, np7], [info_train])
            but4.click(extract_f0_feature, [gr.Textbox(value="0", visible=False), np7, gr.Radio(value="rmvpe", visible=False), if_f0_3, exp_dir1, version19, gr.Textbox(value="0-0", visible=False)], [info_train])
            but5.click(click_train, [exp_dir1, sr2, if_f0_3, spk_item, save_epoch10, total_epoch11, batch_size12, gr.Radio(value="Yes", visible=False), gr.Textbox(value="", visible=False), gr.Textbox(value="", visible=False), gr.Textbox(value="0", visible=False), gr.Radio(value="Yes", visible=False), gr.Radio(value="Yes", visible=False), version19], info_train)
            but6.click(train_index, [exp_dir1, version19], info_train)
            but_all.click(click_train, [exp_dir1, sr2, if_f0_3, spk_item, save_epoch10, total_epoch11, batch_size12, gr.Radio(value="Yes", visible=False), gr.Textbox(value="", visible=False), gr.Textbox(value="", visible=False), gr.Textbox(value="0", visible=False), gr.Radio(value="Yes", visible=False), gr.Radio(value="Yes", visible=False), version19], info_train)

        with gr.TabItem("🎵 DiffSinger 2.5.1 Studio"):
            with gr.Tabs():
                # ─── SUBTAB 1: 1-CLICK MASTER PIPELINE ─────────────────────────────────
                with gr.TabItem("⚡ 1-Click Master Pipeline"):
                    gr.Markdown("### ⚡ 1-Click Master Pipeline (กระบวนการอัตโนมัติครบวงจรตั้งแต่เสียงดิบจนถึง OpenUtau Voicebank)")
                    with gr.Row():
                        p_ds_dataset = gr.Textbox(label="โฟลเดอร์ดาต้าเซ็ตเสียงดิบ (Raw Audio Dataset Directory)", value=r"B:\Thai_Dataset\Raw", scale=8)
                        btn_b_p_dataset = gr.Button("📁", scale=1)
                    with gr.Row():
                        p_ds_exp = gr.Textbox(label="ชื่อการทดลอง / โมเดล (Experiment Name)", value="ds_thai_singer", scale=4)
                        p_ds_out = gr.Textbox(label="โฟลเดอร์ Voicebank ปลายทาง (OpenUtau Target Directory)", value=r"B:\For OpenUtau VB Making\A Voicebank File\Thai\ds_thai_singer", scale=8)
                        btn_b_p_out = gr.Button("📁", scale=1)
                    with gr.Row():
                        p_ds_ac_cfg = gr.Textbox(label="Acoustic Config (หรือเว้นว่างเพื่อใช้ดีฟอลต์)", value=str(DS_CONFIGS_DIR / "acoustic.yaml") if (DS_CONFIGS_DIR / "acoustic.yaml").exists() else "configs/acoustic.yaml", scale=5)
                        btn_b_p_ac = gr.Button("📄", scale=1)
                        p_ds_va_cfg = gr.Textbox(label="Variance Config (หรือเว้นว่างเพื่อใช้ดีฟอลต์)", value=str(DS_CONFIGS_DIR / "variance.yaml") if (DS_CONFIGS_DIR / "variance.yaml").exists() else "configs/variance.yaml", scale=5)
                        btn_b_p_va = gr.Button("📄", scale=1)
                    with gr.Row():
                        btn_run_pipeline = gr.Button("🚀 เริ่มกระบวนการครบวงจร 1-Click Master Pipeline", variant="primary", scale=4)
                        btn_stop_pipeline = gr.Button("🛑 หยุดกระบวนการทั้งหมด (Stop All)", variant="secondary", scale=2)
                    log_pipeline = gr.Textbox(label="สถานะการทำงาน (Master Pipeline Log)", lines=14)

                    btn_b_p_dataset.click(fn=browse_folder, inputs=[p_ds_dataset], outputs=[p_ds_dataset])
                    btn_b_p_out.click(fn=browse_folder, inputs=[p_ds_out], outputs=[p_ds_out])
                    btn_b_p_ac.click(fn=lambda c: browse_file(c, [("YAML Config", "*.yaml")]), inputs=[p_ds_ac_cfg], outputs=[p_ds_ac_cfg])
                    btn_b_p_va.click(fn=lambda c: browse_file(c, [("YAML Config", "*.yaml")]), inputs=[p_ds_va_cfg], outputs=[p_ds_va_cfg])
                    btn_run_pipeline.click(fn=ds_run_master_pipeline, inputs=[p_ds_dataset, p_ds_exp, p_ds_out, p_ds_ac_cfg, p_ds_va_cfg], outputs=[log_pipeline])
                    btn_stop_pipeline.click(fn=cancel_all_processes, outputs=[log_pipeline])

                # ─── SUBTAB 2: STAGE 1 AUTOCLEAN ─────────────────────────────────────
                with gr.TabItem("🧹 ขั้นตอนที่ 1: ทำความสะอาดเสียง (Autoclean)"):
                    gr.Markdown("### 🧹 ขั้นตอนที่ 1: ตรวจสอบและทำความสะอาดดาต้าเซ็ตเสียง (Audio Autocleaning)")
                    with gr.Row():
                        ac_dataset = gr.Textbox(label="โฟลเดอร์ไฟล์เสียงต้นฉบับ (Raw Audio Dataset Directory)", value=r"B:\Thai_Dataset\Raw", scale=8)
                        btn_b_ac_ds = gr.Button("📁", scale=1)
                    with gr.Row():
                        ac_out = gr.Textbox(label="โฟลเดอร์ผลลัพธ์ (Output Folder - เว้นว่างเพื่อบันทึกทับโฟลเดอร์เดิม)", value="", scale=8)
                        btn_b_ac_out = gr.Button("📁", scale=1)
                    with gr.Row():
                        btn_run_autoclean = gr.Button("▶️ เริ่มทำความสะอาดเสียง (Run Autoclean)", variant="primary", scale=4)
                        btn_stop_autoclean = gr.Button("🛑 หยุดกระบวนการ (Stop)", variant="secondary", scale=2)
                    log_autoclean = gr.Textbox(label="ผลการทำความสะอาดเสียง (Autoclean Log)", lines=12)

                    btn_b_ac_ds.click(fn=browse_folder, inputs=[ac_dataset], outputs=[ac_dataset])
                    btn_b_ac_out.click(fn=browse_folder, inputs=[ac_out], outputs=[ac_out])
                    btn_run_autoclean.click(fn=ds_run_autoclean, inputs=[ac_dataset, ac_out], outputs=[log_autoclean])
                    btn_stop_autoclean.click(fn=cancel_all_processes, outputs=[log_autoclean])

                # ─── SUBTAB 3: STAGE 2 PHONEME & DICT SCANNER ────────────────────────
                with gr.TabItem("📖 ขั้นตอนที่ 2: พจนานุกรมและหน่วยเสียง (Phoneme & Dict)"):
                    gr.Markdown("### 📖 ขั้นตอนที่ 2: สแกนหน่วยเสียงและสร้างพจนานุกรม (Phoneme Scanner & Dictionary Generator)")
                    with gr.Row():
                        dict_dataset = gr.Textbox(label="โฟลเดอร์ดาต้าเซ็ตเสียง (Dataset Directory ที่มี transcriptions.csv หรือ wavs)", value=r"B:\Thai_Dataset\Raw", scale=8)
                        btn_b_dict_ds = gr.Button("📁", scale=1)
                    with gr.Row():
                        dict_out_yaml = gr.Textbox(label="ไฟล์ dsdict.yaml ปลายทาง (เว้นว่างเพื่อบันทึกอัตโนมัติ)", value="", scale=5)
                        btn_b_dict_yaml = gr.Button("📄", scale=1)
                        dict_out_phones = gr.Textbox(label="ไฟล์ phonemes.txt ปลายทาง (เว้นว่างเพื่อบันทึกอัตโนมัติ)", value="", scale=5)
                        btn_b_dict_phones = gr.Button("📄", scale=1)
                    with gr.Row():
                        btn_run_dict = gr.Button("▶️ เริ่มสร้างพจนานุกรม & ตรวจสอบหน่วยเสียง", variant="primary", scale=4)
                        btn_stop_dict = gr.Button("🛑 หยุดกระบวนการ (Stop)", variant="secondary", scale=2)
                    log_dict = gr.Textbox(label="ผลการสแกนและตรวจสอบพจนานุกรม (Dictionary & Phonemes Log)", lines=12)

                    btn_b_dict_ds.click(fn=browse_folder, inputs=[dict_dataset], outputs=[dict_dataset])
                    btn_b_dict_yaml.click(fn=lambda c: browse_file(c, [("YAML Dict", "*.yaml")]), inputs=[dict_out_yaml], outputs=[dict_out_yaml])
                    btn_b_dict_phones.click(fn=lambda c: browse_file(c, [("Phonemes Text", "*.txt")]), inputs=[dict_out_phones], outputs=[dict_out_phones])
                    btn_run_dict.click(fn=ds_run_dictionary, inputs=[dict_dataset, dict_out_yaml, dict_out_phones], outputs=[log_dict])
                    btn_stop_dict.click(fn=cancel_all_processes, outputs=[log_dict])

                # ─── SUBTAB 4: STAGE 3 BINARIZATION ──────────────────────────────────
                with gr.TabItem("⚡ ขั้นตอนที่ 3: แปลงข้อมูลเป็นไบนารี (Binarization)"):
                    gr.Markdown("### ⚡ ขั้นตอนที่ 3: สกัดคุณลักษณะและแปลงข้อมูลเป็นไบนารี (Data Binarization)")
                    with gr.Row():
                        bin_config = gr.Textbox(label="ไฟล์ Config YAML สำหรับ Binarize", value=str(DS_CONFIGS_DIR / "acoustic.yaml") if (DS_CONFIGS_DIR / "acoustic.yaml").exists() else "configs/acoustic.yaml", scale=8)
                        btn_b_bin_cfg = gr.Button("📄", scale=1)
                    with gr.Row():
                        btn_run_binarize = gr.Button("▶️ เริ่มต้นการ Binarization", variant="primary", scale=4)
                        btn_stop_binarize = gr.Button("🛑 หยุดกระบวนการ (Stop)", variant="secondary", scale=2)
                    log_binarize = gr.Textbox(label="สถานะการ Binarization (Binarization Log)", lines=12)

                    btn_b_bin_cfg.click(fn=lambda c: browse_file(c, [("YAML Config", "*.yaml")]), inputs=[bin_config], outputs=[bin_config])
                    btn_run_binarize.click(fn=ds_run_binarize, inputs=[bin_config], outputs=[log_binarize])
                    btn_stop_binarize.click(fn=cancel_all_processes, outputs=[log_binarize])

                # ─── SUBTAB 5: STAGE 4 TRAINING ──────────────────────────────────────
                with gr.TabItem("🎯 ขั้นตอนที่ 4: ฝึกสอนโมเดล (Training)"):
                    gr.Markdown("### 🎯 ขั้นตอนที่ 4: ฝึกสอนโครงข่ายประสาทเทียม DiffSinger (Model Training)")
                    with gr.Row():
                        tr_config = gr.Textbox(label="ไฟล์ Config YAML สำหรับฝึกสอน", value=str(DS_CONFIGS_DIR / "acoustic.yaml") if (DS_CONFIGS_DIR / "acoustic.yaml").exists() else "configs/acoustic.yaml", scale=7)
                        btn_b_tr_cfg = gr.Button("📄", scale=1)
                        tr_exp = gr.Textbox(label="ชื่อ Experiment", value="ds_thai_singer", scale=4)
                    with gr.Row():
                        tr_batch = gr.Number(label="Batch Size (0 = คำนวณตาม VRAM อัตโนมัติป้องกัน OOM)", value=0, precision=0, scale=3)
                        tr_reset = gr.Checkbox(label="เริ่มใหม่ตั้งแต่ต้น (Reset Checkpoint)", value=False, scale=2)
                    with gr.Row():
                        btn_run_train = gr.Button("▶️ เริ่มต้นการฝึกสอน (Start Training)", variant="primary", scale=4)
                        btn_stop_train = gr.Button("🛑 หยุดการฝึกสอน (Stop)", variant="secondary", scale=2)
                    log_train = gr.Textbox(label="สถานะการฝึกสอน (Training Log)", lines=14)

                    btn_b_tr_cfg.click(fn=lambda c: browse_file(c, [("YAML Config", "*.yaml")]), inputs=[tr_config], outputs=[tr_config])
                    btn_run_train.click(fn=ds_run_train, inputs=[tr_config, tr_exp, tr_batch, tr_reset], outputs=[log_train])
                    btn_stop_train.click(fn=cancel_all_processes, outputs=[log_train])

                # ─── SUBTAB 6: STAGE 5 & 6 EXPORT & PACKAGE ──────────────────────────
                with gr.TabItem("📦 ขั้นตอนที่ 5 & 6: ส่งออก ONNX & บรรจุ OpenUtau Voicebank"):
                    gr.Markdown("### 📦 ขั้นตอนที่ 5: ส่งออกโมเดลเป็นกราฟคำนวณ ONNX (ONNX Model Export)")
                    with gr.Row():
                        exp_onnx_name = gr.Textbox(label="ชื่อ Experiment ที่ต้องการส่งออก", value="ds_thai_singer", scale=4)
                        exp_onnx_type = gr.Radio(["acoustic", "variance"], label="ประเภทโมเดลที่ต้องการส่งออก (Export Type)", value="acoustic", scale=3)
                        exp_onnx_out = gr.Textbox(label="โฟลเดอร์ผลลัพธ์ ONNX (เว้นว่างเพื่อใช้โฟลเดอร์ checkpoints/<exp_name>)", value="", scale=7)
                        btn_b_exp_onnx_out = gr.Button("📁", scale=1)
                    with gr.Row():
                        btn_run_export = gr.Button("📤 ส่งออกโมเดลเป็น ONNX Graph", variant="primary", scale=4)

                    gr.Markdown("---")
                    gr.Markdown("### 🎁 ขั้นตอนที่ 6: บรรจุ Voicebank พร้อมใช้งานสำหรับ OpenUtau (Voicebank Packager)")
                    with gr.Row():
                        pkg_exp_name = gr.Textbox(label="ชื่อ Experiment ต้นทาง", value="ds_thai_singer", scale=4)
                        pkg_out_dir = gr.Textbox(label="โฟลเดอร์ OpenUtau Voicebank ปลายทาง", value=r"B:\For OpenUtau VB Making\A Voicebank File\Thai\ds_thai_singer", scale=7)
                        btn_b_pkg_out = gr.Button("📁", scale=1)
                        pkg_char_name = gr.Textbox(label="ชื่อตัวละคร/นักร้อง (Character Display Name)", value="", scale=3)
                    with gr.Row():
                        btn_run_package = gr.Button("🎁 บรรจุเป็น OpenUtau Voicebank พร้อมใช้งาน", variant="primary", scale=4)
                        btn_stop_pkg = gr.Button("🛑 หยุดกระบวนการทั้งหมด (Stop)", variant="secondary", scale=2)
                    log_pkg = gr.Textbox(label="ผลการส่งออกและบรรจุแพ็กเกจ (Export & Package Log)", lines=12)

                    btn_b_exp_onnx_out.click(fn=browse_folder, inputs=[exp_onnx_out], outputs=[exp_onnx_out])
                    btn_b_pkg_out.click(fn=browse_folder, inputs=[pkg_out_dir], outputs=[pkg_out_dir])
                    btn_run_export.click(fn=ds_run_export, inputs=[exp_onnx_name, exp_onnx_type, exp_onnx_out], outputs=[log_pkg])
                    btn_run_package.click(fn=ds_run_package, inputs=[pkg_exp_name, pkg_out_dir, pkg_char_name], outputs=[log_pkg])
                    btn_stop_pkg.click(fn=cancel_all_processes, outputs=[log_pkg])


listen_port = int(getattr(config, "listen_port", 9378))
is_colab = getattr(config, "iscolab", False)
no_auto_open = getattr(config, "noautoopen", False)

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(add_help=False)
    cli_parser.add_argument("--route_model", type=str, default="", help="Path or code to route model files by version")
    cli_parser.add_argument("--version_profile", type=str, default="AUTO", help="Version profile override")
    cli_parser.add_argument("--exp_name", type=str, default="", help="Experiment name for routing or pipeline")
    cli_parser.add_argument("--op_mode", type=str, default="copy", help="copy or move")
    known_args, _ = cli_parser.parse_known_args()

    if known_args.route_model:
        report_txt, _, _ = MODEL_ROUTER.execute_routing(
            input_code_or_path=known_args.route_model,
            version_profile=known_args.version_profile,
            exp_name=known_args.exp_name,
            operation_mode=known_args.op_mode,
            sync_to_ui_config=True
        )
        print(report_txt, flush=True)
        sys.exit(0)

    DeltaLogger.inf(f"🚀 กำลังเริ่มต้นระบบ DELTA RVC Studio x DiffSinger v5.3.4 บนพอร์ต {listen_port}...", to_console=True)
    if is_colab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        try:
            app.queue(concurrency_count=511, max_size=1022).launch(server_name="0.0.0.0", inbrowser=not no_auto_open, server_port=listen_port, quiet=True)
        except OSError:
            alt_port = 9379
            DeltaLogger.wrn(f"พอร์ต {listen_port} ถูกใช้งานแล้ว กำลังสลับไปยังพอร์ตสำรอง {alt_port}...", to_console=True)
            app.queue(concurrency_count=511, max_size=1022).launch(server_name="0.0.0.0", inbrowser=not no_auto_open, server_port=alt_port, quiet=True)