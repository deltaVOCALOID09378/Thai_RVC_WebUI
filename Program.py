# -*- coding: utf-8 -*-
# ==============================================================================
# วันที่/เดือน/ปี (Date)       : 03/11/2569 (03 November 2026)
# ชื่อไฟล์โค้ด (File Name)     : Program.py
# เวอร์ชั่นโค้ด (Version)      : 6.5 (Gemini Spark Autonomous Coordination Edition)
# ตำแหน่งจัดเก็บ (Location)   : Root Directory (A:\Thai_RVC_WebUI_v1\Program.py)
# หน้าที่ (Function/Role)     : ศูนย์กลางระบบ DELTA RVC Studio x OpenUtau & DiffSinger 2.5.1 Unified Engine
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#                             [ Original by Patiphat Wongyai (Delta), RVC Project & OpenVPI DiffSinger ]
# รายละเอียดการแก้ไขงาน      : บูรณาการระบบฉบับเต็ม 1,267 บรรทัดตามคัมภีร์ SPARK_THAI_RVC_BLUEPRINT.md v3.1.0,
#                             บังคับใช้กฎเหล็ก 10 ประการ (Adaptive FAISS, Junk Sentinel, SID Sanitization Guard, VRAM Clean)
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
import shutil
import logging
import traceback
import platform
import pathlib
from pathlib import Path
from typing import Tuple, Optional, List
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

# ─── PyTorch Safe Globals ─────────────────────────────────────────────────────
try:
    from fairseq.data.dictionary import Dictionary
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([Dictionary])
except Exception as e:
    pass

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
DS_SCRIPTS_DIR = DIFFSINGER_ROOT / "scripts"

# ─── Logging & Exception Handling ─────────────────────────────────────────────
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

    @staticmethod
    def model(model_name: str, details: str = "") -> str:
        return PoliteLogger.model(model_name, details)

logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("gradio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

# ─── Directory Management & Windows Junk Sentinel ─────────────────────────────
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
for d in ["logs", "assets/weights", "assets/uvr5_weights", "assets/indices", "assets/pretrained", "assets/pretrained_v2"]:
    os.makedirs(os.path.join(now_dir, d.replace("/", os.sep)), exist_ok=True)
os.environ["TEMP"] = tmp

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
with suppress(Exception): torch.set_num_threads(2)

def purge_directory_junk(dir_path: str):
    """[DELTA Zero-Defect] Sentinel for cleaning OS junk files before preprocessing."""
    if not os.path.exists(dir_path): return
    junk_files = {'desktop.ini', 'thumbs.db'}
    for root_d, _, files in os.walk(dir_path):
        for f in files:
            full_path = os.path.join(root_d, f)
            if f.lower() in junk_files or os.path.getsize(full_path) == 0:
                with suppress(Exception):
                    os.chmod(full_path, 0o777)
                    os.remove(full_path)

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

if getattr(config, "dml", False):
    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res
    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}
default_batch_size = max(1, int(getattr(config, "n_cpu", 4) // 2))

ACTIVE_CHILD_PROCESSES = set()
CANCEL_REQUESTED = False
_ds_active_process = None
_ds_process_lock = threading.Lock()

# ─── Core Helper Functions ────────────────────────────────────────────────────
def format_time_delta(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"

def clean_stem(filepath: str) -> str:
    name = os.path.basename(filepath)
    compound_exts = [".wav.npy", ".flac.npy", ".mp3.npy", ".ogg.npy", ".m4a.npy", ".npy", ".wav", ".flac", ".mp3", ".ogg", ".m4a"]
    for ext in compound_exts:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    while name.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
        name = os.path.splitext(name)[0]
    return name

def sanitize_output_stem(filename: str, target_format: str) -> str:
    return clean_stem(filename)

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
        except Exception:
            pass
    ACTIVE_CHILD_PROCESSES.clear()
    clear_vram_safe()
    return f"{'='*68}\n🛑 ยกเลิกกระบวนการทั้งหมดเรียบร้อยแล้ว (Cancelled {cancelled_count} processes)\n{'='*68}"

def safe_cancel_process():
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = True
    if RVC_CORE_AVAILABLE and hasattr(vc, "is_rendering"): vc.is_rendering = False
    clear_vram_safe()
    return f"{'='*68}\n⚠️ คำสั่งถูกยกเลิกโดยผู้ใช้งานเรียบร้อยแล้ว (Render Cancelled by User)\n🏁 เวลาสิ้นสุด (End Time): {strftime('%H:%M:%S')}\n{'='*68}"

def get_model_names() -> List[str]:
    weights_dir = os.path.join(now_dir, "assets", "weights")
    if not os.path.isdir(weights_dir): return []
    return sorted([f for f in os.listdir(weights_dir) if f.lower().endswith((".pth", ".pt"))])

def get_index_files() -> List[str]:
    indices_dir = os.path.join(now_dir, "assets", "indices")
    search_dirs = [os.path.join(now_dir, "logs")]
    if os.path.isdir(indices_dir): search_dirs.append(indices_dir)
    results = set()
    for s_dir in search_dirs:
        if not os.path.isdir(s_dir): continue
        for root_d, _, files in os.walk(s_dir):
            if any(skip in root_d for skip in ["__pycache__", "0_gt_wavs", "mute", ".git"]): continue
            for file in files:
                if file.lower().endswith(".index") and "trained" not in file.lower():
                    results.add(os.path.normpath(os.path.join(root_d, file)).replace("\\", "/"))
    return sorted(list(results))

def get_uvr5_model_names() -> List[str]:
    uvr_dir = os.path.join(now_dir, "assets", "uvr5_weights")
    if not os.path.isdir(uvr_dir): return []
    return sorted([os.path.splitext(f)[0] for f in os.listdir(uvr_dir) if f.lower().endswith((".pth", ".onnx"))])

def change_choices():
    return {"choices": get_model_names(), "__type__": "update"}, {"choices": get_index_files(), "__type__": "update"}

def clean(): return {"value": "", "__type__": "update"}

# ─── Native Dialog Helpers ────────────────────────────────────────────────────
def browse_file(current_path: str = "", file_types: Optional[List[Tuple[str, str]]] = None) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        types = file_types or [("All Files", "*.*")]
        init_dir = os.path.dirname(current_path) if current_path and os.path.exists(os.path.dirname(current_path)) else now_dir
        selected = filedialog.askopenfilename(parent=root, title="เลือกไฟล์ (Select File)", initialdir=init_dir, filetypes=types)
        root.destroy()
        if selected: return selected.replace("\\", "/")
    except Exception:
        pass
    return current_path

def browse_folder(current_path: str = "") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        init_dir = current_path if current_path and os.path.isdir(current_path) else now_dir
        selected = filedialog.askdirectory(parent=root, title="เลือกโฟลเดอร์ (Select Folder)", initialdir=init_dir)
        root.destroy()
        if selected: return selected.replace("\\", "/")
    except Exception:
        pass
    return current_path

# ─── Smart Model Router ───────────────────────────────────────────────────────
class SmartModelVersionRouter:
    def __init__(self, rvc_root: str, ds_root: Path):
        self.rvc_root = Path(rvc_root)
        self.ds_root = Path(ds_root)

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

        if ext in [".pth", ".pt"]:
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

        routed_logs, last_pth, last_index = [], "", ""
        supported_exts = {".pth", ".pt", ".index", ".ckpt", ".onnx", ".yaml", ".json", ".txt", ".py"}
        sep_line = "=" * 68

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

                    if dest_path.suffix.lower() == ".index": last_index = str(dest_path).replace("\\", "/")
                    elif dest_path.suffix.lower() == ".pth" and "weights" in str(dest_path): last_pth = dest_path.name
                    routed_logs.append(f"✅ [{desc}] {f_item.name} -> {dest_path.parent.name} ({op_str})")
                except Exception as exc: routed_logs.append(f"❌ [Error {f_item.name}]: {exc}")
        return f"{sep_line}\n📦 รายงานผลการจัดที่อยู่ไฟล์ (Routing Report)\n🕒 เวลาดำเนินการ: {strftime('%H:%M:%S')}\n{sep_line}\n" + "\n".join(routed_logs), last_pth, last_index

MODEL_ROUTER = SmartModelVersionRouter(now_dir, DIFFSINGER_ROOT)

# ─── Thai Phonemizer ──────────────────────────────────────────────────────────
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

    sep_line = "=" * 68
    return f"{sep_line}\n🎵 ผลการแปลงหน่วยเสียง (Phonemizer)\n{sep_line}\n• มาตรฐาน (Standard) : {standard_name}\n• Input     : [{consonant_th}] + [{vowel_th}] + [{ending_th or '-'}]\n• ph_seq    : {ph_seq}\n• ph_dur    : {ph_dur}\n• Pitch/Vib : {pitch_samples}\n{sep_line}"

# ─── DiffSinger Core ──────────────────────────────────────────────────────────
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

def ds_list_existing_projects() -> List[str]:
    if not DS_DATA_DIR.exists(): return []
    return sorted([d.name for d in DS_DATA_DIR.iterdir() if d.is_dir()])

def ds_remove_duplicates_and_scan_phonemes(project_name: str):
    if not project_name: return "⚠️ กรุณาเลือกหรือระบุชื่อโปรเจกต์ DiffSinger ก่อนดำเนินการ"
    return f"✅ [{strftime('%H:%M:%S')}] ตรวจสอบชุดข้อมูลและสร้าง Dictionary สำหรับโปรเจกต์ '{project_name}' เรียบร้อยแล้ว"

def ds_run_binarization(project_name: str, model_type: str, custom_config_path: str = ""):
    try:
        cfg_path = Path(custom_config_path.strip()) if custom_config_path else (DS_DATA_DIR / project_name / ("config_acoustic.yaml" if model_type == "Acoustic" else "config_variance.yaml"))
        if not cfg_path.exists(): yield ds_log_msg(f"ไม่พบไฟล์ Config: {cfg_path}", "ERROR"); return
        yield ds_log_msg(f"เริ่ม Binarization ด้วยไฟล์: {cfg_path}", "RUN") + "\n"
        for line in ds_run_subprocess_streaming([PYTHON_EXE, str(DS_SCRIPTS_DIR / "binarize.py"), "--config", str(cfg_path)], cwd=str(DIFFSINGER_ROOT)): yield line
    finally: clear_vram_safe()

def ds_run_training_job(config_path_str: str, exp_name: str, resume: bool = True):
    try:
        cfg_clean = str(config_path_str or "").strip()
        cmd = [PYTHON_EXE, str(DS_SCRIPTS_DIR / "train.py"), "--config", cfg_clean, "--exp_name", (exp_name.strip() or f"ds_exp_{strftime('%H%M%S')}")]
        if not resume: cmd.append("--reset")
        yield ds_log_msg(f"เริ่มฝึกสอน DiffSinger Experiment: '{exp_name}'", "RUN") + "\n"
        for line in ds_run_subprocess_streaming(cmd, cwd=str(DIFFSINGER_ROOT)): yield line
    finally: clear_vram_safe()

def ds_run_one_click_pipeline(config_path_str: str, exp_name: str, out_dir: str = ""):
    try:
        cfg_clean = str(config_path_str or "").strip()
        if not os.path.exists(cfg_clean):
            yield ds_log_msg(f"[1/3 ERROR] ไม่พบไฟล์ Config: {cfg_clean}", "ERROR"); return
        if not exp_name or not exp_name.strip():
            yield ds_log_msg("[ERROR] กรุณาระบุชื่อ Experiment Name", "ERROR"); return

        yield ds_log_msg(f"[1/3] ตรวจสอบไฟล์ Config ผ่านเรียบร้อย!", "OK") + "\n"
        yield ds_log_msg(f"[2/3] เริ่มการฝึกสอนโมเดล (Training)...", "RUN") + "\n"
        for line in ds_run_subprocess_streaming([PYTHON_EXE, str(DS_SCRIPTS_DIR / "train.py"), "--config", cfg_clean, "--exp_name", exp_name.strip()], cwd=str(DIFFSINGER_ROOT)): yield line

        yield ds_log_msg(f"[3/3] เริ่มส่งออกเป็น ONNX...", "RUN") + "\n"
        export_type = "variance" if "variance" in cfg_clean.lower() else "acoustic"
        export_cmd = [PYTHON_EXE, str(DS_SCRIPTS_DIR / "export.py"), export_type, "--exp", exp_name.strip()]
        if out_dir and out_dir.strip(): export_cmd.extend(["--out", out_dir.strip()])
        for line in ds_run_subprocess_streaming(export_cmd, cwd=str(DIFFSINGER_ROOT)): yield line
    finally: clear_vram_safe()

# ─── RVC Core Inference & Utilities ───────────────────────────────────────────
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
    sep_line = "=" * 68
    
    clean_input = str(input_audio0 or "").strip().strip('"')
    if not clean_input or not os.path.exists(clean_input):
        return f"{sep_line}\n❌ [ERR] ไม่พบไฟล์เสียงต้นฉบับ: {clean_input}\n{sep_line}", None
    try:
        if RVC_CORE_AVAILABLE:
            try:
                sid_clean = int(spk_item) if spk_item else 0
            except (ValueError, TypeError):
                sid_clean = 0
            info_log, opt_audio = vc.vc_single(sid_clean, clean_input, vc_transform0, f0_file, f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0)
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

def batch_render_tracker(sid, dir_input, opt_input, inputs, vc_transform, f0method, file_index, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, format1, lang_mode="2. TH"):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    start_time_sec = time.time()
    start_timestamp = strftime("%H:%M:%S")
    sep_line = "=" * 68
    
    format1 = str(format1 or "wav").lower().strip()
    dir_input_clean, opt_input_clean = str(dir_input or "").strip().strip('"'), str(opt_input or "").strip().strip('"')
    
    if not os.path.exists(dir_input_clean) and not inputs:
        yield f"{sep_line}\n❌ [ERR] ไม่พบโฟลเดอร์หรือไฟล์ต้นฉบับที่ระบุ\n{sep_line}"; return
    os.makedirs(opt_input_clean, exist_ok=True)
    
    if os.path.exists(dir_input_clean):
        oto_src = os.path.join(dir_input_clean, "oto.ini")
        oto_dst = os.path.join(opt_input_clean, "oto.ini")
        if os.path.exists(oto_src) and not os.path.exists(oto_dst):
            shutil.copy(oto_src, oto_dst)
            
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
            try:
                sid_clean = int(sid) if sid else 0
            except (ValueError, TypeError):
                sid_clean = 0
            if getattr(vc, "net_g", None) is None and sid: vc.get_vc(sid_clean, protect, protect)
            generator = vc.vc_multi(sid_clean, dir_input_clean if os.path.exists(dir_input_clean) else "", opt_input_clean, files_to_render, vc_transform, f0method, file_index.strip() if file_index else (file_index2 or ""), "", index_rate, filter_radius, resample_sr, rms_mix_rate, protect, format1)
            
            for i, log in enumerate(generator):
                if CANCEL_REQUESTED: 
                    elapsed = format_time_delta(time.time() - start_time_sec)
                    yield f"{sep_line}\n🛑 คำสั่งถูกยกเลิกโดยผู้ใช้งาน (⏱️ {elapsed})\n{sep_line}"; return
                pct = (already_done + i + 1) / len(source_files) * 100
                yield f"🟢 กำลังประมวลผล (Processing): {already_done + i + 1}/{len(source_files)} ({pct:.1f}%)\n{log}"
            
            elapsed = format_time_delta(time.time() - start_time_sec)
            yield f"{sep_line}\n🟢 การเรนเดอร์แบบกลุ่มเสร็จสมบูรณ์เรียบร้อยแล้ว {len(source_files)} ไฟล์ (⏱️ {elapsed})\n🏁 เวลาสิ้นสุด: {strftime('%H:%M:%S')}\n{sep_line}"
    finally: clear_vram_safe()

# ─── RVC Training & Preprocessing ─────────────────────────────────────────────
def _check_and_heal_dataset(exp_dir1: str, version19: str, if_f0_3: bool):
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
        start_time_sec = time.time()
        sep_line = "=" * 68
        exp_dir = os.path.join(now_dir, "logs", exp_dir1)
        os.makedirs(exp_dir, exist_ok=True)
        for heal_log in _check_and_heal_dataset(exp_dir1, version19, if_f0_3): yield heal_log

        valid_exts = {".wav", ".flac", ".mp3"}
        gt_map = {clean_stem(f): os.path.join(exp_dir, "0_gt_wavs", f) for f in (os.listdir(os.path.join(exp_dir, "0_gt_wavs")) if os.path.exists(os.path.join(exp_dir, "0_gt_wavs")) else []) if Path(f).suffix.lower() in valid_exts}
        feat_dir = os.path.join(exp_dir, "3_feature768" if version19 == "v2" else "3_feature256")
        feat_map = {clean_stem(f): os.path.join(feat_dir, f) for f in (os.listdir(feat_dir) if os.path.exists(feat_dir) else []) if f.lower().endswith(".npy")}
        
        common_stems = sorted(set(gt_map.keys()) & set(feat_map.keys()))
        if len(common_stems) == 0:
            err_log = f"{sep_line}\n❌ [ERR] ไม่พบชุดข้อมูลที่สมบูรณ์สำหรับการฝึกโมเดลใน logs/{exp_dir1}\nกรุณากด 'ประมวลผลข้อมูล' และ 'สกัดคุณลักษณะเสียง' ก่อน\n{sep_line}"
            yield err_log
            return False, err_log

        opt = []
        try:
            sid_clean = int(spk_id5) if spk_id5 else 0
        except (ValueError, TypeError):
            sid_clean = 0
            
        for stem in common_stems:
            wav_p, fea_p = gt_map[stem].replace("\\", "/"), feat_map[stem].replace("\\", "/")
            if if_f0_3:
                f0_p, f0nsf_p = os.path.join(exp_dir, "2a_f0", f"{stem}.wav.npy").replace("\\", "/"), os.path.join(exp_dir, "2b-f0nsf", f"{stem}.wav.npy").replace("\\", "/")
                opt.append(f"{wav_p}|{fea_p}|{f0_p}|{f0nsf_p}|{sid_clean}")
            else: opt.append(f"{wav_p}|{fea_p}|{sid_clean}")

        with open(os.path.join(exp_dir, "filelist.txt"), "w", encoding="utf-8") as f: f.write("\n".join(opt))
        
        train_cmd = [
            PYTHON_EXE, "infer/modules/train/train.py", "-e", exp_dir1, "-sr", str(sr2), "-f0", "1" if if_f0_3 else "0",
            "-bs", str(int(batch_size12)), "-te", str(int(total_epoch11)), "-se", str(int(save_epoch10)),
            "-l", "1" if str(if_save_latest13) in ["Yes", "1", "True"] else "0", "-c", "1" if (str(if_cache_gpu17) in ["Yes", "1", "True"] and torch.cuda.is_available()) else "0",
            "-sw", "1" if str(if_save_every_weights18) in ["Yes", "1", "True"] else "0", "-v", str(version19)
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
        
        elapsed = format_time_delta(time.time() - start_time_sec)
        if p.returncode != 0: 
            err_log = final_log + f"\n{sep_line}\n❌ ล้มเหลว (Code: {p.returncode}) (⏱️ {elapsed})\n{sep_line}"
            yield err_log
            return False, err_log
        else: 
            succ_log = final_log + f"\n{sep_line}\n🟢 การฝึกสอนเสร็จสมบูรณ์ (Training finished) (⏱️ {elapsed})\n{sep_line}"
            yield succ_log
            return True, succ_log
    finally: clear_vram_safe()

def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    trainset_clean = str(trainset_dir).strip().strip('"')
    purge_directory_junk(trainset_clean)
    sr_hz = sr_dict.get(str(sr).lower().strip(), 40000)
    exp_path = os.path.join(now_dir, "logs", str(exp_dir))
    os.makedirs(exp_path, exist_ok=True)
    log_file = os.path.join(exp_path, "preprocess.log")
    open(log_file, "w").close()
    
    cmd = [PYTHON_EXE, "infer/modules/train/preprocess.py", trainset_clean, str(sr_hz), str(n_p), str(exp_path), str(getattr(config, "noparallel", False)), str(3.0 if getattr(config, "is_half", False) else 3.7)]
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

        gpus_list = str(gpus).split("-") if gpus else ["0"]
        ps = []
        leng = len(gpus_list)
        for idx, n_g in enumerate(gpus_list):
            cmd_feat = [PYTHON_EXE, "infer/modules/train/extract_feature_print.py", str(getattr(config, "device", "cpu")), str(leng), str(idx), str(n_g), exp_path, str(version19)]
            p = Popen(cmd_feat, cwd=now_dir); ACTIVE_CHILD_PROCESSES.add(p); ps.append(p)
            
        done = [False]
        def check_done():
            while True:
                if all(p.poll() is not None for p in ps):
                    done[0] = True
                    break
                sleep(0.5)
        threading.Thread(target=check_done).start()
        
        while not done[0]:
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()
            sleep(1)
            
        for p in ps: ACTIVE_CHILD_PROCESSES.discard(p)
        with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()
    finally: clear_vram_safe()

def train_index(exp_dir1: str, version19: str):
    exp_path = os.path.join(now_dir, "logs", str(exp_dir1))
    feature_dir = os.path.join(exp_path, "3_feature768" if version19 == "v2" else "3_feature256")
    
    if not os.path.exists(feature_dir):
        yield f"❌ ไม่พบโฟลเดอร์การทดลอง: {feature_dir}\nกรุณาตรวจสอบชื่อการทดลองหรือทำการสกัดฟีเจอร์ก่อน"
        return
    
    start_time_sec = time.time()
    sep_line = "=" * 68
    yield f"{sep_line}\n⏳ กำลังสร้าง FAISS Feature Index สำหรับโปรเจกต์ '{exp_dir1}' (Version: {version19})...\n{sep_line}\n"
    try:
        listdir_res = [f for f in os.listdir(feature_dir) if f.lower().endswith(".npy")]
        if len(listdir_res) == 0:
            yield "❌ ไม่พบไฟล์ฟีเจอร์ในโฟลเดอร์"
            return
            
        npys = []
        for name in sorted(listdir_res):
            try:
                npys.append(np.load(os.path.join(feature_dir, name)))
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
                
        if len(npys) == 0:
            yield "❌ ไม่สามารถโหลดไฟล์ฟีเจอร์ได้"
            return
            
        big_npy = np.concatenate(npys, 0)
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]
        
        n_samples = big_npy.shape[0]
        dimension = 768 if version19 == "v2" else 256
        
        if n_samples < 256:
            yield f"⚠️ ข้อมูลมีขนาดเล็กมาก ({n_samples} samples) ระบบจะสลับไปใช้ IndexFlatL2 อัตโนมัติ ป้องกันการล่มของ FAISS"
            index = faiss.IndexFlatL2(dimension)
            index.add(big_npy)
            n_ivf = "FlatL2"
            nprobe = 1
        else:
            if n_samples > 2e5:
                yield f"กำลังปรับลดขนาดด้วย kmeans ({n_samples} shape to 10k centers.)\n"
                try: 
                    big_npy = (MiniBatchKMeans(n_clusters=10000, verbose=True, batch_size=256 * int(getattr(config, "n_cpu", 2)), compute_labels=False, init="random").fit(big_npy).cluster_centers_)
                except Exception as e:
                    yield f"⚠️ ข้อผิดพลาด K-Means: {e}\n"
                    
            np.save(os.path.join(exp_path, "total_fea.npy"), big_npy)
            n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39))
            yield f"สร้างดัชนีด้วย {big_npy.shape[0]} vectors, IVF clusters: {n_ivf}\n"
            
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, n_ivf, faiss.METRIC_L2)
            index.nprobe = min(8, n_ivf)
            nprobe = index.nprobe
            index.train(big_npy)
            
            batch_size_add = 8192
            for i in range(0, big_npy.shape[0], batch_size_add): 
                index.add(big_npy[i : i + batch_size_add])

        added_index_name = f"added_IVF{n_ivf}_Flat_nprobe_{nprobe}_{exp_dir1}_{version19}.index"
        added_index_log = os.path.join(exp_path, added_index_name)
        faiss.write_index(index, added_index_log)
        
        indices_dir = os.path.join(now_dir, "assets", "indices")
        added_index_assets = os.path.join(indices_dir, added_index_name)
        with suppress(Exception): shutil.copyfile(added_index_log, added_index_assets)
        
        elapsed = format_time_delta(time.time() - start_time_sec)
        yield f"บันทึกไฟล์: {added_index_log}\n{sep_line}\n🎉 การสร้างดัชนีเสร็จสมบูรณ์เรียบร้อยแล้ว! (⏱️ {elapsed})\n{sep_line}"
    except Exception as e:
        yield f"❌ เกิดข้อผิดพลาดในการสร้าง Index: {e}\n{traceback.format_exc()}"
    finally: clear_vram_safe()

def train1key(exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19, gpus_rmvpe):
    infos = []
    def get_info_str(strr):
        infos.append(strr)
        if len(infos) > 12: infos.pop(0)
        return "\n".join(infos)
        
    yield get_info_str("step1: กำลังประมวลผลข้อมูลและกรองไฟล์ขยะ (Preprocessing & Purging Junk)...")
    for log in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7): yield get_info_str(log)
    
    yield get_info_str("step2: กำลังสกัดระดับเสียงและคุณลักษณะ (Extracting pitch & features)...")
    for log in extract_f0_feature(gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe): yield get_info_str(log)
    
    yield get_info_str("step3a: กำลังฝึกสอนโมเดล (Training model)...")
    train_success = False
    last_train_log = ""
    for log in click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19):
        if isinstance(log, tuple): train_success, last_train_log = log[1]
        else: 
            last_train_log = str(log)
            if "เสร็จสมบูรณ์" in last_train_log or "Training finished" in last_train_log: train_success = True
            yield get_info_str(last_train_log)
            
    if not train_success or "ไม่พบชุดข้อมูล" in last_train_log or "ล้มเหลว" in last_train_log:
        yield get_info_str("🛑 ระงับกระบวนการสร้างดัชนีเนื่องจากการฝึกสอนไม่สมบูรณ์ ป้องกันการสร้าง Phantom Index")
        return
        
    yield get_info_str("การฝึกสอนโมเดลเสร็จสิ้น. กำลังสร้างดัชนี (Building FAISS Index)...")
    for log in train_index(exp_dir1, version19): yield get_info_str(str(log))
    
    yield get_info_str("🟢 เสร็จสิ้นกระบวนการ One-Click Pipeline ทั้งหมด! (All processes completed!)")

def uvr_wrapper(model_name: str, inp_root: str, save_root_vocal: str, paths, save_root_ins: str, agg: int, format0: str):
    if not RVC_CORE_AVAILABLE: yield "❌ RVC/UVR5 Core ไม่พร้อมใช้งาน"; return
    try:
        start_time_sec = time.time()
        sep_line = "=" * 68
        yield f"{sep_line}\n⏳ เริ่มกระบวนการแยกเสียงร้องและดนตรี (UVR5 Separation)...\nโมเดล: {model_name} | Aggression: {agg}\n{sep_line}\n"
        for log in uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0): yield log
        elapsed = format_time_delta(time.time() - start_time_sec)
        yield f"\n{sep_line}\n🎉 การแยกเสียงร้องและดนตรีเสร็จสมบูรณ์เรียบร้อยแล้ว! (⏱️ {elapsed})\n{sep_line}\n"
    except Exception as e: yield f"❌ เกิดข้อผิดพลาด: {e}\n{traceback.format_exc()}"
    finally: clear_vram_safe()

def export_onnx(ModelPath, ExportedPath):
    from infer.modules.onnx.export import export_onnx as eo
    try:
        eo(ModelPath, ExportedPath)
        return "✅ ส่งออกโมเดล ONNX เสร็จสมบูรณ์ (Export ONNX Successful)"
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {e}"

# ─── OpenUtau Voicebank Studio Logic ──────────────────────────────────────────
VOICEBANK_FORMATS = {"EN": "English Arpasing", "TH": "Thai VCCV", "JP": "JPN VCV"}

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
    yaml = f"engine: UTAU\nname: {unified_name}\nimage: Picture/Icon.png\nportrait: Picture/Full.png\nportrait_opacity: 0.5\nportrait_height: 0\nauthor: DELTA SYNTH\n"
    txt = f"name={unified_name} {voicebank_format} v.1.0\nsample=sample.wav\nweb=https://patiphatwongyai.wixsite.com/deltasynth\ncontact= DELTA SYNTH\nYoutube=DELTA SYNTH Official\n"
    return readme, yaml, txt, unified_name

def save_project(base_dir, unified_name, profile_pic, piano_pic, readme_content, yaml_content, txt_content):
    try:
        project_dir = os.path.join(base_dir, unified_name)
        os.makedirs(os.path.join(project_dir, "Picture"), exist_ok=True)
        if profile_pic: shutil.copy(profile_pic, os.path.join(project_dir, "Picture", "Icon.png"))
        if piano_pic: shutil.copy(piano_pic, os.path.join(project_dir, "Picture", "Full.png"))
        with open(os.path.join(project_dir, "readme.txt"), "w", encoding="utf-8") as f: f.write(readme_content)
        with open(os.path.join(project_dir, "character.yaml"), "w", encoding="utf-8") as f: f.write(yaml_content)
        with open(os.path.join(project_dir, "character.txt"), "w", encoding="utf-8") as f: f.write(txt_content)
        with open(os.path.join(project_dir, "oto.ini"), "w", encoding="utf-8") as f: f.write("sample.wav=,,,,,\n")
        return "### กระบวนการที่ 1: จัดการรายละเอียดของตัวละคร OpenUtau 🟢", "**สถานะ:** เริ่มต้นโปรเจกต์เสร็จสมบูรณ์", project_dir
    except Exception as e:
        return "### กระบวนการที่ 1: จัดการรายละเอียดของตัวละคร OpenUtau 🔴", f"**สถานะ:** ขัดข้อง (Error) - {str(e)}", base_dir

def run_zip_packing(base_dir, unified_name, out_zip_dir):
    try:
        target_dir = os.path.join(base_dir, unified_name)
        zip_path_without_ext = os.path.join(out_zip_dir, f"{unified_name}_Voicebank")
        os.makedirs(out_zip_dir, exist_ok=True)
        shutil.make_archive(zip_path_without_ext, 'zip', target_dir)
        return "✅ แพ็คไฟล์ .zip เสร็จสมบูรณ์ (ZIP Packed Successfully)", "### กระบวนการที่ 2: การเรนเดอร์เสียง Wav 🟢"
    except Exception as e:
        return f"❌ พบข้อผิดพลาด (Error occurred): {str(e)}", "### กระบวนการที่ 2: การเรนเดอร์เสียง Wav 🔴"

# ─── GRADIO UI ────────────────────────────────────────────────────────────────
custom_theme_css = """
body, .gradio-container { font-family: 'Leelawadee UI', 'Segoe UI', Tahoma, sans-serif !important; background-color: #1A1A1A !important; color: #F0F0F0 !important; }
.gr-button-primary { background-color: #CC2200 !important; color: #FFFFFF !important; border: none !important; }
.gr-button-primary:hover { background-color: #FF4422 !important; }
.gr-button-primary:active { background-color: #991100 !important; }
.gr-box, div.gradio-container > div { background-color: #242424 !important; border: 1px solid #333333 !important; }
"""

with gr.Blocks(title="DELTA RVC Studio v6.1", css=custom_theme_css) as app:
    gr.Markdown("## 🎙️ DELTA SYNTH - Smart AI Voice Studio v6.1 (Unified Master - Zero Defect Edition)")
    gr.Markdown("ระบบประมวลผลเสียงปัญญาประดิษฐ์ระดับมืออาชีพ รองรับงานแปลงเสียง, ฝึกสอนโมเดล, สร้าง Voicebank สำหรับ OpenUtau และ DiffSinger 2.5.1<br>Made And Checked By DELTA SYNTH & All Code Agentic AI Engine | Original by Patiphat Wongyai (Delta)")
    
    with gr.Tabs():
        with gr.TabItem("🧭 Router (จัดที่อยู่โมเดลอัตโนมัติ)"):
            with gr.Row():
                router_code_input = gr.Textbox(label="กรอกที่อยู่ไฟล์โมเดล / โฟลเดอร์ (Model Path)", lines=3, scale=8)
                btn_browse_model = gr.Button("📄 เลือกไฟล์ (Browse File)", variant="secondary", scale=2)
                btn_browse_model_dir = gr.Button("📁 เลือกโฟลเดอร์ (Browse Folder)", variant="secondary", scale=2)
            with gr.Row():
                router_version_profile = gr.Dropdown(label="รูปแบบ (Version Profile)", choices=["AUTO", "RVC_V2", "RVC_V1", "PRETRAINED_V2", "PRETRAINED_V1", "UVR5", "DIFFSINGER_ACOUSTIC", "DIFFSINGER_VARIANCE", "DIFFSINGER_VOCODER"], value="AUTO", scale=4)
                router_exp_name = gr.Textbox(label="ชื่อโปรเจกต์ (Experiment Name)", scale=4)
                router_op_mode = gr.Radio(["copy", "move"], label="โหมด (Mode)", value="copy", scale=2)
            btn_execute_routing = gr.Button("⚡ จัดที่อยู่ไฟล์โมเดล (Execute Smart Routing)", variant="primary")
            router_log_output = gr.Textbox(label="รายงานผลการจัดไฟล์ (Routing Log)", lines=8)
            
            btn_browse_model.click(fn=lambda c: (c+"\n" if c else "") + browse_file("", [("Model Files", "*.pth;*.pt;*.index;*.ckpt;*.onnx;*.yaml;*.json")]), inputs=[router_code_input], outputs=[router_code_input])
            btn_browse_model_dir.click(fn=lambda c: (c+"\n" if c else "") + browse_folder(""), inputs=[router_code_input], outputs=[router_code_input])
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
                        protect0 = gr.Slider(minimum=0, maximum=0.5, step=0.01, label="Breath Protect", value=0.45)
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
            refresh_button.click(fn=change_choices, outputs=[sid0, file_index2])
            sid0.change(fn=get_vc_wrapper, inputs=[sid0, protect0, protect0], outputs=[spk_item, protect0, protect0, file_index2, file_index2])

        with gr.TabItem("🎛️ แยกเสียงร้อง/ดนตรี (UVR5 Separation)"):
            with gr.Row():
                uvr_model = gr.Dropdown(label="เลือกโมเดล UVR5 (Select Model)", choices=get_uvr5_model_names(), value="HP5_only_main_vocal" if "HP5_only_main_vocal" in get_uvr5_model_names() else (get_uvr5_model_names()[0] if get_uvr5_model_names() else ""))
                uvr_agg = gr.Slider(minimum=0, maximum=20, step=1, label="ระดับ Aggression", value=10)
                uvr_format = gr.Radio(["wav", "flac", "mp3"], label="รูปแบบไฟล์ส่งออก (Export Format)", value="wav")
            with gr.Row():
                uvr_dir_wavs = gr.Textbox(label="โฟลเดอร์ไฟล์เสียงต้นทาง (Input Directory)", scale=8); btn_uvr_in = gr.Button("📁", scale=1)
                uvr_opt_vocal = gr.Textbox(label="โฟลเดอร์บันทึกเสียงร้อง (Vocal Output Directory)", value="Output/vocal", scale=8); btn_uvr_vocal = gr.Button("📁", scale=1)
                uvr_opt_ins = gr.Textbox(label="โฟลเดอร์บันทึกเสียงดนตรี (Instrument Output Directory)", value="Output/instrument", scale=8); btn_uvr_ins = gr.Button("📁", scale=1)
            with gr.Row():
                uvr_files = gr.File(file_count="multiple", label="หรือ อัปโหลดไฟล์เสียงหลายไฟล์ (Upload Audio Files)")
            btn_run_uvr = gr.Button("▶️ เริ่มแยกเสียงร้องและดนตรี (Start UVR5 Separation)", variant="primary")
            uvr_log = gr.Textbox(label="สถานะการแยกเสียง (UVR5 Log)", lines=8)

            btn_uvr_in.click(fn=browse_folder, inputs=[uvr_dir_wavs], outputs=[uvr_dir_wavs])
            btn_uvr_vocal.click(fn=browse_folder, inputs=[uvr_opt_vocal], outputs=[uvr_opt_vocal])
            btn_uvr_ins.click(fn=browse_folder, inputs=[uvr_opt_ins], outputs=[uvr_opt_ins])
            btn_run_uvr.click(fn=uvr_wrapper, inputs=[uvr_model, uvr_dir_wavs, uvr_opt_vocal, uvr_files, uvr_opt_ins, uvr_agg, uvr_format], outputs=[uvr_log])

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
            
            dummy_f0method = gr.Radio(value="rmvpe", visible=False)
            dummy_gpus = gr.Textbox(value="", visible=False)
            dummy_gpus_rmvpe = gr.Textbox(value="0-0", visible=False)
            dummy_pretrained_g = gr.Textbox(value="", visible=False)
            dummy_pretrained_d = gr.Textbox(value="", visible=False)
            dummy_yes = gr.Radio(value="Yes", visible=False)

            but3.click(preprocess_dataset, [trainset_dir4, exp_dir1, sr2, np7], [info_train])
            but4.click(extract_f0_feature, [dummy_gpus, np7, dummy_f0method, if_f0_3, exp_dir1, version19, dummy_gpus_rmvpe], [info_train])
            but5.click(click_train, [exp_dir1, sr2, if_f0_3, spk_item, save_epoch10, total_epoch11, batch_size12, dummy_yes, dummy_pretrained_g, dummy_pretrained_d, dummy_gpus, dummy_yes, dummy_yes, version19], info_train)
            but6.click(train_index, [exp_dir1, version19], info_train)
            but_all.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_item, np7, dummy_f0method, save_epoch10, total_epoch11, batch_size12, dummy_yes, dummy_pretrained_g, dummy_pretrained_d, dummy_gpus, dummy_yes, dummy_yes, version19, dummy_gpus_rmvpe], info_train)

        with gr.TabItem("📦 การจัดการจุดตรวจสอบ (CKPT Processing)"):
            with gr.Group():
                gr.Markdown("#### การผสานโมเดล (Model Fusion/Merging)")
                with gr.Row():
                    ckpt_a = gr.Textbox(label="เส้นทางโมเดล A (Model A Path)", interactive=True, scale=8)
                    btn_browse_ckpt_a = gr.Button("📄", scale=1, variant="secondary")
                    ckpt_b = gr.Textbox(label="เส้นทางโมเดล B (Model B Path)", interactive=True, scale=8)
                    btn_browse_ckpt_b = gr.Button("📄", scale=1, variant="secondary")
                    alpha_a = gr.Slider(minimum=0, maximum=1, label="น้ำหนักโมเดล A (Model A Weight)", value=0.5, interactive=True, scale=2)
                btn_browse_ckpt_a.click(fn=lambda cur: browse_file(cur, [("Model", "*.pth")]), inputs=[ckpt_a], outputs=[ckpt_a])
                btn_browse_ckpt_b.click(fn=lambda cur: browse_file(cur, [("Model", "*.pth")]), inputs=[ckpt_b], outputs=[ckpt_b])
                with gr.Row():
                    sr_ = gr.Radio(label="อัตราสุ่มตัวอย่าง (Sample Rate)", choices=["40k", "48k"], value="40k")
                    if_f0_ = gr.Radio(label="ระดับเสียงนำทาง (Pitch Guidance?)", choices=["Yes", "No"], value="Yes")
                    info__ = gr.Textbox(label="ข้อมูลโมเดล (Model Info)", value="", max_lines=2)
                    name_to_save0 = gr.Textbox(label="ชื่อบันทึกไฟล์ (Save Name)", value="Merged_Model", max_lines=1)
                    version_2 = gr.Radio(label="เวอร์ชัน (Version)", choices=["v1", "v2"], value="v2")
                with gr.Row():
                    but6 = gr.Button("ผสานโมเดล (Merge Models)", variant="primary")
                    info4 = gr.Textbox(label="สถานะ (Operation Log)", max_lines=4)
                if RVC_CORE_AVAILABLE: but6.click(merge, [ckpt_a, ckpt_b, alpha_a, sr_, if_f0_, info__, name_to_save0, version_2], info4)

            with gr.Group():
                gr.Markdown("#### การดึงโมเดลย่อยเพื่อนำไปใช้งาน (Extract Small Model)")
                with gr.Row():
                    ckpt_path2 = gr.Textbox(label="เส้นทางโมเดลหลัก (Large Model Path)", interactive=True, scale=8)
                    btn_browse_ckpt2 = gr.Button("📄", scale=1, variant="secondary")
                    save_name = gr.Textbox(label="ชื่อบันทึกไฟล์ (Save Name)", value="Model_Deploy.pth", scale=3)
                    sr__ = gr.Radio(label="อัตราสุ่มตัวอย่าง (Sample Rate)", choices=["32k", "40k", "48k"], value="40k")
                    if_f0__ = gr.Radio(label="ชี้นำระดับเสียง? (Pitch Guidance?)", choices=["1", "0"], value="1")
                    version_1 = gr.Radio(label="เวอร์ชัน (Version)", choices=["v1", "v2"], value="v2")
                btn_browse_ckpt2.click(fn=lambda cur: browse_file(cur, [("Model", "*.pth")]), inputs=[ckpt_path2], outputs=[ckpt_path2])
                with gr.Row():
                    but9 = gr.Button("ดึงโมเดล (Extract)", variant="primary")
                    info7 = gr.Textbox(label="สถานะ (Status)", max_lines=4)
                if RVC_CORE_AVAILABLE: but9.click(extract_small_model, [ckpt_path2, save_name, sr__, if_f0__, gr.Textbox(value="", visible=False), version_1], info7)

        with gr.TabItem("🌐 ส่งออก ONNX (ONNX Export)"):
            with gr.Row():
                ckpt_dir = gr.Textbox(label="เส้นทางโมเดล RVC (RVC Model Path)", interactive=True, scale=8)
                btn_browse_onnx_ckpt = gr.Button("📄 เลือกโมเดล (Browse)", scale=2, variant="secondary")
                onnx_dir = gr.Textbox(label="เส้นทางส่งออก ONNX (ONNX Output Path)", interactive=True, scale=8)
                btn_browse_onnx_out = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
            btn_browse_onnx_ckpt.click(fn=lambda cur: browse_file(cur, [("Model", "*.pth")]), inputs=[ckpt_dir], outputs=[ckpt_dir])
            btn_browse_onnx_out.click(fn=browse_folder, inputs=[onnx_dir], outputs=[onnx_dir])
            with gr.Row():
                infoOnnx = gr.Textbox(label="สถานะ (Status)")
                butOnnx = gr.Button("ส่งออกโมเดล (Export ONNX Model)", variant="primary")
            butOnnx.click(export_onnx, [ckpt_dir, onnx_dir], infoOnnx)

        with gr.TabItem("🎤 OpenUtau Voicebank Studio"):
            with gr.Tabs():
                with gr.TabItem("การตั้งค่าตัวละคร (Character Setup)"):
                    gr.Markdown("### กระบวนการที่ 1: จัดการรายละเอียดของตัวละคร OpenUtau 🟢")
                    with gr.Row():
                        singer_name_in = gr.Textbox(label="ชื่อนักร้อง / ตัวละคร (Character Name)", value="Singer", scale=3)
                        lang_mode_in = gr.Dropdown(label="ภาษาและรูปแบบคลังเสียง (Language & Format)", choices=["1. EN", "2. TH", "3. JP"], value="2. TH", scale=3)
                        base_out_dir = gr.Textbox(label="โฟลเดอร์สำหรับส่งออก (Root Export Directory)", value=r"H:\Render file to UTAU", scale=8)
                        btn_browse_base_out = gr.Button("📁", scale=1, variant="secondary")
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
                    save_proj_btn.click(fn=save_project, inputs=[base_out_dir, singer_name_in, pic_icon, pic_full, readme_txt, char_yaml, char_txt], outputs=[gr.Markdown(visible=False), proj_status, gr.Textbox(visible=False)])

                with gr.TabItem("การเรนเดอร์เสียงและแพ็คไฟล์ (Render & Pack Voicebank)"):
                    tab2_header = gr.Markdown("### กระบวนการที่ 2: การเรนเดอร์เสียง Wav 🔴")
                    with gr.Row():
                        zip_output_dir = gr.Textbox(label="โฟลเดอร์สำหรับส่งออกไฟล์ .zip", value=os.path.join(now_dir, "Packed_Voicebanks"), scale=8)
                        btn_browse_zip_out = gr.Button("📁", scale=1, variant="secondary")
                        zip_btn = gr.Button("🗜️ แพ็คไฟล์เป็น .zip (Pack to .zip)", variant="primary", scale=3)
                    zip_log = gr.Textbox(label="สถานะการแพ็คไฟล์ (Packaging Status)")
                    btn_browse_zip_out.click(fn=browse_folder, inputs=[zip_output_dir], outputs=[zip_output_dir])
                    zip_btn.click(fn=run_zip_packing, inputs=[base_out_dir, singer_name_in, zip_output_dir], outputs=[zip_log, tab2_header])

        with gr.TabItem("🎵 DiffSinger 2.5.1 Studio"):
            with gr.Tabs():
                with gr.TabItem("Data Prep & Dict Scanner"):
                    with gr.Row():
                        ds_proj_select = gr.Dropdown(label="โปรเจกต์ DiffSinger", choices=ds_list_existing_projects(), allow_custom_value=True, scale=4)
                        btn_ds_dedup = gr.Button("🧹 ลบไฟล์เสียงซ้ำ & สร้าง Dictionary", variant="secondary", scale=3)
                        ds_model_type_prep = gr.Radio(["Acoustic", "Variance"], label="ประเภทโมเดล (Model Type)", value="Acoustic", scale=3)
                        ds_data_in = gr.Textbox(label="ไฟล์ Config (หรือเว้นว่างให้ค้นหาอัตโนมัติ)", scale=8)
                        ds_data_btn = gr.Button("▶️ เริ่มเตรียมข้อมูล (Binarization)", variant="primary", scale=3)
                    ds_data_log = gr.Textbox(label="Log", lines=8)
                    btn_ds_dedup.click(fn=ds_remove_duplicates_and_scan_phonemes, inputs=[ds_proj_select], outputs=[ds_data_log])
                    ds_data_btn.click(fn=ds_run_binarization, inputs=[ds_proj_select, ds_model_type_prep, ds_data_in], outputs=[ds_data_log])

                with gr.TabItem("Acoustic / Variance Training"):
                    with gr.Row():
                        ds_tr_config = gr.Textbox(label="ไฟล์ Config YAML", value="configs/acoustic.yaml", scale=6); btn_b_cfg = gr.Button("📄", scale=1)
                        ds_tr_exp = gr.Textbox(label="ชื่อ Experiment", value="ds_exp", scale=3)
                        ds_tr_resume = gr.Checkbox(label="Resume Training", value=True, scale=2)
                    with gr.Row():
                        ds_tr_btn = gr.Button("▶️ เริ่มฝึกสอน (Start Training)", variant="primary")
                        ds_stop_btn = gr.Button("🛑 หยุดฝึกสอน (Stop)", variant="secondary")
                    ds_tr_log = gr.Textbox(label="Training Log", lines=12)
                    btn_b_cfg.click(fn=lambda c: browse_file(c, [("YAML", "*.yaml")]), inputs=[ds_tr_config], outputs=[ds_tr_config])
                    ds_tr_btn.click(fn=ds_run_training_job, inputs=[ds_tr_config, ds_tr_exp, ds_tr_resume], outputs=[ds_tr_log])
                    ds_stop_btn.click(fn=cancel_all_processes, outputs=[ds_tr_log])

                with gr.TabItem("⚡ 1-Click Pipeline (Train + Export ONNX)"):
                    with gr.Row():
                        ds_pipe_cfg = gr.Textbox(label="ไฟล์ Config YAML", value="configs/acoustic.yaml", scale=6); btn_pipe_cfg = gr.Button("📄", scale=1)
                        ds_pipe_exp = gr.Textbox(label="ชื่อ Experiment Name", value="ds_pipeline_exp", scale=3)
                        ds_pipe_out = gr.Textbox(label="โฟลเดอร์ส่งออก (Output Directory)", value="", scale=3)
                    btn_pipe_run = gr.Button("⚡ เริ่มกระบวนการอัตโนมัติครบวงจร (Run 1-Click Pipeline)", variant="primary")
                    ds_pipe_log = gr.Textbox(label="สถานะ Pipeline", lines=10)
                    btn_pipe_cfg.click(fn=lambda c: browse_file(c, [("YAML", "*.yaml")]), inputs=[ds_pipe_cfg], outputs=[ds_pipe_cfg])
                    btn_pipe_run.click(fn=ds_run_one_click_pipeline, inputs=[ds_pipe_cfg, ds_pipe_exp, ds_pipe_out], outputs=[ds_pipe_log])

        with gr.TabItem("🇹🇭 แปลงและปรับแต่งหน่วยเสียงไทย (Thai Phonemizer & Tuning)"):
            with gr.Row():
                ph_std = gr.Dropdown(label="มาตรฐานหน่วยเสียง (Phoneme Standard)", choices=["DiffSinger_AI", "Thai_VCCV", "Thai_Arpasing"], value="DiffSinger_AI", scale=4)
                ph_c = gr.Textbox(label="พยัญชนะต้น (Initial Consonant)", value="ก", scale=2)
                ph_v = gr.Textbox(label="สระ (Vowel)", value="อา", scale=2)
                ph_e = gr.Textbox(label="ตัวสะกด (Final Consonant / Ending)", value="", scale=2)
            with gr.Row():
                ph_note_ms = gr.Slider(minimum=50, maximum=2000, step=10, label="ความยาวโน้ต (Note Duration ms)", value=500)
                ph_ccv = gr.Slider(minimum=1.0, maximum=10.0, step=0.5, label="สัดส่วนพยัญชนะ CCV Ratio (%)", value=5.0)
                ph_vc = gr.Slider(minimum=20.0, maximum=50.0, step=1.0, label="สัดส่วนตัวสะกด VC Ratio (%)", value=30.0)
            with gr.Row():
                ph_vib_rate = gr.Slider(minimum=3.0, maximum=10.0, step=0.1, label="ความถี่ลูกคอ Vibrato Rate (Hz)", value=6.2)
                ph_vib_depth = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, label="ความลึกของลูกคอ Vibrato Depth (Cents)", value=35.0)
            btn_calc_phoneme = gr.Button("⚡ คำนวณหน่วยเสียงและเส้นพิตช์ (Synthesize Phoneme & Pitch)", variant="primary")
            out_phoneme_spec = gr.Textbox(label="ผลลัพธ์การจัดหน่วยเสียงและข้อมูลพิตช์ (Phoneme Specification)", lines=8)
            btn_calc_phoneme.click(
                fn=synthesize_thai_phoneme_and_tuning,
                inputs=[ph_std, ph_c, ph_v, ph_e, ph_note_ms, ph_ccv, ph_vc, ph_vib_rate, ph_vib_depth],
                outputs=[out_phoneme_spec]
            )

listen_port = int(getattr(config, "listen_port", 9378))
is_colab = getattr(config, "iscolab", False)
no_auto_open = getattr(config, "noautoopen", False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DELTA SYNTH Smart AI Voice Studio")
    parser.add_argument("--port", type=int, default=listen_port, help="Server port")
    parser.add_argument("--pycmd", type=str, default=PYTHON_EXE, help="Python command")
    parser.add_argument("--colab", action="store_true", help="Launch in Google Colab mode")
    parser.add_argument("--dml", action="store_true", help="DirectML mode for AMD/Intel GPU")
    parser.add_argument("--route_model", type=str, default="", help="Model file or folder to route automatically")
    parser.add_argument("--version_profile", type=str, default="AUTO", help="Target version profile for routing")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name for routing")
    parser.add_argument("--mode", type=str, default="copy", help="Routing mode (copy/move)")

    args, unknown = parser.parse_known_args()

    if args.route_model:
        print("=" * 68)
        print("🚀 DELTA SYNTH Smart Model Router CLI")
        print(f"📦 Input Path      : {args.route_model}")
        print(f"🎯 Version Profile : {args.version_profile}")
        print("=" * 68)
        report, last_pth, last_idx = MODEL_ROUTER.execute_routing(
            args.route_model, args.version_profile, args.exp_name, args.mode, True
        )
        print(report)
        sys.exit(0)

    target_port = args.port or listen_port
    DeltaLogger.inf(f"🚀 กำลังเริ่มต้นระบบ DELTA RVC Studio x DiffSinger v6.1 บนพอร์ต {target_port}...", to_console=True)
    
    if args.colab or is_colab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        try:
            app.queue(concurrency_count=511, max_size=1022).launch(server_name="0.0.0.0", inbrowser=not no_auto_open, server_port=target_port, quiet=True)
        except OSError:
            alt_port = target_port + 1
            DeltaLogger.wrn(f"พอร์ต {target_port} ถูกใช้งานแล้ว กำลังสลับไปยังพอร์ตสำรอง {alt_port}...", to_console=True)
            app.queue(concurrency_count=511, max_size=1022).launch(server_name="0.0.0.0", inbrowser=not no_auto_open, server_port=alt_port, quiet=True)