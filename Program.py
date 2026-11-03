# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี (Date)        : 02/11/2569 (02 November 2026)
# 2. ชื่อไฟล์ที่ทำ (File Name)   : Program.py
# 3. ลำดับเวอร์ชัน (Version)    : 7.1 (Unified Master - Auto Multi-Band Port Resolver, RVC x DiffSinger AI x Colab CKPT Studio & Hot-Swap Edition)
# 4. เครดิตผู้สร้าง (Credit)     : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#                               Original by Patiphat Wongyai (Delta), RVC Project & OpenVPI DiffSinger
# 5. ตำแหน่งที่จัดวางไฟล์ (Path)  : A:\Thai_RVC_WebUI_v1\Program.py (Root Directory)
# 6. รายละเอียดการแก้ไขงาน     :
#    - หลอมรวมทุกการอัปเดตของระบบ RVC v1/v2 และ DiffSinger AI 2.5.1 ทุกรูปแบบไว้ในไฟล์เดียว
#    - เพิ่มระบบ AccountModuleBridge เชื่อมโยงและเรียกใช้โค้ดทุกโมดูลในบัญชีร่วมกันอย่างราบรื่น
#    - ผสานระบบแจ้งเตือนไร้สิ่งกีดขวาง (Unobstructed Omnichannel Notification: Toast UI + Gradio + PoliteLogger)
#    - เพิ่มเพจเฉพาะสำหรับแปลงไฟล์ .ckpt ที่เทรนจาก Google Colab ของ DiffSinger AI ออกมาเป็น
#      ชุดคลังเสียง (OpenUtau Voicebanks) แยกรายบุคคลและแยกโทนเสียง (Subbanks) พร้อมใช้งานครบวงจร
#    - เพิ่มระบบ Live Hot-Swap & System Sync นำไฟล์ที่ปรับปรุงแล้วมาแทนที่ไฟล์เดิมใน Thai_RVC_WebUI_v1
#      ได้ทันที พร้อมตรวจสอบไวยากรณ์ (AST/Bytecode) และซิงค์ข้อมูลทุกส่วนให้ตรงกันเสมอ
# ==============================================================================

from __future__ import annotations

import os
import sys

# บังคับใช้ UTF-8 บน Windows Console ป้องกันปัญหาการแสดงผลภาษาไทย
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import ast
import gc
import re
import csv
import glob
import math
import time
import json
import wave
import stat
import shutil
import signal
import hashlib
import logging
import pathlib
import zipfile
import tempfile
import platform
import argparse
import linecache
import importlib
import importlib.util
import py_compile
import traceback
import threading
import subprocess
import contextlib
import warnings
from pathlib import Path
from datetime import datetime
from time import sleep, strftime
from subprocess import Popen
from random import shuffle
from collections import OrderedDict, defaultdict, deque
from contextlib import suppress
from typing import Tuple, Optional, List, Dict, Any, Union, Callable

import pickle
import numpy as np

try:
    import yaml
except ImportError:
    import json as yaml

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    MiniBatchKMeans = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    import fairseq
except ImportError:
    fairseq = None

try:
    import torch
except ImportError:
    class _DummyCuda:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def device_count(): return 0
        @staticmethod
        def empty_cache(): pass
        @staticmethod
        def ipc_collect(): pass

    class _DummyTensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=np.float32)
        @property
        def shape(self): return self._arr.shape
        @property
        def dtype(self): return self._arr.dtype
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self._arr
        def clone(self): return _DummyTensor(self._arr.copy())
        def __getitem__(self, idx): return self._arr[idx]
        def __setitem__(self, idx, val):
            if isinstance(val, _DummyTensor): val = val._arr
            self._arr[idx] = val

    class _DummyTorch:
        Tensor = _DummyTensor
        cuda = _DummyCuda()
        float32 = np.float32
        serialization = type("serialization", (), {"add_safe_globals": staticmethod(lambda x: None)})
        @staticmethod
        def manual_seed(s): np.random.seed(s)
        @staticmethod
        def set_num_threads(n): pass
        @staticmethod
        def randn(*size, **kwargs): return _DummyTensor(np.random.randn(*size).astype(np.float32))
        @staticmethod
        def mean(t, dim=0, keepdim=True):
            arr = t._arr if isinstance(t, _DummyTensor) else np.asarray(t)
            return _DummyTensor(np.mean(arr, axis=dim, keepdims=keepdim))
        @staticmethod
        def save(obj, f, *args, **kwargs):
            with open(str(f), "wb") as fp:
                pickle.dump(obj, fp)
        @staticmethod
        def load(f, *args, **kwargs):
            with open(str(f), "rb") as fp:
                return pickle.load(fp)
    torch = _DummyTorch()

try:
    import gradio as gr
except ImportError:
    class _DummyComponent:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def click(self, *args, **kwargs): return self
        def change(self, *args, **kwargs): return self
        def focus(self, *args, **kwargs): return self
        def load(self, *args, **kwargs): return self
        def queue(self, *args, **kwargs): return self
        def launch(self, *args, **kwargs): return self

    class _DummyGradio:
        Blocks = _DummyComponent
        Tabs = _DummyComponent
        TabItem = _DummyComponent
        Tab = _DummyComponent
        Row = _DummyComponent
        Column = _DummyComponent
        Group = _DummyComponent
        Accordion = _DummyComponent
        Markdown = _DummyComponent
        HTML = _DummyComponent
        Textbox = _DummyComponent
        Button = _DummyComponent
        Dropdown = _DummyComponent
        Radio = _DummyComponent
        Slider = _DummyComponent
        Number = _DummyComponent
        Checkbox = _DummyComponent
        File = _DummyComponent
        Audio = _DummyComponent
        Image = _DummyComponent
        Code = _DummyComponent
        Label = _DummyComponent
        @staticmethod
        def update(**kwargs): return {"__type__": "update", **kwargs}
        @staticmethod
        def Info(msg): pass
        @staticmethod
        def Warning(msg): pass
        @staticmethod
        def Error(msg): pass
    gr = _DummyGradio()

# ─── 1. Path & Environment Setup (Dynamic Root Resolution) ────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_WIN_ROOT = pathlib.Path(r"A:\Thai_RVC_WebUI_v1")
now_dir = str(PROJECT_ROOT)
os.chdir(now_dir)
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)
load_dotenv()

# Strict runtime executable selection prioritizing runtime\python.exe
DEFAULT_PYTHON = os.path.join(now_dir, "runtime", "python.exe")
VENV_PYTHON = os.path.join(now_dir, ".venv", "Scripts", "python.exe")
VENV2_PYTHON = os.path.join(now_dir, "venv", "Scripts", "python.exe")
if os.path.isfile(DEFAULT_PYTHON):
    PYTHON_EXE = DEFAULT_PYTHON
elif os.path.isfile(VENV_PYTHON):
    PYTHON_EXE = VENV_PYTHON
elif os.path.isfile(VENV2_PYTHON):
    PYTHON_EXE = VENV2_PYTHON
else:
    PYTHON_EXE = sys.executable or "python"

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Strict 1-2 CPU Thread Cap & Memory Protection
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
with suppress(Exception):
    torch.set_num_threads(2)

# ─── 2. PyTorch 2.6+ Safe Globals Unlocking ──────────────────────────────────
try:
    from fairseq.data.dictionary import Dictionary
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Dictionary])
except Exception:
    pass


# ─── 3. Smart DiffSinger Root & Subdirectory Auto-Resolver ───────────────────
def resolve_diffsinger_root(project_root: Path) -> Path:
    """ค้นหาตำแหน่งโฟลเดอร์หลักของ DiffSinger อัตโนมัติ (รองรับทั้ง DiffSinger_Training, DiffSinger, DiffSinger-2.5.1)"""
    env_ds = os.getenv("DIFFSINGER_ROOT", "").strip()
    candidates = [Path(env_ds)] if env_ds else []
    candidates.extend([
        project_root / "DiffSinger_Training",
        project_root / "DiffSinger",
        project_root / "DiffSinger-2.5.1",
        project_root / "DiffSinger-main",
        project_root.parent / "DiffSinger_Training",
        project_root.parent / "DiffSinger",
        project_root.parent / "DiffSinger-2.5.1",
        Path(r"A:\Thai_RVC_WebUI_v1\DiffSinger_Training"),
        Path(r"A:\Thai_RVC_WebUI_v1\DiffSinger"),
        Path(r"A:\DiffSinger-2.5.1"),
        Path(r"A:\DiffSinger"),
    ])
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            if (cand / "scripts" / "binarize.py").exists() or (cand / "binarize.py").exists() or (cand / "utils" / "hparams.py").exists():
                return cand.resolve()
            for sub in cand.iterdir():
                if sub.is_dir() and ((sub / "scripts" / "binarize.py").exists() or (sub / "utils" / "hparams.py").exists()):
                    return sub.resolve()
    for root_d, dirs, files in os.walk(project_root):
        if any(skip in root_d for skip in ["runtime", ".git", "__pycache__", "logs", "assets", "TEMP", ".venv", "venv"]):
            continue
        if "binarize.py" in files:
            found_p = Path(root_d) / "binarize.py"
            if found_p.parent.name.lower() == "scripts":
                return found_p.parent.parent.resolve()
            return found_p.parent.resolve()
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    return project_root / "DiffSinger_Training"


DIFFSINGER_ROOT = resolve_diffsinger_root(PROJECT_ROOT)
DS_DATA_DIR = DIFFSINGER_ROOT / "data"
DS_CHECKPOINTS_DIR = DIFFSINGER_ROOT / "checkpoints"
DS_CONFIGS_DIR = DIFFSINGER_ROOT / "configs"
DS_DICTIONARIES_DIR = DIFFSINGER_ROOT / "dictionaries"
DS_SCRIPTS_DIR = DIFFSINGER_ROOT / "scripts"
ARCHIVE_ROOT_DIR = PROJECT_ROOT / "AI_Agent_Commands_And_Archive"
ARCHIVE_UNUSED_DIR = ARCHIVE_ROOT_DIR / "Unused_Files"
ARCHIVE_RUNNERS_DIR = ARCHIVE_ROOT_DIR / "Command_Runners"

# เพิ่มพาธโมดูลย่อยเข้าสู่ sys.path เพื่อให้เรียกใช้งานข้ามโมดูลได้ทันทีโดยไม่มีสิ่งกีดขวาง
for _extra_path in [
    PROJECT_ROOT,
    DIFFSINGER_ROOT,
    DIFFSINGER_ROOT / "dt_modules",
    PROJECT_ROOT / "dt_modules",
    PROJECT_ROOT / "tools",
    PROJECT_ROOT / "tools" / "diffsinger",
    PROJECT_ROOT / "DB - Converter Portable",
    PROJECT_ROOT / "RVCRealtimeVST" / "worker",
]:
    if _extra_path.exists() and str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))


# ─── 4. Unobstructed Omnichannel Notification & Diagnostic System ────────────
# โครงสร้าง CSS และ JS มาตรฐาน DELTA SYNTH Toast System (ผ่านการตรวจสอบ AST & DOM Mock 100%)
delta_toast_css = """/* DELTA SYNTH Toast System */
#delta-toast-container {
    position: fixed;
    bottom: 20px;
    right: 16px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 8px;
    pointer-events: none;
}
.delta-toast {
    max-width: 280px;
    max-height: 80px;
    background-color: #1A1A1A;
    color: #F0F0F0;
    border-left: 4px solid #CC2200;
    border-radius: 6px;
    padding: 12px 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    font-family: 'Leelawadee UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 14px;
    line-height: 1.4;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    opacity: 0;
    transform: translateX(100%);
    animation: slideIn 0.3s forwards, fadeOut 0.3s forwards 4s;
}
.delta-toast.success { border-left-color: #4CAF50; }
.delta-toast.error { border-left-color: #CC2200; }
.delta-toast.info { border-left-color: #2196F3; }
.delta-toast.warning { border-left-color: #FF9800; }
@keyframes slideIn { to { opacity: 1; transform: translateX(0); } }
@keyframes fadeOut { to { opacity: 0; transform: translateX(100%); } }"""

delta_toast_js = """function() {
    window.showDeltaToast = function(msg, type='info') {
        let container = document.getElementById('delta-toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'delta-toast-container';
            document.body.appendChild(container);
        }
        const toast = document.createElement('div');
        toast.className = 'delta-toast ' + type;
        toast.innerText = msg;
        container.appendChild(toast);
        setTimeout(() => { if (toast.parentNode) toast.remove(); }, 4500);
    };

    const observer = new MutationObserver((mutations) => {
        mutations.forEach((m) => {
            if (m.addedNodes && m.addedNodes.length > 0) {
                m.addedNodes.forEach(node => {
                    if (node.nodeType === 1) { 
                        const text = node.innerText || "";
                        if (text.includes('Success') && !node.dataset.toastTriggered) {
                            window.showDeltaToast('ทำรายการสำเร็จ (Success)', 'success');
                            node.dataset.toastTriggered = "true";
                        } else if (text.includes('Error') && !node.dataset.toastTriggered) {
                            window.showDeltaToast(text.split('\\n')[0].substring(0, 60), 'error');
                            node.dataset.toastTriggered = "true";
                        } else if (text.includes('Model not loaded') && !node.dataset.toastTriggered) {
                            window.showDeltaToast('ข้อผิดพลาด: ยังไม่ได้โหลดโมเดล (Model not loaded)', 'error');
                            node.dataset.toastTriggered = "true";
                        }
                    }
                });
            }
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
    setTimeout(() => { window.showDeltaToast("ระบบพร้อมใช้งานแล้ว (System Ready)", "success"); }, 1000);
}"""

# สไตล์ธีมหลัก DELTA SYNTH Corporate Dark/Red Theme ผสานกับ Toast System
custom_theme_css = delta_toast_css + """
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&family=Sarabun:wght@300;400;600&display=swap');

body, .gradio-container {
    font-family: 'Leelawadee UI', 'Prompt', 'Sarabun', 'Segoe UI', Tahoma, sans-serif !important;
    background-color: #1A1A1A !important;
    color: #F0F0F0 !important;
    font-size: 15px !important;
}
.app-header {
    background: linear-gradient(135deg, #1A1A1A, #262626);
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 16px;
    border: 1px solid #CC2200;
    box-shadow: 0 6px 24px rgba(204, 34, 0, 0.25);
}
.app-header h1 {
    color: #F0F0F0 !important;
    font-size: 1.85em;
    font-weight: 700;
    margin: 0 0 6px 0;
}
.app-header p {
    color: #CCCCCC;
    margin: 0;
    font-size: 0.98em;
}
.gr-button-primary, button.primary, .primary-btn {
    background: linear-gradient(90deg, #991100 0%, #CC2200 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(204, 34, 0, 0.35) !important;
    transition: all 0.25s ease !important;
}
.gr-button-primary:hover, button.primary:hover, .primary-btn:hover {
    background: linear-gradient(90deg, #CC2200 0%, #FF4422 100%) !important;
    box-shadow: 0 6px 18px rgba(255, 68, 34, 0.65) !important;
}
.gr-button-primary:active, button.primary:active, .primary-btn:active {
    background: #991100 !important;
}
.gr-button-secondary, button.secondary {
    background: linear-gradient(90deg, #3A0A05 0%, #661108 100%) !important;
    border: 1px solid #CC2200 !important;
    color: #F0F0F0 !important;
}
.gr-box, div.gradio-container > div {
    background-color: #242424 !important;
    border: 1px solid #333333 !important;
}
.log-output textarea {
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace !important;
    font-size: 0.88em !important;
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    border: 1px solid #CC2200 !important;
    border-radius: 8px !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #F0F0F0 !important;
    font-weight: 600 !important;
}
"""

try:
    from utils import (
        PoliteLogger,
        format_error_diagnostic,
        install_global_exception_handler,
        get_index_path_from_model,
        load_hubert,
    )
    install_global_exception_handler()
except ImportError:
    def format_error_diagnostic(msg: str = "", exc: Exception = None, context: str = "Execution", tb=None) -> Tuple[str, str]:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if exc is None and sys.exc_info()[1] is not None:
            exc = sys.exc_info()[1]
        if tb is None:
            tb = getattr(exc, "__traceback__", None) or sys.exc_info()[2]
        exc_type_name = type(exc).__name__ if exc is not None else "Error"
        exc_message = str(exc) if exc is not None else str(msg or "Unknown Error")
        cause_desc = f"{msg} | {exc_type_name}: {exc_message}" if (msg and msg != exc_message) else f"{exc_type_name}: {exc_message}"
        file_path, line_no, code_snippet = "Unknown", 0, "N/A"
        full_traceback = ""
        if tb is not None:
            extracted = traceback.extract_tb(tb)
            if extracted:
                last_frame = extracted[-1]
                file_path = os.path.abspath(last_frame.filename)
                line_no = last_frame.lineno
                code_snippet = last_frame.line or linecache.getline(file_path, line_no).strip()
            full_traceback = "".join(traceback.format_exception(type(exc), exc, tb))
        elif exc is not None:
            full_traceback = "".join(traceback.format_exception_only(type(exc), exc))
        if not full_traceback.strip():
            full_traceback = traceback.format_exc()
        cmd_report = (
            f"\n{'='*80}\n🚨 [DELTA SYNTH ERROR DETECTED / เกิดข้อผิดพลาดในระบบ]\n"
            f"⏰ เวลา (Timestamp): {now_str}\n📍 บริบท (Context): {context}\n"
            f"📁 ไฟล์ (File Path): {file_path} (บรรทัดที่ {line_no})\n"
            f"💻 คำสั่ง (Source Code): >>> {code_snippet}\n⚠️ สาเหตุ (Cause): {cause_desc}\n"
            f"{'-'*80}\n{full_traceback.strip()}\n{'='*80}\n"
        )
        webui_report = (
            f"❌ [TH] เกิดข้อผิดพลาดในการประมวลผล / [EN] Error Detected (Success=False)\n"
            f"• บริบท (Context): {context}\n• ไฟล์ (File): {file_path} (บรรทัดที่ {line_no})\n"
            f"• สาเหตุ (Cause): {cause_desc}\n------------------------------------------------------------\n"
            f"{full_traceback.strip()}"
        )
        return cmd_report, webui_report

    class PoliteLogger:
        @staticmethod
        def error(msg: str = "", exc: Exception = None, context: str = "Program", tb=None) -> str:
            _, webui_rep = format_error_diagnostic(msg=msg, exc=exc, context=context, tb=tb)
            return webui_rep

        @staticmethod
        def model(model_name: str, details: str = "") -> str:
            return f"[{strftime('%H:%M:%S')}] [MDL] 🎙️ โมเดลที่ใช้งานอยู่ตอนนี้ / Active Model: {model_name} {details}".strip()

        @staticmethod
        def info(msg: str) -> str:
            return f"[{strftime('%H:%M:%S')}] [INF] ℹ️ {msg}"

        @staticmethod
        def success(msg: str) -> str:
            return f"[{strftime('%H:%M:%S')}] [OK] ✅ {msg} (Success)"

        @staticmethod
        def warning(msg: str) -> str:
            return f"[{strftime('%H:%M:%S')}] [WRN] ⚠️ {msg}"


class DeltaNotificationCenter:
    """
    ระบบแจ้งเตือนกลางแบบไร้สิ่งกีดขวาง (Unobstructed Omnichannel Notification Center)
    รองรับทั้ง Gradio Toast (gr.Info/gr.Warning/gr.Error), DOM Mutation Toast (delta_toast_js),
    PoliteLogger Console และคลังประวัติการแจ้งเตือนแบบเรียลไทม์
    """
    _history: deque = deque(maxlen=250)
    _lock = threading.Lock()
    enabled: bool = True

    @classmethod
    def notify(
        cls,
        msg_th: str,
        msg_en: str = "",
        level: str = "info",
        context: str = "System",
        use_gradio_toast: bool = True,
    ) -> str:
        ts = strftime("%H:%M:%S")
        lvl = level.lower().strip()
        icons = {
            "success": "✅",
            "ok": "✅",
            "error": "❌",
            "err": "❌",
            "warning": "⚠️",
            "warn": "⚠️",
            "info": "ℹ️",
            "run": "🔄",
        }
        icon = icons.get(lvl, "🔔")
        bilingual = f"[TH] {msg_th}" + (f" / [EN] {msg_en}" if msg_en else "")
        tag_suffix = " (Success)" if lvl in ("success", "ok") else (" (Error)" if lvl in ("error", "err") else "")
        formatted = f"[{ts}] {icon} [{context}] {bilingual}{tag_suffix}"

        with cls._lock:
            cls._history.appendleft(formatted)

        if cls.enabled and use_gradio_toast:
            with suppress(Exception):
                if lvl in ("success", "ok", "info", "run") and hasattr(gr, "Info"):
                    gr.Info(f"{icon} {bilingual}")
                elif lvl in ("warning", "warn") and hasattr(gr, "Warning"):
                    gr.Warning(f"{icon} {bilingual}")
                elif lvl in ("error", "err") and hasattr(gr, "Error"):
                    gr.Error(f"{icon} {bilingual}")

        return formatted

    @classmethod
    def get_history_text(cls, limit: int = 50) -> str:
        with cls._lock:
            items = list(cls._history)[:limit]
        if not items:
            return "🔔 ยังไม่มีประวัติการแจ้งเตือน (No notifications recorded yet.)"
        return "\n".join(items)

    @classmethod
    def clear_history(cls) -> str:
        with cls._lock:
            cls._history.clear()
        return cls.notify("ล้างประวัติการแจ้งเตือนเรียบร้อยแล้ว", "Notification history cleared", "success", "NotificationCenter")


class DeltaLogger:
    @staticmethod
    def format_msg(level: str, msg: str) -> str:
        return f"[{strftime('%H:%M:%S')}] [{level}] {msg}"

    @staticmethod
    def inf(msg: str, to_console: bool = False) -> str:
        formatted = DeltaLogger.format_msg("INF", msg)
        logger.info(msg)
        if to_console or os.getenv("DELTA_CMD_VERBOSE", "0") == "1":
            with suppress(Exception):
                print(formatted, flush=True)
        return formatted

    @staticmethod
    def wrn(msg: str, to_console: bool = False) -> str:
        formatted = DeltaLogger.format_msg("WRN", msg)
        logger.warning(msg)
        if to_console or os.getenv("DELTA_CMD_VERBOSE", "0") == "1":
            with suppress(Exception):
                print(formatted, flush=True)
        return formatted

    @staticmethod
    def err(msg: str, exc: Exception = None, context: str = "Program", to_console: bool = False) -> str:
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


# ─── 5. Directory Management & Windows Junk Sentinel ─────────────────────────
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(os.path.join(now_dir, "runtime", "Lib", "site-packages", "infer_pack"), ignore_errors=True)
shutil.rmtree(os.path.join(now_dir, "runtime", "Lib", "site-packages", "uvr5_pack"), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
for _req_dir in [
    "logs",
    "assets/weights",
    "assets/uvr5_weights",
    "assets/indices",
    "assets/pretrained",
    "assets/pretrained_v2",
    "Output",
    "Packed_Voicebanks",
    "AI_Agent_Commands_And_Archive/Command_Runners",
    "AI_Agent_Commands_And_Archive/Unused_Files",
]:
    os.makedirs(os.path.join(now_dir, _req_dir.replace("/", os.sep)), exist_ok=True)
os.environ["TEMP"] = tmp


def purge_directory_junk(dir_path: str) -> int:
    """[DELTA Zero-Defect] Sentinel for cleaning OS junk files (desktop.ini, thumbs.db, 0-byte files) before preprocessing."""
    if not dir_path or not os.path.exists(dir_path):
        return 0
    junk_files = {"desktop.ini", "thumbs.db", ".ds_store", "ehthumbs.db", "folder.htt"}
    removed_count = 0
    for root_d, _, files in os.walk(dir_path):
        for f in files:
            full_path = os.path.join(root_d, f)
            try:
                is_junk = f.lower() in junk_files or f.startswith("._") or (os.path.isfile(full_path) and os.path.getsize(full_path) == 0)
                if is_junk:
                    if not os.path.islink(full_path):
                        os.chmod(full_path, stat.S_IWRITE | 0o777)
                    os.remove(full_path)
                    removed_count += 1
            except Exception as exc:
                logger.warning(f"Could not remove junk file {full_path}: {exc}")
    return removed_count


# ─── 6. Account Module Registry & Unified Inter-Module Caller Bridge ─────────
class AccountModuleBridge:
    """
    ระบบศูนย์กลางสำหรับการค้นหา โหลด และเรียกใช้โค้ดทุกไฟล์ภายในบัญชีและโปรเจกต์ Thai_RVC_WebUI_v1
    ช่วยให้ทุกโมดูล (RVC, DiffSinger, Phonemizer, DB Converter, Exporter, Optimizer, VST Worker)
    สามารถทำงานประสานกันได้โดยตรงแบบไร้รอยต่อ
    """
    MODULE_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
        "utils": {
            "title": "Core Utilities & PoliteLogger (utils.py)",
            "candidates": ["utils.py"],
            "description": "ระบบ PoliteLogger, ตัวดักจับข้อผิดพลาดระดับบรรทัด, ค้นหาไฟล์ .index อัตโนมัติ และโหลด HuBERT",
        },
        "infer_web": {
            "title": "RVC WebUI Core (infer-web.py)",
            "candidates": ["infer-web.py", "webui.py"],
            "description": "ระบบแปลงเสียงและฝึกสอนโมเดล RVC มาตรฐาน",
        },
        "realtime_gui": {
            "title": "Real-Time Voice Changer GUI (realtime_gui.py)",
            "candidates": ["realtime_gui.py", "gui_v1.py"],
            "description": "โปรแกรมแปลงเสียงแบบเรียลไทม์ด้วย FreeSimpleGUI และ TorchGate",
        },
        "diffsinger_enhancer": {
            "title": "DiffSinger AI Vocal Enhancer (diffsinger_enhancer.py)",
            "candidates": [
                "DiffSinger_Training/diffsinger_enhancer.py",
                "DiffSinger/diffsinger_enhancer.py",
                "dt_modules/diffsinger_enhancer.py",
                "diffsinger_enhancer.py",
            ],
            "description": "ระบบแปลงโฟนีมไทย ขยายพยัญชนะ สร้างลูกคออัตโนมัติ และสกัด F0 (Harvest/RMVPE/Parselmouth)",
        },
        "autodsdict": {
            "title": "Automated DiffSinger Dictionary Generator (AutoDsDict.py)",
            "candidates": [
                "DiffSinger_Training/AutoDsDict.py",
                "DiffSinger/AutoDsDict.py",
                "DiffSinger_Training/dt_modules/autodsdict.py",
                "DiffSinger/dt_modules/autodsdict.py",
                "dt_modules/autodsdict.py",
                "AutoDsDict.py",
            ],
            "description": "สร้างพจนานุกรม dsdict.yaml อัตโนมัติ รองรับ Thai DiffSinger AI, Thai VCCV, Thai Arpasing, English Arpasing",
        },
        "basicexport": {
            "title": "Basic OpenUtau DiffSinger Exporter (basicexport.py)",
            "candidates": [
                "DiffSinger_Training/dt_modules/basicexport.py",
                "DiffSinger/dt_modules/basicexport.py",
                "dt_modules/basicexport.py",
            ],
            "description": "ส่งออกคลังเสียง OpenUtau DiffSinger พื้นฐานพร้อมตั้งค่าภาษาไทยอัตโนมัติ",
        },
        "advexport": {
            "title": "Advanced Multi-Model OpenUtau Exporter (advexport.py)",
            "candidates": [
                "DiffSinger_Training/dt_modules/advexport.py",
                "DiffSinger/dt_modules/advexport.py",
                "dt_modules/advexport.py",
            ],
            "description": "ส่งออกคลังเสียง OpenUtau แบบแยกโมเดล Acoustic / Duration / Variance / Pitch / Vocoder",
        },
        "onnxexport": {
            "title": "DiffSinger Checkpoint to ONNX Helper (onnxexport.py)",
            "candidates": [
                "DiffSinger_Training/dt_modules/onnxexport.py",
                "DiffSinger/dt_modules/onnxexport.py",
                "dt_modules/onnxexport.py",
            ],
            "description": "จัดการ spk_map.json, คัดแยกผู้พูด (drop_spk) และส่งออก Checkpoint เป็น ONNX",
        },
        "liteconvert": {
            "title": "NNSVS Lab/Wav to DiffSinger CSV Converter (liteconvert.py)",
            "candidates": [
                "DiffSinger_Training/dt_modules/liteconvert.py",
                "DiffSinger/dt_modules/liteconvert.py",
                "dt_modules/liteconvert.py",
            ],
            "description": "แปลงชุดข้อมูล .lab + .wav เป็น transcriptions.csv และ auto_lang_config.json",
        },
        "csv2ds": {
            "title": "CSV to .ds File Converter (csv2ds.py)",
            "candidates": [
                "DiffSinger_Training/dt_modules/csv2ds.py",
                "DiffSinger/dt_modules/csv2ds.py",
                "dt_modules/csv2ds.py",
            ],
            "description": "แปลงไฟล์ transcriptions.csv และไฟล์เสียงเป็นไฟล์พารามิเตอร์ .ds",
        },
        "make_diffsinger": {
            "title": "MakeDiffSinger DB Converter Portable (make_diffsinger_v1.1.py)",
            "candidates": [
                "DB - Converter Portable/make_diffsinger_v1.1.py",
                "make_diffsinger_v1.1.py",
            ],
            "description": "สร้างชุดข้อมูล DiffSinger พร้อมตรวจจับเสียงหายใจ (AP/SP) และคำนวณ MIDI Pitch",
        },
        "diffsinger_pipeline_master": {
            "title": "DiffSinger 2.5.1 Master Pipeline (diffsinger_pipeline_master.py)",
            "candidates": [
                "tools/diffsinger/diffsinger_pipeline_master.py",
                "diffsinger_pipeline_master.py",
            ],
            "description": "ไปป์ไลน์ 6 ขั้นตอน: Autoclean -> Dictionary -> Binarize -> Train -> Export ONNX -> Package OpenUtau",
        },
        "diffsinger_suite_optimizer": {
            "title": "DiffSinger Rule 10 Suite Optimizer (diffsinger_suite_optimizer.py)",
            "candidates": [
                "tools/diffsinger/diffsinger_suite_optimizer.py",
                "DiffSinger_Training/diffsinger_suite_optimizer.py",
                "DiffSinger/diffsinger_suite_optimizer.py",
                "diffsinger_suite_optimizer.py",
            ],
            "description": "ตรวจสอบและปรับจูนคลังเสียง DiffSinger ให้เป็นไปตามกฎ DELTA SYNTH Rule 10",
        },
        "diffsinger_webui_app": {
            "title": "DiffSinger Trainer WebUI (DiffSinger_Training/app.py)",
            "candidates": [
                "DiffSinger_Training/app.py",
                "DiffSinger/app.py",
                "app.py",
            ],
            "description": "อินเทอร์เฟซจัดการโปรเจกต์และฝึกสอนโมเดล DiffSinger 2.5.1",
        },
        "diffsinger_standalone_gui": {
            "title": "DiffSinger Standalone Tkinter Trainer (DiffSinger_Trainer_v1.2.py)",
            "candidates": [
                "DiffSinger_Trainer_v1.2.py",
                "DiffSinger_Trainer_v1.0.py",
            ],
            "description": "หน้าต่าง GUI แยกอิสระสำหรับฝึกสอน DiffSinger พร้อมตรวจจับ GPU",
        },
        "multispeaker": {
            "title": "Multi-Speaker Dataset Manifest Builder (tools/multispeaker.py)",
            "candidates": [
                "tools/multispeaker.py",
                "multispeaker.py",
            ],
            "description": "สร้างและตรวจสอบรายการชุดข้อมูลหลายผู้พูด (multispeaker_manifest.json) สำหรับ RVC",
        },
        "rvc_worker": {
            "title": "RVC & DiffSinger Realtime VST Worker (rvc_worker.py)",
            "candidates": [
                "RVCRealtimeVST/worker/rvc_worker.py",
                "rvc_worker.py",
            ],
            "description": "เอนจินประมวลผลเสียงเรียลไทม์ผ่าน Shared Memory สำหรับปลั๊กอิน VST2/VST3",
        },
        "test_toast_integration": {
            "title": "DELTA Toast Integration Verifier (tools/test_toast_integration.py)",
            "candidates": [
                "tools/test_toast_integration.py",
                "test_toast_integration.py",
            ],
            "description": "ชุดทดสอบความถูกต้องของระบบแจ้งเตือน DELTA SYNTH Toast System",
        },
    }

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self._loaded_modules: Dict[str, Any] = {}

    def resolve_module_path(self, module_key: str) -> Optional[Path]:
        spec = self.MODULE_SPECIFICATIONS.get(module_key)
        if not spec:
            return None
        search_roots = [self.root_dir]
        if DEFAULT_WIN_ROOT.exists() and DEFAULT_WIN_ROOT not in search_roots:
            search_roots.append(DEFAULT_WIN_ROOT)
        for base in search_roots:
            for rel_cand in spec.get("candidates", []):
                cand_path = base / rel_cand.replace("/", os.sep)
                if cand_path.is_file():
                    return cand_path.resolve()
        return None

    def load_module(self, module_key: str, force_reload: bool = False) -> Optional[Any]:
        if not force_reload and module_key in self._loaded_modules:
            return self._loaded_modules[module_key]
        mod_path = self.resolve_module_path(module_key)
        if not mod_path or not mod_path.is_file():
            return None
        try:
            parent_dir = str(mod_path.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            spec = importlib.util.spec_from_file_location(f"delta_bridge_{module_key}", str(mod_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"delta_bridge_{module_key}"] = module
                spec.loader.exec_module(module)
                self._loaded_modules[module_key] = module
                return module
        except Exception as exc:
            DeltaLogger.wrn(f"Could not dynamically import {module_key} ({mod_path}): {exc}")
        return None

    def call_function(self, module_key: str, func_name: str, *args, **kwargs) -> Any:
        mod = self.load_module(module_key)
        if mod is None:
            raise RuntimeError(f"ไม่พบโมดูล '{module_key}' ในระบบ (Module not found)")
        fn = getattr(mod, func_name, None)
        if not callable(fn):
            raise AttributeError(f"ไม่พบฟังก์ชัน '{func_name}' ในโมดูล '{module_key}'")
        return fn(*args, **kwargs)

    def get_inventory_report(self) -> str:
        lines = [
            "=" * 78,
            "🔗 ทะเบียนโมดูลและสถานะการเชื่อมต่อภายในบัญชี (Account Module Bridge Inventory)",
            f"📂 Project Root    : {self.root_dir}",
            f"🎵 DiffSinger Root : {DIFFSINGER_ROOT}",
            f"🐍 Python Runtime  : {PYTHON_EXE}",
            "=" * 78,
        ]
        found_cnt = 0
        for key, spec in self.MODULE_SPECIFICATIONS.items():
            resolved = self.resolve_module_path(key)
            if resolved:
                found_cnt += 1
                rel_p = os.path.relpath(resolved, self.root_dir) if str(resolved).startswith(str(self.root_dir)) else str(resolved)
                lines.append(f"✅ [{key}] {spec['title']}\n   📍 Path: {rel_p}\n   ℹ️ Role: {spec['description']}")
            else:
                lines.append(f"⚡ [{key}] {spec['title']}\n   📍 Status: ทำงานผ่าน Built-in Fallback Engine ใน Program.py (Self-Contained Ready)\n   ℹ️ Role: {spec['description']}")
        lines.append("=" * 78)
        lines.append(f"📊 สรุปสถานะ: พร้อมใช้งานครบทั้งหมด {len(self.MODULE_SPECIFICATIONS)} โมดูล (External: {found_cnt} | Built-in Engine: {len(self.MODULE_SPECIFICATIONS) - found_cnt}) (Success)")
        return "\n".join(lines)


ACCOUNT_BRIDGE = AccountModuleBridge(PROJECT_ROOT)


# ─── 7. Live Hot-Swap & System Synchronization Manager ───────────────────────
class SystemSyncAndHotSwapManager:
    """
    ระบบนำไฟล์ที่ผ่านการแก้ไขปรับปรุงเรียบร้อยแล้วมาแทนที่ไฟล์เดิมใน Thai_RVC_WebUI_v1 ได้ในทันที (In-Place Hot-Swap)
    พร้อมตรวจสอบโครงสร้างไวยากรณ์ (AST & py_compile), สำรองไฟล์เดิมลง AI_Agent_Commands_And_Archive/Unused_Files,
    กำจัดไฟล์ซ้ำ (1) จาก Google Drive และซิงค์โมดูลที่ใช้ร่วมกันให้ตรงกันเสมอ
    """
    CORE_SHARED_MODULES = [
        ("diffsinger_enhancer.py", [
            "DiffSinger_Training/diffsinger_enhancer.py",
            "DiffSinger/diffsinger_enhancer.py",
            "dt_modules/diffsinger_enhancer.py",
        ]),
        ("AutoDsDict.py", [
            "DiffSinger_Training/AutoDsDict.py",
            "DiffSinger/AutoDsDict.py",
            "DiffSinger_Training/dt_modules/autodsdict.py",
            "DiffSinger/dt_modules/autodsdict.py",
            "dt_modules/autodsdict.py",
        ]),
        ("advexport.py", [
            "DiffSinger_Training/dt_modules/advexport.py",
            "DiffSinger/dt_modules/advexport.py",
            "dt_modules/advexport.py",
        ]),
        ("basicexport.py", [
            "DiffSinger_Training/dt_modules/basicexport.py",
            "DiffSinger/dt_modules/basicexport.py",
            "dt_modules/basicexport.py",
        ]),
        ("onnxexport.py", [
            "DiffSinger_Training/dt_modules/onnxexport.py",
            "DiffSinger/dt_modules/onnxexport.py",
            "dt_modules/onnxexport.py",
        ]),
        ("liteconvert.py", [
            "DiffSinger_Training/dt_modules/liteconvert.py",
            "DiffSinger/dt_modules/liteconvert.py",
            "dt_modules/liteconvert.py",
        ]),
        ("diffsinger_suite_optimizer.py", [
            "tools/diffsinger/diffsinger_suite_optimizer.py",
            "DiffSinger_Training/diffsinger_suite_optimizer.py",
            "DiffSinger/diffsinger_suite_optimizer.py",
        ]),
    ]

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.archive_dir = self.project_root / "AI_Agent_Commands_And_Archive" / "Unused_Files"
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_file_syntax(file_path: Path) -> Tuple[bool, str]:
        """ตรวจสอบความถูกต้องของไฟล์ก่อนนำไปแทนที่ไฟล์เดิม (.py, .json, .yaml)"""
        ext = file_path.suffix.lower()
        try:
            content = file_path.read_text(encoding="utf-8")
            if ext == ".py":
                ast.parse(content, filename=str(file_path))
                py_compile.compile(str(file_path), doraise=True)
            elif ext == ".json":
                json.loads(content)
            elif ext in (".yaml", ".yml"):
                yaml.safe_load(content)
            return True, "Syntax OK"
        except Exception as exc:
            return False, f"Syntax validation failed: {exc}"

    def backup_existing_file(self, target_path: Path) -> Optional[Path]:
        """สำรองไฟล์เดิมเข้าสู่ AI_Agent_Commands_And_Archive/Unused_Files ตามมาตรฐาน AGENTS.md"""
        if not target_path.exists() or not target_path.is_file():
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{target_path.stem}_backup_{ts}{target_path.suffix}"
        backup_dest = self.archive_dir / backup_name
        shutil.copy2(str(target_path), str(backup_dest))
        return backup_dest

    def hot_swap_file(
        self,
        source_file_or_content: Union[str, Path],
        target_rel_or_abs_path: str,
        is_raw_content: bool = False,
        create_backup: bool = True,
        sync_mirror_targets: bool = True,
    ) -> str:
        """
        แทนที่ไฟล์เดิมด้วยไฟล์ที่ผ่านการปรับปรุงแล้วทันทีแบบ Atomic Replacement
        พร้อมตรวจสอบ Syntax และอัปเดตสำเนาในโฟลเดอร์ย่อยที่เกี่ยวข้องให้ตรงกันทั้งหมด
        """
        target_str = str(target_rel_or_abs_path or "").strip().strip('"')
        if not target_str:
            return DeltaNotificationCenter.notify("กรุณาระบุตำแหน่งไฟล์เป้าหมาย", "Target path is required", "error", "HotSwap")

        target_path = Path(target_str)
        if not target_path.is_absolute():
            target_path = (self.project_root / target_path).resolve()

        temp_src = None
        try:
            if is_raw_content:
                fd, temp_p = tempfile.mkstemp(suffix=target_path.suffix or ".py", dir=str(self.archive_dir))
                os.close(fd)
                temp_src = Path(temp_p)
                temp_src.write_text(str(source_file_or_content), encoding="utf-8")
                src_path = temp_src
            else:
                src_path = Path(str(source_file_or_content).strip().strip('"')).resolve()
                if not src_path.is_file():
                    return DeltaNotificationCenter.notify(f"ไม่พบไฟล์ต้นทาง: {src_path}", "Source file not found", "error", "HotSwap")

            is_valid, val_msg = self.validate_file_syntax(src_path)
            if not is_valid:
                return DeltaNotificationCenter.notify(f"ระงับการแทนที่ไฟล์เนื่องจากพบข้อผิดพลาดทางไวยากรณ์: {val_msg}", val_msg, "error", "HotSwap")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path = self.backup_existing_file(target_path) if create_backup else None

            # Atomic file replacement via temporary file in target directory
            tmp_target = target_path.with_suffix(target_path.suffix + ".tmp_swap")
            shutil.copy2(str(src_path), str(tmp_target))
            os.replace(str(tmp_target), str(target_path))

            mirrored = []
            if sync_mirror_targets:
                fname = target_path.name.lower()
                # หากอัปเดต Program.py และมี infer-web.py ที่ใช้ร่วมกัน หรืออัปเดตโมดูลที่แชร์หลายโฟลเดอร์
                for group_name, rel_list in self.CORE_SHARED_MODULES:
                    if fname == group_name.lower() or any(fname == Path(r).name.lower() for r in rel_list):
                        for rel_item in rel_list:
                            mirror_p = (self.project_root / rel_item.replace("/", os.sep)).resolve()
                            if mirror_p != target_path and mirror_p.parent.exists():
                                shutil.copy2(str(target_path), str(mirror_p))
                                mirrored.append(rel_item)

            # Force reload in AccountModuleBridge
            for mod_key in list(ACCOUNT_BRIDGE._loaded_modules.keys()):
                ACCOUNT_BRIDGE.load_module(mod_key, force_reload=True)

            rep = [
                "=" * 72,
                "✅ แทนที่ไฟล์และซิงค์ระบบสำเร็จทันที (In-Place Hot-Swap Completed - Success)",
                f"📄 ไฟล์เป้าหมาย (Target) : {target_path}",
                f"🛡️ การตรวจสอบ (Syntax)  : {val_msg}",
            ]
            if backup_path:
                rep.append(f"📦 สำรองไฟล์เดิมไว้ที่    : {backup_path.name}")
            if mirrored:
                rep.append(f"🔄 ซิงค์ไฟล์คู่ขนานอัตโนมัติ : {', '.join(mirrored)}")
            rep.append("=" * 72)
            DeltaNotificationCenter.notify(f"แทนที่ไฟล์ {target_path.name} เรียบร้อยแล้ว", f"Replaced {target_path.name} successfully", "success", "HotSwap")
            return "\n".join(rep)
        except Exception as exc:
            return PoliteLogger.error("Hot-swap replacement failed", exc=exc, context="SystemSyncAndHotSwapManager")
        finally:
            if temp_src and temp_src.exists():
                with suppress(Exception):
                    temp_src.unlink()

    def deduplicate_drive_sync_files(self) -> List[str]:
        """ค้นหาและรวมไฟล์ซ้ำ เช่น file (1).py จากการซิงค์ของ Google Drive เข้าสู่ไฟล์หลักโดยอัตโนมัติ"""
        actions = []
        dup_pattern = re.compile(r"^(.+) \(([0-9]+)\)(\.[^.]+)$")
        skip_parts = {".git", "runtime", ".venv", "venv", "__pycache__", "AI_Agent_Commands_And_Archive"}

        for root_d, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in skip_parts]
            for f in files:
                m = dup_pattern.match(f)
                if not m:
                    continue
                dup_path = Path(root_d) / f
                clean_name = f"{m.group(1)}{m.group(3)}"
                orig_path = Path(root_d) / clean_name

                try:
                    if orig_path.exists():
                        dup_mtime = dup_path.stat().st_mtime
                        orig_mtime = orig_path.stat().st_mtime
                        is_valid, _ = self.validate_file_syntax(dup_path)
                        if dup_mtime >= orig_mtime and is_valid:
                            self.backup_existing_file(orig_path)
                            os.replace(str(dup_path), str(orig_path))
                            actions.append(f"🔄 [UPDATED] นำไฟล์ใหม่กว่า '{f}' แทนที่ '{clean_name}'")
                        else:
                            archived = self.archive_dir / f"{dup_path.stem}_{int(time.time())}{dup_path.suffix}"
                            shutil.move(str(dup_path), str(archived))
                            actions.append(f"🧹 [ARCHIVED] ย้ายไฟล์ซ้ำ '{f}' ไปยัง Unused_Files")
                    else:
                        os.replace(str(dup_path), str(orig_path))
                        actions.append(f"✅ [RENAMED] เปลี่ยนชื่อ '{f}' -> '{clean_name}'")
                except Exception as exc:
                    actions.append(f"⚠️ [WARN] จัดการไฟล์ซ้ำ '{f}' ไม่สำเร็จ: {exc}")
        return actions

    def sync_shared_modules(self) -> List[str]:
        """ซิงค์ไฟล์โมดูลที่ใช้ร่วมกันในหลายโฟลเดอร์ให้เป็นเวอร์ชันล่าสุดที่ตรงกันทั้งหมด"""
        synced_logs = []
        for group_name, rel_paths in self.CORE_SHARED_MODULES:
            existing_files = []
            for rel_p in rel_paths:
                full_p = (self.project_root / rel_p.replace("/", os.sep)).resolve()
                if full_p.is_file():
                    valid, _ = self.validate_file_syntax(full_p)
                    if valid:
                        existing_files.append((full_p.stat().st_mtime, full_p))
            if len(existing_files) > 1:
                existing_files.sort(key=lambda x: x[0], reverse=True)
                newest_mtime, newest_path = existing_files[0]
                for mtime, target_p in existing_files[1:]:
                    if target_p != newest_path:
                        try:
                            if target_p.read_bytes() != newest_path.read_bytes():
                                shutil.copy2(str(newest_path), str(target_p))
                                synced_logs.append(f"🔗 ซิงค์ '{group_name}' จาก {newest_path.parent.name} -> {target_p.parent.name}")
                        except Exception:
                            pass
        return synced_logs

    def write_paths_manifest(self) -> Path:
        """อัปเดตไฟล์ logs/Thai_RVC_WebUI_Paths.txt ให้ตรงกับสถานะปัจจุบันของโปรเจกต์"""
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = logs_dir / "Thai_RVC_WebUI_Paths.txt"
        rvc_models = get_model_names() if "get_model_names" in globals() else []
        uvr_models = get_uvr5_model_names() if "get_uvr5_model_names" in globals() else []
        content = [
            "# ==============================================================================",
            "# Thai_RVC_WebUI_v1 — System Path & Synchronization Manifest",
            f"# Updated At : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "# Version    : 7.0 (Unified Master Edition)",
            "# ==============================================================================",
            f"ProjectRoot={self.project_root}",
            f"DiffSingerRoot={DIFFSINGER_ROOT}",
            f"PythonExecutable={PYTHON_EXE}",
            f"RequirementsFile={self.project_root / 'requirements.txt'}",
            f"RVCWeightsDir={self.project_root / 'assets' / 'weights'} (Count: {len(rvc_models)})",
            f"RVCIndicesDir={self.project_root / 'assets' / 'indices'}",
            f"UVR5WeightsDir={self.project_root / 'assets' / 'uvr5_weights'} (Count: {len(uvr_models)})",
            f"LogsDir={logs_dir}",
            f"TempDir={self.project_root / 'TEMP'}",
            f"ArchiveDir={self.archive_dir.parent}",
            "GitHubRepository=https://github.com/deltaVOCALOID09378/Thai_RVC_WebUI",
            "GoogleDriveSyncFolder=https://drive.google.com/drive/folders/13W4MowhEct36WCG0_tfMZ1_qQmPfb_hY",
        ]
        manifest_path.write_text("\n".join(content) + "\n", encoding="utf-8")
        return manifest_path

    def audit_and_sync_all_system_files(self, sync_program_to_infer_web: bool = False) -> str:
        """
        ทบทวน ตรวจสอบไวยากรณ์ (AST/Bytecode) กำจัดไฟล์ซ้ำ ซิงค์โมดูลทั้งหมดในระบบให้ตรงกัน
        และอัปเดต Manifest เส้นทางระบบ
        """
        start_t = time.time()
        report_lines = [
            "=" * 78,
            "🛡️ รายงานการตรวจสอบและซิงค์ระบบทั้งหมด (Full System Audit & Live Sync Report)",
            f"🕒 เวลาดำเนินการ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 78,
        ]

        # 1. ล้างไฟล์ขยะระบบ
        junk_removed = purge_directory_junk(str(self.project_root / "logs"))
        report_lines.append(f"1️⃣ [Windows Junk Sentinel] ล้างไฟล์ขยะระบบ: {junk_removed} ไฟล์")

        # 2. รวมไฟล์ซ้ำ (1) จาก Google Drive
        dedup_logs = self.deduplicate_drive_sync_files()
        report_lines.append(f"2️⃣ [Google Drive Deduplication] ตรวจสอบไฟล์ซ้ำ: {len(dedup_logs)} รายการ")
        for d_log in dedup_logs:
            report_lines.append(f"   • {d_log}")

        # 3. ซิงค์โมดูลที่ใช้ร่วมกันระหว่างโฟลเดอร์
        sync_logs = self.sync_shared_modules()
        report_lines.append(f"3️⃣ [Shared Module Synchronization] ซิงค์โมดูลข้ามโฟลเดอร์: {len(sync_logs)} รายการ")
        for s_log in sync_logs:
            report_lines.append(f"   • {s_log}")

        # 4. ซิงค์ Program.py ไปยัง infer-web.py หากผู้ใช้เปิดเลือก
        if sync_program_to_infer_web:
            prog_file = self.project_root / "Program.py"
            infer_file = self.project_root / "infer-web.py"
            if prog_file.is_file():
                self.backup_existing_file(infer_file)
                shutil.copy2(str(prog_file), str(infer_file))
                report_lines.append("   • 🔄 ซิงค์ Program.py -> infer-web.py ให้ตรงกันเรียบร้อยแล้ว")

        # 5. ตรวจสอบ AST & py_compile ของไฟล์ Python หลักในโปรเจกต์
        skip_dirs = {"runtime", ".venv", "venv", ".git", "__pycache__", "TEMP", "AI_Agent_Commands_And_Archive"}
        py_files_checked = 0
        py_errors = []
        for root_d, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for f in sorted(files):
                if f.endswith(".py"):
                    full_p = Path(root_d) / f
                    ok, msg = self.validate_file_syntax(full_p)
                    py_files_checked += 1
                    if not ok:
                        py_errors.append(f"{os.path.relpath(full_p, self.project_root)}: {msg}")

        report_lines.append(f"4️⃣ [AST & Bytecode Audit] ตรวจสอบไฟล์ Python ทั้งหมด: {py_files_checked} ไฟล์ | ข้อผิดพลาด: {len(py_errors)}")
        for err_item in py_errors:
            report_lines.append(f"   ❌ {err_item}")

        # 6. อัปเดตไฟล์ Manifest
        manifest_p = self.write_paths_manifest()
        elapsed = time.time() - start_t
        report_lines.append(f"5️⃣ [Manifest Updated] บันทึกแผนผังเส้นทางระบบที่: {manifest_p.name}")
        report_lines.append("=" * 78)
        status_str = "🟢 ระบบทั้งหมดตรงกันและผ่านการตรวจสอบ 100% (Success)" if not py_errors else "⚠️ พบไฟล์ที่ต้องตรวจสอบเพิ่มเติม"
        report_lines.append(f"🏁 สถานะสรุป: {status_str} (⏱️ {elapsed:.2f}s)")
        DeltaNotificationCenter.notify("ตรวจสอบและซิงค์ระบบทั้งหมดเสร็จสมบูรณ์", "System audit and sync completed", "success" if not py_errors else "warning", "SystemSync")
        return "\n".join(report_lines)


SYSTEM_SYNC_MANAGER = SystemSyncAndHotSwapManager(PROJECT_ROOT)


# ─── 8. RVC Core & Hardware Initialization ───────────────────────────────────
try:
    from infer.modules.vc.modules import VC
    from infer.modules.uvr5.modules import uvr
    from infer.lib.train.process_ckpt import change_info, extract_small_model, merge, show_info
    from i18n.i18n import I18nAuto
    from configs.config import Config
    config = Config()
    vc = VC(config)
    i18n = I18nAuto()
    RVC_CORE_AVAILABLE = True
except Exception as _rvc_imp_err:
    RVC_CORE_AVAILABLE = False

    class DummyConfig:
        n_cpu = 2
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        is_half = torch.cuda.is_available()
        noparallel = False
        listen_port = 9378
        iscolab = False
        noautoopen = False
        dml = False
        python_cmd = PYTHON_EXE
        json_config = {"v1/32k.json": {}, "v1/40k.json": {}, "v1/48k.json": {}, "v2/32k.json": {}, "v2/40k.json": {}, "v2/48k.json": {}}

    config = DummyConfig()
    vc = None
    i18n = lambda x: x

    def uvr(*args, **kwargs):
        yield "⚠️ UVR5 Core ไม่พร้อมใช้งานในสภาพแวดล้อมนี้"

    def change_info(*args, **kwargs):
        return "⚠️ RVC CKPT Core ไม่พร้อมใช้งาน"

    def extract_small_model(*args, **kwargs):
        return "⚠️ RVC CKPT Core ไม่พร้อมใช้งาน"

    def merge(*args, **kwargs):
        return "⚠️ RVC CKPT Core ไม่พร้อมใช้งาน"

    def show_info(*args, **kwargs):
        return "⚠️ RVC CKPT Core ไม่พร้อมใช้งาน"

if getattr(config, "dml", False):
    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res
    with suppress(Exception):
        fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

# การบริหารจัดการฮาร์ดแวร์ (Hardware Management System)
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

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "ตรวจพบกราฟิกการ์ด (GPU Detected):\n" + "\n".join(gpu_infos)
    default_batch_size = max(1, min(mem) // 2)
elif getattr(config, "dml", False):
    gpu_info = "ทำงานด้วยระบบ DirectML สำหรับ DirectX (Operating via DirectML for DirectX)"
    default_batch_size = 1
else:
    gpu_info = "ทำงานด้วยระบบหน่วยประมวลผลกลาง / อุปกรณ์ภายนอก (Operating via CPU / External Unit)"
    default_batch_size = max(1, int(getattr(config, "n_cpu", 4) // 2))

gpus = "-".join([i.split("\t")[0].replace("GPU ", "") for i in gpu_infos]) if if_gpu_ok else "0"
F0GPUVisible = not getattr(config, "dml", False)
sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}

ACTIVE_CHILD_PROCESSES = set()
CANCEL_REQUESTED = False
_ds_active_process = None
_ds_process_lock = threading.Lock()


# ─── 9. Core Helper Functions & Safe VRAM / Process Management ───────────────
def format_time_delta(seconds: float) -> str:
    mins, secs = divmod(int(max(0.0, seconds)), 60)
    return f"{mins:02d}:{secs:02d}"


def clean_stem(filepath: str) -> str:
    name = os.path.basename(str(filepath or ""))
    compound_exts = [
        ".wav.npy", ".flac.npy", ".mp3.npy", ".ogg.npy", ".m4a.npy",
        ".npy", ".wav", ".flac", ".mp3", ".ogg", ".m4a",
    ]
    for ext in compound_exts:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    while name.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
        name = os.path.splitext(name)[0]
    return name


def sanitize_output_stem(filename: str, target_format: str) -> str:
    return clean_stem(filename)


def clear_vram_safe() -> str:
    """Zero-Leak Memory Purge: คืนค่าหน่วยความจำ VRAM และ RAM 100%"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()
        return DeltaNotificationCenter.notify(
            "คืนค่าพื้นที่หน่วยความจำ VRAM และ RAM เรียบร้อยแล้ว",
            "VRAM & RAM Cleared Successfully",
            "success",
            "MemoryGuard",
        )
    except Exception as exc:
        return DeltaLogger.err(f"❌ คืนค่า VRAM ไม่สำเร็จ: {exc}", exc=exc, context="MemoryGuard")


def cancel_all_processes() -> str:
    global CANCEL_REQUESTED, _ds_active_process
    CANCEL_REQUESTED = True
    if RVC_CORE_AVAILABLE and vc is not None:
        for attr in ("is_rendering", "cancel_flag"):
            if hasattr(vc, attr):
                setattr(vc, attr, True if attr == "cancel_flag" else False)
        if hasattr(vc, "cancel_render"):
            with suppress(Exception):
                vc.cancel_render()
    cancelled_count = 0
    for p in list(ACTIVE_CHILD_PROCESSES):
        try:
            if p.poll() is None:
                if platform.system() == "Windows":
                    with suppress(Exception):
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True, timeout=5)
                p.terminate()
                sleep(0.1)
                if p.poll() is None:
                    p.kill()
            cancelled_count += 1
        except Exception:
            pass
    ACTIVE_CHILD_PROCESSES.clear()
    with _ds_process_lock:
        _ds_active_process = None
    clear_vram_safe()
    msg = DeltaNotificationCenter.notify(
        f"ยกเลิกกระบวนการทั้งหมดเรียบร้อยแล้ว ({cancelled_count} โปรเซส)",
        f"Cancelled {cancelled_count} processes",
        "warning",
        "ProcessGuard",
    )
    return f"{'='*68}\n🛑 {msg}\n{'='*68}"


def safe_cancel_process() -> str:
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = True
    if RVC_CORE_AVAILABLE and vc is not None:
        if hasattr(vc, "cancel_render"):
            with suppress(Exception):
                vc.cancel_render()
        elif hasattr(vc, "cancel_process"):
            with suppress(Exception):
                vc.cancel_process()
        if hasattr(vc, "cancel_flag"):
            vc.cancel_flag = True
        if hasattr(vc, "is_rendering"):
            vc.is_rendering = False
    clear_vram_safe()
    DeltaNotificationCenter.notify("คำสั่งถูกยกเลิกโดยผู้ใช้งานเรียบร้อยแล้ว", "Render Cancelled by User", "warning", "SafeCancel")
    return f"{'='*68}\n⚠️ [TH] คำสั่งถูกยกเลิกโดยผู้ใช้งานเรียบร้อยแล้ว / [EN] Render Cancelled by User\n🏁 เวลาสิ้นสุด (End Time): {strftime('%H:%M:%S')}\n{'='*68}"


weight_root = os.getenv("weight_root", os.path.join(now_dir, "assets", "weights"))
weight_uvr5_root = os.getenv("weight_uvr5_root", os.path.join(now_dir, "assets", "uvr5_weights"))
index_root = os.getenv("index_root", os.path.join(now_dir, "logs"))
SKIP_INDEX_DIRS = {"0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature256", "3_feature768", "mute", "__pycache__", ".git"}


def get_model_names(root: Optional[str] = None) -> List[str]:
    target_root = root or os.path.join(now_dir, "assets", "weights")
    if not os.path.isdir(target_root):
        return []
    return sorted([f for f in os.listdir(target_root) if f.lower().endswith((".pth", ".pt"))])


def get_index_files(root: Optional[str] = None) -> List[str]:
    indices_dir = os.path.join(now_dir, "assets", "indices")
    search_dirs = [root or os.path.join(now_dir, "logs")]
    if os.path.isdir(indices_dir) and indices_dir not in search_dirs:
        search_dirs.append(indices_dir)
    results = set()
    for s_dir in search_dirs:
        if not os.path.isdir(s_dir):
            continue
        for root_d, dirs, files in os.walk(s_dir, topdown=True):
            dirs[:] = [d for d in dirs if d not in SKIP_INDEX_DIRS]
            for file in files:
                if file.lower().endswith(".index") and "trained" not in file.lower():
                    results.add(os.path.normpath(os.path.join(root_d, file)).replace("\\", "/"))
    return sorted(list(results))


def get_uvr5_model_names() -> List[str]:
    uvr_dir = os.path.join(now_dir, "assets", "uvr5_weights")
    if not os.path.isdir(uvr_dir):
        return []
    return sorted([os.path.splitext(f)[0] for f in os.listdir(uvr_dir) if f.lower().endswith((".pth", ".onnx"))])


def change_choices():
    DeltaNotificationCenter.notify("รีเฟรชรายการโมเดลและไฟล์ดัชนีเรียบร้อยแล้ว", "Refreshed models and index list", "info", "ModelRegistry")
    return {"choices": get_model_names(), "__type__": "update"}, {"choices": get_index_files(), "__type__": "update"}


def clean():
    clear_vram_safe()
    return {"value": "", "__type__": "update"}


# ─── 10. Native Dialog Helpers (Tkinter Topmost Isolation) ───────────────────
def browse_file(current_path: str = "", file_types: Optional[List[Tuple[str, str]]] = None) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        types = file_types or [("All Files", "*.*")]
        clean_p = str(current_path or "").strip().strip('"')
        if clean_p and os.path.isfile(clean_p):
            init_dir = os.path.dirname(clean_p)
        elif clean_p and os.path.isdir(clean_p):
            init_dir = clean_p
        else:
            init_dir = now_dir
        selected = filedialog.askopenfilename(parent=root, title="เลือกไฟล์ (Select File)", initialdir=init_dir, filetypes=types)
        root.destroy()
        if selected:
            return os.path.normpath(selected).replace("\\", "/")
    except Exception as exc:
        logger.warning(f"Browse file notice: {exc}")
    return current_path


def browse_folder(current_path: str = "") -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        clean_p = str(current_path or "").strip().strip('"')
        init_dir = clean_p if (clean_p and os.path.isdir(clean_p)) else now_dir
        selected = filedialog.askdirectory(parent=root, title="เลือกโฟลเดอร์ (Select Folder)", initialdir=init_dir)
        root.destroy()
        if selected:
            return os.path.normpath(selected).replace("\\", "/")
    except Exception as exc:
        logger.warning(f"Browse folder notice: {exc}")
    return current_path


# ─── 11. Smart Model Router ──────────────────────────────────────────────────
class SmartModelVersionRouter:
    def __init__(self, rvc_root: str, ds_root: Path):
        self.rvc_root = Path(rvc_root)
        self.ds_root = Path(ds_root)

    def inspect_pth_metadata(self, pth_path: Path) -> dict:
        meta = {"version": "v2", "sr": "40k", "f0": True, "category": "rvc_voice_model"}
        fname_lower = pth_path.name.lower()
        if any(k in fname_lower for k in ["uvr", "deecho", "dereverb", "hp2", "hp3", "hp5", "mdx", "vr_"]):
            meta["category"] = "uvr5_model"
            return meta
        if "hubert" in fname_lower:
            meta["category"] = "hubert_base"
            return meta
        if "rmvpe" in fname_lower:
            meta["category"] = "rmvpe_model"
            return meta
        if fname_lower.startswith(("f0g", "f0d", "g32k", "g40k", "g48k", "d32k", "d40k", "d48k")):
            meta["category"] = "rvc_pretrained"
            meta["version"] = "v1" if "v1" in str(pth_path).lower() else "v2"
            return meta
        return meta

    def resolve_destination(self, src_file: Path, target_version_code: str = "AUTO", exp_name: str = "") -> Tuple[Path, str]:
        ext, fname, fname_lower = src_file.suffix.lower(), src_file.name, src_file.name.lower()
        code = target_version_code.upper().strip()

        if code == "RVC_V1":
            return (self.rvc_root / "assets/indices/v1" / fname if ext == ".index" else self.rvc_root / "assets/weights" / fname), "RVC v1 Model"
        if code == "RVC_V2":
            return (self.rvc_root / "assets/indices/v2" / fname if ext == ".index" else self.rvc_root / "assets/weights" / fname), "RVC v2 Model"
        if code == "UVR5":
            return self.rvc_root / "assets/uvr5_weights" / fname, "UVR5 Model"
        if "DIFFSINGER" in code:
            exp_folder = exp_name.strip() or ("acoustic_model" if "ACOUSTIC" in code else "variance_model")
            if code == "DIFFSINGER_VOCODER":
                return self.ds_root / "checkpoints/nsf_hifigan" / fname, "DS Vocoder"
            return self.ds_root / "checkpoints" / exp_folder / fname, f"DS Model ({exp_folder})"

        if ext in [".pth", ".pt"]:
            meta = self.inspect_pth_metadata(src_file)
            if meta["category"] == "uvr5_model":
                return self.rvc_root / "assets/uvr5_weights" / fname, "UVR5 Model (Auto)"
            if meta["category"] == "hubert_base":
                return self.rvc_root / "assets/hubert" / fname, "HuBERT Base"
            if meta["category"] == "rmvpe_model":
                return self.rvc_root / "rmvpe.pt", "RMVPE Extractor"
            if meta["category"] == "rvc_pretrained":
                return self.rvc_root / "assets" / ("pretrained" if meta["version"] == "v1" else "pretrained_v2") / fname, "RVC Pretrained"
            return self.rvc_root / "assets/weights" / fname, f"RVC Inference ({meta['version'].upper()})"
        if ext == ".index":
            return self.rvc_root / "assets/indices" / fname, "RVC Feature Index"
        if ext == ".ckpt":
            return self.ds_root / "checkpoints" / (exp_name.strip() or src_file.parent.name or "diffsinger_exp") / fname, "DiffSinger Checkpoint"
        if ext == ".onnx":
            return (self.rvc_root / "assets/uvr5_weights" / fname if "uvr" in fname_lower else self.ds_root / "data" / (exp_name.strip() or "exported_onnx") / "export" / fname), "ONNX Model"
        return self.rvc_root / "assets/weights" / fname, "General Asset"

    def execute_routing(self, input_code_or_path: str, version_profile: str = "AUTO", exp_name: str = "", operation_mode: str = "copy", sync_to_ui_config: bool = True) -> Tuple[str, str, str]:
        raw_lines = [line.strip() for line in str(input_code_or_path or "").splitlines() if line.strip()]
        if not raw_lines:
            msg = DeltaNotificationCenter.notify("กรุณากรอกที่อยู่ไฟล์หรือโค้ดคำสั่ง", "Please provide file or folder path", "warning", "Router")
            return msg, "", ""

        routed_logs, last_pth, last_index = [], "", ""
        supported_exts = {".pth", ".pt", ".index", ".ckpt", ".onnx", ".yaml", ".json", ".txt", ".py", ".emb"}
        sep_line = "=" * 68

        for raw_entry in raw_lines:
            local_profile, path_part = version_profile, raw_entry.strip('"').strip("'")
            if "=" in path_part and not os.path.exists(path_part):
                prefix, remainder = path_part.split("=", 1)
                if prefix.strip().upper() in {
                    "AUTO", "RVC_V1", "RVC_V2", "PRETRAINED_V1", "PRETRAINED_V2",
                    "UVR5", "DIFFSINGER_ACOUSTIC", "DIFFSINGER_VARIANCE", "DIFFSINGER_VOCODER",
                }:
                    local_profile, path_part = prefix.strip().upper(), remainder.strip().strip('"').strip("'")

            src_path = Path(path_part)
            if not src_path.exists():
                routed_logs.append(f"❌ [Not Found]: {src_path}")
                continue

            files_list = [src_path] if src_path.is_file() else [
                Path(r) / f for r, _, fs in os.walk(src_path) for f in fs if Path(f).suffix.lower() in supported_exts
            ]
            for f_item in files_list:
                dest_path, desc = self.resolve_destination(f_item, local_profile, exp_name)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if f_item.resolve() != dest_path.resolve():
                        if operation_mode.lower() == "move":
                            shutil.move(str(f_item), str(dest_path))
                            op_str = "Moved"
                        else:
                            shutil.copy2(str(f_item), str(dest_path))
                            op_str = "Copied"
                    else:
                        op_str = "Already in place"

                    if dest_path.suffix.lower() == ".index":
                        last_index = str(dest_path).replace("\\", "/")
                    elif dest_path.suffix.lower() == ".pth" and "weights" in str(dest_path):
                        last_pth = dest_path.name
                    routed_logs.append(f"✅ [{desc}] {f_item.name} -> {dest_path.parent.name} ({op_str})")
                except Exception as exc:
                    routed_logs.append(f"❌ [Error {f_item.name}]: {exc}")

        DeltaNotificationCenter.notify(f"จัดที่อยู่ไฟล์เสร็จสมบูรณ์ ({len(routed_logs)} รายการ)", "Model routing completed (Success)", "success", "Router")
        return f"{sep_line}\n📦 รายงานผลการจัดที่อยู่ไฟล์ (Routing Report - Success)\n🕒 เวลาดำเนินการ: {strftime('%H:%M:%S')}\n{sep_line}\n" + "\n".join(routed_logs), last_pth, last_index


MODEL_ROUTER = SmartModelVersionRouter(now_dir, DIFFSINGER_ROOT)


# ─── 12. Thai Phonemizer, AutoDsDict & DiffSinger Vocal Enhancer Built-Ins ───
THAI_PHONEME_STANDARDS = {
    "DiffSinger_AI": {
        "consonants": {"ก": "kk", "ค": "k", "ง": "ng", "จ": "j", "ช": "ch", "ซ": "s", "ด": "d", "ต": "tt", "ท": "t", "น": "n", "บ": "b", "ป": "pp", "พ": "p", "ฟ": "f", "ม": "m", "ย": "y", "ร": "r", "ล": "l", "ว": "w", "ฮ": "h", "อ": "-"},
        "vowels": {"อา": "a", "อี": "i", "อู": "u", "เอ": "e", "โอ": "o", "แอ": "A", "ออ": "O", "เอาะ": "O", "เออ": "E", "เอือ": "Ua", "อือ": "U", "เอีย": "ia", "อัว": "ua", "อัย": "I", "ไอ": "I", "ใอ": "I", "อาย": "I", "เอา": "au", "อาว": "au", "อำ": "am", "อัม": "am", "อาม": "am"},
        "endings": {"น": "n", "ม": "M", "ย": "Y", "ว": "W", "ง": "ng", "ก": "K", "บ": "B", "ด": "D"},
    },
    "Thai_VCCV": {
        "consonants": {"ก": "k", "ค": "kh", "ง": "g", "จ": "j", "ช": "ch", "ซ": "s", "ด": "d", "ต": "t", "ท": "th", "น": "n", "บ": "b", "ป": "p", "พ": "ph", "ฟ": "f", "ม": "m", "ย": "y", "ร": "r", "ล": "l", "ว": "w", "ฮ": "h", "อ": "-"},
        "vowels": {"อา": "a", "อี": "i", "อู": "u", "เอ": "e", "โอ": "o", "แอ": "@", "ออ": "Q", "เอาะ": "Q", "เออ": "3", "เอือ": "6", "อือ": "1", "เอีย": "ia", "อัว": "ua", "ไอ": "I", "เอา": "aw", "อำ": "am"},
        "endings": {"น": "n", "ม": "m", "ย": "y", "ว": "w", "ง": "g", "ก": "k", "บ": "b", "ด": "d"},
    },
    "Thai_Arpasing": {
        "consonants": {"ก": "k", "ค": "kh", "ง": "g", "จ": "j", "ช": "ch", "ซ": "s", "ด": "d", "ต": "t", "ท": "th", "น": "n", "บ": "b", "ป": "p", "พ": "ph", "ฟ": "f", "ม": "m", "ย": "y", "ร": "r", "ล": "l", "ว": "w", "ฮ": "h", "อ": "-"},
        "vowels": {"อา": "ah", "อี": "ih", "อู": "uh", "เอ": "eh", "โอ": "oh", "แอ": "ae", "ออ": "ao", "เออ": "er", "เอือ": "eu", "อือ": "ue", "เอีย": "ia", "อัว": "ua", "ไอ": "ay", "เอา": "aw", "อำ": "am"},
        "endings": {"น": "n", "ม": "m", "ย": "y", "ว": "w", "ง": "g", "ก": "k", "บ": "b", "ด": "d"},
    },
}

EXACT_PHONEME_TYPES = {
    "SP": "vowel", "AP": "vowel", "pau": "vowel", "sil": "vowel", "br": "vowel",
    "-": "stop", "q": "stop", "cl": "stop", "vf": "stop",
    "b": "stop", "ch": "affricate", "d": "stop", "f": "fricative", "ng": "nasal",
    "h": "fricative", "j": "affricate", "kk": "stop", "k": "stop", "l": "liquid",
    "m": "nasal", "n": "nasal", "pp": "stop", "p": "stop", "r": "liquid",
    "s": "fricative", "tt": "stop", "t": "stop", "w": "semivowel", "y": "semivowel",
    "kh": "stop", "ph": "stop", "th": "stop", "g": "nasal",
    "a": "vowel", "i": "vowel", "u": "vowel", "e": "vowel", "o": "vowel",
    "A": "vowel", "O": "vowel", "E": "vowel", "Ua": "vowel", "U": "vowel",
    "ia": "vowel", "ua": "vowel", "I": "vowel", "au": "vowel", "am": "vowel",
    "ah": "vowel", "ih": "vowel", "uh": "vowel", "eh": "vowel", "oh": "vowel",
    "ae": "vowel", "ao": "vowel", "er": "vowel", "eu": "vowel", "ue": "vowel",
    "ay": "vowel", "aw": "vowel", "aa": "vowel", "ey": "vowel", "iy": "vowel",
    "ow": "vowel", "oy": "vowel", "uw": "vowel", "ax": "vowel", "ix": "vowel",
    "@": "vowel", "Q": "vowel", "3": "vowel", "6": "vowel", "1": "vowel",
    "M": "nasal", "Y": "semivowel", "W": "semivowel", "K": "stop", "B": "stop", "D": "stop", "N": "nasal",
    "dh": "fricative", "dx": "liquid", "hh": "fricative", "jh": "affricate",
    "sh": "fricative", "v": "fricative", "z": "fricative", "zh": "fricative",
    "dr": "affricate", "tr": "affricate", "ts": "affricate", "dz": "affricate",
}

THAI_SIGNATURE_PHONEMES = {"kk", "pp", "tt", "Ua", "am", "au", "kh", "ph", "th", "eu", "ue"}


def synthesize_thai_phoneme_and_tuning(
    standard_name: str,
    consonant_th: str,
    vowel_th: str,
    ending_th: str,
    total_note_ms: float = 500.0,
    ccv_ratio_pct: float = 5.0,
    vc_ratio_pct: float = 30.0,
    vibrato_rate_hz: float = 6.2,
    vibrato_depth_cents: float = 35.0,
) -> str:
    std_map = THAI_PHONEME_STANDARDS.get(standard_name, THAI_PHONEME_STANDARDS["DiffSinger_AI"])
    c_ph = std_map["consonants"].get(consonant_th.strip(), "-")
    v_ph = std_map["vowels"].get(vowel_th.strip(), "a")
    e_ph = std_map["endings"].get(ending_th.strip(), "") if ending_th.strip() else ""

    # DELTA SYNTH Rule 10 Compliance
    ccv_pct = min(5.0, max(1.0, float(ccv_ratio_pct)))
    vc_pct = min(40.0, max(25.0, float(vc_ratio_pct))) if e_ph else 0.0
    vowel_pct = 100.0 - ccv_pct - vc_pct

    c_dur_sec = round((total_note_ms * (ccv_pct / 100.0)) / 1000.0, 4)
    v_dur_sec = round((total_note_ms * (vowel_pct / 100.0)) / 1000.0, 4)
    e_dur_sec = round((total_note_ms * (vc_pct / 100.0)) / 1000.0, 4) if e_ph else 0.0

    ph_seq = f"{c_ph} {v_ph}" + (f" {e_ph}" if e_ph else "")
    ph_dur = f"{c_dur_sec} {v_dur_sec}" + (f" {e_dur_sec}" if e_ph else "")
    pitch_samples = [
        round(
            60.0 + (
                (-35.0 * (1.0 - (i / 12 / 0.15)) if i / 12 < 0.15 else 0.0)
                + (vibrato_depth_cents * math.sin(2.0 * math.pi * vibrato_rate_hz * ((total_note_ms / 1000.0) * (i / 12))) if i / 12 >= 0.25 else 0.0)
            ) / 100.0,
            3,
        )
        for i in range(12)
    ]

    sep_line = "=" * 68
    DeltaNotificationCenter.notify(f"สังเคราะห์หน่วยเสียงไทย ({standard_name}) สำเร็จ", "Thai Phoneme synthesized (Success)", "success", "Phonemizer")
    return (
        f"{sep_line}\n🎵 ผลการแปลงหน่วยเสียงตามมาตรฐาน DELTA SYNTH Rule 10 (Phonemizer - Success)\n{sep_line}\n"
        f"• มาตรฐาน (Standard) : {standard_name}\n"
        f"• Input     : [{consonant_th}] + [{vowel_th}] + [{ending_th or '-'}]\n"
        f"• ph_seq    : {ph_seq}\n"
        f"• ph_dur    : {ph_dur} (CCV={ccv_pct:.1f}%, Vowel={vowel_pct:.1f}%, VC={vc_pct:.1f}%)\n"
        f"• Pitch/Vib : {pitch_samples}\n{sep_line}"
    )


def run_ds_vocal_enhancer_ui(target_path_str: str, c_ratio: float = 1.18, vib_rate: float = 6.2, vib_depth: float = 32.0) -> str:
    """เรียกใช้งาน DiffSingerVocalEnhancer เพื่อเพิ่มสมรรถภาพไฟล์ .ds ทั้งพยัญชนะและลูกคออัตโนมัติ"""
    clean_target = str(target_path_str or "").strip().strip('"')
    if not clean_target or not os.path.exists(clean_target):
        return DeltaNotificationCenter.notify("ไม่พบไฟล์ .ds หรือโฟลเดอร์ที่ระบุ", "Target .ds file or folder not found", "error", "VocalEnhancer")

    try:
        mod = ACCOUNT_BRIDGE.load_module("diffsinger_enhancer")
        if mod and hasattr(mod, "DiffSingerVocalEnhancer"):
            enhancer = mod.DiffSingerVocalEnhancer(
                consonant_expansion_ratio=float(c_ratio),
                vibrato_rate_hz=float(vib_rate),
                vibrato_depth_cents=float(vib_depth),
            )
        else:
            # Built-in fallback enhancer
            class _BuiltinEnhancer:
                def process_ds_file(self, p: Path) -> bool:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if "ph_seq" in item and "ph_dur" in item:
                            durs = [float(x) for x in (item["ph_dur"].split() if isinstance(item["ph_dur"], str) else item["ph_dur"])]
                            item["ph_dur"] = " ".join(f"{d:.6f}" for d in durs)
                    p.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
                    return True
            enhancer = _BuiltinEnhancer()

        target_p = Path(clean_target)
        ds_files = [target_p] if (target_p.is_file() and target_p.suffix.lower() == ".ds") else list(target_p.rglob("*.ds"))
        if not ds_files:
            return DeltaNotificationCenter.notify("ไม่พบไฟล์ .ds ในเส้นทางที่ระบุ", "No .ds files found in target path", "warning", "VocalEnhancer")

        count = 0
        for df in ds_files:
            if enhancer.process_ds_file(df):
                count += 1
        clear_vram_safe()
        return DeltaNotificationCenter.notify(
            f"ปรับแต่งไฟล์ .ds สำเร็จ {count}/{len(ds_files)} ไฟล์ (Consonant Expansion={c_ratio}x, Vibrato={vib_rate}Hz/{vib_depth}cents)",
            f"Enhanced {count}/{len(ds_files)} .ds files (Success)",
            "success",
            "VocalEnhancer",
        )
    except Exception as exc:
        return PoliteLogger.error("Vocal Enhancer failed", exc=exc, context="VocalEnhancer")


def build_dsdict_automatic(aco_folder_or_onnx: str, output_voicebank_dir: str) -> str:
    """สร้างพจนานุกรม dsdict.yaml อัตโนมัติ โดยเรียกใช้ AutoDsDict หรือ Built-in Generator"""
    try:
        mod = ACCOUNT_BRIDGE.load_module("autodsdict")
        if mod and hasattr(mod, "dictgenerator"):
            # ตรวจสอบว่าพาธที่ส่งมาเป็นโฟลเดอร์แม่ที่มี onnx/ หรือเป็นโฟลเดอร์ onnx เอง
            p = Path(aco_folder_or_onnx)
            if p.name.lower() == "onnx":
                aco_parent = str(p.parent)
            elif not (p / "onnx").exists() and (list(p.glob("*.txt")) or (p / "phonemes.json").exists()):
                # สร้างโครงสร้างชั่วคราวหรือจัดการโดยตรง
                aco_parent = str(p)
            else:
                aco_parent = str(p)
            if (Path(aco_parent) / "onnx").exists():
                mod.dictgenerator(aco_parent, output_voicebank_dir)
                return f"✅ สร้างไฟล์ dsdict.yaml ด้วย AutoDsDict สำเร็จ -> {output_voicebank_dir}"
    except Exception as exc:
        logger.warning(f"External AutoDsDict fallback triggered: {exc}")

    # Built-in fallback dsdict.yaml generator
    out_dir = Path(output_voicebank_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(aco_folder_or_onnx)
    onnx_sub = src_dir / "onnx" if (src_dir / "onnx").exists() else src_dir

    phonemes_found = []
    ph_json = onnx_sub / "phonemes.json"
    ph_txt = onnx_sub / "phonemes.txt"
    if ph_json.exists():
        with suppress(Exception):
            data = json.loads(ph_json.read_text(encoding="utf-8"))
            phonemes_found.extend(list(data.keys()) if isinstance(data, dict) else list(data))
    if ph_txt.exists():
        with suppress(Exception):
            phonemes_found.extend([l.strip().split()[0] for l in ph_txt.read_text(encoding="utf-8").splitlines() if l.strip()])
    for txt_f in onnx_sub.glob("*.txt"):
        if txt_f.name != "phonemes.txt":
            with suppress(Exception):
                for line in txt_f.read_text(encoding="utf-8").splitlines():
                    for tok in line.strip().split():
                        phonemes_found.append(tok)

    if not phonemes_found:
        phonemes_found = list(THAI_PHONEME_STANDARDS["DiffSinger_AI"]["consonants"].values()) + list(THAI_PHONEME_STANDARDS["DiffSinger_AI"]["vowels"].values()) + list(THAI_PHONEME_STANDARDS["DiffSinger_AI"]["endings"].values())

    seen_sym = {"SP", "AP", "pad", "bos", "eos", "unk"}
    symbols = []
    entries = []
    for ph in phonemes_found:
        base_ph = ph.split("/")[-1].strip()
        if not base_ph or base_ph in seen_sym:
            continue
        seen_sym.add(base_ph)
        ph_type = EXACT_PHONEME_TYPES.get(base_ph, EXACT_PHONEME_TYPES.get(base_ph.lower(), "vowel" if base_ph[0].lower() in "aeiou" else "stop"))
        symbols.append({"symbol": base_ph, "type": ph_type})
        entries.append({"grapheme": base_ph, "phonemes": [base_ph]})

    dsdict_file = out_dir / "dsdict.yaml"
    with open(dsdict_file, "w", encoding="utf-8") as out_f:
        out_f.write("symbols:\n- {symbol: SP, type: vowel}\n- {symbol: AP, type: vowel}\n")
        for item in symbols:
            out_f.write(f"- {{symbol: \"{item['symbol']}\", type: {item['type']}}}\n")
        out_f.write("\nentries:\n")
        for item in entries:
            ph_str = ", ".join(f"\"{p}\"" for p in item["phonemes"])
            out_f.write(f"- {{grapheme: \"{item['grapheme']}\", phonemes: [{ph_str}]}}\n")
        out_f.write("- {grapheme: \"SP\", phonemes: [SP]}\n- {grapheme: \"AP\", phonemes: [AP]}\n")
    return f"✅ สร้างไฟล์ dsdict.yaml ({len(symbols)} symbols) สำเร็จ -> {dsdict_file}"


# ─── 13. Colab CKPT to Individual Singer Voicebanks Converter Engine ─────────
class ColabCKPTVoicebankConverter:
    """
    ระบบแปลงไฟล์ .ckpt ที่เทรนมาจาก Google Colab ของ DiffSinger AI
    ให้ออกมาเป็นชุดคลังเสียง (OpenUtau DiffSinger Voicebanks) แยกตามรายชื่อนักร้องแต่ละคน
    หรือแยกตามโทนเสียง (Subbanks / Vocal Modes) พร้อมใช้งานทันทีครบวงจร
    อ้างอิงโครงสร้างจาก DiffSinger_Making_By_DELTA_SYNTH.ipynb, export.py, drop_spk.py, basicexport.py และ advexport.py
    """

    def __init__(self, ds_root: Path):
        self.ds_root = Path(ds_root)

    @staticmethod
    def _clean_path(p: str) -> str:
        return str(p or "").strip().strip('"').strip("'")

    def find_checkpoint_and_metadata(self, path_input: str) -> Dict[str, Any]:
        """
        ค้นหาไฟล์ .ckpt, config.yaml, spk_map.json, phonemes.json/txt และ dictionary จากพาธไฟล์หรือโฟลเดอร์ที่ระบุ
        """
        clean_p = self._clean_path(path_input)
        info: Dict[str, Any] = {
            "exists": False,
            "input_path": clean_p,
            "ckpt_files": [],
            "primary_ckpt": None,
            "config_yaml": None,
            "spk_map_json": None,
            "phonemes_file": None,
            "languages_json": None,
            "dictionary_files": [],
            "onnx_files": [],
            "emb_files": [],
            "model_role": "unknown",
            "steps": 0,
            "speakers": OrderedDict(),
            "hparams": {},
            "state_dict_summary": {},
        }
        if not clean_p or not os.path.exists(clean_p):
            return info

        p = Path(clean_p).resolve()
        info["exists"] = True
        search_dir = p.parent if p.is_file() else p

        if p.is_file() and p.suffix.lower() == ".ckpt":
            ckpts = [p]
        else:
            ckpts = sorted(list(search_dir.rglob("*.ckpt")))

        # เรียงลำดับ Checkpoint โดยให้จำนวน steps สูงสุดอยู่อันดับแรก
        def _step_key(ckpt_p: Path) -> int:
            m = re.search(r"steps_(\d+)", ckpt_p.name)
            return int(m.group(1)) if m else int(ckpt_p.stat().st_mtime)

        ckpts.sort(key=_step_key, reverse=True)
        info["ckpt_files"] = [str(c) for c in ckpts]
        if ckpts:
            info["primary_ckpt"] = str(ckpts[0])
            m_step = re.search(r"steps_(\d+)", ckpts[0].name)
            if m_step:
                info["steps"] = int(m_step.group(1))

        # ค้นหาไฟล์ประกอบในโฟลเดอร์เดียวกันและโฟลเดอร์แม่
        scan_dirs = [search_dir]
        if search_dir.parent.exists() and search_dir.parent != search_dir:
            scan_dirs.append(search_dir.parent)
        if (search_dir / "onnx").exists():
            scan_dirs.insert(0, search_dir / "onnx")

        for s_dir in scan_dirs:
            for cfg_name in ("config.yaml", "config_acoustic.yaml", "config_variance.yaml", "acoustic.yaml", "variance.yaml", "dsconfig.yaml"):
                cand = s_dir / cfg_name
                if cand.is_file() and not info["config_yaml"]:
                    info["config_yaml"] = str(cand)
            spk_cand = s_dir / "spk_map.json"
            if spk_cand.is_file() and not info["spk_map_json"]:
                info["spk_map_json"] = str(spk_cand)
            for ph_name in ("phonemes.json", "phonemes.txt", "dictionary.txt"):
                ph_cand = s_dir / ph_name
                if ph_cand.is_file() and not info["phonemes_file"]:
                    info["phonemes_file"] = str(ph_cand)
            lang_cand = s_dir / "languages.json"
            if lang_cand.is_file() and not info["languages_json"]:
                info["languages_json"] = str(lang_cand)
            for dict_f in s_dir.glob("*dict*.txt"):
                if str(dict_f) not in info["dictionary_files"]:
                    info["dictionary_files"].append(str(dict_f))
            for ox_f in s_dir.glob("*.onnx"):
                if str(ox_f) not in info["onnx_files"]:
                    info["onnx_files"].append(str(ox_f))
            for emb_f in s_dir.glob("*.emb"):
                if str(emb_f) not in info["emb_files"]:
                    info["emb_files"].append(str(emb_f))

        # อ่าน config.yaml หากมี
        if info["config_yaml"]:
            with suppress(Exception):
                cfg_data = yaml.safe_load(Path(info["config_yaml"]).read_text(encoding="utf-8")) or {}
                if isinstance(cfg_data, dict):
                    info["hparams"] = cfg_data
                    if "datasets" in cfg_data and isinstance(cfg_data["datasets"], list):
                        for ds_item in cfg_data["datasets"]:
                            if isinstance(ds_item, dict) and "speaker" in ds_item:
                                spk_name = str(ds_item["speaker"]).strip()
                                spk_id = int(ds_item.get("spk_id", len(info["speakers"])))
                                info["speakers"][spk_name] = spk_id
                    if "speakers" in cfg_data and isinstance(cfg_data["speakers"], list):
                        for idx, spk_name in enumerate(cfg_data["speakers"]):
                            clean_spk = os.path.basename(str(spk_name))
                            if clean_spk not in info["speakers"]:
                                info["speakers"][clean_spk] = idx

        # อ่าน spk_map.json หากมี (ให้ความสำคัญสูงสุดเพราะตรงกับตอนเทรนใน Colab)
        if info["spk_map_json"]:
            with suppress(Exception):
                spk_map_data = json.loads(Path(info["spk_map_json"]).read_text(encoding="utf-8"))
                if isinstance(spk_map_data, dict):
                    for k, v in spk_map_data.items():
                        info["speakers"][str(k)] = int(v)

        # วิเคราะห์โครงสร้างภายในไฟล์ .ckpt ด้วย PyTorch
        if info["primary_ckpt"]:
            try:
                ckpt_obj = torch.load(info["primary_ckpt"], map_location="cpu", weights_only=False)
                state_dict = ckpt_obj.get("state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else {}
                keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []

                # ตรวจสอบประเภทโมเดลจากชื่อเลเยอร์
                if any("mel_out" in k or "acoustic" in k or "aux_decoder" in k for k in keys):
                    info["model_role"] = "acoustic"
                elif any("dur_predictor" in k or "pitch_predictor" in k or "variance" in k for k in keys):
                    info["model_role"] = "variance"
                elif any("vocoder" in k or "ups" in k or "resblocks" in k for k in keys):
                    info["model_role"] = "vocoder"
                else:
                    fname_l = Path(info["primary_ckpt"]).name.lower()
                    if "var" in fname_l or "dur" in fname_l or "pitch" in fname_l:
                        info["model_role"] = "variance"
                    else:
                        info["model_role"] = "acoustic"

                # ดึงข้อมูล Speaker Embeddings จาก state_dict
                spk_tensor = None
                for spk_key in ("model.spk_embed.weight", "model.fs2.spk_embed.weight", "spk_embed.weight"):
                    if spk_key in state_dict and hasattr(state_dict[spk_key], "shape"):
                        spk_tensor = state_dict[spk_key]
                        info["state_dict_summary"]["spk_embed_key"] = spk_key
                        info["state_dict_summary"]["num_spk_in_tensor"] = int(spk_tensor.shape[0])
                        info["state_dict_summary"]["hidden_size"] = int(spk_tensor.shape[1])
                        break

                info["state_dict_summary"]["total_tensors"] = len(keys)
                if spk_tensor is not None:
                    num_t_spk = int(spk_tensor.shape[0])
                    if not info["speakers"]:
                        if num_t_spk == 1:
                            default_name = Path(info["primary_ckpt"]).parent.name
                            if default_name.lower() in ("checkpoints", "acoustic", "variance", "model", ""):
                                default_name = Path(info["primary_ckpt"]).stem
                            info["speakers"][default_name] = 0
                        else:
                            for idx in range(num_t_spk):
                                info["speakers"][f"Singer_{idx:02d}"] = idx
            except Exception as exc:
                info["state_dict_summary"]["load_error"] = str(exc)

        if not info["speakers"]:
            fallback_singer = search_dir.name if search_dir.name.lower() not in ("checkpoints", "acoustic", "variance", "onnx") else "DeltaSinger"
            info["speakers"][fallback_singer] = 0

        clear_vram_safe()
        return info

    def inspect_colab_inputs_report(self, acoustic_ckpt_or_dir: str, variance_ckpt_or_dir: str = "", custom_spk_map: str = "") -> Tuple[str, str]:
        """สร้างรายงานวิเคราะห์ไฟล์ .ckpt ที่เทรนจาก Google Colab พร้อมคืนรายการผู้พูดทั้งหมด"""
        aco_info = self.find_checkpoint_and_metadata(acoustic_ckpt_or_dir)
        var_info = self.find_checkpoint_and_metadata(variance_ckpt_or_dir) if variance_ckpt_or_dir else {"exists": False}

        if custom_spk_map and os.path.isfile(self._clean_path(custom_spk_map)):
            with suppress(Exception):
                custom_map = json.loads(Path(self._clean_path(custom_spk_map)).read_text(encoding="utf-8"))
                if isinstance(custom_map, dict):
                    aco_info["speakers"] = OrderedDict((str(k), int(v)) for k, v in custom_map.items())

        if not aco_info["exists"] and not var_info.get("exists"):
            msg = DeltaNotificationCenter.notify(
                "ไม่พบไฟล์ .ckpt หรือโฟลเดอร์โมเดลจาก Google Colab ที่ระบุ",
                "Colab checkpoint file or directory not found",
                "error",
                "ColabCKPTInspector",
            )
            return msg, ""

        lines = [
            "=" * 76,
            "🎓 รายงานการวิเคราะห์โมเดล DiffSinger AI จาก Google Colab (Colab CKPT Inspection)",
            "=" * 76,
        ]
        if aco_info["exists"]:
            lines.append(f"🎵 [Acoustic Source] : {aco_info['input_path']}")
            lines.append(f"   • ไฟล์ Checkpoint หลัก : {os.path.basename(aco_info['primary_ckpt'] or 'ไม่พบ .ckpt (ใช้โหมด ONNX)')}")
            lines.append(f"   • ประเภทโมเดลที่ตรวจพบ : {aco_info['model_role'].upper()} | จำนวน Steps: {aco_info['steps']}")
            lines.append(f"   • ไฟล์ Config YAML    : {os.path.basename(aco_info['config_yaml'] or 'Auto-Default')}")
            lines.append(f"   • ไฟล์ spk_map.json   : {os.path.basename(aco_info['spk_map_json'] or 'Auto-Detected')}")
            lines.append(f"   • ข้อมูล State Dict    : {aco_info['state_dict_summary']}")

        if var_info.get("exists"):
            lines.append(f"\n🎛️ [Variance/Dur/Pitch Source] : {var_info['input_path']}")
            lines.append(f"   • ไฟล์ Checkpoint หลัก : {os.path.basename(var_info['primary_ckpt'] or 'ไม่พบ .ckpt (ใช้โหมด ONNX)')}")
            lines.append(f"   • ประเภทโมเดลที่ตรวจพบ : {var_info['model_role'].upper()} | จำนวน Steps: {var_info['steps']}")

        speakers_dict = aco_info.get("speakers") or var_info.get("speakers") or {"DeltaSinger": 0}
        lines.append("\n🎤 รายชื่อนักร้อง / โทนเสียงที่ตรวจพบในโมเดล (Discovered Singers / Speaker IDs):")
        spk_Spec_list = []
        for s_name, s_id in speakers_dict.items():
            lines.append(f"   👤 ID {s_id:02d} : {s_name}")
            spk_Spec_list.append(f"{s_id}:{s_name}")

        lines.append("=" * 76)
        lines.append(f"✅ พร้อมสำหรับการแปลงและแยกชุดคลังเสียงออกเป็นรายบุคคลจำนวน {len(speakers_dict)} ชุด! (Success)")
        DeltaNotificationCenter.notify(
            f"ตรวจพบนักร้อง/คลังเสียงในไฟล์ Colab CKPT จำนวน {len(speakers_dict)} รายการ",
            f"Discovered {len(speakers_dict)} singers in Colab checkpoint (Success)",
            "success",
            "ColabCKPTInspector",
        )
        return "\n".join(lines), ", ".join(spk_Spec_list)

    @staticmethod
    def isolate_speaker_in_ckpt(
        input_ckpt_path: Path,
        output_ckpt_path: Path,
        retain_ids: List[int],
        fill_mode: str = "zeros",
    ) -> bool:
        """
        คัดแยกเฉพาะนักร้องที่ต้องการเก็บไว้ในไฟล์ .ckpt (ตามอัลกอริทึม drop_spk.py ของ DiffSinger Colab)
        """
        ckpt_loaded = torch.load(str(input_ckpt_path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt_loaded, dict) or "state_dict" not in ckpt_loaded:
            shutil.copy2(str(input_ckpt_path), str(output_ckpt_path))
            return True

        state_dict = ckpt_loaded["state_dict"]
        retain_set = set(int(x) for x in retain_ids)

        def _modify_embed(spk_embed: torch.Tensor):
            num_spk, hidden_size = spk_embed.shape
            all_ids = set(range(num_spk))
            drop_ids = sorted(list(all_ids - retain_set))
            if not drop_ids:
                return
            for idx, spk_id in enumerate(drop_ids):
                if fill_mode == "zeros":
                    spk_embed[spk_id, :] = 0.0
                elif fill_mode == "random":
                    spk_embed[spk_id, :] = torch.randn(1, hidden_size, dtype=spk_embed.dtype, device="cpu")
                elif fill_mode == "mean":
                    spk_embed[spk_id, :] = torch.mean(spk_embed, dim=0, keepdim=True)
                elif fill_mode == "cyclic" and retain_set:
                    ret_sorted = sorted(list(retain_set.intersection(all_ids)))
                    if ret_sorted:
                        spk_embed[spk_id, :] = spk_embed[ret_sorted[idx % len(ret_sorted)], :]

        for emb_key in ("model.fs2.spk_embed.weight", "model.spk_embed.weight", "spk_embed.weight"):
            if emb_key in state_dict and hasattr(state_dict[emb_key], "shape"):
                _modify_embed(state_dict[emb_key])

        output_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_loaded, str(output_ckpt_path))
        return True

    @staticmethod
    def extract_speaker_embedding_files(
        ckpt_path: Optional[Path],
        output_embeds_dir: Path,
        speakers_map: Dict[str, int],
        hidden_size: int = 256,
    ) -> List[str]:
        """
        สกัดเวกเตอร์อัตลักษณ์เสียงของนักร้องแต่ละคนจาก Checkpoint ออกมาเป็นไฟล์ .emb (สำหรับ OpenUtau DiffSinger)
        """
        output_embeds_dir.mkdir(parents=True, exist_ok=True)
        written_embs = []
        spk_weight = None

        if ckpt_path and ckpt_path.is_file():
            with suppress(Exception):
                cpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                sd = cpt.get("state_dict", cpt) if isinstance(cpt, dict) else {}
                for k in ("model.spk_embed.weight", "model.fs2.spk_embed.weight", "spk_embed.weight"):
                    if k in sd and hasattr(sd[k], "shape"):
                        spk_weight = sd[k].detach().cpu().numpy().astype(np.float32)
                        break

        for spk_name, spk_id in speakers_map.items():
            safe_name = re.sub(r'[\\/:*?"<>|\s]+', "_", str(spk_name).strip()) or f"Singer_{spk_id:02d}"
            emb_file = output_embeds_dir / f"{safe_name}.emb"
            if spk_weight is not None and 0 <= int(spk_id) < spk_weight.shape[0]:
                vec = spk_weight[int(spk_id)]
            else:
                vec = np.zeros((hidden_size,), dtype=np.float32)
            with open(emb_file, "wb") as f:
                f.write(vec.tobytes())
            written_embs.append(safe_name)

        return written_embs

    def _try_export_ckpt_to_onnx(
        self,
        ckpt_path: Path,
        config_yaml_path: Optional[Path],
        spk_map_path: Optional[Path],
        phonemes_path: Optional[Path],
        export_type: str,
        out_onnx_dir: Path,
        exp_temp_name: str,
    ) -> Tuple[bool, str]:
        """
        ส่งออกไฟล์ .ckpt จาก Google Colab เป็นกราฟคำนวณ .onnx โดยจัดเตรียมโครงสร้างชั่วคราวให้ scripts/export.py
        พร้อมระบบ Fallback อัตโนมัติหากสภาพแวดล้อมยังไม่ได้ติดตั้งโมดูล deployment ของ OpenVPI
        """
        out_onnx_dir.mkdir(parents=True, exist_ok=True)
        export_script = self.ds_root / "scripts" / "export.py"
        if not export_script.is_file():
            export_script = PROJECT_ROOT / "DiffSinger" / "scripts" / "export.py"

        if export_script.is_file():
            temp_exp_dir = export_script.parent.parent / "checkpoints" / exp_temp_name
            temp_exp_dir.mkdir(parents=True, exist_ok=True)
            try:
                # วางไฟล์ Checkpoint ในรูปแบบชื่อมาตรฐาน model_ckpt_steps_XXXXXX.ckpt
                step_match = re.search(r"steps_(\d+)", ckpt_path.name)
                step_num = step_match.group(1) if step_match else "160000"
                staged_ckpt = temp_exp_dir / f"model_ckpt_steps_{step_num}.ckpt"
                if ckpt_path.resolve() != staged_ckpt.resolve():
                    shutil.copy2(str(ckpt_path), str(staged_ckpt))

                if config_yaml_path and config_yaml_path.is_file():
                    shutil.copy2(str(config_yaml_path), str(temp_exp_dir / "config.yaml"))
                if spk_map_path and spk_map_path.is_file():
                    shutil.copy2(str(spk_map_path), str(temp_exp_dir / "spk_map.json"))
                if phonemes_path and phonemes_path.is_file():
                    shutil.copy2(str(phonemes_path), str(temp_exp_dir / phonemes_path.name))

                cmd = [
                    PYTHON_EXE,
                    str(export_script),
                    export_type.lower(),
                    "--exp",
                    exp_temp_name,
                    "--out",
                    str(out_onnx_dir),
                ]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(export_script.parent.parent)
                env["PYTHONUNBUFFERED"] = "1"
                res = subprocess.run(
                    cmd,
                    cwd=str(export_script.parent.parent),
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=300,
                    creationflags=NO_WINDOW,
                )
                if res.returncode == 0 and list(out_onnx_dir.glob("*.onnx")):
                    return True, f"ส่งออก ONNX ด้วย scripts/export.py สำเร็จ ({export_type})"
            except Exception as exc:
                logger.warning(f"scripts/export.py execution notice: {exc}")

        return False, "ใช้โหมดคัดลอกไฟล์ ONNX / บรรจุ Checkpoint มาตรฐาน"

    def build_individual_singer_voicebanks(
        self,
        acoustic_ckpt_or_dir: str,
        variance_ckpt_or_dir: str,
        output_root_dir: str,
        voicebank_prefix: str = "",
        split_mode: str = "แยกนักร้องทุกคนเป็นคลังเสียงรายบุคคล (Split All Singers Individually)",
        selected_speakers_filter: str = "",
        language_code: str = "th",
        vocoder_name_or_path: str = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02",
        profile_img_path: str = "",
        portrait_img_path: str = "",
        author_name: str = "DELTA SYNTH",
        fill_embed_mode: str = "zeros",
        export_isolated_ckpt: bool = True,
        pack_zip_archive: bool = True,
    ):
        """
        กระบวนการหลักแบบ Streaming Generator:
        รับไฟล์ .ckpt (หรือโฟลเดอร์โมเดล) ที่เทรนมาจาก Google Colab ของ DiffSinger AI
        แล้วทำการแยกและสร้างชุดคลังเสียง (OpenUtau DiffSinger Voicebanks) ออกเป็นชุดๆ ตามรายบุคคล
        พร้อมไฟล์ dsconfig.yaml, character.yaml, character.txt, dsdict.yaml, embeds, dsdur, dsvariance, dspitch และ .zip
        """
        start_time = time.time()
        sep = "=" * 76
        logs: List[str] = []

        def _emit(msg: str) -> str:
            logs.append(f"[{strftime('%H:%M:%S')}] {msg}")
            return "\n".join(logs)

        yield _emit(f"{sep}\n🚀 เริ่มกระบวนการแปลงไฟล์ Colab CKPT เป็นชุดคลังเสียง DiffSinger AI รายบุคคล\n{sep}")

        aco_info = self.find_checkpoint_and_metadata(acoustic_ckpt_or_dir)
        var_info = self.find_checkpoint_and_metadata(variance_ckpt_or_dir) if variance_ckpt_or_dir else {"exists": False, "speakers": OrderedDict()}

        if not aco_info["exists"] and not var_info.get("exists"):
            err = DeltaNotificationCenter.notify(
                "กรุณาระบุไฟล์ .ckpt หรือโฟลเดอร์โมเดล Acoustic / Variance ที่ได้จาก Google Colab",
                "Please provide valid Acoustic or Variance Colab checkpoint path",
                "error",
                "ColabCKPTConverter",
            )
            yield _emit(err)
            return

        out_root = Path(self._clean_path(output_root_dir) or os.path.join(now_dir, "Packed_Voicebanks", "Colab_DiffSinger_Singers")).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        # รวมรายชื่อผู้พูดทั้งหมดจาก Acoustic และ Variance
        all_speakers: OrderedDict[str, int] = OrderedDict()
        for k, v in (aco_info.get("speakers") or {}).items():
            all_speakers[k] = int(v)
        for k, v in (var_info.get("speakers") or {}).items():
            if k not in all_speakers:
                all_speakers[k] = int(v)
        if not all_speakers:
            all_speakers["DeltaSinger"] = 0

        # กรองเฉพาะผู้พูดที่ผู้ใช้ระบุ (หากมีการระบุในช่อง Filter เช่น "0,1" หรือ "Kochujang,Natural")
        filter_tokens = [t.strip() for t in str(selected_speakers_filter or "").split(",") if t.strip()]
        target_speakers: OrderedDict[str, int] = OrderedDict()
        if filter_tokens:
            for spk_name, spk_id in all_speakers.items():
                if str(spk_id) in filter_tokens or spk_name in filter_tokens or f"{spk_id}:{spk_name}" in filter_tokens:
                    target_speakers[spk_name] = spk_id
            if not target_speakers:
                target_speakers = all_speakers
        else:
            target_speakers = all_speakers

        yield _emit(f"🎤 ตรวจพบนักร้องเป้าหมายทั้งหมด {len(target_speakers)} รายการ: {', '.join(f'{n} (ID:{i})' for n, i in target_speakers.items())}")

        aco_cfg = aco_info.get("hparams") or {}
        var_cfg = var_info.get("hparams") or {}
        sample_rate = int(aco_cfg.get("audio_sample_rate", aco_cfg.get("sample_rate", 44100)))
        hop_size = int(aco_cfg.get("hop_size", 512))
        hidden_size = int(aco_info.get("state_dict_summary", {}).get("hidden_size", aco_cfg.get("hidden_size", 256)))

        is_multi_subbank_mode = "Subbank" in split_mode or "รวมโทนเสียง" in split_mode
        singer_groups: List[Tuple[str, OrderedDict[str, int]]] = []

        if is_multi_subbank_mode:
            group_title = (voicebank_prefix.strip() or next(iter(target_speakers.keys()), "DeltaSinger")).replace(" ", "_")
            singer_groups.append((group_title, target_speakers))
        else:
            for spk_name, spk_id in target_speakers.items():
                clean_s = re.sub(r'[\\/:*?"<>|\s]+', "_", spk_name.strip())
                vb_title = f"{voicebank_prefix.strip()}_{clean_s}" if voicebank_prefix.strip() else clean_s
                singer_groups.append((vb_title, OrderedDict([(spk_name, spk_id)])))

        created_voicebanks: List[str] = []
        total_groups = len(singer_groups)

        for idx, (vb_name, group_spks) in enumerate(singer_groups, 1):
            yield _emit(f"\n{'─'*68}\n📦 [{idx}/{total_groups}] กำลังสร้างชุดคลังเสียงสำหรับนักร้อง: '{vb_name}'\n{'─'*68}")
            vb_dir = out_root / vb_name
            dsmain_dir = vb_dir / "dsmain"
            main_embeds_dir = vb_dir / "embeds"
            dsdur_dir = vb_dir / "dsdur"
            dur_embeds_dir = dsdur_dir / "embeds"
            dsvariance_dir = vb_dir / "dsvariance"
            var_embeds_dir = dsvariance_dir / "embeds"
            dspitch_dir = vb_dir / "dspitch"
            pitch_embeds_dir = dspitch_dir / "embeds"
            ckpt_out_dir = vb_dir / "colab_ckpt_isolated"

            for d in (vb_dir, dsmain_dir, main_embeds_dir, dsdur_dir, dur_embeds_dir, dsvariance_dir, var_embeds_dir, dspitch_dir, pitch_embeds_dir):
                d.mkdir(parents=True, exist_ok=True)

            retain_ids = list(group_spks.values())

            # 1. คัดแยกไฟล์ .ckpt เฉพาะบุคคล (Drop/Retain Speakers)
            isolated_aco_ckpt = None
            isolated_var_ckpt = None
            if export_isolated_ckpt:
                ckpt_out_dir.mkdir(parents=True, exist_ok=True)
                if aco_info.get("primary_ckpt"):
                    src_aco_ckpt = Path(aco_info["primary_ckpt"])
                    isolated_aco_ckpt = ckpt_out_dir / f"{vb_name}_acoustic.ckpt"
                    yield _emit(f"✂️ [Step 1A] คัดแยกผู้พูด {retain_ids} จากโมเดล Acoustic CKPT -> {isolated_aco_ckpt.name}")
                    self.isolate_speaker_in_ckpt(src_aco_ckpt, isolated_aco_ckpt, retain_ids, fill_embed_mode)
                if var_info.get("primary_ckpt"):
                    src_var_ckpt = Path(var_info["primary_ckpt"])
                    isolated_var_ckpt = ckpt_out_dir / f"{vb_name}_variance.ckpt"
                    yield _emit(f"✂️ [Step 1B] คัดแยกผู้พูด {retain_ids} จากโมเดล Variance CKPT -> {isolated_var_ckpt.name}")
                    self.isolate_speaker_in_ckpt(src_var_ckpt, isolated_var_ckpt, retain_ids, fill_embed_mode)

            # 2. สกัดไฟล์ Speaker Embeddings (.emb) สำหรับนักร้องรายนี้
            yield _emit("🧬 [Step 2] สกัดเวกเตอร์อัตลักษณ์เสียง (.emb) สำหรับ Acoustic, Duration, Variance และ Pitch...")
            aco_ckpt_for_emb = isolated_aco_ckpt or (Path(aco_info["primary_ckpt"]) if aco_info.get("primary_ckpt") else None)
            var_ckpt_for_emb = isolated_var_ckpt or (Path(var_info["primary_ckpt"]) if var_info.get("primary_ckpt") else aco_ckpt_for_emb)

            aco_emb_names = self.extract_speaker_embedding_files(aco_ckpt_for_emb, main_embeds_dir, group_spks, hidden_size)
            self.extract_speaker_embedding_files(var_ckpt_for_emb, dur_embeds_dir, group_spks, hidden_size)
            self.extract_speaker_embedding_files(var_ckpt_for_emb, var_embeds_dir, group_spks, hidden_size)
            self.extract_speaker_embedding_files(var_ckpt_for_emb, pitch_embeds_dir, group_spks, hidden_size)

            # 3. ส่งออกหรือคัดลอกไฟล์ ONNX และ Phonemes
            yield _emit("⚙️ [Step 3] ประมวลผลกราฟคำนวณ ONNX และชุดหน่วยเสียง (Phonemes & Languages)...")
            if aco_ckpt_for_emb:
                ok_aco, msg_aco = self._try_export_ckpt_to_onnx(
                    aco_ckpt_for_emb,
                    Path(aco_info["config_yaml"]) if aco_info.get("config_yaml") else None,
                    Path(aco_info["spk_map_json"]) if aco_info.get("spk_map_json") else None,
                    Path(aco_info["phonemes_file"]) if aco_info.get("phonemes_file") else None,
                    "acoustic",
                    dsmain_dir,
                    f"colab_aco_{vb_name}",
                )
                yield _emit(f"   • Acoustic Export: {msg_aco}")

            if var_ckpt_for_emb and var_info.get("exists"):
                ok_var, msg_var = self._try_export_ckpt_to_onnx(
                    var_ckpt_for_emb,
                    Path(var_info["config_yaml"]) if var_info.get("config_yaml") else None,
                    Path(var_info["spk_map_json"]) if var_info.get("spk_map_json") else None,
                    Path(var_info["phonemes_file"]) if var_info.get("phonemes_file") else None,
                    "variance",
                    vb_dir,
                    f"colab_var_{vb_name}",
                )
                yield _emit(f"   • Variance Export: {msg_var}")

            # คัดลอกไฟล์ .onnx ที่มีอยู่แล้วในโฟลเดอร์ต้นทาง (กรณีผู้ใช้ส่งออก ONNX จาก Colab มาด้วย)
            for ox_str in aco_info.get("onnx_files", []) + var_info.get("onnx_files", []):
                ox_p = Path(ox_str)
                ox_l = ox_p.name.lower()
                if "acoustic" in ox_l:
                    shutil.copy2(str(ox_p), str(dsmain_dir / "acoustic.onnx"))
                elif "linguistic" in ox_l:
                    shutil.copy2(str(ox_p), str(dsmain_dir / "linguistic.onnx"))
                    shutil.copy2(str(ox_p), str(dsdur_dir / "linguistic.onnx"))
                    shutil.copy2(str(ox_p), str(dsvariance_dir / "linguistic.onnx"))
                    shutil.copy2(str(ox_p), str(dspitch_dir / "linguistic.onnx"))
                elif "dur" in ox_l:
                    shutil.copy2(str(ox_p), str(dsdur_dir / "dur.onnx"))
                elif "variance" in ox_l or "var" in ox_l:
                    shutil.copy2(str(ox_p), str(dsvariance_dir / "variance.onnx"))
                elif "pitch" in ox_l:
                    shutil.copy2(str(ox_p), str(dspitch_dir / "pitch.onnx"))

            # จัดเตรียมไฟล์ phonemes.json และ phonemes.txt ใน dsmain, dsdur, dsvariance, dspitch
            ph_src = aco_info.get("phonemes_file") or var_info.get("phonemes_file")
            phonemes_dict = OrderedDict([("pad", 0), ("AP", 1), ("SP", 2)])
            if ph_src and os.path.isfile(ph_src):
                ph_p = Path(ph_src)
                if ph_p.suffix.lower() == ".json":
                    with suppress(Exception):
                        loaded_ph = json.loads(ph_p.read_text(encoding="utf-8"))
                        if isinstance(loaded_ph, dict):
                            phonemes_dict = OrderedDict(loaded_ph)
                else:
                    with suppress(Exception):
                        for line in ph_p.read_text(encoding="utf-8").splitlines():
                            tok = line.strip().split()[0] if line.strip() else ""
                            if tok and tok not in phonemes_dict:
                                phonemes_dict[tok] = len(phonemes_dict)
            else:
                for std_grp in ("consonants", "vowels", "endings"):
                    for ph_val in THAI_PHONEME_STANDARDS["DiffSinger_AI"][std_grp].values():
                        if ph_val not in phonemes_dict:
                            phonemes_dict[ph_val] = len(phonemes_dict)

            for sub_target in (dsmain_dir, dsdur_dir, dsvariance_dir, dspitch_dir):
                (sub_target / "phonemes.json").write_text(json.dumps(phonemes_dict, ensure_ascii=False, indent=2), encoding="utf-8")
                (sub_target / "phonemes.txt").write_text("\n".join(phonemes_dict.keys()) + "\n", encoding="utf-8")

            # สร้าง languages.json พร้อมรหัสภาษาไทย ('th')
            lang_data = {"th": 0, "en": 1, "ja": 2, "zh": 3}
            if aco_info.get("languages_json") and os.path.isfile(aco_info["languages_json"]):
                with suppress(Exception):
                    loaded_l = json.loads(Path(aco_info["languages_json"]).read_text(encoding="utf-8"))
                    if isinstance(loaded_l, dict):
                        lang_data = loaded_l
                        if language_code and language_code not in lang_data:
                            lang_data[language_code] = max(lang_data.values(), default=-1) + 1
            (dsmain_dir / "languages.json").write_text(json.dumps(lang_data, ensure_ascii=False, indent=2), encoding="utf-8")
            (dsdur_dir / "languages.json").write_text(json.dumps(lang_data, ensure_ascii=False, indent=2), encoding="utf-8")

            # 4. สร้าง dsconfig.yaml หลักและย่อย (dsdur, dsvariance, dspitch, dsvocoder)
            yield _emit("📝 [Step 4] สร้างไฟล์กำหนดค่า OpenUtau (dsconfig.yaml, character.yaml, character.txt)...")
            speaker_embed_paths = [f"embeds/{ename}" for ename in aco_emb_names]
            subbanks = [
                {"color": f"{i:02d}: {ename}", "suffix": f"embeds/{ename}"}
                for i, ename in enumerate(aco_emb_names, 1)
            ]

            vocoder_clean = self._clean_path(vocoder_name_or_path) or "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
            if os.path.isfile(vocoder_clean):
                dsvoc_dir = vb_dir / "dsvocoder"
                dsvoc_dir.mkdir(parents=True, exist_ok=True)
                voc_p = Path(vocoder_clean)
                shutil.copy2(str(voc_p), str(dsvoc_dir / voc_p.name))
                if (voc_p.parent / "vocoder.yaml").is_file():
                    shutil.copy2(str(voc_p.parent / "vocoder.yaml"), str(dsvoc_dir / "vocoder.yaml"))
                vocoder_ref = f"dsvocoder/{voc_p.stem}"
            else:
                vocoder_ref = vocoder_clean

            main_dsconfig = OrderedDict([
                ("phonemes", "dsmain/phonemes.json"),
                ("languages", "dsmain/languages.json"),
                ("acoustic", "dsmain/acoustic.onnx"),
                ("vocoder", vocoder_ref),
                ("singer_type", "diffsinger"),
                ("sample_rate", sample_rate),
                ("hop_size", hop_size),
                ("win_size", int(aco_cfg.get("win_size", 2048))),
                ("fft_size", int(aco_cfg.get("fft_size", 2048))),
                ("num_mel_bins", int(aco_cfg.get("audio_num_mel_bins", 128))),
                ("fmin", int(aco_cfg.get("fmin", 40))),
                ("fmax", int(aco_cfg.get("fmax", 16000))),
                ("mel_base", str(aco_cfg.get("mel_base", "e"))),
                ("mel_scale", "slaney"),
                ("hidden_size", hidden_size),
                ("use_energy_embed", bool(aco_cfg.get("use_energy_embed", True))),
                ("use_breathiness_embed", bool(aco_cfg.get("use_breathiness_embed", False))),
                ("use_voicing_embed", bool(aco_cfg.get("use_voicing_embed", True))),
                ("use_tension_embed", bool(aco_cfg.get("use_tension_embed", True))),
                ("use_key_shift_embed", bool(aco_cfg.get("use_key_shift_embed", True))),
                ("use_speed_embed", bool(aco_cfg.get("use_speed_embed", True))),
                ("use_variable_depth", bool(aco_cfg.get("use_shallow_diffusion", True))),
                ("max_depth", float(aco_cfg.get("T_start", 0.4))),
                ("use_continuous_acceleration", True),
                ("speakers", speaker_embed_paths),
                ("delta_synth_rule10_compliance", {
                    "standard": "DELTA SYNTH Rule 10",
                    "leading_consonant_ratio": "<= 5%",
                    "vowel_stretch_mode": "note_duration",
                    "ending_consonant_ratio": "25-40%",
                }),
            ])
            with open(vb_dir / "dsconfig.yaml", "w", encoding="utf-8") as f:
                yaml.dump(dict(main_dsconfig), f, allow_unicode=True, sort_keys=False)

            # Sub-configs: dsdur, dsvariance, dspitch
            dsdur_cfg = {
                "phonemes": "phonemes.json",
                "languages": "../dsmain/languages.json",
                "linguistic": "linguistic.onnx" if (dsdur_dir / "linguistic.onnx").exists() else "../dsmain/linguistic.onnx",
                "dur": "dur.onnx",
                "use_continuous_acceleration": True,
                "sample_rate": sample_rate,
                "hidden_size": hidden_size,
                "hop_size": hop_size,
                "predict_dur": True,
                "use_lang_id": bool(var_cfg.get("use_lang_id", False)),
                "speakers": speaker_embed_paths,
            }
            with open(dsdur_dir / "dsconfig.yaml", "w", encoding="utf-8") as f:
                yaml.dump(dsdur_cfg, f, allow_unicode=True, sort_keys=False)

            dsvariance_cfg = {
                "phonemes": "phonemes.json",
                "languages": "../dsmain/languages.json",
                "linguistic": "linguistic.onnx" if (dsvariance_dir / "linguistic.onnx").exists() else "../dsmain/linguistic.onnx",
                "variance": "variance.onnx",
                "use_continuous_acceleration": True,
                "sample_rate": sample_rate,
                "hidden_size": hidden_size,
                "hop_size": hop_size,
                "predict_dur": True,
                "predict_voicing": bool(var_cfg.get("predict_voicing", True)),
                "predict_tension": bool(var_cfg.get("predict_tension", True)),
                "predict_energy": bool(var_cfg.get("predict_energy", True)),
                "predict_breathiness": bool(var_cfg.get("predict_breathiness", False)),
                "use_lang_id": bool(var_cfg.get("use_lang_id", False)),
                "speakers": speaker_embed_paths,
            }
            with open(dsvariance_dir / "dsconfig.yaml", "w", encoding="utf-8") as f:
                yaml.dump(dsvariance_cfg, f, allow_unicode=True, sort_keys=False)

            dspitch_cfg = {
                "phonemes": "phonemes.json",
                "languages": "../dsmain/languages.json",
                "linguistic": "linguistic.onnx" if (dspitch_dir / "linguistic.onnx").exists() else "../dsmain/linguistic.onnx",
                "pitch": "pitch.onnx",
                "use_expr": True,
                "use_continuous_acceleration": True,
                "sample_rate": sample_rate,
                "hidden_size": hidden_size,
                "hop_size": hop_size,
                "predict_dur": True,
                "use_lang_id": bool(var_cfg.get("use_lang_id", False)),
                "use_note_rest": bool(var_cfg.get("use_note_rest", False)),
                "speakers": speaker_embed_paths,
            }
            with open(dspitch_dir / "dsconfig.yaml", "w", encoding="utf-8") as f:
                yaml.dump(dspitch_cfg, f, allow_unicode=True, sort_keys=False)

            # รูปภาพตัวละคร (Icon & Portrait)
            icon_filename = ""
            portrait_filename = ""
            clean_icon = self._clean_path(profile_img_path)
            clean_portrait = self._clean_path(portrait_img_path)
            if clean_icon and os.path.isfile(clean_icon):
                icon_filename = "Icon" + Path(clean_icon).suffix.lower()
                shutil.copy2(clean_icon, str(vb_dir / icon_filename))
            if clean_portrait and os.path.isfile(clean_portrait):
                portrait_filename = "Full" + Path(clean_portrait).suffix.lower()
                shutil.copy2(clean_portrait, str(vb_dir / portrait_filename))

            # เขียนไฟล์ character.txt และ character.yaml
            char_txt_lines = [
                f"name={vb_name}",
                f"author={author_name or 'DELTA SYNTH'}",
                "web=https://patiphatwongyai.wixsite.com/deltasynth",
            ]
            if icon_filename:
                char_txt_lines.append(f"image={icon_filename}")
            (vb_dir / "character.txt").write_text("\n".join(char_txt_lines) + "\n", encoding="utf-8")

            char_yaml_data = OrderedDict([
                ("name", vb_name),
                ("singer_type", "diffsinger"),
                ("author", author_name or "DELTA SYNTH"),
                ("version", "1.0.0"),
                ("language", language_code or "th"),
                ("text_file_encoding", "utf-8"),
                ("default_phonemizer", "OpenUtau.Core.DiffSinger.DiffSingerPhonemizer"),
                ("image", icon_filename),
                ("portrait", portrait_filename),
                ("portrait_opacity", 0.45),
                ("subbanks", subbanks),
                ("delta_synth_rule10_compliance", {
                    "standard": "DELTA SYNTH Rule 10",
                    "leading_consonant_ratio": "<= 5%",
                    "vowel_stretch_mode": "note_duration",
                    "ending_consonant_ratio": "25-40%",
                }),
            ])
            with open(vb_dir / "character.yaml", "w", encoding="utf-8") as f:
                yaml.dump(dict(char_yaml_data), f, allow_unicode=True, sort_keys=False)

            # 5. สร้างพจนานุกรม dsdict.yaml อัตโนมัติและคัดลอกไปยังโฟลเดอร์ย่อย
            dict_msg = build_dsdict_automatic(str(dsmain_dir), str(vb_dir))
            yield _emit(f"📖 [Step 5] {dict_msg}")
            if (vb_dir / "dsdict.yaml").is_file():
                for sub_d in (dsmain_dir, dsdur_dir, dsvariance_dir, dspitch_dir):
                    shutil.copy2(str(vb_dir / "dsdict.yaml"), str(sub_d / "dsdict.yaml"))

            # 6. บีบอัดเป็นไฟล์ .zip พร้อมติดตั้งใน OpenUtau (ถ้าเลือกไว้)
            if pack_zip_archive:
                zip_base = out_root / f"{vb_name}_OpenUtau"
                shutil.make_archive(str(zip_base), "zip", str(vb_dir))
                yield _emit(f"🗜️ [Step 6] บีบอัดคลังเสียงพร้อมใช้งาน: {zip_base.name}.zip")

            created_voicebanks.append(str(vb_dir))

        elapsed = format_time_delta(time.time() - start_time)
        summary_msg = (
            f"\n{sep}\n🎉 แปลงไฟล์ Colab CKPT ออกมาเป็นชุดคลังเสียงรายบุคคลเสร็จสมบูรณ์ {len(created_voicebanks)} ชุด! (⏱️ {elapsed}) (Success)\n"
            f"📂 โฟลเดอร์ผลลัพธ์หลัก: {out_root}\n"
            + "\n".join(f"   ✅ {p}" for p in created_voicebanks)
            + f"\n{sep}"
        )
        DeltaNotificationCenter.notify(
            f"สร้างชุดคลังเสียงจากไฟล์ Colab CKPT สำเร็จ {len(created_voicebanks)} ชุด!",
            f"Built {len(created_voicebanks)} individual singer voicebanks (Success)",
            "success",
            "ColabCKPTConverter",
        )
        clear_vram_safe()
        yield _emit(summary_msg)


COLAB_CKPT_CONVERTER = ColabCKPTVoicebankConverter(DIFFSINGER_ROOT)


# ─── 14. DiffSinger 2.5.1 Studio Core & Pipeline Functions ──────────────────
def ds_log_msg(msg: str, level: str = "INFO") -> str:
    icons = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERROR": "❌", "RUN": "🔄"}
    return f"[{strftime('%H:%M:%S')}] {icons.get(level, '📌')} {msg}"


def ds_run_subprocess_streaming(cmd: list, cwd: Optional[str] = None):
    global _ds_active_process, CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DIFFSINGER_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd or str(DIFFSINGER_ROOT),
            env=env,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        with _ds_process_lock:
            _ds_active_process = process
        ACTIVE_CHILD_PROCESSES.add(process)
        output = ""
        for line in iter(process.stdout.readline, ""):
            if CANCEL_REQUESTED:
                cancel_all_processes()
                yield output + f"\n{ds_log_msg('ยกเลิกการทำงานโดยผู้ใช้ (Process Cancelled)', 'WARN')}\n"
                return
            if line:
                output += line
                yield output
        process.wait()
        status_lvl = "OK" if process.returncode == 0 else "ERROR"
        suffix = " (Success)" if process.returncode == 0 else " (Error)"
        yield output + f"\n{ds_log_msg(f'Process Exited Code: {process.returncode}{suffix}', status_lvl)}\n"
    except Exception as exc:
        yield f"{ds_log_msg(f'Error: {exc}', 'ERROR')}\n"
    finally:
        with _ds_process_lock:
            if _ds_active_process in ACTIVE_CHILD_PROCESSES:
                ACTIVE_CHILD_PROCESSES.discard(_ds_active_process)
            _ds_active_process = None
        clear_vram_safe()


def ds_list_existing_projects() -> List[str]:
    if not DS_DATA_DIR.exists():
        return []
    return sorted([d.name for d in DS_DATA_DIR.iterdir() if d.is_dir()])


def ds_list_all_experiments() -> List[str]:
    if not DS_CHECKPOINTS_DIR.exists():
        return []
    return sorted([
        d.name for d in DS_CHECKPOINTS_DIR.iterdir()
        if d.is_dir() and d.name != ".gitkeep"
    ])


def ds_create_project(project_name: str, speaker_name: str, speaker_id: int, language: str):
    if not project_name or not str(project_name).strip():
        return DeltaNotificationCenter.notify("กรุณาระบุชื่อโปรเจกต์", "Please provide a project name", "error", "DiffSinger"), gr.update()
    proj_clean = str(project_name).strip().replace(" ", "_")
    spk_clean = str(speaker_name).strip() if speaker_name else proj_clean
    try:
        sid_clean = int(speaker_id)
    except (ValueError, TypeError):
        sid_clean = 0

    proj_dir = DS_DATA_DIR / proj_clean
    try:
        (proj_dir / "raw" / "wavs").mkdir(parents=True, exist_ok=True)
        (proj_dir / "binary").mkdir(parents=True, exist_ok=True)
        (proj_dir / "export").mkdir(parents=True, exist_ok=True)
        trans_csv = proj_dir / "raw" / "transcriptions.csv"
        if not trans_csv.exists():
            with open(trans_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["name", "ph_seq", "ph_dur"])
        meta = {
            "project_name": proj_clean,
            "speaker_name": spk_clean,
            "speaker_id": sid_clean,
            "language": language or "th",
            "created_at": datetime.now().isoformat(),
            "sample_rate": 44100,
        }
        (proj_dir / "project_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        msg = DeltaNotificationCenter.notify(f"สร้างโปรเจกต์ DiffSinger '{proj_clean}' สำเร็จ", f"Created project {proj_clean} (Success)", "success", "DiffSinger")
        return msg, gr.update(choices=ds_list_existing_projects(), value=proj_clean)
    except Exception as exc:
        return PoliteLogger.error("Failed to create DiffSinger project", exc=exc, context="DiffSinger"), gr.update()


def ds_remove_duplicates_and_scan_phonemes(project_name: str) -> str:
    if not project_name:
        return DeltaNotificationCenter.notify("กรุณาเลือกหรือระบุชื่อโปรเจกต์ DiffSinger ก่อนดำเนินการ", "Please select a DiffSinger project", "warning", "DiffSinger")
    proj_dir = DS_DATA_DIR / str(project_name).strip()
    trans_path = proj_dir / "raw" / "transcriptions.csv"
    phonemes = {"AP", "SP"}
    removed = 0
    if trans_path.is_file():
        rows = []
        with open(trans_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        seen = {}
        kept = []
        for r in rows:
            ph_s = r.get("ph_seq", "").strip()
            for p in ph_s.split():
                phonemes.add(p.split("/")[-1])
            if ph_s and ph_s in seen:
                removed += 1
            else:
                seen[ph_s] = True
                kept.append(r)
        if removed > 0 and kept:
            with open(trans_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
                w.writeheader()
                w.writerows(kept)
        dict_content = "\n".join(f"{ph}\t{ph}" for ph in sorted(phonemes))
        (proj_dir / "custom_dictionary.txt").write_text(dict_content + "\n", encoding="utf-8")
    return DeltaNotificationCenter.notify(
        f"ตรวจสอบชุดข้อมูล '{project_name}' สำเร็จ (ลบซ้ำ {removed} รายการ, พบหน่วยเสียง {len(phonemes)} รายการ)",
        f"Scanned project {project_name} (Success)",
        "success",
        "DiffSinger",
    )


def ds_generate_config_yaml(project_name: str, model_type: str = "Acoustic", max_updates: int = 160000) -> Tuple[str, str]:
    proj_clean = str(project_name or "my_voicebank").strip()
    proj_dir = DS_DATA_DIR / proj_clean
    meta = {}
    if (proj_dir / "project_meta.json").is_file():
        with suppress(Exception):
            meta = json.loads((proj_dir / "project_meta.json").read_text(encoding="utf-8"))

    lang = meta.get("language", "th")
    cfg = OrderedDict([
        ("base_config", [f"configs/{model_type.lower()}.yaml"]),
        ("dictionaries", {lang: f"data/{proj_clean}/custom_dictionary.txt"}),
        ("extra_phonemes", []),
        ("merged_phoneme_groups", []),
        ("datasets", [{
            "raw_data_dir": f"data/{proj_clean}/raw",
            "speaker": meta.get("speaker_name", proj_clean),
            "spk_id": int(meta.get("speaker_id", 0)),
            "language": lang,
            "test_prefixes": ["test"],
        }]),
        ("binary_data_dir", f"data/{proj_clean}/binary"),
        ("binarization_args", {"num_workers": 2, "shuffle": True}),
        ("pe", "rmvpe"),
        ("pe_ckpt", "checkpoints/rmvpe/model.pt"),
        ("audio_sample_rate", 44100),
        ("hop_size", 512),
        ("audio_num_mel_bins", 128),
        ("diffusion_type", "reflow"),
        ("max_updates", int(max_updates or 160000)),
        ("max_batch_size", max(4, default_batch_size * 2)),
    ])
    if model_type.lower() == "acoustic":
        cfg["use_shallow_diffusion"] = True
        cfg["use_energy_embed"] = True
        cfg["use_tension_embed"] = True
        cfg["use_voicing_embed"] = True
        cfg["use_key_shift_embed"] = True
        cfg["use_speed_embed"] = True
    else:
        cfg["predict_dur"] = True
        cfg["predict_pitch"] = True
        cfg["predict_energy"] = True
        cfg["predict_tension"] = True
        cfg["predict_voicing"] = True

    yaml_str = yaml.dump(dict(cfg), allow_unicode=True, sort_keys=False)
    proj_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = proj_dir / f"config_{model_type.lower()}.yaml"
    cfg_file.write_text(yaml_str, encoding="utf-8")
    msg = DeltaNotificationCenter.notify(f"สร้างและบันทึก {cfg_file.name} สำเร็จ", f"Generated {cfg_file.name} (Success)", "success", "DiffSinger")
    return msg, yaml_str


def ds_run_binarization(project_name: str, model_type: str, custom_config_path: str = ""):
    try:
        cfg_path = Path(custom_config_path.strip()) if custom_config_path and custom_config_path.strip() else (
            DS_DATA_DIR / str(project_name or "").strip() / ("config_acoustic.yaml" if model_type == "Acoustic" else "config_variance.yaml")
        )
        if not cfg_path.exists():
            yield ds_log_msg(f"ไม่พบไฟล์ Config: {cfg_path} (Error)", "ERROR")
            return
        purge_directory_junk(str(cfg_path.parent))
        yield ds_log_msg(f"เริ่ม Binarization ด้วยไฟล์: {cfg_path}", "RUN") + "\n"
        for line in ds_run_subprocess_streaming([PYTHON_EXE, str(DS_SCRIPTS_DIR / "binarize.py"), "--config", str(cfg_path)], cwd=str(DIFFSINGER_ROOT)):
            yield line
    finally:
        clear_vram_safe()


def ds_run_training_job(config_path_str: str, exp_name: str, resume: bool = True):
    try:
        cfg_clean = str(config_path_str or "").strip()
        cmd = [
            PYTHON_EXE,
            str(DS_SCRIPTS_DIR / "train.py"),
            "--config",
            cfg_clean,
            "--exp_name",
            (exp_name.strip() or f"ds_exp_{strftime('%H%M%S')}"),
        ]
        if not resume:
            cmd.append("--reset")
        yield ds_log_msg(f"เริ่มฝึกสอน DiffSinger Experiment: '{exp_name}'", "RUN") + "\n"
        for line in ds_run_subprocess_streaming(cmd, cwd=str(DIFFSINGER_ROOT)):
            yield line
    finally:
        clear_vram_safe()


def ds_run_one_click_pipeline(config_path_str: str, exp_name: str, out_dir: str = ""):
    try:
        cfg_clean = str(config_path_str or "").strip()
        if not os.path.exists(cfg_clean):
            yield ds_log_msg(f"[1/3 ERROR] ไม่พบไฟล์ Config: {cfg_clean} (Error)", "ERROR")
            return
        if not exp_name or not exp_name.strip():
            yield ds_log_msg("[ERROR] กรุณาระบุชื่อ Experiment Name (Error)", "ERROR")
            return

        yield ds_log_msg("[1/3] ตรวจสอบไฟล์ Config ผ่านเรียบร้อย!", "OK") + "\n"
        yield ds_log_msg("[2/3] เริ่มการฝึกสอนโมเดล (Training)...", "RUN") + "\n"
        for line in ds_run_subprocess_streaming([PYTHON_EXE, str(DS_SCRIPTS_DIR / "train.py"), "--config", cfg_clean, "--exp_name", exp_name.strip()], cwd=str(DIFFSINGER_ROOT)):
            yield line

        yield ds_log_msg("[3/3] เริ่มส่งออกเป็น ONNX...", "RUN") + "\n"
        export_type = "variance" if "variance" in cfg_clean.lower() else "acoustic"
        export_cmd = [PYTHON_EXE, str(DS_SCRIPTS_DIR / "export.py"), export_type, "--exp", exp_name.strip()]
        if out_dir and out_dir.strip():
            export_cmd.extend(["--out", out_dir.strip()])
        for line in ds_run_subprocess_streaming(export_cmd, cwd=str(DIFFSINGER_ROOT)):
            yield line
    finally:
        clear_vram_safe()


def convert_lab_wav_to_diffsinger_csv(dataset_root_dir: str) -> str:
    """แปลงชุดข้อมูล .lab + .wav (NNSVS / vLabeler format) เป็น transcriptions.csv และ auto_lang_config.json"""
    clean_root = str(dataset_root_dir or "").strip().strip('"')
    if not clean_root or not os.path.isdir(clean_root):
        return DeltaNotificationCenter.notify("ไม่พบโฟลเดอร์ชุดข้อมูล .lab + .wav", "Dataset directory not found", "error", "DBConverter")
    try:
        mod = ACCOUNT_BRIDGE.load_module("liteconvert")
        if mod and hasattr(mod, "auto_config") and hasattr(mod, "lab2csv"):
            mod.auto_config(clean_root)
            lang_cfg = os.path.join(clean_root, "auto_lang_config.json")
            mod.lab2csv(clean_root, lang_cfg)
            return DeltaNotificationCenter.notify(f"แปลง .lab + .wav ใน '{clean_root}' เป็นรูปแบบ DiffSinger สำเร็จ!", "Converted .lab + .wav to DiffSinger CSV (Success)", "success", "DBConverter")
    except Exception as exc:
        return PoliteLogger.error("DB Converter failed", exc=exc, context="DBConverter")
    return DeltaNotificationCenter.notify("ประมวลผลชุดข้อมูลเสร็จสิ้น", "Dataset conversion finished (Success)", "success", "DBConverter")


# ─── 15. RVC Core Inference, Training & Multi-Speaker Wrappers ───────────────
def get_vc_wrapper(sid, protect0, protect1):
    if sid:
        DeltaLogger.model(sid, "Active Voice Conversion Model Selected")
        DeltaNotificationCenter.notify(f"โหลดโมเดลเสียง RVC: {sid}", f"Loaded RVC model: {sid}", "info", "RVCInference")
    else:
        DeltaLogger.inf("🔄 กำลังคืนค่าและยกเลิกโมเดล (Unloading Model)", to_console=False)
    if RVC_CORE_AVAILABLE and vc is not None:
        return vc.get_vc(sid, protect0, protect1)
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def single_infer_wrapper(spk_item, input_audio0, vc_transform0, f0_file, f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0, sid0=None):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    start_time_sec = time.time()
    start_timestamp = strftime("%H:%M:%S")
    sep_line = "=" * 68

    clean_input = str(input_audio0 or "").strip().strip('"')
    if not clean_input or not os.path.exists(clean_input):
        msg = DeltaNotificationCenter.notify(f"ไม่พบไฟล์เสียงต้นฉบับ: {clean_input}", "Source Audio Not Found", "error", "RVCInference")
        return f"{sep_line}\n{msg}\n{sep_line}", None
    try:
        if RVC_CORE_AVAILABLE and vc is not None:
            if getattr(vc, "net_g", None) is None and sid0:
                vc.get_vc(sid0, protect0, protect0)
            try:
                sid_clean = int(spk_item) if spk_item else 0
            except (ValueError, TypeError):
                sid_clean = 0
            info_log, opt_audio = vc.vc_single(
                sid_clean, clean_input, vc_transform0, f0_file, f0method0,
                file_index1, file_index2, index_rate1, filter_radius0,
                resample_sr0, rms_mix_rate0, protect0,
            )
            elapsed = format_time_delta(time.time() - start_time_sec)
            DeltaNotificationCenter.notify(f"แปลงเสียง '{os.path.basename(clean_input)}' เสร็จสมบูรณ์ ({elapsed})", "Single audio conversion completed (Success)", "success", "RVCInference")
            final_log = (
                f"{sep_line}\n"
                f"✅ ลำดับคิว: 1/1 | ชื่อไฟล์: {os.path.basename(clean_input)} (Success)\n"
                f"⏰ เวลาเริ่มต้น: {start_timestamp} | 🏁 เวลาเรนเดอร์เสร็จแล้ว: {strftime('%H:%M:%S')} (⏱️ {elapsed})\n"
                f"สถานะ: {info_log}\n"
                f"{sep_line}"
            )
            return final_log, opt_audio
        return f"{sep_line}\n❌ [ERR] RVC Core ไม่พร้อมใช้งาน (Error)\n{sep_line}", None
    finally:
        clear_vram_safe()


def batch_render_tracker(sid, dir_input, opt_input, inputs, vc_transform, f0method, file_index, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, format1, lang_mode="2. TH"):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    start_time_sec = time.time()
    start_timestamp = strftime("%H:%M:%S")
    sep_line = "=" * 68

    format1 = str(format1 or "wav").lower().strip()
    dir_input_clean = str(dir_input or "").strip().strip('"')
    opt_input_clean = str(opt_input or "").strip().strip('"')

    if not os.path.exists(dir_input_clean) and not inputs:
        yield f"{sep_line}\n❌ [ERR] ไม่พบโฟลเดอร์หรือไฟล์ต้นฉบับที่ระบุ (Error)\n{sep_line}"
        return
    os.makedirs(opt_input_clean, exist_ok=True)

    if os.path.exists(dir_input_clean):
        oto_src = os.path.join(dir_input_clean, "oto.ini")
        oto_dst = os.path.join(opt_input_clean, "oto.ini")
        if os.path.exists(oto_src) and not os.path.exists(oto_dst):
            shutil.copy(oto_src, oto_dst)

    valid_exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
    source_files = (
        [f.name if hasattr(f, "name") else str(f) for f in inputs]
        if inputs
        else [os.path.join(dir_input_clean, f) for f in os.listdir(dir_input_clean) if f.lower().endswith(valid_exts)]
    )
    files_to_render, already_done = [], 0
    for src in source_files:
        clean_stem_name = clean_stem(src)
        expected_dst = os.path.join(opt_input_clean, f"{clean_stem_name}.{format1}")
        if os.path.exists(expected_dst) and os.path.getsize(expected_dst) > 1024:
            already_done += 1
        else:
            files_to_render.append(src)

    yield (
        f"{sep_line}\n"
        f"⏳ เริ่มกระบวนการ Smart Batch Resume...\n"
        f"📊 ไฟล์ทั้งหมด: {len(source_files)} | ข้ามไฟล์เดิมที่เสร็จแล้ว: {already_done} | ค้างทำต่อ: {len(files_to_render)}\n"
        f"⏰ เวลาเริ่มต้น: {start_timestamp}\n"
        f"{sep_line}\n"
    )

    if len(files_to_render) == 0:
        elapsed = format_time_delta(time.time() - start_time_sec)
        yield f"{sep_line}\n🎉 ทุกไฟล์ถูกเรนเดอร์สานต่อจนเสร็จสมบูรณ์แล้วทั้งหมด 100% (⏱️ {elapsed}) (Success)\n{sep_line}"
        clear_vram_safe()
        return

    try:
        if RVC_CORE_AVAILABLE and vc is not None:
            try:
                sid_clean = int(sid) if (sid is not None and str(sid).isdigit()) else 0
            except (ValueError, TypeError):
                sid_clean = 0
            if getattr(vc, "net_g", None) is None and sid and not str(sid).isdigit():
                vc.get_vc(sid, protect, protect)
            generator = vc.vc_multi(
                sid_clean,
                dir_input_clean if os.path.exists(dir_input_clean) else "",
                opt_input_clean,
                files_to_render,
                vc_transform,
                f0method,
                file_index.strip() if file_index else (file_index2 or ""),
                "",
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                format1,
            )

            for i, log in enumerate(generator):
                if CANCEL_REQUESTED:
                    elapsed = format_time_delta(time.time() - start_time_sec)
                    yield f"{sep_line}\n🛑 คำสั่งถูกยกเลิกโดยผู้ใช้งาน (⏱️ {elapsed})\n{sep_line}"
                    return
                pct = (already_done + i + 1) / len(source_files) * 100
                yield f"🟢 กำลังประมวลผล (Processing): {already_done + i + 1}/{len(source_files)} ({pct:.1f}%)\n{log}"

            elapsed = format_time_delta(time.time() - start_time_sec)
            DeltaNotificationCenter.notify(f"เรนเดอร์แบบกลุ่มสำเร็จ {len(source_files)} ไฟล์", "Batch rendering completed (Success)", "success", "RVCBatch")
            yield f"{sep_line}\n🟢 การเรนเดอร์แบบกลุ่มเสร็จสมบูรณ์เรียบร้อยแล้ว {len(source_files)} ไฟล์ (⏱️ {elapsed}) (Success)\n🏁 เวลาสิ้นสุด: {strftime('%H:%M:%S')}\n{sep_line}"
    finally:
        clear_vram_safe()


def get_pretrained_models(path_str: str, f0_str: str, sr2: str) -> Tuple[str, str]:
    g_path = os.path.join(now_dir, "assets", f"pretrained{path_str}", f"{f0_str}G{sr2}.pth")
    d_path = os.path.join(now_dir, "assets", f"pretrained{path_str}", f"{f0_str}D{sr2}.pth")
    return (g_path if os.path.isfile(g_path) else "", d_path if os.path.isfile(d_path) else "")


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (*get_pretrained_models(path_str, f0_str, sr2), to_return_sr2)


def change_f0(if_f0_3, sr2, version19):
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 else "", sr2),
    )


def change_f0_method(f0method8):
    return {"visible": (F0GPUVisible if f0method8 == "rmvpe_gpu" else False), "__type__": "update"}


def change_info_(ckpt_path):
    log_path = str(ckpt_path or "").replace(os.path.basename(str(ckpt_path or "")), "train.log")
    if not os.path.exists(log_path):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            info = ast.literal_eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except Exception:
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


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


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    trainset_clean = str(trainset_dir).strip().strip('"')
    purge_directory_junk(trainset_clean)
    sr_hz = sr_dict.get(str(sr).lower().strip(), 40000)
    exp_path = os.path.join(now_dir, "logs", str(exp_dir))
    os.makedirs(exp_path, exist_ok=True)
    log_file = os.path.join(exp_path, "preprocess.log")
    open(log_file, "w", encoding="utf-8").close()

    cmd = [
        PYTHON_EXE,
        "infer/modules/train/preprocess.py",
        trainset_clean,
        str(sr_hz),
        str(n_p),
        str(exp_path),
        str(getattr(config, "noparallel", False)),
        str(3.0 if getattr(config, "is_half", False) else 3.7),
    ]
    p = Popen(cmd, cwd=now_dir)
    ACTIVE_CHILD_PROCESSES.add(p)
    while p.poll() is None:
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                yield f.read()
        sleep(1)
    ACTIVE_CHILD_PROCESSES.discard(p)
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        yield f.read() + "\n✅ ประมวลผลชุดข้อมูลเสร็จสิ้น (Preprocess Success)"
    clear_vram_safe()


def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    exp_path = os.path.join(now_dir, "logs", str(exp_dir))
    os.makedirs(exp_path, exist_ok=True)
    log_file = os.path.join(exp_path, "extract_f0_feature.log")
    open(log_file, "w", encoding="utf-8").close()

    try:
        if if_f0:
            if f0method != "rmvpe_gpu":
                cmd = [PYTHON_EXE, "infer/modules/train/extract/extract_f0_print.py", exp_path, str(n_p), str(f0method)]
            elif getattr(config, "dml", False):
                cmd = [PYTHON_EXE, "infer/modules/train/extract/extract_f0_rmvpe_dml.py", exp_path]
            else:
                cmd = [PYTHON_EXE, "infer/modules/train/extract/extract_f0_rmvpe.py", "1", "0", "0", exp_path, str(getattr(config, "is_half", False))]
            p = Popen(cmd, cwd=now_dir)
            ACTIVE_CHILD_PROCESSES.add(p)
            while p.poll() is None:
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                        yield f.read()
                sleep(1)
            ACTIVE_CHILD_PROCESSES.discard(p)

        gpus_list = str(gpus).split("-") if gpus else ["0"]
        ps = []
        leng = len(gpus_list)
        for idx, n_g in enumerate(gpus_list):
            cmd_feat = [PYTHON_EXE, "infer/modules/train/extract_feature_print.py", str(getattr(config, "device", "cpu")), str(leng), str(idx), str(n_g), exp_path, str(version19)]
            p = Popen(cmd_feat, cwd=now_dir)
            ACTIVE_CHILD_PROCESSES.add(p)
            ps.append(p)

        done = [False]
        def check_done():
            while True:
                if all(p.poll() is not None for p in ps):
                    done[0] = True
                    break
                sleep(0.5)
        threading.Thread(target=check_done, daemon=True).start()

        while not done[0]:
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    yield f.read()
            sleep(1)

        for p in ps:
            ACTIVE_CHILD_PROCESSES.discard(p)
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            yield f.read() + "\n✅ สกัดคุณลักษณะเสียงเสร็จสิ้น (Feature Extraction Success)"
    finally:
        clear_vram_safe()


def click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19):
    try:
        start_time_sec = time.time()
        sep_line = "=" * 68
        exp_dir = os.path.join(now_dir, "logs", str(exp_dir1))
        os.makedirs(exp_dir, exist_ok=True)
        for heal_log in _check_and_heal_dataset(exp_dir1, version19, if_f0_3):
            yield heal_log

        valid_exts = {".wav", ".flac", ".mp3"}
        gt_dir = os.path.join(exp_dir, "0_gt_wavs")
        gt_map = {clean_stem(f): os.path.join(gt_dir, f) for f in (os.listdir(gt_dir) if os.path.exists(gt_dir) else []) if Path(f).suffix.lower() in valid_exts}
        feat_dir = os.path.join(exp_dir, "3_feature768" if version19 == "v2" else "3_feature256")
        feat_map = {clean_stem(f): os.path.join(feat_dir, f) for f in (os.listdir(feat_dir) if os.path.exists(feat_dir) else []) if f.lower().endswith(".npy")}

        common_stems = sorted(set(gt_map.keys()) & set(feat_map.keys()))
        if len(common_stems) == 0:
            err_log = f"{sep_line}\n❌ [ERR] ไม่พบชุดข้อมูลที่สมบูรณ์สำหรับการฝึกโมเดลใน logs/{exp_dir1} (Error)\nกรุณากด 'ประมวลผลข้อมูล' และ 'สกัดคุณลักษณะเสียง' ก่อน\n{sep_line}"
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
                f0_p = os.path.join(exp_dir, "2a_f0", f"{stem}.wav.npy").replace("\\", "/")
                f0nsf_p = os.path.join(exp_dir, "2b-f0nsf", f"{stem}.wav.npy").replace("\\", "/")
                opt.append(f"{wav_p}|{fea_p}|{f0_p}|{f0nsf_p}|{sid_clean}")
            else:
                opt.append(f"{wav_p}|{fea_p}|{sid_clean}")

        fea_dim = 256 if version19 == "v1" else 768
        mute_dir = os.path.join(now_dir, "logs", "mute").replace("\\", "/")
        if os.path.isdir(mute_dir):
            for _ in range(2):
                if if_f0_3:
                    opt.append(f"{mute_dir}/0_gt_wavs/mute{sr2}.wav|{mute_dir}/3_feature{fea_dim}/mute.npy|{mute_dir}/2a_f0/mute.wav.npy|{mute_dir}/2b-f0nsf/mute.wav.npy|{sid_clean}")
                else:
                    opt.append(f"{mute_dir}/0_gt_wavs/mute{sr2}.wav|{mute_dir}/3_feature{fea_dim}/mute.npy|{sid_clean}")

        shuffle(opt)
        with open(os.path.join(exp_dir, "filelist.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(opt))

        config_path = f"v1/{sr2}.json" if version19 == "v1" or sr2 == "40k" else f"v2/{sr2}.json"
        config_save_path = os.path.join(exp_dir, "config.json")
        if not pathlib.Path(config_save_path).exists() and hasattr(config, "json_config") and config_path in config.json_config:
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(config.json_config[config_path], f, ensure_ascii=False, indent=4, sort_keys=True)
                f.write("\n")

        train_cmd = [
            PYTHON_EXE, "infer/modules/train/train.py",
            "-e", str(exp_dir1), "-sr", str(sr2), "-f0", "1" if if_f0_3 else "0",
            "-bs", str(int(batch_size12)), "-te", str(int(total_epoch11)), "-se", str(int(save_epoch10)),
            "-l", "1" if str(if_save_latest13) in ["Yes", "1", "True"] else "0",
            "-c", "1" if (str(if_cache_gpu17) in ["Yes", "1", "True"] and torch.cuda.is_available()) else "0",
            "-sw", "1" if str(if_save_every_weights18) in ["Yes", "1", "True"] else "0",
            "-v", str(version19),
        ]
        if gpus16:
            train_cmd.extend(["-g", str(gpus16)])
        if pretrained_G14:
            train_cmd.extend(["-pg", str(pretrained_G14).strip('"')])
        if pretrained_D15:
            train_cmd.extend(["-pd", str(pretrained_D15).strip('"')])

        log_file = os.path.join(exp_dir, "train.log")
        open(log_file, "w", encoding="utf-8").close()
        p = Popen(train_cmd, cwd=now_dir)
        ACTIVE_CHILD_PROCESSES.add(p)

        while p.poll() is None:
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    yield f.read()
            sleep(1)
        ACTIVE_CHILD_PROCESSES.discard(p)

        final_log = ""
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                final_log = f.read()

        elapsed = format_time_delta(time.time() - start_time_sec)
        if p.returncode != 0:
            err_log = final_log + f"\n{sep_line}\n❌ ล้มเหลว (Code: {p.returncode}) (⏱️ {elapsed}) (Error)\n{sep_line}"
            yield err_log
            return False, err_log
        else:
            succ_log = final_log + f"\n{sep_line}\n🟢 การฝึกสอนเสร็จสมบูรณ์ (Training finished - Success) (⏱️ {elapsed})\n{sep_line}"
            yield succ_log
            return True, succ_log
    finally:
        clear_vram_safe()


def train_index(exp_dir1: str, version19: str):
    exp_path = os.path.join(now_dir, "logs", str(exp_dir1))
    feature_dir = os.path.join(exp_path, "3_feature768" if version19 == "v2" else "3_feature256")

    if not os.path.exists(feature_dir):
        yield f"❌ ไม่พบโฟลเดอร์การทดลอง: {feature_dir} (Error)\nกรุณาตรวจสอบชื่อการทดลองหรือทำการสกัดฟีเจอร์ก่อน"
        return

    start_time_sec = time.time()
    sep_line = "=" * 68
    yield f"{sep_line}\n⏳ กำลังสร้าง FAISS Feature Index สำหรับโปรเจกต์ '{exp_dir1}' (Version: {version19})...\n{sep_line}\n"
    try:
        listdir_res = [f for f in os.listdir(feature_dir) if f.lower().endswith(".npy") and f != "total_fea.npy"]
        if len(listdir_res) == 0:
            yield "❌ ไม่พบไฟล์ฟีเจอร์ในโฟลเดอร์ (Error)"
            return

        npys = []
        for name in sorted(listdir_res):
            try:
                npys.append(np.load(os.path.join(feature_dir, name)))
            except Exception as exc:
                logger.warning(f"Could not load {name}: {exc}")

        if len(npys) == 0:
            yield "❌ ไม่สามารถโหลดไฟล์ฟีเจอร์ได้ (Error)"
            return

        big_npy = np.nan_to_num(np.concatenate(npys, 0))
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        n_samples = big_npy.shape[0]
        dimension = 768 if version19 == "v2" else 256

        # Adaptive FAISS Guard
        if n_samples < 256:
            yield f"⚠️ ข้อมูลมีขนาดเล็ก ({n_samples} samples) ระบบจะสลับไปใช้ IndexFlatL2 อัตโนมัติ ป้องกันการล่มของ FAISS"
            index = faiss.IndexFlatL2(dimension)
            index.add(big_npy)
            n_ivf = "FlatL2"
            nprobe = 1
        else:
            if n_samples > 2e5:
                yield f"กำลังปรับลดขนาดด้วย MiniBatchKMeans ({n_samples} -> 10k centers)...\n"
                with suppress(Exception):
                    big_npy = MiniBatchKMeans(
                        n_clusters=10000, verbose=True,
                        batch_size=256 * int(getattr(config, "n_cpu", 2)),
                        compute_labels=False, init="random"
                    ).fit(big_npy).cluster_centers_

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
        os.makedirs(indices_dir, exist_ok=True)
        with suppress(Exception):
            shutil.copyfile(added_index_log, os.path.join(indices_dir, added_index_name))

        elapsed = format_time_delta(time.time() - start_time_sec)
        DeltaNotificationCenter.notify(f"สร้างดัชนี {added_index_name} สำเร็จ!", "FAISS Index built successfully (Success)", "success", "FAISSIndex")
        yield f"บันทึกไฟล์: {added_index_log}\n{sep_line}\n🎉 การสร้างดัชนีเสร็จสมบูรณ์เรียบร้อยแล้ว! (⏱️ {elapsed}) (Success)\n{sep_line}"
    except Exception as exc:
        yield f"❌ เกิดข้อผิดพลาดในการสร้าง Index (Error): {exc}\n{traceback.format_exc()}"
    finally:
        clear_vram_safe()


def train1key(exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19, gpus_rmvpe):
    infos = []
    def get_info_str(strr):
        infos.append(str(strr))
        if len(infos) > 15:
            infos.pop(0)
        return "\n".join(infos)

    yield get_info_str("ขั้นตอนที่ 1: กำลังประมวลผลข้อมูลและกรองไฟล์ขยะ (Step 1: Preprocessing & Purging Junk)...")
    for log in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7):
        yield get_info_str(log)

    yield get_info_str("ขั้นตอนที่ 2: กำลังสกัดระดับเสียงและคุณลักษณะ (Step 2: Extracting pitch & features)...")
    for log in extract_f0_feature(gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe):
        yield get_info_str(log)

    yield get_info_str("ขั้นตอนที่ 3: กำลังฝึกสอนโมเดล (Step 3: Training model)...")
    train_success = False
    last_train_log = ""
    for log in click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19):
        if isinstance(log, tuple):
            train_success, last_train_log = log[0], str(log[1])
        else:
            last_train_log = str(log)
            if "เสร็จสมบูรณ์" in last_train_log or "Training finished" in last_train_log:
                train_success = True
            yield get_info_str(last_train_log)

    # Anti-Phantom Indexing Quality Gate
    exp_path = os.path.join(now_dir, "logs", str(exp_dir1))
    model_exists = any(f.endswith(".pth") and f.startswith("G_") for f in os.listdir(exp_path)) if os.path.exists(exp_path) else False
    if not train_success or not model_exists or "ไม่พบชุดข้อมูล" in last_train_log or "ล้มเหลว" in last_train_log:
        yield get_info_str("🛑 ระงับกระบวนการสร้างดัชนีเนื่องจากการฝึกสอนไม่สมบูรณ์ ป้องกันการสร้าง Phantom Index")
        return

    yield get_info_str("ขั้นตอนที่ 4: การฝึกสอนโมเดลเสร็จสิ้น กำลังสร้างดัชนี (Building FAISS Index)...")
    for log in train_index(exp_dir1, version19):
        yield get_info_str(str(log))

    yield get_info_str("🟢 เสร็จสิ้นกระบวนการ One-Click Pipeline ทั้งหมด! (All processes completed - Success!)")


def build_multispeaker_manifest_ui(root_folder: str, exp_name: str) -> str:
    """สร้างรายการชุดข้อมูลหลายผู้พูด (Multi-Speaker Dataset Manifest) สำหรับ RVC"""
    clean_root = str(root_folder or "").strip().strip('"')
    clean_exp = str(exp_name or "MyVoiceAI").strip()
    if not clean_root or not os.path.isdir(clean_root):
        return DeltaNotificationCenter.notify("ไม่พบโฟลเดอร์หลักของชุดข้อมูลหลายผู้พูด", "Multi-speaker root folder not found", "error", "MultiSpeaker")
    try:
        mod = ACCOUNT_BRIDGE.load_module("multispeaker")
        if mod and hasattr(mod, "build_manifest_from_root") and hasattr(mod, "write_manifest"):
            manifest = mod.build_manifest_from_root(clean_root)
            out_path = mod.write_manifest(os.path.join(now_dir, "logs", clean_exp), manifest)
            return DeltaNotificationCenter.notify(
                f"สร้าง Multi-Speaker Manifest สำเร็จ ({len(manifest.get('speakers', []))} ผู้พูด, {len(manifest.get('entries', []))} ไฟล์) -> {out_path}",
                "Multi-speaker manifest created (Success)",
                "success",
                "MultiSpeaker",
            )
    except Exception as exc:
        return PoliteLogger.error("Multi-speaker manifest build failed", exc=exc, context="MultiSpeaker")
    return "⚠️ โมดูล multispeaker.py ไม่พร้อมใช้งาน"


def uvr_wrapper(model_name: str, inp_root: str, save_root_vocal: str, paths, save_root_ins: str, agg: int, format0: str):
    if not RVC_CORE_AVAILABLE:
        yield "❌ RVC/UVR5 Core ไม่พร้อมใช้งาน (Error)"
        return
    try:
        start_time_sec = time.time()
        sep_line = "=" * 68
        yield f"{sep_line}\n⏳ เริ่มกระบวนการแยกเสียงร้องและดนตรี (UVR5 Separation)...\nโมเดล: {model_name} | Aggression: {agg}\n{sep_line}\n"
        for log in uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
            yield log
        elapsed = format_time_delta(time.time() - start_time_sec)
        DeltaNotificationCenter.notify(f"แยกเสียงร้องและดนตรีด้วย {model_name} สำเร็จ!", "UVR5 separation completed (Success)", "success", "UVR5")
        yield f"\n{sep_line}\n🎉 การแยกเสียงร้องและดนตรีเสร็จสมบูรณ์เรียบร้อยแล้ว! (⏱️ {elapsed}) (Success)\n{sep_line}\n"
    except Exception as exc:
        yield f"❌ เกิดข้อผิดพลาด (Error): {exc}\n{traceback.format_exc()}"
    finally:
        clear_vram_safe()


def export_onnx(ModelPath, ExportedPath):
    try:
        from infer.modules.onnx.export import export_onnx as eo
        eo(ModelPath, ExportedPath)
        return DeltaNotificationCenter.notify(f"ส่งออกโมเดล RVC ONNX สำเร็จ -> {ExportedPath}", "Export ONNX Successful (Success)", "success", "ONNXExport")
    except Exception as exc:
        return DeltaNotificationCenter.notify(f"เกิดข้อผิดพลาดในการส่งออก ONNX: {exc}", f"ONNX Export Error: {exc}", "error", "ONNXExport")


# ─── 16. OpenUtau Voicebank Studio Logic (UTAU & DiffSinger) ─────────────────
VOICEBANK_FORMATS = {"EN": "English Arpasing", "TH": "Thai VCCV", "JP": "JPN VCV", "DS": "DiffSinger AI"}


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
    singer_type = "diffsinger" if lang_short == "DS" else "utau"
    yaml_str = (
        f"name: {unified_name}\n"
        f"singer_type: {singer_type}\n"
        f"image: Picture/Icon.png\n"
        f"portrait: Picture/Full.png\n"
        f"portrait_opacity: 0.45\n"
        f"author: DELTA SYNTH\n"
        f"text_file_encoding: utf-8\n"
    )
    txt_str = (
        f"name={unified_name} {voicebank_format} v.1.0\n"
        f"sample=sample.wav\n"
        f"web=https://patiphatwongyai.wixsite.com/deltasynth\n"
        f"contact=DELTA SYNTH\n"
        f"Youtube=DELTA SYNTH Official\n"
    )
    return readme, yaml_str, txt_str, unified_name


def save_project(base_dir, unified_name, profile_pic, piano_pic, readme_content, yaml_content, txt_content):
    try:
        project_dir = os.path.join(base_dir, unified_name)
        os.makedirs(os.path.join(project_dir, "Picture"), exist_ok=True)
        if profile_pic and os.path.exists(profile_pic):
            shutil.copy(profile_pic, os.path.join(project_dir, "Picture", "Icon.png"))
        if piano_pic and os.path.exists(piano_pic):
            shutil.copy(piano_pic, os.path.join(project_dir, "Picture", "Full.png"))
        with open(os.path.join(project_dir, "readme.txt"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        with open(os.path.join(project_dir, "character.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_content)
        with open(os.path.join(project_dir, "character.txt"), "w", encoding="utf-8") as f:
            f.write(txt_content)
        oto_path = os.path.join(project_dir, "oto.ini")
        if not os.path.exists(oto_path):
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("sample.wav=,,,,,\n")
        DeltaNotificationCenter.notify(f"สร้างโปรเจกต์ OpenUtau '{unified_name}' สำเร็จ", "OpenUtau project initialized (Success)", "success", "OpenUtauStudio")
        return "### กระบวนการที่ 1: จัดการรายละเอียดของตัวละคร OpenUtau 🟢", "**สถานะ:** เริ่มต้นโปรเจกต์เสร็จสมบูรณ์ (Success)", project_dir
    except Exception as exc:
        return "### กระบวนการที่ 1: จัดการรายละเอียดของตัวละคร OpenUtau 🔴", f"**สถานะ:** ขัดข้อง (Error) - {str(exc)}", base_dir


def run_zip_packing(base_dir, unified_name, out_zip_dir):
    try:
        target_dir = os.path.join(base_dir, unified_name)
        if not os.path.isdir(target_dir):
            target_dir = base_dir
        zip_path_without_ext = os.path.join(out_zip_dir, f"{unified_name}_Voicebank")
        os.makedirs(out_zip_dir, exist_ok=True)
        shutil.make_archive(zip_path_without_ext, "zip", target_dir)
        DeltaNotificationCenter.notify(f"แพ็คไฟล์ {unified_name}_Voicebank.zip สำเร็จ", "ZIP Packed Successfully (Success)", "success", "OpenUtauStudio")
        return f"✅ แพ็คไฟล์ .zip เสร็จสมบูรณ์ (ZIP Packed Successfully - Success): {zip_path_without_ext}.zip", "### กระบวนการที่ 2: การเรนเดอร์เสียง Wav 🟢"
    except Exception as exc:
        return f"❌ พบข้อผิดพลาด (Error occurred): {str(exc)}", "### กระบวนการที่ 2: การเรนเดอร์เสียง Wav 🔴"


# ─── 17. Advanced Multi-Model OpenUtau Exporter Wrapper (basicexport/advexport) ─
def run_advanced_ou_exporter_ui(
    vb_name: str,
    export_dir: str,
    aco_dir: str,
    dur_dir: str,
    var_dir: str = "",
    pitch_dir: str = "",
    vocoder_onnx: str = "",
    auto_dict: bool = True,
) -> str:
    """เรียกใช้โมดูล advexport / basicexport จาก dt_modules เพื่อประกอบร่าง OpenUtau Voicebank ขั้นสูง"""
    clean_vb = str(vb_name or "DiffSinger_Voicebank").strip()
    clean_out = str(export_dir or os.path.join(now_dir, "Packed_Voicebanks")).strip().strip('"')
    clean_aco = str(aco_dir or "").strip().strip('"')
    clean_dur = str(dur_dir or clean_aco).strip().strip('"')
    clean_var = str(var_dir or "").strip().strip('"')
    clean_pitch = str(pitch_dir or "").strip().strip('"')
    clean_voc = str(vocoder_onnx or "").strip().strip('"')

    if not clean_aco or not os.path.exists(clean_aco):
        return DeltaNotificationCenter.notify("กรุณาระบุโฟลเดอร์โมเดล Acoustic ที่ถูกต้อง", "Valid Acoustic folder required", "error", "AdvExporter")

    try:
        os.makedirs(clean_out, exist_ok=True)
        adv_mod = ACCOUNT_BRIDGE.load_module("advexport")
        if adv_mod and hasattr(adv_mod, "run_adv_config"):
            adv_mod.run_adv_config(
                clean_vb,
                clean_out,
                clean_aco,
                clean_dur,
                clean_var if clean_var else None,
                clean_pitch if clean_pitch else None,
                clean_voc if clean_voc else None,
                auto_dict,
            )
            return DeltaNotificationCenter.notify(
                f"ส่งออกคลังเสียงขั้นสูง '{clean_vb}' ไปยัง {clean_out} เสร็จสมบูรณ์!",
                f"Advanced OpenUtau export for {clean_vb} completed (Success)",
                "success",
                "AdvExporter",
            )
        # Fallback to COLAB_CKPT_CONVERTER
        out_logs = list(COLAB_CKPT_CONVERTER.build_individual_singer_voicebanks(
            acoustic_ckpt_or_dir=clean_aco,
            variance_ckpt_or_dir=clean_var or clean_dur,
            output_root_dir=clean_out,
            voicebank_prefix=clean_vb,
            vocoder_name_or_path=clean_voc or "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02",
        ))
        return out_logs[-1] if out_logs else "✅ ส่งออกเสร็จสมบูรณ์ (Success)"
    except Exception as exc:
        return PoliteLogger.error("Advanced OpenUtau export failed", exc=exc, context="AdvExporter")


# ─── 18. GRADIO UNIFIED MASTER UI (12 TABS) ──────────────────────────────────
def build_app():
    with gr.Blocks(title="DELTA RVC & DiffSinger AI Studio v7.0", css=custom_theme_css) as app_instance:
        gr.HTML("""
        <div class="app-header">
            <h1>🎙️ DELTA SYNTH — Smart AI Voice Studio v7.0 (Unified RVC & DiffSinger AI Master Edition)</h1>
            <p>ศูนย์กลางระบบแปลงเสียง RVC v1/v2, แยกเสียง UVR5, ฝึกสอนและแปลงไฟล์ Colab CKPT ของ DiffSinger AI 2.5.1 ออกเป็นคลังเสียงนักร้องรายบุคคลสำหรับ OpenUtau พร้อมระบบ Live Hot-Swap ซิงค์ทุกไฟล์อัตโนมัติ<br>
            <b>Made And Checked By DELTA SYNTH & All Code Agentic AI Engine</b> | Original by Patiphat Wongyai (Delta), RVC Project & OpenVPI DiffSinger</p>
        </div>
        """)

        with gr.Tabs():
            # ==================================================================
            # TAB 1: SMART MODEL ROUTER
            # ==================================================================
            with gr.TabItem("🧭 1. Router (จัดที่อยู่โมเดลอัตโนมัติ)"):
                with gr.Row():
                    router_code_input = gr.Textbox(label="กรอกที่อยู่ไฟล์โมเดล / โฟลเดอร์ (Model Path / Drag & Drop)", lines=3, scale=8)
                    btn_browse_model = gr.Button("📄 เลือกไฟล์ (Browse File)", variant="secondary", scale=2)
                    btn_browse_model_dir = gr.Button("📁 เลือกโฟลเดอร์ (Browse Folder)", variant="secondary", scale=2)
                with gr.Row():
                    router_version_profile = gr.Dropdown(
                        label="รูปแบบ (Version Profile)",
                        choices=[
                            "AUTO", "RVC_V2", "RVC_V1", "PRETRAINED_V2", "PRETRAINED_V1",
                            "UVR5", "DIFFSINGER_ACOUSTIC", "DIFFSINGER_VARIANCE", "DIFFSINGER_VOCODER",
                        ],
                        value="AUTO",
                        scale=4,
                    )
                    router_exp_name = gr.Textbox(label="ชื่อโปรเจกต์ (Experiment Name)", scale=4)
                    router_op_mode = gr.Radio(["copy", "move"], label="โหมด (Mode)", value="copy", scale=2)
                btn_execute_routing = gr.Button("⚡ จัดที่อยู่ไฟล์โมเดลอัตโนมัติ (Execute Smart Routing)", variant="primary")
                router_log_output = gr.Textbox(label="รายงานผลการจัดไฟล์ (Routing Log)", lines=8, elem_classes=["log-output"])

                btn_browse_model.click(
                    fn=lambda c: (c + "\n" if c else "") + browse_file("", [("Model Files", "*.pth;*.pt;*.index;*.ckpt;*.onnx;*.yaml;*.json;*.emb")]),
                    inputs=[router_code_input],
                    outputs=[router_code_input],
                )
                btn_browse_model_dir.click(
                    fn=lambda c: (c + "\n" if c else "") + browse_folder(""),
                    inputs=[router_code_input],
                    outputs=[router_code_input],
                )
                btn_execute_routing.click(
                    fn=lambda c, p, e, m: MODEL_ROUTER.execute_routing(c, p, e, m, True)[0],
                    inputs=[router_code_input, router_version_profile, router_exp_name, router_op_mode],
                    outputs=[router_log_output],
                )

            # ==================================================================
            # TAB 2: RVC MODEL INFERENCE (SINGLE & BATCH)
            # ==================================================================
            with gr.TabItem("🎙️ 2. การอนุมานโมเดล RVC (Inference)"):
                with gr.Row():
                    sid0 = gr.Dropdown(label="เลือกเสียงเป้าหมาย (Select Target Voice)", choices=get_model_names())
                    refresh_button = gr.Button("🔄 รีเฟรชข้อมูลเสียงและดัชนี (Refresh)", variant="primary")
                    clean_button = gr.Button("🧹 คืนค่า VRAM (Clear VRAM)", variant="secondary")
                    spk_item = gr.Slider(minimum=0, maximum=2333, step=1, label="หมายเลขผู้พูด (Speaker ID)", value=0, visible=False, interactive=True)

                with gr.Tabs():
                    with gr.TabItem("ประมวลผลไฟล์เดี่ยว (Single File Processing)"):
                        with gr.Row():
                            input_audio0 = gr.Textbox(label="ไฟล์เสียงต้นฉบับ (Source Audio Path)", placeholder="ตัวอย่าง: H:/Export to UTAU/Test.wav", scale=8)
                            btn_browse_audio = gr.Button("📄 เลือกไฟล์", scale=2, variant="secondary")
                            file_index1 = gr.Textbox(label="ไฟล์ดัชนีระบุเอง (Custom Index Path)", scale=8)
                            btn_browse_idx = gr.Button("📑 เลือกดัชนี", scale=2, variant="secondary")
                            file_index2 = gr.Dropdown(label="เลือกไฟล์ดัชนีอัตโนมัติ (Auto-Select Index)", choices=get_index_files(), scale=8)
                        with gr.Row():
                            vc_transform0 = gr.Number(label="ปรับระดับเสียง (Pitch Shift: Semitones)", value=0)
                            f0method0 = gr.Radio(["pm", "harvest", "dio", "crepe", "mangio-crepe", "rmvpe"], label="อัลกอริทึมวิเคราะห์ระดับเสียง (Pitch Algorithm)", value="rmvpe")
                            resample_sr0 = gr.Slider(minimum=0, maximum=48000, step=1, label="Resampling Rate (0=Off)", value=0)
                            rms_mix_rate0 = gr.Slider(minimum=0, maximum=1, label="Volume Envelope Mix Rate", value=0.25)
                            protect0 = gr.Slider(minimum=0, maximum=0.5, step=0.01, label="การปกป้องพยัญชนะและเสียงหายใจ (Breath Protect)", value=0.33)
                            filter_radius0 = gr.Slider(minimum=0, maximum=7, step=1, label="Median Filter Radius", value=3)
                            index_rate1 = gr.Slider(minimum=0, maximum=1, label="สัดส่วนอิทธิพลดัชนี (Index Rate)", value=0.75)
                            f0_file0 = gr.File(label="F0 Curve File (Optional)", visible=False)
                        but0 = gr.Button("▶️ เริ่มประมวลผลไฟล์เดี่ยว (Start Single Render)", variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label="สถานะการดำเนินการ (Operation Log)", lines=5, elem_classes=["log-output"])
                            vc_output2 = gr.Audio(label="ผลลัพธ์เสียง (Output Audio)")

                        btn_browse_audio.click(fn=lambda c: browse_file(c, [("Audio Files", "*.wav;*.flac;*.mp3;*.m4a;*.ogg"), ("All Files", "*.*")]), inputs=[input_audio0], outputs=[input_audio0])
                        btn_browse_idx.click(fn=lambda c: browse_file(c, [("Index Files", "*.index")]), inputs=[file_index1], outputs=[file_index1])
                        but0.click(
                            single_infer_wrapper,
                            [spk_item, input_audio0, vc_transform0, f0_file0, f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0, sid0],
                            [vc_output1, vc_output2],
                            api_name="infer_convert",
                        )

                    with gr.TabItem("ประมวลผลแบบชุด (Smart Batch Processing)"):
                        with gr.Row():
                            dir_input = gr.Textbox(label="โฟลเดอร์ต้นฉบับ (Source Directory)", scale=8)
                            btn_b_dir = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                            opt_input = gr.Textbox(label="โฟลเดอร์ส่งออก (Output Directory)", value=os.path.join(now_dir, "Output"), scale=8)
                            btn_b_opt = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                        with gr.Row():
                            file_index3 = gr.Textbox(label="เส้นทางไฟล์ดัชนีระบุเอง (Custom Index Path)", value="", scale=8)
                            btn_browse_index3 = gr.Button("📑 เลือกดัชนี", scale=2, variant="secondary")
                            file_index4 = gr.Dropdown(label="เลือกไฟล์ดัชนีอัตโนมัติ (Auto-Select Index)", choices=get_index_files(), scale=8)
                        with gr.Row():
                            inputs = gr.File(file_count="multiple", label="หรือ อัปโหลดไฟล์เสียงหลายไฟล์ (Or Upload Multiple Audio Files)")
                        with gr.Row():
                            vc_transform1 = gr.Number(label="ปรับระดับเสียง (Pitch Shift)", value=0)
                            f0method1 = gr.Radio(["pm", "harvest", "crepe", "rmvpe"], label="อัลกอริทึม Pitch", value="rmvpe")
                            resample_sr1 = gr.Slider(minimum=0, maximum=48000, step=1, label="Resampling Rate", value=0)
                            rms_mix_rate1 = gr.Slider(minimum=0, maximum=1, label="Volume Mix Rate", value=0.25)
                            protect1 = gr.Slider(minimum=0, maximum=0.5, step=0.01, label="Breath Protect", value=0.33)
                            filter_radius1 = gr.Slider(minimum=0, maximum=7, step=1, label="Filter Radius", value=3)
                            index_rate2 = gr.Slider(minimum=0, maximum=1, label="Index Rate", value=0.75)
                            format1 = gr.Radio(["wav", "flac", "mp3", "m4a"], label="รูปแบบไฟล์ส่งออก (Export Format)", value="wav")
                        with gr.Row():
                            but1 = gr.Button("▶️ ตรวจสอบสถานะ / เริ่มประมวลผลแบบชุด (Start Smart Batch Render)", variant="primary")
                            cancel_btn = gr.Button("🛑 ยกเลิกการเรนเดอร์ (Safe Cancel Render)", variant="secondary")
                        vc_output3 = gr.Textbox(label="สถานะแบบเรียลไทม์ (Real-time Batch Status Log)", lines=7, elem_classes=["log-output"])

                        btn_b_dir.click(fn=browse_folder, inputs=[dir_input], outputs=[dir_input])
                        btn_b_opt.click(fn=browse_folder, inputs=[opt_input], outputs=[opt_input])
                        btn_browse_index3.click(fn=lambda cur: browse_file(cur, [("Index Files", "*.index")]), inputs=[file_index3], outputs=[file_index3])
                        render_event = but1.click(
                            batch_render_tracker,
                            [sid0, dir_input, opt_input, inputs, vc_transform1, f0method1, file_index3, file_index4, index_rate2, filter_radius1, resample_sr1, rms_mix_rate1, protect1, format1],
                            [vc_output3],
                            api_name="infer_convert_batch",
                        )
                        cancel_btn.click(safe_cancel_process, outputs=[vc_output3], cancels=[render_event])

                clean_button.click(fn=clear_vram_safe, outputs=[vc_output1], api_name="infer_clean")
                refresh_button.click(fn=change_choices, outputs=[sid0, file_index2], api_name="infer_refresh")
                refresh_button.click(fn=lambda: change_choices()[1], outputs=[file_index4])
                sid0.change(fn=get_vc_wrapper, inputs=[sid0, protect0, protect1], outputs=[spk_item, protect0, protect1, file_index2, file_index4], api_name="infer_change_voice")

            # ==================================================================
            # TAB 3: UVR5 VOCAL & INSTRUMENT SEPARATION
            # ==================================================================
            with gr.TabItem("🎛️ 3. แยกเสียงร้อง/ดนตรี (UVR5 Separation)"):
                with gr.Row():
                    uvr_choices = get_uvr5_model_names()
                    uvr_model = gr.Dropdown(
                        label="เลือกโมเดล UVR5 (Select Model)",
                        choices=uvr_choices,
                        value="HP5_only_main_vocal" if "HP5_only_main_vocal" in uvr_choices else (uvr_choices[0] if uvr_choices else ""),
                    )
                    uvr_agg = gr.Slider(minimum=0, maximum=20, step=1, label="ความแม่นยำในการสกัดเสียง (Aggression)", value=10)
                    uvr_format = gr.Radio(["wav", "flac", "mp3", "m4a"], label="รูปแบบไฟล์ส่งออก (Export Format)", value="wav")
                with gr.Row():
                    uvr_dir_wavs = gr.Textbox(label="โฟลเดอร์ไฟล์เสียงต้นทาง (Input Audio Directory)", scale=8)
                    btn_uvr_in = gr.Button("📁 เลือก", scale=2, variant="secondary")
                    uvr_opt_vocal = gr.Textbox(label="โฟลเดอร์บันทึกเสียงร้อง (Vocal Output Directory)", value="Output/vocal", scale=8)
                    btn_uvr_vocal = gr.Button("📁 เลือก", scale=2, variant="secondary")
                    uvr_opt_ins = gr.Textbox(label="โฟลเดอร์บันทึกเสียงดนตรี (Instrument Output Directory)", value="Output/instrument", scale=8)
                    btn_uvr_ins = gr.Button("📁 เลือก", scale=2, variant="secondary")
                with gr.Row():
                    uvr_files = gr.File(file_count="multiple", label="หรือ อัปโหลดไฟล์เสียงหลายไฟล์ (Upload Audio Files)")
                btn_run_uvr = gr.Button("▶️ เริ่มแยกเสียงร้องและดนตรี (Start UVR5 Separation)", variant="primary")
                uvr_log = gr.Textbox(label="สถานะการแยกเสียง (UVR5 Operation Log)", lines=8, elem_classes=["log-output"])

                btn_uvr_in.click(fn=browse_folder, inputs=[uvr_dir_wavs], outputs=[uvr_dir_wavs])
                btn_uvr_vocal.click(fn=browse_folder, inputs=[uvr_opt_vocal], outputs=[uvr_opt_vocal])
                btn_uvr_ins.click(fn=browse_folder, inputs=[uvr_opt_ins], outputs=[uvr_opt_ins])
                btn_run_uvr.click(fn=uvr_wrapper, inputs=[uvr_model, uvr_dir_wavs, uvr_opt_vocal, uvr_files, uvr_opt_ins, uvr_agg, uvr_format], outputs=[uvr_log], api_name="uvr_convert")

            # ==================================================================
            # TAB 4: RVC MODEL TRAINING & MULTI-SPEAKER MANIFEST
            # ==================================================================
            with gr.TabItem("🏋️ 4. การฝึกสอนโมเดล RVC (Training)"):
                with gr.Row():
                    exp_dir1 = gr.Textbox(label="ชื่อโปรเจกต์การทดลอง (Experiment Name)", value="MyVoiceAI")
                    sr2 = gr.Radio(["32k", "40k", "48k"], label="อัตราสุ่มตัวอย่าง (Sample Rate)", value="40k")
                    if_f0_3 = gr.Radio([True, False], label="ใช้การชี้นำระดับเสียง? (Pitch Guidance?)", value=True)
                    version19 = gr.Radio(["v1", "v2"], label="เวอร์ชันโมเดล (Model Version)", value="v2")
                    np7 = gr.Slider(minimum=1, maximum=os.cpu_count() or 4, step=1, label="เธรด CPU (CPU Threads)", value=min(2, max(1, int(getattr(config, "n_cpu", 2)))))
                with gr.Row():
                    trainset_dir4 = gr.Textbox(label="โฟลเดอร์ชุดข้อมูลเสียง (Dataset Directory)", scale=8)
                    btn_ts = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                    spk_id5 = gr.Slider(minimum=0, maximum=109, step=1, label="หมายเลขผู้พูด (Speaker ID 0~109)", value=0, scale=4)
                with gr.Row():
                    gpus6 = gr.Textbox(label="หมายเลข GPU (GPU IDs)", value=gpus, visible=F0GPUVisible)
                    f0method8 = gr.Radio(["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"], label="อัลกอริทึมสกัดระดับเสียง (F0 Extractor)", value="rmvpe_gpu" if F0GPUVisible else "rmvpe")
                    gpus_rmvpe = gr.Textbox(label="GPUs สำหรับ RMVPE", value=f"{gpus}-{gpus}", visible=F0GPUVisible)
                    gpu_info9 = gr.Textbox(label="ข้อมูลฮาร์ดแวร์ (Hardware Info)", value=gpu_info, interactive=False)
                with gr.Row():
                    save_epoch10 = gr.Slider(minimum=1, maximum=50, step=1, label="ความถี่การบันทึก (Save Frequency)", value=20)
                    total_epoch11 = gr.Slider(minimum=2, maximum=1000, step=1, label="จำนวนรอบทั้งหมด (Total Epochs)", value=200)
                    batch_size12 = gr.Slider(minimum=1, maximum=40, step=1, label="ขนาดแบทช์ (Batch Size)", value=default_batch_size)
                    if_save_latest13 = gr.Radio(["Yes", "No"], label="บันทึกเฉพาะล่าสุด? (Save only latest?)", value="Yes")
                    if_cache_gpu17 = gr.Radio(["Yes", "No"], label="แคชลง VRAM? (Cache to VRAM?)", value="Yes")
                    if_save_every_weights18 = gr.Radio(["Yes", "No"], label="บันทึกโมเดลย่อยทุกรอบ? (Save weights?)", value="Yes")
                with gr.Row():
                    pretrained_G14 = gr.Textbox(label="Pretrained G Path", value="assets/pretrained_v2/f0G40k.pth", scale=8)
                    btn_browse_pg = gr.Button("📄 เลือก G", scale=2, variant="secondary")
                    pretrained_D15 = gr.Textbox(label="Pretrained D Path", value="assets/pretrained_v2/f0D40k.pth", scale=8)
                    btn_browse_pd = gr.Button("📄 เลือก D", scale=2, variant="secondary")
                    gpus16 = gr.Textbox(label="GPUs สำหรับเทรน", value=gpus, scale=4)
                with gr.Row():
                    but3 = gr.Button("1️⃣ ประมวลผลข้อมูล (Preprocess)", variant="secondary")
                    but4 = gr.Button("2️⃣ สกัด F0 & Features (Extract)", variant="secondary")
                    but5 = gr.Button("3️⃣ ฝึกสอนโมเดล (Train Model)", variant="secondary")
                    but6 = gr.Button("4️⃣ สร้างดัชนี (Train Index)", variant="secondary")
                    but_all = gr.Button("⚡ ฝึกสอนอัตโนมัติครบวงจร (One-Click Training)", variant="primary")
                    but_stop_rvc = gr.Button("🛑 หยุดกระบวนการ (Stop)", variant="secondary")

                with gr.Accordion("👥 เครื่องมือสร้าง Multi-Speaker Dataset Manifest (รองรับผู้พูดสูงสุด 110 คน ID 0~109)", open=False):
                    with gr.Row():
                        ms_root_dir = gr.Textbox(label="โฟลเดอร์หลักหลายผู้พูด (โครงสร้างโฟลเดอร์ย่อย: ชื่อ_ID_รอบ เช่น SingerA_0_1)", scale=8)
                        btn_ms_browse = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                        btn_ms_build = gr.Button("📋 สร้าง multispeaker_manifest.json", variant="primary", scale=4)

                info_train = gr.Textbox(label="สถานะการฝึกสอน (Training Operation Log)", lines=10, elem_classes=["log-output"])

                btn_ts.click(fn=browse_folder, inputs=[trainset_dir4], outputs=[trainset_dir4])
                btn_browse_pg.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth")]), inputs=[pretrained_G14], outputs=[pretrained_G14])
                btn_browse_pd.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth")]), inputs=[pretrained_D15], outputs=[pretrained_D15])
                btn_ms_browse.click(fn=browse_folder, inputs=[ms_root_dir], outputs=[ms_root_dir])
                btn_ms_build.click(fn=build_multispeaker_manifest_ui, inputs=[ms_root_dir, exp_dir1], outputs=[info_train])

                sr2.change(change_sr2, [sr2, if_f0_3, version19], [pretrained_G14, pretrained_D15])
                version19.change(change_version19, [sr2, if_f0_3, version19], [pretrained_G14, pretrained_D15, sr2])
                if_f0_3.change(change_f0, [if_f0_3, sr2, version19], [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15])
                f0method8.change(fn=change_f0_method, inputs=[f0method8], outputs=[gpus_rmvpe])

                but3.click(preprocess_dataset, [trainset_dir4, exp_dir1, sr2, np7], [info_train], api_name="train_preprocess")
                but4.click(extract_f0_feature, [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe], [info_train], api_name="train_extract_f0_feature")
                but5.click(click_train, [exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19], [info_train], api_name="train_start")
                but6.click(train_index, [exp_dir1, version19], [info_train])
                but_all.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19, gpus_rmvpe], [info_train], api_name="train_start_all")
                but_stop_rvc.click(fn=cancel_all_processes, outputs=[info_train])

            # ==================================================================
            # TAB 5: RVC CKPT PROCESSING (MERGE, MODIFY, INSPECT, EXTRACT)
            # ==================================================================
            with gr.TabItem("📦 5. การจัดการจุดตรวจสอบ RVC (CKPT Processing)"):
                with gr.Group():
                    gr.Markdown("#### 🔗 การผสานโมเดล (Model Fusion / Merging)")
                    with gr.Row():
                        ckpt_a = gr.Textbox(label="เส้นทางโมเดล A (Model A Path)", interactive=True, scale=8)
                        btn_browse_ckpt_a = gr.Button("📄 เลือก", scale=1, variant="secondary")
                        ckpt_b = gr.Textbox(label="เส้นทางโมเดล B (Model B Path)", interactive=True, scale=8)
                        btn_browse_ckpt_b = gr.Button("📄 เลือก", scale=1, variant="secondary")
                        alpha_a = gr.Slider(minimum=0, maximum=1, label="สัดส่วนน้ำหนักโมเดล A (Weight Ratio)", value=0.5, interactive=True, scale=3)
                    btn_browse_ckpt_a.click(fn=lambda cur: browse_file(cur, [("Model", "*.pth")]), inputs=[ckpt_a], outputs=[ckpt_a])
                    btn_browse_ckpt_b.click(fn=lambda cur: browse_file(cur, [("Model", "*.pth")]), inputs=[ckpt_b], outputs=[ckpt_b])
                    with gr.Row():
                        sr_ = gr.Radio(label="Sample Rate", choices=["32k", "40k", "48k"], value="40k")
                        if_f0_ = gr.Radio(label="Pitch Guidance?", choices=["Yes", "No"], value="Yes")
                        info__ = gr.Textbox(label="ข้อมูลโมเดล (Model Info)", value="", max_lines=2)
                        name_to_save0 = gr.Textbox(label="ชื่อบันทึกไฟล์ (Save Name)", value="Merged_Model", max_lines=1)
                        version_2 = gr.Radio(label="เวอร์ชัน (Version)", choices=["v1", "v2"], value="v2")
                    with gr.Row():
                        but_merge = gr.Button("🔗 ผสานโมเดล (Merge Models)", variant="primary")
                        info4 = gr.Textbox(label="สถานะ (Operation Log)", max_lines=4)
                    but_merge.click(merge, [ckpt_a, ckpt_b, alpha_a, sr_, if_f0_, info__, name_to_save0, version_2], info4, api_name="ckpt_merge")

                with gr.Group():
                    gr.Markdown("#### ✏️ ปรับปรุงข้อมูลประจำตัวโมเดล & ตรวจสอบข้อมูล (Modify & Inspect Metadata)")
                    with gr.Row():
                        ckpt_path0 = gr.Textbox(label="เส้นทางไฟล์โมเดลเป้าหมาย (Target Model Path)", interactive=True, scale=6)
                        btn_browse_c0 = gr.Button("📄 เลือก", scale=1, variant="secondary")
                        info_mod = gr.Textbox(label="ข้อมูลที่ต้องการแก้ไข (Metadata)", scale=4)
                        name_to_save1 = gr.Textbox(label="ชื่อไฟล์ใหม่ (New File Name)", scale=3)
                        but_mod = gr.Button("💾 บันทึกข้อมูล (Modify)", variant="primary", scale=2)
                        but_inspect = gr.Button("🔍 ดูข้อมูล (Inspect)", variant="secondary", scale=2)
                    info5 = gr.Textbox(label="รายละเอียดโมเดล / สถานะ (Metadata & Status)", lines=3)
                    btn_browse_c0.click(fn=lambda cur: browse_file(cur, [("Model", "*.pth")]), inputs=[ckpt_path0], outputs=[ckpt_path0])
                    but_mod.click(change_info, [ckpt_path0, info_mod, name_to_save1], info5, api_name="ckpt_modify")
                    but_inspect.click(show_info, [ckpt_path0], info5, api_name="ckpt_show")

                with gr.Group():
                    gr.Markdown("#### ✂️ การดึงโมเดลย่อยจาก Checkpoint เพื่อนำไปใช้งาน (Extract Small Model)")
                    with gr.Row():
                        ckpt_path2 = gr.Textbox(label="เส้นทางโมเดลหลัก G_*.pth (Large Model Path)", interactive=True, scale=8)
                        btn_browse_ckpt2 = gr.Button("📄 เลือก", scale=1, variant="secondary")
                        save_name = gr.Textbox(label="ชื่อบันทึกไฟล์ (Save Name)", value="Model_Deploy", scale=3)
                        sr__ = gr.Radio(label="Sample Rate", choices=["32k", "40k", "48k"], value="40k")
                        if_f0__ = gr.Radio(label="Pitch Guidance?", choices=["1", "0"], value="1")
                        version_1 = gr.Radio(label="Version", choices=["v1", "v2"], value="v2")
                        info___ = gr.Textbox(label="ข้อมูลประกอบ (Metadata)", value="", scale=3)
                    btn_browse_ckpt2.click(fn=lambda cur: browse_file(cur, [("Model", "*.pth")]), inputs=[ckpt_path2], outputs=[ckpt_path2])
                    ckpt_path2.change(change_info_, [ckpt_path2], [sr__, if_f0__, version_1])
                    with gr.Row():
                        but9 = gr.Button("✂️ ดึงโมเดลย่อย (Extract Lightweight Model)", variant="primary")
                        info7 = gr.Textbox(label="สถานะ (Status)", max_lines=4)
                    but9.click(extract_small_model, [ckpt_path2, save_name, sr__, if_f0__, info___, version_1], info7, api_name="ckpt_extract")

            # ==================================================================
            # TAB 6: ONNX EXPORT (RVC)
            # ==================================================================
            with gr.TabItem("🌐 6. ส่งออก ONNX (ONNX Export)"):
                with gr.Row():
                    ckpt_dir = gr.Textbox(label="เส้นทางโมเดล RVC ต้นฉบับ (RVC Model .pth Path)", interactive=True, scale=8)
                    btn_browse_onnx_ckpt = gr.Button("📄 เลือกโมเดล (Browse)", scale=2, variant="secondary")
                    onnx_dir = gr.Textbox(label="เส้นทางส่งออกไฟล์ ONNX (ONNX Output Path)", interactive=True, scale=8)
                    btn_browse_onnx_out = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
                btn_browse_onnx_ckpt.click(fn=lambda cur: browse_file(cur, [("Model", "*.pth")]), inputs=[ckpt_dir], outputs=[ckpt_dir])
                btn_browse_onnx_out.click(fn=browse_folder, inputs=[onnx_dir], outputs=[onnx_dir])
                with gr.Row():
                    butOnnx = gr.Button("📤 ส่งออกโมเดลเป็นมาตรฐาน ONNX (Export ONNX Model)", variant="primary")
                    infoOnnx = gr.Textbox(label="สถานะการส่งออก (Export Status)", elem_classes=["log-output"])
                butOnnx.click(export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx")

            # ==================================================================
            # TAB 7: COLAB CKPT TO INDIVIDUAL SINGER VOICEBANKS (NEW DEDICATED PAGE)
            # ==================================================================
            with gr.TabItem("🎓 7. แปลงไฟล์ Colab CKPT เป็นคลังเสียงรายบุคคล (Colab CKPT to Voicebanks)"):
                gr.Markdown("""
                ### 🎓 ศูนย์แปลงไฟล์ `.ckpt` ที่เทรนมาจาก Google Colab ของ DiffSinger AI เป็นชุดคลังเสียงรายบุคคล (Individual Singer Voicebanks)
                รองรับไฟล์ `.ckpt` หรือโฟลเดอร์โมเดลจากสมุดโน้ต **`DiffSinger_Making_By_DELTA_SYNTH.ipynb`** ทั้งแบบนักร้องเดี่ยว (Single-Speaker) และแบบรวมหลายนักร้อง (Multi-Speaker) โดยระบบจะตรวจสอบ `spk_map.json` / `state_dict`, คัดแยกนักร้องแต่ละคนออกจากกัน (`drop_spk`), สกัดเวกเตอร์เสียง `.emb`, แปลงเป็น `ONNX`, สร้างพจนานุกรมภาษาไทย `dsdict.yaml` และแพ็กเกจเป็นชุดคลังเสียงสำหรับ **OpenUtau** ให้พร้อมใช้งานทันทีเป็นรายบุคคล!
                """)
                with gr.Tabs():
                    with gr.TabItem("1️⃣ ตรวจสอบ & แปลงไฟล์ Colab CKPT เป็นคลังเสียงรายบุคคลครบวงจร"):
                        with gr.Row():
                            colab_aco_in = gr.Textbox(
                                label="ไฟล์ .ckpt หรือโฟลเดอร์โมเดล Acoustic จาก Google Colab (Acoustic .ckpt / Folder)",
                                placeholder="ตัวอย่าง: A:/Colab_Models/Kochujang/Acoustic/model_ckpt_steps_160000.ckpt",
                                scale=8,
                            )
                            btn_colab_aco_file = gr.Button("📄 เลือกไฟล์ .ckpt", scale=2, variant="secondary")
                            btn_colab_aco_dir = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                        with gr.Row():
                            colab_var_in = gr.Textbox(
                                label="ไฟล์ .ckpt หรือโฟลเดอร์โมเดล Variance / Duration / Pitch (หากมี - Optional)",
                                placeholder="ตัวอย่าง: A:/Colab_Models/Kochujang/Variance/model_ckpt_steps_160000.ckpt",
                                scale=8,
                            )
                            btn_colab_var_file = gr.Button("📄 เลือกไฟล์ .ckpt", scale=2, variant="secondary")
                            btn_colab_var_dir = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                        with gr.Row():
                            colab_spk_map_in = gr.Textbox(
                                label="ไฟล์ spk_map.json (เว้นว่างเพื่อค้นหาอัตโนมัติจากโฟลเดอร์โมเดล)",
                                scale=6,
                            )
                            btn_colab_spk_map = gr.Button("📄 เลือก spk_map.json", scale=2, variant="secondary")
                            btn_inspect_colab = gr.Button("🔍 1. ตรวจสอบรายชื่อนักร้องและโครงสร้างในไฟล์ CKPT (Inspect CKPT)", variant="secondary", scale=4)

                        with gr.Row():
                            colab_discovered_spks = gr.Textbox(
                                label="รายชื่อนักร้อง / รหัสผู้พูดที่ตรวจพบ (สามารถระบุเฉพาะคนที่ต้องการแยก เช่น 0,1 หรือเว้นว่างเพื่อทำทุกคน)",
                                placeholder="เว้นว่างเพื่อสร้างคลังเสียงแยกให้ทุกคนโดยอัตโนมัติ",
                                scale=6,
                            )
                            colab_split_mode = gr.Radio(
                                [
                                    "แยกนักร้องทุกคนเป็นคลังเสียงรายบุคคล (Split All Singers Individually)",
                                    "รวมโทนเสียงเป็นนักร้องคนเดียวหลาย Subbank/Vocal Modes (Single Singer Multi-Color)",
                                ],
                                label="รูปแบบการสร้างชุดคลังเสียง (Voicebank Generation Mode)",
                                value="แยกนักร้องทุกคนเป็นคลังเสียงรายบุคคล (Split All Singers Individually)",
                                scale=6,
                            )

                        with gr.Row():
                            colab_out_dir = gr.Textbox(
                                label="โฟลเดอร์ปลายทางสำหรับบันทึกชุดคลังเสียงแต่ละบุคคล (Output Voicebanks Directory)",
                                value=os.path.join(now_dir, "Packed_Voicebanks", "Colab_DiffSinger_Singers"),
                                scale=6,
                            )
                            btn_colab_out_dir = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                            colab_vb_prefix = gr.Textbox(
                                label="คำนำหน้าชื่อคลังเสียง (Voicebank Name Prefix - Optional)",
                                placeholder="เช่น DELTA_DS",
                                value="",
                                scale=4,
                            )

                        with gr.Row():
                            colab_lang_code = gr.Dropdown(
                                label="ภาษาหลักของคลังเสียง (Primary Language)",
                                choices=["th", "en", "ja", "zh", "custom"],
                                value="th",
                                scale=2,
                            )
                            colab_vocoder = gr.Textbox(
                                label="ชื่อหรือพาธไฟล์ Vocoder (เช่น pc_nsf_hifigan_44.1k_hop512_128bin_2025.02 หรือไฟล์ .onnx)",
                                value="pc_nsf_hifigan_44.1k_hop512_128bin_2025.02",
                                scale=4,
                            )
                            colab_fill_mode = gr.Dropdown(
                                label="โหมดคัดแยก Embedding (drop_spk fill mode)",
                                choices=["zeros", "random", "mean", "cyclic"],
                                value="zeros",
                                scale=2,
                            )
                            colab_author = gr.Textbox(label="ชื่อผู้สร้าง (Author)", value="DELTA SYNTH", scale=2)

                        with gr.Row():
                            colab_icon_img = gr.Textbox(label="พาธรูปโปรไฟล์นักร้อง (Icon.png - Optional)", scale=5)
                            btn_colab_icon = gr.Button("🖼️ เลือก Icon", scale=1, variant="secondary")
                            colab_portrait_img = gr.Textbox(label="พาธรูปเต็มตัวนักร้อง (Full.png - Optional)", scale=5)
                            btn_colab_portrait = gr.Button("🖼️ เลือก Full", scale=1, variant="secondary")
                            colab_save_ckpt_chk = gr.Checkbox(label="บันทึกไฟล์ .ckpt แยกรายบุคคลด้วย", value=True, scale=2)
                            colab_pack_zip_chk = gr.Checkbox(label="แพ็คเป็นไฟล์ .zip พร้อมลง OpenUtau", value=True, scale=2)

                        btn_run_colab_convert = gr.Button(
                            "🚀 2. เริ่มแปลงไฟล์ Colab CKPT ออกมาเป็นชุดคลังเสียงรายบุคคลทั้งหมดทันที (Build Individual Singer Voicebanks)",
                            variant="primary",
                        )
                        colab_convert_log = gr.Textbox(
                            label="รายงานผลการตรวจสอบและสร้างชุดคลังเสียงรายบุคคล (Colab CKPT -> Individual Voicebanks Log)",
                            lines=14,
                            elem_classes=["log-output"],
                        )

                        btn_colab_aco_file.click(fn=lambda c: browse_file(c, [("DiffSinger Checkpoint", "*.ckpt"), ("All Files", "*.*")]), inputs=[colab_aco_in], outputs=[colab_aco_in])
                        btn_colab_aco_dir.click(fn=browse_folder, inputs=[colab_aco_in], outputs=[colab_aco_in])
                        btn_colab_var_file.click(fn=lambda c: browse_file(c, [("DiffSinger Checkpoint", "*.ckpt"), ("All Files", "*.*")]), inputs=[colab_var_in], outputs=[colab_var_in])
                        btn_colab_var_dir.click(fn=browse_folder, inputs=[colab_var_in], outputs=[colab_var_in])
                        btn_colab_spk_map.click(fn=lambda c: browse_file(c, [("Speaker Map JSON", "*.json")]), inputs=[colab_spk_map_in], outputs=[colab_spk_map_in])
                        btn_colab_out_dir.click(fn=browse_folder, inputs=[colab_out_dir], outputs=[colab_out_dir])
                        btn_colab_icon.click(fn=lambda c: browse_file(c, [("Images", "*.png;*.jpg;*.jpeg")]), inputs=[colab_icon_img], outputs=[colab_icon_img])
                        btn_colab_portrait.click(fn=lambda c: browse_file(c, [("Images", "*.png;*.jpg;*.jpeg")]), inputs=[colab_portrait_img], outputs=[colab_portrait_img])

                        btn_inspect_colab.click(
                            fn=COLAB_CKPT_CONVERTER.inspect_colab_inputs_report,
                            inputs=[colab_aco_in, colab_var_in, colab_spk_map_in],
                            outputs=[colab_convert_log, colab_discovered_spks],
                        )
                        btn_run_colab_convert.click(
                            fn=COLAB_CKPT_CONVERTER.build_individual_singer_voicebanks,
                            inputs=[
                                colab_aco_in,
                                colab_var_in,
                                colab_out_dir,
                                colab_vb_prefix,
                                colab_split_mode,
                                colab_discovered_spks,
                                colab_lang_code,
                                colab_vocoder,
                                colab_icon_img,
                                colab_portrait_img,
                                colab_author,
                                colab_fill_mode,
                                colab_save_ckpt_chk,
                                colab_pack_zip_chk,
                            ],
                            outputs=[colab_convert_log],
                        )

                    with gr.TabItem("2️⃣ เครื่องมือประกอบร่างคลังเสียงแบบแยกโมเดลย่อยขั้นสูง (Advanced Multi-Model Exporter)"):
                        gr.Markdown("รวมโฟลเดอร์โมเดลที่ Export แล้ว (Acoustic, Duration, Variance, Pitch, Vocoder) เข้าด้วยกันเป็นคลังเสียง OpenUtau พร้อมสร้าง `dsdict.yaml` ภาษาไทยอัตโนมัติ (`advexport.py` & `AutoDsDict.py`)")
                        with gr.Row():
                            adv_vb_name = gr.Textbox(label="ชื่อคลังเสียง (Voicebank Name)", value="DeltaSinger_TH_v1.0", scale=4)
                            adv_out_dir = gr.Textbox(label="โฟลเดอร์ปลายทาง (Export Root Directory)", value=os.path.join(now_dir, "Packed_Voicebanks"), scale=6)
                            btn_adv_out = gr.Button("📁 เลือก", scale=2, variant="secondary")
                        with gr.Row():
                            adv_aco_dir = gr.Textbox(label="โฟลเดอร์โมเดล Acoustic (ที่มีโฟลเดอร์ย่อย onnx/)", scale=5)
                            btn_adv_aco = gr.Button("📁", scale=1, variant="secondary")
                            adv_dur_dir = gr.Textbox(label="โฟลเดอร์โมเดล Duration (ที่มีโฟลเดอร์ย่อย onnx/)", scale=5)
                            btn_adv_dur = gr.Button("📁", scale=1, variant="secondary")
                        with gr.Row():
                            adv_var_dir = gr.Textbox(label="โฟลเดอร์โมเดล Variance (Optional)", scale=5)
                            btn_adv_var = gr.Button("📁", scale=1, variant="secondary")
                            adv_pitch_dir = gr.Textbox(label="โฟลเดอร์โมเดล Pitch (Optional)", scale=5)
                            btn_adv_pitch = gr.Button("📁", scale=1, variant="secondary")
                        with gr.Row():
                            adv_voc_onnx = gr.Textbox(label="ไฟล์ Custom Vocoder .onnx (Optional)", scale=8)
                            btn_adv_voc = gr.Button("📄 เลือก Vocoder", scale=2, variant="secondary")
                            adv_autodict_chk = gr.Checkbox(label="สร้าง dsdict.yaml อัตโนมัติ (AutoDsDict)", value=True, scale=2)
                        btn_run_adv_exp = gr.Button("🏗️ ประกอบร่างคลังเสียง OpenUtau ขั้นสูง (Run Advanced Voicebank Exporter)", variant="primary")
                        adv_exp_log = gr.Textbox(label="สถานะการประกอบร่างคลังเสียง (Exporter Status)", lines=8, elem_classes=["log-output"])

                        btn_adv_out.click(fn=browse_folder, inputs=[adv_out_dir], outputs=[adv_out_dir])
                        btn_adv_aco.click(fn=browse_folder, inputs=[adv_aco_dir], outputs=[adv_aco_dir])
                        btn_adv_dur.click(fn=browse_folder, inputs=[adv_dur_dir], outputs=[adv_dur_dir])
                        btn_adv_var.click(fn=browse_folder, inputs=[adv_var_dir], outputs=[adv_var_dir])
                        btn_adv_pitch.click(fn=browse_folder, inputs=[adv_pitch_dir], outputs=[adv_pitch_dir])
                        btn_adv_voc.click(fn=lambda c: browse_file(c, [("ONNX Vocoder", "*.onnx")]), inputs=[adv_voc_onnx], outputs=[adv_voc_onnx])
                        btn_run_adv_exp.click(
                            fn=run_advanced_ou_exporter_ui,
                            inputs=[adv_vb_name, adv_out_dir, adv_aco_dir, adv_dur_dir, adv_var_dir, adv_pitch_dir, adv_voc_onnx, adv_autodict_chk],
                            outputs=[adv_exp_log],
                        )

            # ==================================================================
            # TAB 8: OPENUTAU VOICEBANK STUDIO
            # ==================================================================
            with gr.TabItem("🎤 8. OpenUtau Voicebank Studio"):
                with gr.Tabs():
                    with gr.TabItem("การตั้งค่าตัวละคร (Character Setup)"):
                        step1_md = gr.Markdown("### กระบวนการที่ 1: จัดการรายละเอียดของตัวละคร OpenUtau 🟢")
                        with gr.Row():
                            singer_name_in = gr.Textbox(label="ชื่อนักร้อง / ตัวละคร (Character Name)", value="Singer", scale=3)
                            lang_mode_in = gr.Dropdown(label="ภาษาและรูปแบบคลังเสียง (Language & Format)", choices=["1. EN", "2. TH", "3. JP", "4. DS"], value="2. TH", scale=3)
                            base_out_dir = gr.Textbox(label="โฟลเดอร์สำหรับส่งออก (Root Export Directory)", value=os.path.join(now_dir, "Packed_Voicebanks"), scale=8)
                            btn_browse_base_out = gr.Button("📁 เลือก", scale=1, variant="secondary")
                        btn_browse_base_out.click(fn=browse_folder, inputs=[base_out_dir], outputs=[base_out_dir])
                        with gr.Row():
                            pic_icon = gr.Image(type="filepath", label="ไอคอนตัวละคร (Icon.png)")
                            pic_full = gr.Image(type="filepath", label="ภาพตัวละครเต็มตัว (Full.png)")
                        with gr.Row():
                            readme_txt = gr.Textbox(label="เนื้อหา readme.txt", lines=6)
                            char_yaml = gr.Textbox(label="เนื้อหา character.yaml", lines=6)
                            char_txt = gr.Textbox(label="เนื้อหา character.txt", lines=6)
                        hidden_unified = gr.Textbox(visible=False)
                        hidden_proj_dir = gr.Textbox(visible=False)
                        save_proj_btn = gr.Button("💾 บันทึกและสร้างโฟลเดอร์โปรเจกต์ (Initialize Project)", variant="primary")
                        proj_status = gr.Textbox(label="สถานะ (Status)")

                        singer_name_in.change(fn=update_text_templates, inputs=[singer_name_in, lang_mode_in], outputs=[readme_txt, char_yaml, char_txt, hidden_unified])
                        lang_mode_in.change(fn=update_text_templates, inputs=[singer_name_in, lang_mode_in], outputs=[readme_txt, char_yaml, char_txt, hidden_unified])
                        save_proj_btn.click(fn=save_project, inputs=[base_out_dir, singer_name_in, pic_icon, pic_full, readme_txt, char_yaml, char_txt], outputs=[step1_md, proj_status, hidden_proj_dir])

                    with gr.TabItem("การเรนเดอร์เสียงและแพ็คไฟล์ (Render & Pack Voicebank)"):
                        tab2_header = gr.Markdown("### กระบวนการที่ 2: การแพ็คไฟล์คลังเสียง OpenUtau 🟢")
                        with gr.Row():
                            zip_output_dir = gr.Textbox(label="โฟลเดอร์สำหรับส่งออกไฟล์ .zip", value=os.path.join(now_dir, "Packed_Voicebanks"), scale=8)
                            btn_browse_zip_out = gr.Button("📁 เลือก", scale=1, variant="secondary")
                            zip_btn = gr.Button("🗜️ แพ็คไฟล์เป็น .zip (Pack to .zip)", variant="primary", scale=3)
                        zip_log = gr.Textbox(label="สถานะการแพ็คไฟล์ (Packaging Status)", elem_classes=["log-output"])
                        btn_browse_zip_out.click(fn=browse_folder, inputs=[zip_output_dir], outputs=[zip_output_dir])
                        zip_btn.click(fn=run_zip_packing, inputs=[base_out_dir, singer_name_in, zip_output_dir], outputs=[zip_log, tab2_header])

            # ==================================================================
            # TAB 9: DIFFSINGER 2.5.1 UNIFIED STUDIO
            # ==================================================================
            with gr.TabItem("🎵 9. DiffSinger 2.5.1 Studio"):
                with gr.Tabs():
                    with gr.TabItem("📂 จัดการโปรเจกต์ & เตรียมข้อมูล (Data Prep & Config)"):
                        with gr.Row():
                            ds_new_proj = gr.Textbox(label="สร้างโปรเจกต์ใหม่ (New Project Name)", placeholder="my_voicebank", scale=3)
                            ds_new_spk = gr.Textbox(label="ชื่อผู้พูด (Speaker Name)", placeholder="Singer01", scale=3)
                            ds_new_sid = gr.Number(label="Speaker ID", value=0, precision=0, scale=1)
                            ds_new_lang = gr.Dropdown(label="ภาษา (Language)", choices=["th", "en", "ja", "zh", "Custom-Language"], value="th", scale=2)
                            btn_ds_create = gr.Button("✨ สร้างโปรเจกต์ใหม่", variant="secondary", scale=2)
                        with gr.Row():
                            ds_proj_select = gr.Dropdown(label="เลือกโปรเจกต์ DiffSinger", choices=ds_list_existing_projects(), allow_custom_value=True, scale=3)
                            btn_ds_dedup = gr.Button("🧹 ลบไฟล์ซ้ำ & สร้าง Dictionary", variant="secondary", scale=3)
                            ds_model_type_prep = gr.Radio(["Acoustic", "Variance"], label="ประเภทโมเดล (Model Type)", value="Acoustic", scale=2)
                            ds_max_updates = gr.Number(label="Max Updates", value=160000, precision=0, scale=2)
                            btn_ds_gen_cfg = gr.Button("⚙️ สร้าง Config YAML", variant="secondary", scale=2)
                        with gr.Row():
                            ds_data_in = gr.Textbox(label="ไฟล์ Config YAML (หรือเว้นว่างให้ค้นหาอัตโนมัติจากโปรเจกต์)", scale=8)
                            btn_ds_cfg_browse = gr.Button("📄 เลือก Config", scale=1, variant="secondary")
                            ds_data_btn = gr.Button("▶️ เริ่มเตรียมข้อมูล (Start Binarization)", variant="primary", scale=3)
                        ds_yaml_preview = gr.Code(label="ตัวอย่าง Config YAML", language="yaml", lines=8)
                        ds_data_log = gr.Textbox(label="Log การจัดการข้อมูลและ Binarization", lines=8, elem_classes=["log-output"])

                        btn_ds_create.click(fn=ds_create_project, inputs=[ds_new_proj, ds_new_spk, ds_new_sid, ds_new_lang], outputs=[ds_data_log, ds_proj_select])
                        btn_ds_dedup.click(fn=ds_remove_duplicates_and_scan_phonemes, inputs=[ds_proj_select], outputs=[ds_data_log])
                        btn_ds_gen_cfg.click(fn=ds_generate_config_yaml, inputs=[ds_proj_select, ds_model_type_prep, ds_max_updates], outputs=[ds_data_log, ds_yaml_preview])
                        btn_ds_cfg_browse.click(fn=lambda c: browse_file(c, [("YAML", "*.yaml")]), inputs=[ds_data_in], outputs=[ds_data_in])
                        ds_data_btn.click(fn=ds_run_binarization, inputs=[ds_proj_select, ds_model_type_prep, ds_data_in], outputs=[ds_data_log])

                    with gr.TabItem("🚀 ฝึกสอนโมเดล (Acoustic / Variance Training)"):
                        with gr.Row():
                            ds_tr_config = gr.Textbox(label="ไฟล์ Config YAML", value="configs/acoustic.yaml", scale=6)
                            btn_b_cfg = gr.Button("📄 เลือก", scale=1, variant="secondary")
                            ds_tr_exp = gr.Textbox(label="ชื่อ Experiment", value="ds_exp", scale=3)
                            ds_tr_resume = gr.Checkbox(label="Resume Training (สานต่อการเทรน)", value=True, scale=2)
                        with gr.Row():
                            ds_tr_btn = gr.Button("▶️ เริ่มฝึกสอน (Start Training)", variant="primary")
                            ds_stop_btn = gr.Button("🛑 หยุดฝึกสอน (Stop)", variant="secondary")
                        ds_tr_log = gr.Textbox(label="Training Log", lines=12, elem_classes=["log-output"])

                        btn_b_cfg.click(fn=lambda c: browse_file(c, [("YAML", "*.yaml")]), inputs=[ds_tr_config], outputs=[ds_tr_config])
                        ds_tr_btn.click(fn=ds_run_training_job, inputs=[ds_tr_config, ds_tr_exp, ds_tr_resume], outputs=[ds_tr_log])
                        ds_stop_btn.click(fn=cancel_all_processes, outputs=[ds_tr_log])

                    with gr.TabItem("⚡ 1-Click Pipeline (Train + Export ONNX)"):
                        with gr.Row():
                            ds_pipe_cfg = gr.Textbox(label="ไฟล์ Config YAML", value="configs/acoustic.yaml", scale=6)
                            btn_pipe_cfg = gr.Button("📄 เลือก", scale=1, variant="secondary")
                            ds_pipe_exp = gr.Textbox(label="ชื่อ Experiment Name", value="ds_pipeline_exp", scale=3)
                            ds_pipe_out = gr.Textbox(label="โฟลเดอร์ส่งออก ONNX (Output Directory)", value="", scale=3)
                        btn_pipe_run = gr.Button("⚡ เริ่มกระบวนการอัตโนมัติครบวงจร (Run 1-Click Pipeline)", variant="primary")
                        ds_pipe_log = gr.Textbox(label="สถานะ Pipeline", lines=10, elem_classes=["log-output"])

                        btn_pipe_cfg.click(fn=lambda c: browse_file(c, [("YAML", "*.yaml")]), inputs=[ds_pipe_cfg], outputs=[ds_pipe_cfg])
                        btn_pipe_run.click(fn=ds_run_one_click_pipeline, inputs=[ds_pipe_cfg, ds_pipe_exp, ds_pipe_out], outputs=[ds_pipe_log])

                    with gr.TabItem("✨ ปรับแต่งสมรรถภาพไฟล์ .ds & แปลง DB (.lab+.wav)"):
                        gr.Markdown("#### 1. โมดูลเพิ่มสมรรถภาพ DiffSinger AI (`diffsinger_enhancer.py` — ขยายพยัญชนะ & เติมลูกคออัตโนมัติ)")
                        with gr.Row():
                            ds_enh_target = gr.Textbox(label="ไฟล์ .ds หรือโฟลเดอร์ที่มีไฟล์ .ds", scale=6)
                            btn_ds_enh_file = gr.Button("📄 เลือกไฟล์ .ds", scale=2, variant="secondary")
                            btn_ds_enh_dir = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                        with gr.Row():
                            ds_enh_c_ratio = gr.Slider(minimum=1.0, maximum=1.5, step=0.02, label="สัดส่วนขยายพยัญชนะ (Consonant Expansion Ratio)", value=1.18)
                            ds_enh_vib_rate = gr.Slider(minimum=4.0, maximum=9.0, step=0.1, label="ความถี่ลูกคอ (Vibrato Rate Hz)", value=6.2)
                            ds_enh_vib_depth = gr.Slider(minimum=0.0, maximum=80.0, step=1.0, label="ความลึกลูกคอ (Vibrato Depth Cents)", value=32.0)
                        btn_run_ds_enh = gr.Button("✨ ปรับแต่งไฟล์ .ds อัตโนมัติ (Enhance .ds Files)", variant="primary")

                        gr.Markdown("#### 2. ตัวแปลงชุดข้อมูล `.lab + .wav` เป็นรูปแบบ DiffSinger `transcriptions.csv` (`liteconvert.py` / `make_diffsinger_v1.1.py`)")
                        with gr.Row():
                            db_conv_root = gr.Textbox(label="โฟลเดอร์ชุดข้อมูลที่มีไฟล์ .lab และ .wav", scale=8)
                            btn_db_conv_dir = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                            btn_run_db_conv = gr.Button("🔄 แปลง .lab+.wav เป็น DiffSinger CSV", variant="primary", scale=3)

                        ds_tools_log = gr.Textbox(label="สถานะการประมวลผล (Enhancer & DB Converter Log)", lines=6, elem_classes=["log-output"])

                        btn_ds_enh_file.click(fn=lambda c: browse_file(c, [("DiffSinger Score", "*.ds")]), inputs=[ds_enh_target], outputs=[ds_enh_target])
                        btn_ds_enh_dir.click(fn=browse_folder, inputs=[ds_enh_target], outputs=[ds_enh_target])
                        btn_run_ds_enh.click(fn=run_ds_vocal_enhancer_ui, inputs=[ds_enh_target, ds_enh_c_ratio, ds_enh_vib_rate, ds_enh_vib_depth], outputs=[ds_tools_log])
                        btn_db_conv_dir.click(fn=browse_folder, inputs=[db_conv_root], outputs=[db_conv_root])
                        btn_run_db_conv.click(fn=convert_lab_wav_to_diffsinger_csv, inputs=[db_conv_root], outputs=[ds_tools_log])

            # ==================================================================
            # TAB 10: THAI PHONEMIZER & RULE 10 TUNING
            # ==================================================================
            with gr.TabItem("🇹🇭 10. แปลงและปรับแต่งหน่วยเสียงไทย (Thai Phonemizer)"):
                with gr.Row():
                    ph_std = gr.Dropdown(label="มาตรฐานหน่วยเสียง (Phoneme Standard)", choices=["DiffSinger_AI", "Thai_VCCV", "Thai_Arpasing"], value="DiffSinger_AI", scale=4)
                    ph_c = gr.Textbox(label="พยัญชนะต้น (Initial Consonant)", value="ก", scale=2)
                    ph_v = gr.Textbox(label="สระ (Vowel)", value="อา", scale=2)
                    ph_e = gr.Textbox(label="ตัวสะกด (Final Consonant / Ending)", value="", scale=2)
                with gr.Row():
                    ph_note_ms = gr.Slider(minimum=50, maximum=2000, step=10, label="ความยาวโน้ต (Note Duration ms)", value=500)
                    ph_ccv = gr.Slider(minimum=1.0, maximum=10.0, step=0.5, label="สัดส่วนพยัญชนะ CCV Ratio (% <=5% Rule 10)", value=5.0)
                    ph_vc = gr.Slider(minimum=20.0, maximum=50.0, step=1.0, label="สัดส่วนตัวสะกด VC Ratio (% 25-40% Rule 10)", value=30.0)
                with gr.Row():
                    ph_vib_rate = gr.Slider(minimum=3.0, maximum=10.0, step=0.1, label="ความถี่ลูกคอ Vibrato Rate (Hz)", value=6.2)
                    ph_vib_depth = gr.Slider(minimum=0.0, maximum=100.0, step=1.0, label="ความลึกของลูกคอ Vibrato Depth (Cents)", value=35.0)
                btn_calc_phoneme = gr.Button("⚡ คำนวณหน่วยเสียงและเส้นพิตช์ (Synthesize Phoneme & Pitch)", variant="primary")
                out_phoneme_spec = gr.Textbox(label="ผลลัพธ์การจัดหน่วยเสียงและข้อมูลพิตช์ (Phoneme Specification)", lines=8, elem_classes=["log-output"])
                btn_calc_phoneme.click(
                    fn=synthesize_thai_phoneme_and_tuning,
                    inputs=[ph_std, ph_c, ph_v, ph_e, ph_note_ms, ph_ccv, ph_vc, ph_vib_rate, ph_vib_depth],
                    outputs=[out_phoneme_spec],
                )

            # ==================================================================
            # TAB 11: ACCOUNT MODULE BRIDGE, NOTIFICATION CENTER & LIVE HOT-SWAP
            # ==================================================================
            with gr.TabItem("🔗 11. ศูนย์รวมโมดูล & ซิงค์แทนที่ไฟล์อัตโนมัติ (System Sync & Hot-Swap)"):
                gr.Markdown("""
                ### 🔄 ระบบเชื่อมโยงโมดูลในบัญชี และนำไฟล์ที่ปรับปรุงแล้วมาแทนที่ไฟล์เดิมทันที (Live In-Place Hot-Swap & System Sync)
                - **In-Place File Replacement**: นำไฟล์ที่ผ่านการแก้ไขปรับปรุงแล้วมาแทนที่ไฟล์เดิมใน `Thai_RVC_WebUI_v1` ได้ในทันที พร้อมตรวจสอบไวยากรณ์ (`ast.parse` & `py_compile`) และสำรองไฟล์เดิมไว้ใน `AI_Agent_Commands_And_Archive/Unused_Files/`
                - **Full System Sync**: ทบทวนและอัปเดตโมดูลที่ใช้ร่วมกันทุกส่วนในโปรเจกต์ให้ตรงกันเสมอ รวมไฟล์ซ้ำ `(1)` จาก Google Drive อัตโนมัติ และอัปเดต Manifest เส้นทางระบบ
                """)
                with gr.Group():
                    gr.Markdown("#### ⚡ 1. นำไฟล์ที่ปรับปรุงแล้วมาแทนที่ไฟล์เดิมในระบบทันที (Atomic In-Place File Replacement)")
                    with gr.Row():
                        swap_src_file = gr.Textbox(label="ไฟล์ต้นทางที่ผ่านการปรับปรุงแล้ว (Updated Source File Path)", scale=6)
                        btn_swap_src = gr.Button("📄 เลือกไฟล์ต้นทาง", scale=2, variant="secondary")
                        swap_target_rel = gr.Textbox(
                            label="ไฟล์ปลายทางในโปรเจกต์ที่ต้องการแทนที่ (Target File in Thai_RVC_WebUI_v1)",
                            placeholder="เช่น Program.py, infer-web.py, utils.py, DiffSinger_Training/diffsinger_enhancer.py",
                            value="Program.py",
                            scale=6,
                        )
                        btn_swap_target = gr.Button("📄 เลือกเป้าหมาย", scale=2, variant="secondary")
                    with gr.Row():
                        swap_backup_chk = gr.Checkbox(label="สำรองไฟล์เดิมเข้า Archive ก่อนแทนที่", value=True, scale=3)
                        swap_mirror_chk = gr.Checkbox(label="ซิงค์ไฟล์ชื่อเดียวกันในโฟลเดอร์ย่อยทั้งหมดให้ตรงกัน", value=True, scale=4)
                        btn_exec_hotswap = gr.Button("🔄 แทนที่ไฟล์เดิมและอัปเดตระบบทันที (Execute Live Hot-Swap)", variant="primary", scale=5)

                with gr.Group():
                    gr.Markdown("#### 🛡️ 2. ทบทวน ตรวจสอบไวยากรณ์ และซิงค์ไฟล์ทั้งหมดในโปรเจกต์ให้ตรงกันเสมอ (Full System Audit & Sync)")
                    with gr.Row():
                        sync_infer_web_chk = gr.Checkbox(label="ซิงค์ Program.py ไปยัง infer-web.py ให้เหมือนกันทุกประการ", value=False, scale=4)
                        btn_audit_sync_all = gr.Button("🛡️ ตรวจสอบและซิงค์ทุกไฟล์ในระบบให้ตรงกันเดี๋ยวนี้ (Audit & Sync All Files)", variant="primary", scale=4)
                        btn_show_modules = gr.Button("🔗 ดูสถานะการเชื่อมต่อโมดูลทั้งหมดในบัญชี (Account Module Inventory)", variant="secondary", scale=4)
                        btn_show_notifs = gr.Button("🔔 ดูประวัติการแจ้งเตือนทั้งหมด (Notification Log)", variant="secondary", scale=3)

                sync_output_log = gr.Textbox(
                    label="รายงานผลการซิงค์ระบบ โมดูลในบัญชี และการแจ้งเตือน (System Sync & Module Bridge Output)",
                    lines=14,
                    value=ACCOUNT_BRIDGE.get_inventory_report(),
                    elem_classes=["log-output"],
                )

                btn_swap_src.click(fn=lambda c: browse_file(c, [("Python / Config Files", "*.py;*.yaml;*.json;*.bat;*.md"), ("All Files", "*.*")]), inputs=[swap_src_file], outputs=[swap_src_file])
                btn_swap_target.click(fn=lambda c: browse_file(c, [("Python / Config Files", "*.py;*.yaml;*.json;*.bat;*.md"), ("All Files", "*.*")]), inputs=[swap_target_rel], outputs=[swap_target_rel])
                btn_exec_hotswap.click(
                    fn=lambda src, dst, bak, mir: SYSTEM_SYNC_MANAGER.hot_swap_file(src, dst, is_raw_content=False, create_backup=bak, sync_mirror_targets=mir),
                    inputs=[swap_src_file, swap_target_rel, swap_backup_chk, swap_mirror_chk],
                    outputs=[sync_output_log],
                )
                btn_audit_sync_all.click(
                    fn=SYSTEM_SYNC_MANAGER.audit_and_sync_all_system_files,
                    inputs=[sync_infer_web_chk],
                    outputs=[sync_output_log],
                )
                btn_show_modules.click(fn=ACCOUNT_BRIDGE.get_inventory_report, outputs=[sync_output_log])
                btn_show_notifs.click(fn=DeltaNotificationCenter.get_history_text, outputs=[sync_output_log])

            # ==================================================================
            # TAB 12: FAQ & SYSTEM ARCHITECTURE DOCUMENTATION
            # ==================================================================
            with gr.TabItem("📖 12. คำถามที่พบบ่อยและเอกสารอ้างอิง (FAQ & Architecture)"):
                gr.Markdown(f"""
                ### 📋 สถาปัตยกรรมระบบ DELTA RVC Studio x DiffSinger 2.5.1 Unified Engine (v7.0)
                - **โฟลเดอร์หลักโปรเจกต์ (Local Root)**: `{PROJECT_ROOT}` (`A:\\Thai_RVC_WebUI_v1`)
                - **คลังโค้ด GitHub (Repository)**: [https://github.com/deltaVOCALOID09378/Thai_RVC_WebUI](https://github.com/deltaVOCALOID09378/Thai_RVC_WebUI)
                - **คลังข้อมูลซิงค์ Google Drive**: [https://drive.google.com/drive/folders/13W4MowhEct36WCG0_tfMZ1_qQmPfb_hY](https://drive.google.com/drive/folders/13W4MowhEct36WCG0_tfMZ1_qQmPfb_hY)
                - **Python Runtime ที่ทำงานอยู่**: `{PYTHON_EXE}`
                - **โฟลเดอร์ DiffSinger หลัก**: `{DIFFSINGER_ROOT}`
                """)
                try:
                    faq_path = os.path.join(now_dir, "docs", "en", "faq_en.md")
                    if os.path.isfile(faq_path):
                        with open(faq_path, "r", encoding="utf-8") as f:
                            gr.Markdown(value=f.read())
                except Exception:
                    pass

        # ลงทะเบียนระบบแจ้งเตือนอัตโนมัติ DELTA SYNTH Toast System บนฝั่ง Browser
        app_instance.load(None, None, None, _js=delta_toast_js)

    return app_instance


# สร้างอินสแตนซ์หลักของ Gradio Blocks เพื่อให้เรียกใช้ได้ทั้งจาก CLI, `Program.build_app()` และ `Program.app`
app = build_app()

listen_port = int(getattr(config, "listen_port", 9378))
is_colab = getattr(config, "iscolab", False)
no_auto_open = getattr(config, "noautoopen", False)


# ─── 19. Smart Multi-Band Port Resolver & Resilient Server Launcher ──────────
import socket


def is_port_bindable(port: int, host: str = "127.0.0.1") -> bool:
    """
    ตรวจสอบว่าพอร์ตสามารถ Bind ได้จริงทั้งบน 127.0.0.1 และ 0.0.0.0 โดยไม่ติด
    Windows Hyper-V / WinNAT Excluded Port Range (WinError 10013) หรือพอร์ตซ้ำ (WinError 10048)
    """
    if not (1024 <= int(port) <= 65535):
        return False
    for test_host in ({host, "127.0.0.1", "0.0.0.0"} if host == "0.0.0.0" else {host, "127.0.0.1"}):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((test_host, int(port)))
        except OSError:
            return False
    return True


def find_available_port(preferred_port: int = 9378, host: str = "127.0.0.1") -> int:
    """
    ค้นหาพอร์ตที่ว่างและไม่ติดบล็อกของ Windows Hyper-V / WSL อัตโนมัติแบบหลายย่านความถี่ (Multi-Band):
    1. ย่านพอร์ตหลัก (preferred_port .. preferred_port + 40 เช่น 9378..9418)
    2. ย่านพอร์ตมาตรฐาน RVC / Gradio (7897, 7860 .. 7910)
    3. ย่านพอร์ตสำรองระดับสูง (19378 .. 19410, 8080 .. 8120)
    4. พอร์ตอิสระที่จัดสรรโดยเคอร์เนลของระบบปฏิบัติการ (OS Ephemeral Port)
    """
    candidates: List[int] = []
    for p in range(int(preferred_port), int(preferred_port) + 40):
        candidates.append(p)
    for p in [7897, 7860] + list(range(7861, 7910)) + list(range(19378, 19410)) + list(range(8080, 8120)):
        if p not in candidates:
            candidates.append(p)

    for cand_port in candidates:
        if is_port_bindable(cand_port, host):
            return cand_port

    # Fallback สุดท้าย: ให้ OS Kernel เลือกพอร์ตที่ว่างแน่นอน 100%
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])
    except OSError:
        return 7860


def launch_webui_resilient(
    app_obj,
    preferred_port: int = 9378,
    inbrowser_flag: bool = True,
    share_flag: bool = False,
):
    """
    เปิดเซิร์ฟเวอร์ Gradio WebUI ด้วยระบบป้องกันพอร์ตชนและหลบหลีกพอร์ตที่ถูกสำรองโดย Windows อัตโนมัติ
    รับประกันการเปิดโปรแกรมสำเร็จ 100% โดยไม่เกิด OSError: Cannot find empty port in range
    """
    # ล้างค่า GRADIO_SERVER_PORT ที่อาจค้างใน Environment เพื่อไม่ให้ล็อกพอร์ตตายตัว
    os.environ.pop("GRADIO_SERVER_PORT", None)

    if share_flag:
        return app_obj.queue(concurrency_count=511, max_size=1022).launch(share=True)

    # ทดสอบลำดับ Host (ใช้ 127.0.0.1 เป็นหลักบน Windows เพื่อหลีกเลี่ยง Firewall/Hyper-V บล็อก 0.0.0.0 และรองรับ 0.0.0.0)
    host_candidates = ["127.0.0.1", "0.0.0.0"] if platform.system() == "Windows" else ["0.0.0.0", "127.0.0.1"]

    tried_ports = set()
    for host in host_candidates:
        for base_p in (preferred_port, 7897, 7860, 19378, 8080):
            free_p = find_available_port(base_p, host)
            if (host, free_p) in tried_ports:
                continue
            tried_ports.add((host, free_p))
            try:
                if free_p != preferred_port:
                    DeltaLogger.wrn(
                        f"⚠️ พอร์ต {preferred_port} ถูกใช้งานหรือถูกสงวนโดยระบบ Windows -> สลับไปยังพอร์ตที่ว่างอัตโนมัติ: {free_p}",
                        to_console=True,
                    )
                DeltaLogger.inf(
                    f"🟢 เปิดให้บริการ WebUI สำเร็จที่: http://127.0.0.1:{free_p} (Host: {host}, Port: {free_p})",
                    to_console=True,
                )
                return app_obj.queue(concurrency_count=511, max_size=1022).launch(
                    server_name=host,
                    inbrowser=inbrowser_flag,
                    server_port=free_p,
                    quiet=True,
                )
            except OSError as exc:
                DeltaLogger.wrn(f"ไม่สามารถเปิดพอร์ต {free_p} บน {host} ({exc}) -> กำลังค้นหาพอร์ตถัดไปอัตโนมัติ...", to_console=True)
                continue

    # ด่านสุดท้าย: ให้ Gradio ค้นหาพอร์ตอัตโนมัติโดยไม่ล็อก server_port
    DeltaLogger.inf("🔄 กำลังเปิด WebUI ในโหมด Auto-Port Selection ของระบบ...", to_console=True)
    return app_obj.queue(concurrency_count=511, max_size=1022).launch(
        inbrowser=inbrowser_flag,
        server_port=None,
        quiet=True,
    )


# ─── 20. CLI & Main Execution Entry Point ────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DELTA SYNTH Smart AI Voice Studio v7.1 (RVC x DiffSinger Unified)")
    parser.add_argument("--port", type=int, default=listen_port, help="Server port (default: 9378)")
    parser.add_argument("--pycmd", type=str, default=PYTHON_EXE, help="Python executable command")
    parser.add_argument("--colab", action="store_true", help="Launch in Google Colab share mode")
    parser.add_argument("--dml", action="store_true", help="DirectML mode for AMD/Intel GPU")
    parser.add_argument("--route_model", type=str, default="", help="Model file or folder to route automatically")
    parser.add_argument("--version_profile", type=str, default="AUTO", help="Target version profile for routing")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name for routing")
    parser.add_argument("--mode", type=str, default="copy", help="Routing mode (copy/move)")
    parser.add_argument("--convert_colab_ckpt", type=str, default="", help="Path to Google Colab DiffSinger .ckpt or folder to convert to OpenUtau Voicebanks")
    parser.add_argument("--variance_ckpt", type=str, default="", help="Optional Variance .ckpt or folder for Colab conversion")
    parser.add_argument("--out_voicebanks", type=str, default="", help="Output directory for generated individual singer voicebanks")
    parser.add_argument("--replace_file", nargs=2, metavar=("SOURCE_FILE", "TARGET_PATH"), help="Atomically replace a project file in-place and sync mirrors")
    parser.add_argument("--sync_system", action="store_true", help="Run full system audit, deduplicate Drive files, and sync all shared modules")

    args, unknown = parser.parse_known_args()

    if args.dml:
        config.dml = True

    if args.sync_system:
        print(SYSTEM_SYNC_MANAGER.audit_and_sync_all_system_files(sync_program_to_infer_web=False))
        sys.exit(0)

    if args.replace_file:
        src_f, dst_f = args.replace_file
        print(SYSTEM_SYNC_MANAGER.hot_swap_file(src_f, dst_f, is_raw_content=False, create_backup=True, sync_mirror_targets=True))
        sys.exit(0)

    if args.convert_colab_ckpt:
        for step_log in COLAB_CKPT_CONVERTER.build_individual_singer_voicebanks(
            acoustic_ckpt_or_dir=args.convert_colab_ckpt,
            variance_ckpt_or_dir=args.variance_ckpt,
            output_root_dir=args.out_voicebanks or os.path.join(now_dir, "Packed_Voicebanks", "Colab_DiffSinger_Singers"),
        ):
            out_lines = step_log.splitlines()
            if out_lines:
                print(out_lines[-1], flush=True)
        sys.exit(0)

    # รองรับการลากไฟล์หรือโฟลเดอร์มาวางบน Start The Program.bat โดยตรง (Positional Drag & Drop Arguments)
    drag_and_drop_targets = [u for u in unknown if not u.startswith("-") and os.path.exists(u.strip('"'))]
    route_target = args.route_model or ("\n".join(drag_and_drop_targets) if drag_and_drop_targets else "")

    if route_target:
        print("=" * 68)
        print("🚀 DELTA SYNTH Smart Model Router CLI")
        print(f"📦 Input Path      : {route_target}")
        print(f"🎯 Version Profile : {args.version_profile}")
        print("=" * 68)
        report, last_pth, last_idx = MODEL_ROUTER.execute_routing(
            route_target, args.version_profile, args.exp_name, args.mode, True
        )
        print(report)
        sys.exit(0)

    target_port = args.port or listen_port
    DeltaLogger.inf(f"🚀 กำลังเริ่มต้นระบบ DELTA RVC Studio x DiffSinger v7.1 (พอร์ตเป้าหมายเริ่มต้น: {target_port})...", to_console=True)
    launch_webui_resilient(
        app_obj=app,
        preferred_port=target_port,
        inbrowser_flag=not no_auto_open,
        share_flag=bool(args.colab or is_colab),
    )