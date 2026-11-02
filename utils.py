# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี (Date)       : 02/11/2569 (02 November 2026)
# 2. ชื่อไฟล์ที่ทำ (File Name)   : utils.py
# 3. ลำดับเวอร์ชัน (Version)   : 2.0 (Unified PoliteLogger, Safe HuBERT & Diagnostic Edition)
# 4. เครดิตผู้สร้าง (Credit)    : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#                              Original by Patiphat Wongyai (Delta) & RVC-Project
# 5. ตำแหน่งที่จัดวางไฟล์ (Path) : A:\Thai_RVC_WebUI_v1\utils.py
# 6. รายละเอียดการแก้ไขงาน    : หลอมรวมระบบ PoliteLogger, ตัวดักจับข้อผิดพลาดระดับบรรทัด, 
#                              การค้นหาไฟล์ Index อัตโนมัติ และฟังก์ชันโหลด HuBERT แบบ Safe Globals
# ==============================================================================

import os
import re
import sys
import logging
import linecache
import traceback
import threading
from datetime import datetime
import torch

# บังคับใช้ UTF-8 Without BOM บน Windows Console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SKIP_INDEX_SCAN_DIRS = {
    "0_gt_wavs",
    "1_16k_wavs",
    "2a_f0",
    "2b-f0nsf",
    "3_feature256",
    "3_feature768",
    "mute",
    "__pycache__",
    ".git",
    "eval",
}


def get_index_path_from_model(sid: str) -> str:
    """
    [TH] ระบบค้นหาและจับคู่ไฟล์ดัชนี (.index) อัตโนมัติจากชื่อโมเดล พร้อมข้ามโฟลเดอร์ข้อมูลดิบเพื่อความรวดเร็ว
    [EN] Automatically discover and match the corresponding FAISS .index file based on the model name.
    """
    model_stem = os.path.splitext(os.path.basename(str(sid or "")))[0]
    experiment_name = re.sub(r"_e\d+_s\d+$", "", model_stem, flags=re.IGNORECASE)
    if not experiment_name:
        return ""

    candidates = []
    cwd = os.getcwd()
    outside_root = os.getenv("outside_index_root") or os.path.join(cwd, "assets", "indices")
    inside_root = os.getenv("index_root") or os.path.join(cwd, "logs")
    roots = []
    for r in (outside_root, inside_root, "logs", "assets/indices"):
        if r and os.path.isdir(r):
            abs_r = os.path.abspath(r)
            if abs_r not in roots:
                roots.append(abs_r)

    if not roots:
        return ""

    primary_root = roots[0]
    lower_experiment = experiment_name.lower()
    lower_model_stem = model_stem.lower()

    for index_root in roots:
        for root, dirs, files in os.walk(index_root, topdown=True):
            dirs[:] = [d for d in dirs if d not in SKIP_INDEX_SCAN_DIRS]
            for name in files:
                if not name.lower().endswith(".index") or "trained" in name.lower():
                    continue

                index_stem = os.path.splitext(name)[0]
                lower_index = index_stem.lower()

                standard_match = (
                    lower_index.startswith(lower_experiment + "_added_")
                    or ("_" + lower_experiment + "_v1") in lower_index
                    or ("_" + lower_experiment + "_v2") in lower_index
                    or lower_index.endswith("_" + lower_experiment)
                    or (lower_experiment in lower_index)
                )
                exact_model_match = lower_model_stem in lower_index

                if standard_match or exact_model_match:
                    path = os.path.normpath(os.path.abspath(os.path.join(root, name))).replace("\\", "/")
                    try:
                        mtime = os.path.getmtime(path)
                    except OSError:
                        mtime = 0.0
                    score = (
                        0 if exact_model_match else (1 if standard_match else 2),
                        0 if os.path.abspath(index_root) == primary_root else 1,
                        -mtime,
                        path.lower(),
                    )
                    candidates.append((score, path))

    return min(candidates, default=(None, ""), key=lambda item: item[0])[1]


def load_hubert(config):
    """
    [TH] โหลดโมเดลพื้นฐาน HuBERT เข้าสู่อุปกรณ์ประมวลผล พร้อมปลดล็อก PyTorch 2.6+ Safe Globals
    [EN] Loads the HuBERT base model onto config.device with PyTorch 2.6+ safe globals.
    """
    from fairseq import checkpoint_utils
    try:
        from fairseq.data.dictionary import Dictionary
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([Dictionary])
    except Exception:
        pass

    hubert_path = os.getenv("hubert_path", "assets/hubert/hubert_base.pt")
    if not os.path.isfile(hubert_path) and os.path.isfile("hubert_base.pt"):
        hubert_path = "hubert_base.pt"

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if getattr(config, "is_half", False) and str(config.device) not in ("cpu", "mps") and "privateuseone" not in str(config.device).lower():
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


logger = logging.getLogger("DELTA_SYNTH")
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def format_error_diagnostic(
    msg: str = "",
    exc: Exception = None,
    context: str = "Execution",
    tb=None
) -> tuple[str, str]:
    """
    [TH] สร้างรายงานวิเคราะห์ข้อผิดพลาดเชิงลึก ระบุชื่อไฟล์ เลขบรรทัด คำสั่งที่เกิดปัญหา และ Traceback ฉบับเต็ม
    [EN] Constructs an exhaustive error diagnostic report with exact file, line number, code snippet, and traceback.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if exc is None and sys.exc_info()[1] is not None:
        exc = sys.exc_info()[1]
    if tb is None:
        tb = getattr(exc, "__traceback__", None) or sys.exc_info()[2]

    exc_type_name = type(exc).__name__ if exc is not None else "Error"
    exc_message = str(exc) if exc is not None else str(msg or "Unknown Error")
    if msg and msg != exc_message:
        cause_desc = f"{msg} | {exc_type_name}: {exc_message}"
    else:
        cause_desc = f"{exc_type_name}: {exc_message}"

    file_path = "Unknown"
    line_no = 0
    code_snippet = "N/A"
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
        if not full_traceback.strip() or "NoneType: None" in full_traceback:
            full_traceback = "No active traceback available."

    sep = "=" * 80
    sub_sep = "-" * 80
    cmd_report = (
        f"\n{sep}\n"
        f"🚨 [DELTA SYNTH ERROR DETECTED / เกิดข้อผิดพลาดในระบบ]\n"
        f"⏰ เวลา (Timestamp): {now_str}\n"
        f"📍 บริบท (Context): {context}\n"
        f"📁 ไฟล์ (File Path): {file_path} (บรรทัดที่ {line_no})\n"
        f"💻 บรรทัดคำสั่งที่เกิดปัญหา (Source Code):\n"
        f"   >>> {code_snippet}\n"
        f"⚠️ สาเหตุข้อผิดพลาด (Cause): {cause_desc}\n"
        f"{sub_sep}\n"
        f"📋 รายละเอียด Traceback ฉบับเต็ม:\n"
        f"{full_traceback.strip()}\n"
        f"{sep}\n"
    )

    webui_report = (
        f"❌ [TH] เกิดข้อผิดพลาดในการประมวลผล / [EN] Error Detected\n"
        f"• บริบท (Context): {context}\n"
        f"• ตำแหน่งไฟล์ (File): {file_path} (บรรทัดที่ {line_no})\n"
        f"• คำสั่งที่ขัดข้อง (Code): {code_snippet}\n"
        f"• สาเหตุ (Cause): {cause_desc}\n"
        f"------------------------------------------------------------\n"
        f"📋 Traceback:\n"
        f"{full_traceback.strip()}"
    )

    return cmd_report, webui_report


class PoliteLogger:
    """
    [TH] ระบบจัดการบันทึกสถานะแบบสองภาษา (TH/EN) สำหรับ DELTA SYNTH RVC Studio
    [EN] Polite and structured bilingual studio logger for DELTA SYNTH RVC WebUI.
    """
    CMD_ALLOWED_PREFIXES = {"MODEL", "ERROR", "CRITICAL", "TRAIN", "EPOCH", "SAVE", "FINISH", "DATA", "SYSTEM"}

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @classmethod
    def _safe_print_console(cls, text: str, stream=None) -> None:
        target_stream = stream or sys.stdout
        try:
            print(text, file=target_stream, flush=True)
        except UnicodeEncodeError:
            try:
                encoding = getattr(target_stream, "encoding", None) or "utf-8"
                safe_text = text.encode(encoding, errors="replace").decode(encoding)
                print(safe_text, file=target_stream, flush=True)
            except Exception:
                print(text.encode("ascii", errors="backslashreplace").decode("ascii"), file=target_stream, flush=True)

    @classmethod
    def _safe_print(cls, prefix: str, msg: str) -> None:
        text = f"[{cls._timestamp()}] [{prefix}] {msg}"
        should_print_to_cmd = (
            prefix in cls.CMD_ALLOWED_PREFIXES
            or os.getenv("DELTA_CMD_VERBOSE", "0") == "1"
        )
        if should_print_to_cmd:
            cls._safe_print_console(text)

    @classmethod
    def model(cls, model_name: str, details: str = "") -> str:
        extra = f" {details.strip()}" if details.strip() else ""
        msg = f"🎙️ [TH] โมเดลที่ใช้งานอยู่ตอนนี้ / [EN] Active Model: {model_name}{extra}"
        cls._safe_print("MODEL", msg)
        logger.info(f"[MODEL] {msg}")
        return msg

    @classmethod
    def system(cls, msg: str) -> str:
        formatted = f"⚙️ [SYSTEM] {msg}"
        logger.info(formatted)
        return formatted

    @classmethod
    def info(cls, msg: str) -> str:
        formatted = f"ℹ️ [INFO] {msg}"
        logger.info(formatted)
        return formatted

    @classmethod
    def warning(cls, msg: str) -> str:
        formatted = f"⚠️ [WARN] {msg}"
        logger.warning(formatted)
        return formatted

    @classmethod
    def success(cls, msg: str) -> str:
        formatted = f"✅ [SUCCESS] {msg}"
        logger.info(formatted)
        return formatted

    @classmethod
    def progress(cls, current: int, total: int, msg_th: str, msg_en: str = "") -> str:
        pct = (current / total) * 100 if total > 0 else 0.0
        msg = f"[{current}/{total} ({pct:.1f}%)] [TH] {msg_th}"
        if msg_en:
            msg += f" / [EN] {msg_en}"
        logger.info(f"[PROGRESS] {msg}")
        return msg

    @classmethod
    def error(cls, msg: str = "", exc: Exception = None, context: str = "General", tb=None) -> str:
        cmd_report, webui_report = format_error_diagnostic(msg=msg, exc=exc, context=context, tb=tb)
        cls._safe_print_console(cmd_report, stream=sys.stderr)
        logger.error(f"[ERROR] {msg or str(exc)}\n{cmd_report}")
        return webui_report


def install_global_exception_handler() -> None:
    """
    [TH] ติดตั้งตัวดักจับข้อผิดพลาดระดับ Global เพื่อรายงานบรรทัดที่เกิดปัญหาลงบน Console ทันที
    [EN] Registers global sys.excepthook and threading.excepthook for immediate crash diagnostics.
    """
    def _unhandled_hook(exctype, value, tb):
        PoliteLogger.error(
            msg="Unhandled Exception in Python Runtime",
            exc=value,
            context="System Core Unhandled Crash",
            tb=tb
        )

    sys.excepthook = _unhandled_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = lambda args: _unhandled_hook(args.exc_type, args.exc_value, args.exc_traceback)


install_global_exception_handler()

__all__ = [
    "PoliteLogger",
    "get_index_path_from_model",
    "load_hubert",
    "format_error_diagnostic",
    "install_global_exception_handler",
]