# -*- coding: utf-8 -*-
# ==============================================================================
# File: utils.py
# Path: utils.py
# Description: RVC WebUI Core Module / โมดูลหลัก RVC WebUI
# Version: 1.7
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# Original by RVC-Project
# ==============================================================================
"""
utils.py
Version: 1.7

Description: Studio logging interface (PoliteLogger), diagnostic error sentinel,
and Index file auto-discovery. Strictly silences routine inference notifications
on CMD while ensuring active model notifications and exhaustive error diagnostics
with source code snippets, line numbers, and tracebacks are broadcast to CMD.
"""
import os
import re
import sys
import logging
import linecache
import traceback
import threading
from datetime import datetime

# Windows UTF-8 Console Safety
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_index_path_from_model(sid: str) -> str:
    """
    [TH] ระบบค้นหาและจับคู่ไฟล์ดัชนี (Index) อัตโนมัติจากชื่อโมเดล
    [EN] Automatically discover and match the corresponding Index file based on the model name.
    """
    model_stem = os.path.splitext(os.path.basename(str(sid or "")))[0]
    experiment_name = re.sub(r"_e\d+_s\d+$", "", model_stem, flags=re.IGNORECASE)
    if not experiment_name:
        return ""

    candidates = []
    roots = [os.getenv("outside_index_root"), os.getenv("index_root")]
    
    for index_root in roots:
        if not index_root or not os.path.isdir(index_root):
            continue
            
        for root, _, files in os.walk(index_root, topdown=False):
            for name in files:
                if not name.lower().endswith(".index") or "trained" in name.lower():
                    continue
                    
                index_stem = os.path.splitext(name)[0]
                lower_index = index_stem.lower()
                lower_experiment = experiment_name.lower()
                
                standard_match = (
                    lower_index.startswith(lower_experiment + "_added_")
                    or ("_" + lower_experiment + "_v1") in lower_index
                    or ("_" + lower_experiment + "_v2") in lower_index
                )
                exact_model_match = model_stem.lower() in lower_index
                
                if standard_match or exact_model_match:
                    path = os.path.abspath(os.path.join(root, name))
                    score = (
                        0 if standard_match else 1,
                        0 if os.path.abspath(index_root) == os.path.abspath(roots[0]) else 1,
                        -os.path.getmtime(path),
                        path.lower(),
                    )
                    candidates.append((score, path))
                    
    return min(candidates, default=(None, ""), key=lambda item: item[0])[1]


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
    Constructs an exhaustive error diagnostic report with exact file, line number,
    faulting source code snippet, exception type, cause, and full traceback.
    
    Returns:
        tuple[str, str]: (cmd_report, webui_report)
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
        f"❌ [เกิดข้อผิดพลาดในการประมวลผล (Error Detected)]\n"
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
    Polite and structured studio logger for DELTA SYNTH RVC WebUI.
    
    Logging Contract:
    - Routine inference progress, timers, and success are routed strictly to WebUI.
    - Active model notifications are broadcast explicitly to CMD.
    - System errors and exceptions trigger an exhaustive diagnostic box on CMD
      and are returned for WebUI display.
    """
    # Allowed prefixes that will be output to CMD stdout
    CMD_ALLOWED_PREFIXES = {"MODEL", "ERROR", "CRITICAL", "TRAIN", "EPOCH", "SAVE", "FINISH", "DATA"}

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
    def model(cls, model_name: str, details: str = "") -> None:
        """
        Explicit active model notification sent directly to CMD.
        This is the sole routine informational notification permitted on CMD during inference.
        """
        extra = f" {details.strip()}" if details.strip() else ""
        msg = f"🎙️ โมเดลที่ใช้งานอยู่ตอนนี้ (Active Model): {model_name}{extra}"
        cls._safe_print("MODEL", msg)
        logger.info(f"[MODEL] {msg}")

    @classmethod
    def system(cls, msg: str) -> None:
        # Silent on CMD; logged to internal logger
        logger.info(f"[SYSTEM] {msg}")

    @classmethod
    def info(cls, msg: str) -> None:
        # Silent on CMD; logged to internal logger
        logger.info(f"[INFO] {msg}")

    @classmethod
    def warning(cls, msg: str) -> None:
        # Silent on CMD; logged to internal logger
        logger.warning(f"[WARN] {msg}")

    @classmethod
    def success(cls, msg: str) -> None:
        # Silent on CMD; logged to internal logger
        logger.info(f"[SUCCESS] {msg}")

    @classmethod
    def progress(cls, current: int, total: int, msg_th: str, msg_en: str = "") -> str:
        pct = (current / total) * 100 if total > 0 else 0.0
        msg = f"[{current}/{total} ({pct:.1f}%)] {msg_th}"
        if msg_en:
            msg += f" | {msg_en}"
        logger.info(f"[PROGRESS] {msg}")
        return msg

    @classmethod
    def error(cls, msg: str = "", exc: Exception = None, context: str = "General", tb=None) -> str:
        """
        Broadcasts full diagnostic banner directly to CMD and returns WebUI error text.
        """
        cmd_report, webui_report = format_error_diagnostic(msg=msg, exc=exc, context=context, tb=tb)
        cls._safe_print_console(cmd_report, stream=sys.stderr)
        logger.error(f"[ERROR] {msg or str(exc)}\n{cmd_report}")
        return webui_report


def install_global_exception_handler() -> None:
    """
    Registers global sys.excepthook and threading.excepthook so any unhandled
    exception immediately prints full diagnostics (file, line, code snippet, traceback) to CMD.
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
    "format_error_diagnostic",
    "install_global_exception_handler",
]