# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : utils.py
# เวอร์ชั่นโค้ด (Version)      : 1.6 (High-Speed Directory Pruning & Safe HuBERT Loader)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/vc/utils.py
# หน้าที่ (Function/Role)     : ค้นหาและจับคู่ไฟล์ FAISS Index กับโมเดลอย่างแม่นยำ (ข้ามโฟลเดอร์ขยะอัตโนมัติ)
#                             และโหลดโมเดล HuBERT ด้วยระบบตรวจสอบหลายเส้นทาง (Multi-Path Resolver)
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Agentic AI
#                             Original by RVC-Project & Patiphat Wongyai (Delta)
# ==============================================================================

import os
from pathlib import Path
import torch

SKIP_SUBDIRS = {
    "0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf",
    "3_feature256", "3_feature768", "mute", "__pycache__", ".git"
}


def get_index_path_from_model(sid):
    """
    [TH] ค้นหาและคืนค่าพาธของไฟล์ FAISS Index ที่ตรงกับโมเดลอย่างแม่นยำและรวดเร็ว
    [EN] Return the exact matching FAISS index path for the specified model
    """
    if sid is None or str(sid).strip() == "":
        return ""

    index_roots = (
        os.getenv("outside_index_root"),
        os.getenv("index_root"),
        os.path.join(os.getcwd(), "assets", "indices"),
        os.path.join(os.getcwd(), "logs"),
    )
    index_roots = tuple(
        dict.fromkeys(
            os.path.abspath(root)
            for root in index_roots
            if root and os.path.isdir(root)
        )
    )
    if not index_roots:
        return ""

    model_stem = os.path.splitext(os.path.basename(str(sid)))[0].lower()

    for index_root in index_roots:
        for root, dirs, files in os.walk(index_root, topdown=True):
            # ตัดโฟลเดอร์ที่มีไฟล์ .npy/.wav จำนวนมากออกจากการค้นหาเพื่อเพิ่มความเร็วสูงสุด
            dirs[:] = sorted([d for d in dirs if d not in SKIP_SUBDIRS])
            for name in sorted(files):
                name_lower = name.lower()
                if not name_lower.endswith(".index") or "trained" in name_lower:
                    continue
                file_stem = os.path.splitext(name_lower)[0]
                if (
                    model_stem == file_stem
                    or f"_{model_stem}_" in file_stem
                    or file_stem.endswith(f"_{model_stem}")
                    or file_stem.startswith(f"{model_stem}_")
                    or model_stem in file_stem
                ):
                    return str(Path(root) / name).replace("\\", "/")
    return ""


def load_hubert(config, model_path=None):
    """
    [TH] โหลดและเตรียมโมเดล HuBERT Feature Extractor ขึ้นอุปกรณ์ประมวลผลอย่างปลอดภัย
    [EN] Load and prepare the HuBERT feature extractor for the configured device
    """
    try:
        from fairseq import checkpoint_utils
        import fairseq.data.dictionary
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
    except ImportError:
        raise ImportError("ไม่พบโมดูล fairseq กรุณาติดตั้ง fairseq ก่อนเริ่มใช้งาน (fairseq module not installed)")

    candidates = [
        model_path,
        os.getenv("hubert_path"),
        os.path.join(os.getcwd(), "assets", "hubert", "hubert_base.pt"),
        os.path.join(os.getcwd(), "hubert_base.pt"),
    ]
    resolved_path = None
    for c in candidates:
        if c and os.path.isfile(c):
            resolved_path = c
            break

    if not resolved_path:
        raise FileNotFoundError(
            "ไม่พบไฟล์โมเดล HuBERT (hubert_base.pt) กรุณาวางไฟล์ไว้ที่ assets/hubert/hubert_base.pt"
        )

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [resolved_path],
        suffix="",
    )
    if not models:
        raise RuntimeError(f"ไม่สามารถโหลดโมเดล HuBERT จาก: {resolved_path} (No model was loaded)")

    hubert_model = models[0]
    device = getattr(config, "device", "cpu")
    is_half = getattr(config, "is_half", False)

    hubert_model = hubert_model.to(device)
    if is_half and str(device) not in ["cpu", "mps"]:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()