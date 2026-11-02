# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี (Date)       : 02/11/2569 (02 November 2026)
# 2. ชื่อไฟล์ที่ทำ (File Name)  : utils.py
# 3. ลำดับเวอร์ชัน (Version)   : 2.0 (Unified Blueprint Edition)
# 4. เครดิตผู้สร้าง (Credit)    : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#                              [ Original by "RVC-Project" ]
# 5. ตำแหน่งที่วางของไฟล์ (Path): infer/modules/vc/utils.py
# 6. รายละเอียดการแก้ไขงาน    : ปรับปรุงโครงสร้างหัวกระดาษให้เป็นมาตรฐานและเพิ่มระบบ Safe Globals
# ==============================================================================

import os
import re
import torch

def get_index_path_from_model(sid: str) -> str:
    """Finds the best-matching FAISS .index file for a given model name `sid`."""
    if not sid:
        return ""
    base_stem = os.path.splitext(os.path.basename(str(sid)))[0]
    clean_base = re.sub(r"_e\d+_s\d+$", "", base_stem, flags=re.IGNORECASE)

    search_roots = [
        os.getenv("index_root", "logs"),
        os.getenv("outside_index_root", "assets/indices"),
        "logs",
        "assets/indices",
    ]
    candidates = []
    seen = set()
    for root_dir in search_roots:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        for root, _, files in os.walk(root_dir, topdown=False):
            for name in files:
                if name.lower().endswith(".index") and "trained" not in name.lower():
                    full_p = os.path.normpath(os.path.join(root, name)).replace("\\", "/")
                    if full_p not in seen:
                        seen.add(full_p)
                        candidates.append(full_p)

    # Exact stem match priority, then clean_base match
    for cand in candidates:
        fname = os.path.basename(cand)
        if base_stem in fname:
            return cand
    for cand in candidates:
        fname = os.path.basename(cand)
        if clean_base and clean_base in fname:
            return cand
    return ""


def load_hubert(config):
    """Loads the HuBERT base model onto `config.device` with PyTorch 2.6+ safe globals."""
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
    if config.is_half and str(config.device) not in ("cpu", "mps") and "privateuseone" not in str(config.device).lower():
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()