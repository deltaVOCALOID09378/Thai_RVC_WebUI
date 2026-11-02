# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : extract_f0_print.py
# เวอร์ชั่นโค้ด (Version)      : 2.7 (Multi-Method Studio F0 Extractor)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/train/extract/extract_f0_print.py
# หน้าที่ (Function/Role)     : สคริปต์เริ่มต้นสำหรับสกัด F0 จำกัดเทรดอย่างเคร่งครัดที่ 1-2 เทรด 
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & Staff
# ==============================================================================

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)

from infer.modules.train.extract.f0_extractor import run_extraction


def main() -> int:
    print(f"[INFO] เริ่มต้นระบบสกัดความถี่มูลฐาน (F0 Extraction) Args: {sys.argv}", flush=True)
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/default"
    n_p = min(2, max(1, int(sys.argv[2]))) if len(sys.argv) > 2 else 1
    f0method = sys.argv[3] if len(sys.argv) > 3 else "rmvpe"

    device = "cpu"
    is_half = False
    try:
        import torch
        torch.set_num_threads(2)
        if torch.cuda.is_available():
            device = "cuda"
            is_half = True
    except Exception:
        device = "cpu"

    return run_extraction(
        exp_dir=exp_dir,
        method=f0method,
        workers=n_p,
        device=device,
        is_half=is_half,
    )


if __name__ == "__main__":
    raise SystemExit(main())