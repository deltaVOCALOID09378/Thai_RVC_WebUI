# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : extract_f0_rmvpe_dml.py
# เวอร์ชั่นโค้ด (Version)      : 2.6 (DirectML / CPU Fallback RMVPE Extractor with 1-2 Thread Cap)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/train/extract/extract_f0_rmvpe_dml.py
# หน้าที่ (Function/Role)     : สกัดระดับเสียงด้วย RMVPE บนการ์ดจอ AMD/Intel DirectML พร้อมระบบสำรอง CPU
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Agentic AI
#                             Original by Patiphat Wongyai (Delta) & RVC-Project
# ==============================================================================

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)


def main() -> int:
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/default"
    device = "cpu"
    try:
        import torch
        torch.set_num_threads(2)
        import torch_directml
        device = torch_directml.device(torch_directml.default_device())
    except Exception:
        device = "cpu"

    from infer.modules.train.extract.f0_extractor import run_extraction
    print(sys.argv, flush=True)
    return run_extraction(
        exp_dir=exp_dir,
        method="rmvpe",
        workers=1,
        device=device,
        is_half=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
