# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : extract_f0_rmvpe.py
# เวอร์ชั่นโค้ด (Version)      : 2.7 (GPU/CPU Safe Sharded RMVPE Extractor)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/train/extract/extract_f0_rmvpe.py
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & Staff
# ==============================================================================

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)


def main() -> int:
    if len(sys.argv) < 6:
        exp_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/default"
        shard_count, shard_index, gpu_id, is_half = 1, 0, "0", False
    else:
        shard_count = max(1, min(2, int(sys.argv[1])))
        shard_index = max(0, int(sys.argv[2]))
        gpu_id = str(sys.argv[3])
        exp_dir = sys.argv[4]
        is_half = str(sys.argv[5]).strip().lower() in {"1", "true", "yes", "y", "on"}

    if gpu_id and gpu_id != "-":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    import torch
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        is_half = False

    from infer.modules.train.extract.f0_extractor import run_extraction
    print(f"[INFO] สกัดความถี่ด้วย RMVPE (GPU) Args: {sys.argv}", flush=True)
    return run_extraction(
        exp_dir=exp_dir,
        method="rmvpe",
        shard_count=shard_count,
        shard_index=shard_index,
        device=device,
        is_half=is_half,
    )


if __name__ == "__main__":
    raise SystemExit(main())