# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : train_dataset_inspector.py
# เวอร์ชั่นโค้ด (Version)      : 2.8 (Auto-Mute Padding, Stem Sync & File Integrity Check)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/train/train_dataset_inspector.py
# หน้าที่ (Function/Role)     : ตรวจสอบความพร้อมของชุดข้อมูลใน logs/<exp_name>
#                             และสร้างไฟล์ filelist.txt พร้อมแนบไฟล์เสียงเงียบ (Mute Padding) อัตโนมัติ
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Agentic AI
#                             Original by Patiphat Wongyai (Delta) & RVC-Project
# ==============================================================================

import os
import sys
import glob
import json
import logging
from random import shuffle
from typing import Tuple, List, Dict, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("DatasetInspector")


def clean_stem(filename: str) -> str:
    """ทำความสะอาดชื่อไฟล์เพื่อหานามสกุลฐานร่วม (Clean base stem matching) ให้ตรงกับ Program.py"""
    name = os.path.basename(filename)
    compound_exts = [
        ".wav.npy", ".flac.npy", ".mp3.npy", ".ogg.npy", ".m4a.npy",
        ".npy", ".wav", ".flac", ".mp3", ".ogg", ".m4a"
    ]
    for ext in compound_exts:
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return os.path.splitext(name)[0]


def inspect_and_build_filelist(
    exp_dir: str,
    version: str = "v2",
    if_f0: int = 1,
    spk_id: int = 0,
    sr_tag: str = "40k",
    force_rebuild: bool = True
) -> Tuple[bool, str, int]:
    """
    ตรวจสอบความสมบูรณ์ของชุดข้อมูลการฝึกสอนโมเดล RVC
    และสร้างไฟล์ filelist.txt จากไฟล์ที่มีอยู่จริงบนดิสก์ พร้อมเติมไฟล์เสียงเงียบ (mute)
    """
    exp_dir = os.path.abspath(exp_dir).replace("\\", "/")
    if not os.path.exists(exp_dir):
        return False, f"❌ ไม่พบโฟลเดอร์การทดลอง (Experiment dir not found): {exp_dir}", 0

    # ตรวจสอบ Sample Rate จาก config.json หากมีอยู่แล้วในโฟลเดอร์การทดลอง
    cfg_json_path = os.path.join(exp_dir, "config.json")
    if os.path.isfile(cfg_json_path):
        try:
            with open(cfg_json_path, "r", encoding="utf-8") as cf:
                cfg_data = json.load(cf)
                sr_val = cfg_data.get("data", {}).get("sampling_rate", 40000)
                sr_tag = f"{int(sr_val) // 1000}k"
        except Exception:
            pass

    gt_dir = os.path.join(exp_dir, "0_gt_wavs").replace("\\", "/")
    if not os.path.exists(gt_dir):
        gt_dir = os.path.join(exp_dir, "gt_wavs").replace("\\", "/")

    wav16k_dir = os.path.join(exp_dir, "1_16k_wavs").replace("\\", "/")
    if not os.path.exists(wav16k_dir):
        wav16k_dir = os.path.join(exp_dir, "wavs16k").replace("\\", "/")

    f0_dir = os.path.join(exp_dir, "2a_f0").replace("\\", "/")
    f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf").replace("\\", "/")

    feat_dir_256 = os.path.join(exp_dir, "3_feature256").replace("\\", "/")
    feat_dir_768 = os.path.join(exp_dir, "3_feature768").replace("\\", "/")

    has_768 = os.path.isdir(feat_dir_768) and len(glob.glob(os.path.join(feat_dir_768, "*.npy"))) > 0
    has_256 = os.path.isdir(feat_dir_256) and len(glob.glob(os.path.join(feat_dir_256, "*.npy"))) > 0

    resolved_version = version.lower()
    if has_768 and not has_256:
        feat_dir = feat_dir_768
        feat_dir_name = "3_feature768"
        resolved_version = "v2"
    elif has_256 and not has_768:
        feat_dir = feat_dir_256
        feat_dir_name = "3_feature256"
        resolved_version = "v1"
    else:
        feat_dir_name = "3_feature768" if resolved_version == "v2" else "3_feature256"
        feat_dir = os.path.join(exp_dir, feat_dir_name).replace("\\", "/")

    missing_dirs = []
    if not os.path.exists(gt_dir):
        missing_dirs.append("0_gt_wavs (ตัดเสียงต้นฉบับ / Ground Truth Audio)")
    if not os.path.exists(feat_dir) and not os.path.exists(wav16k_dir):
        missing_dirs.append("1_16k_wavs (เสียง 16kHz / 16k Audio)")
    if if_f0 == 1:
        if not os.path.exists(f0_dir):
            missing_dirs.append("2a_f0 (ความถี่ F0 / Pitch Curve)")
        if not os.path.exists(f0nsf_dir):
            missing_dirs.append("2b-f0nsf (Coarse F0 / Quantized Pitch)")
    if not os.path.exists(feat_dir):
        missing_dirs.append(f"{feat_dir_name} (HuBERT Features)")

    if missing_dirs:
        err_msg = (
            f"-----------------------------------------------------------------\n"
            f"❌ ไม่พบชุดข้อมูลที่สมบูรณ์สำหรับการฝึกโมเดล (Incomplete Dataset Found)\n"
            f"โฟลเดอร์ที่ยังขาดหายไป (Missing Folders):\n" + "\n".join([f"  - {d}" for d in missing_dirs]) +
            f"\n\n💡 คำแนะนำ (Advice): กรุณารัน 'ขั้นตอนที่ 1' และ 'ขั้นตอนที่ 2' ให้เสร็จสมบูรณ์ก่อน\n"
            f"-----------------------------------------------------------------"
        )
        return False, err_msg, 0

    gt_files: Dict[str, str] = {
        clean_stem(f): os.path.abspath(f).replace("\\", "/")
        for f in glob.glob(os.path.join(gt_dir, "*.wav"))
        if os.path.getsize(f) > 64
    }

    feat_files: Dict[str, str] = {
        clean_stem(f): os.path.abspath(f).replace("\\", "/")
        for f in glob.glob(os.path.join(feat_dir, "*.npy"))
        if os.path.getsize(f) > 64
    }

    f0_files: Dict[str, str] = {}
    f0nsf_files: Dict[str, str] = {}
    if if_f0 == 1:
        f0_files = {
            clean_stem(f): os.path.abspath(f).replace("\\", "/")
            for f in glob.glob(os.path.join(f0_dir, "*.npy"))
            if os.path.getsize(f) > 64
        }
        f0nsf_files = {
            clean_stem(f): os.path.abspath(f).replace("\\", "/")
            for f in glob.glob(os.path.join(f0nsf_dir, "*.npy"))
            if os.path.getsize(f) > 64
        }

    common_keys: Set[str] = set(gt_files.keys()) & set(feat_files.keys())
    if if_f0 == 1:
        common_keys = common_keys & set(f0_files.keys()) & set(f0nsf_files.keys())

    sorted_keys: List[str] = sorted(list(common_keys))
    valid_count: int = len(sorted_keys)

    stats = (
        f"-----------------------------------------------------------------\n"
        f"📊 ผลการตรวจสอบชุดข้อมูล (Dataset Audit Report): {exp_dir} [Version: {resolved_version.upper()}]\n"
        f"  - ไฟล์เสียง GT (0_gt_wavs): {len(gt_files)} ไฟล์ (Files)\n"
        f"  - ไฟล์ฟีเจอร์ HuBERT ({feat_dir_name}): {len(feat_files)} ไฟล์ (Files)\n"
    )
    if if_f0 == 1:
        stats += (
            f"  - ไฟล์ระดับเสียง F0 (2a_f0): {len(f0_files)} ไฟล์ (Files)\n"
            f"  - ไฟล์ Coarse F0 (2b-f0nsf): {len(f0nsf_files)} ไฟล์ (Files)\n"
        )
    stats += f"  => 🎯 ไฟล์ที่สมบูรณ์ตรงกันทั้งหมดพร้อมฝึก (Ready to Train): {valid_count} ไฟล์ (Files)\n"
    stats += f"-----------------------------------------------------------------"

    if valid_count == 0:
        err_msg = (
            f"❌ ไม่พบชุดข้อมูลที่สมบูรณ์สำหรับการฝึกโมเดล (No Valid Matched Files Found)\n"
            f"{stats}\n\n"
            f"⚠️ สาเหตุ: มีไฟล์ไม่ครบทุกโฟลเดอร์ กรุณากดรันขั้นตอนที่ 2 'สกัดฟีเจอร์เสียง' ใหม่อีกครั้ง"
        )
        return False, err_msg, 0

    filelist_path = os.path.join(exp_dir, "filelist.txt").replace("\\", "/")
    if force_rebuild or not os.path.exists(filelist_path):
        lines: List[str] = []
        for k in sorted_keys:
            gt_p = gt_files[k]
            feat_p = feat_files[k]
            if if_f0 == 1:
                f0_p = f0_files[k]
                f0nsf_p = f0nsf_files[k]
                lines.append(f"{gt_p}|{feat_p}|{f0_p}|{f0nsf_p}|{spk_id}")
            else:
                lines.append(f"{gt_p}|{feat_p}|{spk_id}")

        # แนบไฟล์เสียงเงียบ (logs/mute) จำนวน 2 รายการเพื่อป้องกันปัญหาเสียงซ่าเมื่อไม่มีสัญญาณพูด
        fea_dim = 256 if resolved_version == "v1" else 768
        logs_root = os.path.dirname(exp_dir)
        mute_dir = os.path.join(logs_root, "mute").replace("\\", "/")
        if os.path.isdir(mute_dir):
            mute_wav = f"{mute_dir}/0_gt_wavs/mute{sr_tag}.wav"
            mute_fea = f"{mute_dir}/3_feature{fea_dim}/mute.npy"
            mute_f0 = f"{mute_dir}/2a_f0/mute.wav.npy"
            mute_f0nsf = f"{mute_dir}/2b-f0nsf/mute.wav.npy"
            if os.path.exists(mute_wav) and os.path.exists(mute_fea):
                for _ in range(2):
                    if if_f0 == 1 and os.path.exists(mute_f0) and os.path.exists(mute_f0nsf):
                        lines.append(f"{mute_wav}|{mute_fea}|{mute_f0}|{mute_f0nsf}|{spk_id}")
                    elif if_f0 == 0:
                        lines.append(f"{mute_wav}|{mute_fea}|{spk_id}")

        shuffle(lines)
        with open(filelist_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    success_msg = (
        f"✅ ตรวจสอบชุดข้อมูลสำเร็จพร้อมฝึกสอนโมเดล! (Dataset Audit Passed!)\n"
        f"{stats}\n"
        f"📝 สร้างรายการไฟล์สำหรับฝึก (filelist.txt): {filelist_path} ({valid_count} รายการหลัก + Mute Padding)"
    )
    return True, success_msg, valid_count


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        ver = sys.argv[2] if len(sys.argv) > 2 else "v2"
        ok, msg, count = inspect_and_build_filelist(target_dir, version=ver)
        print(msg, flush=True)
        sys.exit(0 if ok else 1)
    else:
        print("Usage: python train_dataset_inspector.py <path_to_logs_exp_dir> [v1|v2]")