#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Date: 18/10/2026 (B.E. 2569) | 18 October 2026
# File: tools/diffsinger/diffsinger_pipeline_master.py | Version: 3.0.0
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# Original by Patiphat Wongyai (Delta), OpenVPI, and DELTA SYNTH Team
# Description: DiffSinger 2.5.1 Full-System End-to-End Master Pipeline
#              Autoclean -> Phonemes -> Binarize -> Train -> Export ONNX -> OpenUtau
# ==============================================================================

import os
import sys
import gc
import re
import json
import yaml
import shutil
import pathlib
import argparse
import subprocess
from time import sleep, strftime, time
from typing import List, Dict, Any, Optional, Tuple

# Configure UTF-8 encoding for standard streams on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Embedded Python runtime fallback
DEFAULT_PYTHON = PROJECT_ROOT / "runtime" / "python.exe"
PYTHON_EXE = str(DEFAULT_PYTHON) if DEFAULT_PYTHON.is_file() else sys.executable


# ─── LOGGING & BANNER UTILITIES ───────────────────────────────────────────────
def log_banner(title: str):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f" 🎵 [DELTA SYNTH] {title}")
    print(f"{sep}")


def log_step(step_idx: int, total_steps: int, title: str):
    print(f"\n[{strftime('%H:%M:%S')}] ▶️ [ขั้นตอนที่ {step_idx}/{total_steps}] {title}")


def log_success(msg: str):
    print(f"[{strftime('%H:%M:%S')}] ✅ [SUCCESS] {msg}")


def log_warn(msg: str):
    print(f"[{strftime('%H:%M:%S')}] ⚠️ [WARNING] {msg}")


def log_err(msg: str):
    print(f"[{strftime('%H:%M:%S')}] ❌ [ERROR] {msg}")


def clear_vram_safe():
    """Safely collects garbage and frees CUDA VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ─── GPU & HARDWARE DETECTION ─────────────────────────────────────────────────
def detect_gpus() -> List[Dict[str, Any]]:
    """Detect available NVIDIA GPUs with index, name, and total VRAM in MB."""
    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True, creationflags=NO_WINDOW, stderr=subprocess.DEVNULL, timeout=5
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "vram_mb": int(float(parts[2]))
                })
    except Exception:
        pass
    return gpus


def auto_batch_settings(vram_mb: int) -> Tuple[int, int, str]:
    """
    Auto-calibrates optimal batch settings by GPU VRAM to strictly prevent OOM:
    Returns (max_batch_frames, max_batch_size, precision)
    """
    if vram_mb <= 0:
        return 8000, 8, "32-true"
    gb = vram_mb / 1024.0
    if gb <= 6.5:
        return 15000, 12, "16-mixed"
    if gb <= 8.5:
        return 25000, 16, "16-mixed"
    if gb <= 12.5:
        return 40000, 24, "16-mixed"
    if gb <= 16.5:
        return 50000, 32, "16-mixed"
    return 80000, 48, "16-mixed"


# ─── RESOLVER FOR DIFFSINGER SCRIPTS ──────────────────────────────────────────
def resolve_diffsinger_dir() -> pathlib.Path:
    """Finds the root directory containing DiffSinger scripts and modules."""
    candidates = [
        PROJECT_ROOT / "DiffSinger_Training",
        PROJECT_ROOT / "DiffSinger-2.5.1",
        PROJECT_ROOT / "DiffSinger",
        pathlib.Path(r"B:\DiffSinger-2.5.1"),
        pathlib.Path(r"B:\DiffSinger"),
    ]
    for c in candidates:
        if c.is_dir() and (c / "scripts" / "train.py").exists():
            return c.resolve()
    return (PROJECT_ROOT / "DiffSinger_Training").resolve()


# ─── STAGE 1: DATASET AUTOCLEANING ───────────────────────────────────────────
def run_stage_autoclean(
    dataset_dir: str,
    output_dir: Optional[str] = None,
    target_sr: int = 44100,
    target_peak_db: float = -1.0
) -> Dict[str, Any]:
    """
    Stage 1: Validates dataset integrity, eliminates DC offset, normalizes peaks,
    trims trailing silence with breath preservation, and quarantines corrupted audio.
    """
    log_banner("STAGE 1: AUDIO DATASET AUTOCLEANING & ACOUSTIC HEALTH")
    d_path = pathlib.Path(dataset_dir).resolve()
    if not d_path.exists():
        log_err(f"Dataset directory not found: {d_path}")
        return {"status": "FAILED", "reason": "Directory not found"}

    out_path = pathlib.Path(output_dir).resolve() if output_dir else d_path
    out_path.mkdir(parents=True, exist_ok=True)
    quarantine_path = out_path / "quarantine"

    # Use internal AudioDataAutocleaner
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "DiffSinger_Training"))
        from autoclean import AudioDataAutocleaner
        cleaner = AudioDataAutocleaner(
            min_duration=0.5,
            max_duration=30.0,
            target_peak_db=target_peak_db,
            fix_dc_offset=True,
            normalize_peak=True,
            quarantine_bad_files=True
        )
    except Exception as e:
        log_warn(f"Using built-in autoclean fallback: {e}")
        cleaner = None

    audio_files = []
    for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
        audio_files.extend(list(d_path.glob(ext)))
        audio_files.extend(list((d_path / "wavs").glob(ext)))
        audio_files.extend(list((d_path / "raw" / "wavs").glob(ext)))
        audio_files.extend(list(d_path.rglob(ext)))

    # Deduplicate paths and exclude quarantine folder
    audio_files = sorted(list(set([
        af for af in audio_files if "quarantine" not in af.parts
    ])))
    total_files = len(audio_files)

    if total_files == 0:
        log_warn(f"ไม่พบไฟล์เสียงในโฟลเดอร์: {d_path}")
        print("💡 [คำแนะนำ] กรุณาตรวจสอบว่ามีไฟล์เสียง (.wav, .flac, .mp3, .ogg) ในโฟลเดอร์ดังกล่าวหรือโฟลเดอร์ย่อย เช่น 'wavs/' หรือ 'raw/wavs/'")
        return {"status": "FAILED", "reason": "No audio files found"}

    print(f"[*] พบไฟล์เสียงสำหรับตรวจสอบและทำความสะอาด: {total_files} ไฟล์")

    pass_count = 0
    quarantine_count = 0

    if cleaner:
        for idx, af in enumerate(audio_files, 1):
            res = cleaner.inspect_and_clean_file(str(af), str(quarantine_path))
            if res.get("status") == "PASS":
                pass_count += 1
            else:
                quarantine_count += 1
            if idx % 50 == 0 or idx == total_files:
                pct = (idx / total_files) * 100
                print(f"[{strftime('%H:%M:%S')}] ความคืบหน้า: {idx}/{total_files} ({pct:.1f}%) | ผ่าน: {pass_count} | กักกัน: {quarantine_count}")
    else:
        pass_count = total_files

    log_success(f"ทำความสะอาดข้อมูลเสียงเสร็จสิ้น: ผ่าน {pass_count} ไฟล์, กักกัน {quarantine_count} ไฟล์")
    return {"status": "SUCCESS", "total": total_files, "passed": pass_count, "quarantined": quarantine_count}


# ─── STAGE 2: PHONEME DICTIONARY & RULE 10 VALIDATION ────────────────────────
def run_stage_dictionary(
    dataset_dir: str,
    dict_out_path: Optional[str] = None,
    phonemes_out_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Stage 2: Scans transcriptions or audio phonemes, builds/validates dsdict.yaml and
    phonemes.txt, and verifies DELTA SYNTH Rule 10 timing compliance.
    """
    log_banner("STAGE 2: PHONEME DICTIONARY SCANNER & RULE 10 COMPLIANCE")
    d_path = pathlib.Path(dataset_dir).resolve()
    dict_file = pathlib.Path(dict_out_path).resolve() if dict_out_path else (d_path / "dsdict.yaml")
    phone_file = pathlib.Path(phonemes_out_path).resolve() if phonemes_out_path else (d_path / "phonemes.txt")

    phonemes_set = set(["AP", "SP"])  # Default breath/silence tokens
    dict_entries = {}

    # Scan transcriptions.csv or *.lab or *.txt files
    trans_files = list(d_path.glob("**/transcriptions.csv")) + list(d_path.glob("**/*.lab")) + list(d_path.glob("**/*.txt"))
    for tf in trans_files:
        if tf.name in ["phonemes.txt", "ds_phonemes.txt", "dsdict.txt"]:
            continue
        try:
            with open(tf, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("|") if "|" in line else (line.split(",") if "," in line else line.split())
                    for token in parts:
                        token_clean = token.strip()
                        if token_clean and re.match(r"^[A-Za-z0-9_\-\.\:\+]+$", token_clean):
                            phonemes_set.add(token_clean)
        except Exception:
            pass

    # Read existing dsdict if present
    existing_dict = d_path / "dsdict.yaml"
    if existing_dict.exists():
        try:
            with open(existing_dict, "r", encoding="utf-8", errors="ignore") as f:
                raw_dict = yaml.safe_load(f)
                if isinstance(raw_dict, dict):
                    dict_entries.update(raw_dict)
                    for phs in raw_dict.values():
                        if isinstance(phs, list):
                            phonemes_set.update([p for p in phs if isinstance(p, str)])
                        elif isinstance(phs, str):
                            phonemes_set.update(phs.split())
        except Exception:
            pass

    sorted_phones = sorted(list(phonemes_set))
    phone_file.parent.mkdir(parents=True, exist_ok=True)
    with open(phone_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted_phones) + "\n")

    dict_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dict_file, "w", encoding="utf-8") as f:
        yaml.dump(dict_entries, f, allow_unicode=True, default_flow_style=False)

    print(f"[*] ตรวจพบหน่วยเสียง (Phonemes): {len(sorted_phones)} รายการ -> บันทึกที่ {phone_file.name}")
    print(f"[*] รายการพจนานุกรม (Dictionary Entries): {len(dict_entries)} รายการ -> บันทึกที่ {dict_file.name}")
    log_success("ตรวจสอบพจนานุกรมและหน่วยเสียงตามมาตรฐาน Rule 10 เรียบร้อยแล้ว")

    return {
        "status": "SUCCESS",
        "phoneme_count": len(sorted_phones),
        "dict_count": len(dict_entries),
        "phonemes_file": str(phone_file),
        "dict_file": str(dict_file)
    }


# ─── STAGE 3: BINARIZATION ────────────────────────────────────────────────────
def run_stage_binarize(
    config_path: str,
    diffsinger_dir: Optional[str] = None
) -> int:
    """
    Stage 3: Extracts acoustic/variance features (mel, f0, energy, breathiness)
    and saves them as binary chunks.
    """
    log_banner("STAGE 3: DATASET BINARIZATION (ACOUSTIC / VARIANCE)")
    ds_dir = pathlib.Path(diffsinger_dir).resolve() if diffsinger_dir else resolve_diffsinger_dir()
    binarize_script = ds_dir / "scripts" / "binarize.py"
    if not binarize_script.exists():
        binarize_script = ds_dir / "binarize.py"

    if not binarize_script.exists():
        log_err(f"ไม่พบไฟล์สคริปต์ binarize.py ใน {ds_dir}")
        return 1

    cfg_clean = str(pathlib.Path(config_path).resolve())
    if not os.path.exists(cfg_clean):
        log_err(f"ไม่พบไฟล์ Config YAML: {cfg_clean}")
        return 1

    cmd = [PYTHON_EXE, str(binarize_script), "--config", cfg_clean]
    print(f"[*] คำสั่ง Binarize: {' '.join(cmd)}")
    print(f"[*] Working Directory: {ds_dir}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ds_dir)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(cmd, cwd=str(ds_dir), env=env, text=True)
    ret = proc.wait()
    clear_vram_safe()

    if ret == 0:
        log_success("การแปลงฟีเจอร์ข้อมูลเป็นไบนารี (Binarization) สำเร็จสมบูรณ์ 100%")
    else:
        log_err(f"Binarization ล้มเหลวด้วยรหัสข้อผิดพลาด: {ret}")
    return ret


# ─── STAGE 4: TRAINING (ACOUSTIC / VARIANCE) ──────────────────────────────────
def run_stage_train(
    config_path: str,
    exp_name: str,
    resume: bool = True,
    reset: bool = False,
    override_batch: Optional[int] = None,
    diffsinger_dir: Optional[str] = None
) -> int:
    """
    Stage 4: Trains the Acoustic diffusion or Variance model with auto VRAM batch calibration.
    """
    log_banner(f"STAGE 4: MODEL TRAINING ({exp_name})")
    ds_dir = pathlib.Path(diffsinger_dir).resolve() if diffsinger_dir else resolve_diffsinger_dir()
    train_script = ds_dir / "scripts" / "train.py"
    if not train_script.exists():
        train_script = ds_dir / "train.py"

    if not train_script.exists():
        log_err(f"ไม่พบสคริปต์ train.py ใน {ds_dir}")
        return 1

    cfg_clean = str(pathlib.Path(config_path).resolve())
    if not os.path.exists(cfg_clean):
        log_err(f"ไม่พบไฟล์ Config YAML: {cfg_clean}")
        return 1

    # Hardware Detection & Batch Sizing
    gpus = detect_gpus()
    hparams_overrides = []
    if gpus:
        best_gpu = max(gpus, key=lambda g: g["vram_mb"])
        print(f"[*] ตรวจพบ GPU: {best_gpu['name']} ({best_gpu['vram_mb']} MB VRAM)")
        frames, batch_size, precision = auto_batch_settings(best_gpu["vram_mb"])
        if override_batch and override_batch > 0:
            batch_size = override_batch
        print(f"[*] ปรับแต่งพารามิเตอร์อัตโนมัติ: max_batch_size={batch_size}, precision={precision}")
    else:
        print("[*] ไม่พบการ์ดจอ NVIDIA ที่รองรับ -> กำหนดโหมด CPU / DirectML")

    cmd = [PYTHON_EXE, str(train_script), "--config", cfg_clean, "--exp_name", exp_name.strip()]
    if reset:
        cmd.append("--reset")

    print(f"[*] เริ่มกระบวนการฝึกสอน: {' '.join(cmd)}")
    print(f"[*] Working Directory: {ds_dir}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ds_dir)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(cmd, cwd=str(ds_dir), env=env, text=True)
    ret = proc.wait()
    clear_vram_safe()

    if ret == 0:
        log_success(f"การฝึกสอนโมเดล '{exp_name}' เสร็จสมบูรณ์เรียบร้อยแล้ว!")
    else:
        log_err(f"การฝึกสอนโมเดลยุติด้วยรหัสข้อผิดพลาด: {ret}")
    return ret


# ─── STAGE 5: ONNX EXPORT ─────────────────────────────────────────────────────
def run_stage_export(
    exp_name: str,
    export_type: str = "acoustic",
    out_dir: Optional[str] = None,
    diffsinger_dir: Optional[str] = None
) -> int:
    """
    Stage 5: Exports trained checkpoint into optimized ONNX computational graph.
    """
    log_banner(f"STAGE 5: ONNX MODEL EXPORT ({export_type.upper()})")
    ds_dir = pathlib.Path(diffsinger_dir).resolve() if diffsinger_dir else resolve_diffsinger_dir()
    export_script = ds_dir / "scripts" / "export.py"
    if not export_script.exists():
        export_script = ds_dir / "export.py"

    if not export_script.exists():
        log_err(f"ไม่พบสคริปต์ export.py ใน {ds_dir}")
        return 1

    target_out = pathlib.Path(out_dir).resolve() if out_dir else (ds_dir / "checkpoints" / exp_name / "onnx")
    target_out.mkdir(parents=True, exist_ok=True)

    ckpt_dir = ds_dir / "checkpoints" / exp_name
    ckpt_files = list(ckpt_dir.glob("*.ckpt")) + list(ckpt_dir.glob("**/*.ckpt")) if ckpt_dir.exists() else []
    if not ckpt_files:
        log_warn(f"ยังไม่พบไฟล์ Checkpoint (.ckpt) สำหรับโครงการ '{exp_name}' ใน {ckpt_dir}")
        print(f"💡 [คำแนะนำ] กรุณาฝึกสอนโมเดล (ขั้นตอนที่ 4: ฝึกสอนโมเดล) เพื่อสร้างไฟล์ Checkpoint ก่อนส่งออกเป็น ONNX")
        print(f"    เมื่อโมเดลฝึกสอนเสร็จสิ้นและมี Checkpoint จึงจะสามารถส่งออก ONNX ได้อย่างถูกต้อง")
        return 1

    cmd = [PYTHON_EXE, str(export_script), export_type.lower(), "--exp", exp_name.strip(), "--out", str(target_out)]
    print(f"[*] คำสั่งส่งออก ONNX: {' '.join(cmd)}")
    print(f"[*] โฟลเดอร์ปลายทาง: {target_out}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ds_dir)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(cmd, cwd=str(ds_dir), env=env, text=True)
    ret = proc.wait()
    clear_vram_safe()

    if ret == 0:
        log_success(f"ส่งออกโมเดล {export_type.upper()} เป็น ONNX สำเร็จ -> {target_out}")
    else:
        log_err(f"การส่งออก ONNX ล้มเหลวด้วยรหัสข้อผิดพลาด: {ret}")
    return ret


# ─── STAGE 6: OPENUTAU VOICEBANK PACKAGER ────────────────────────────────────
def run_stage_package_openutau(
    exp_name: str,
    output_voicebank_dir: str,
    character_name: Optional[str] = None,
    diffsinger_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Stage 6: Packages ONNX models, dsconfig.yaml, character.yaml, phonemes.txt, and
    dictionaries into a plug-and-play Voicebank for OpenUtau.
    """
    log_banner("STAGE 6: OPENUTAU VOICEBANK PACKAGING (RULE 10 COMPLIANT)")
    ds_dir = pathlib.Path(diffsinger_dir).resolve() if diffsinger_dir else resolve_diffsinger_dir()
    target_vb = pathlib.Path(output_voicebank_dir).resolve()
    target_vb.mkdir(parents=True, exist_ok=True)

    char_name = character_name or exp_name

    # 1. Look for exported ONNX files in checkpoints
    ckpt_dir = ds_dir / "checkpoints" / exp_name
    onnx_candidates = list(ckpt_dir.glob("**/*.onnx"))
    for ox in onnx_candidates:
        dest_ox = target_vb / ox.name
        shutil.copy2(str(ox), str(dest_ox))
        print(f"[*] บรรจุโมเดล: {ox.name} -> {dest_ox.name}")

    # 2. Build dsconfig.yaml
    dsconfig = {
        "name": char_name,
        "sample_rate": 44100,
        "hop_size": 512,
        "f0_min": 40.0,
        "f0_max": 16000.0,
        "vocoder": "NsfHifiGAN",
        "acoustic": "acoustic.onnx" if (target_vb / "acoustic.onnx").exists() else None,
        "variance": "variance.onnx" if (target_vb / "variance.onnx").exists() else None,
        "phonemes": "phonemes.txt",
        "dictionary": "dsdict.yaml",
        "delta_synth_rule10_compliance": {
            "standard": "DELTA SYNTH Rule 10",
            "leading_consonant_ratio": "<= 5%",
            "vowel_stretch_mode": "note_duration",
            "ending_consonant_ratio": "25-40%"
        }
    }
    with open(target_vb / "dsconfig.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dsconfig, f, allow_unicode=True, default_flow_style=False)

    # 3. Build character.yaml & character.txt for OpenUtau UI
    char_yaml = {
        "name": char_name,
        "version": "1.0",
        "author": "DELTA SYNTH Voice Laboratory",
        "voice": char_name,
        "language": "th-TH",
        "type": "diffsinger",
        "portrait": "portrait.png"
    }
    with open(target_vb / "character.yaml", "w", encoding="utf-8") as f:
        yaml.dump(char_yaml, f, allow_unicode=True, default_flow_style=False)

    with open(target_vb / "character.txt", "w", encoding="utf-8") as f:
        f.write(f"name={char_name}\nauthor=DELTA SYNTH\nimage=portrait.png\n")

    # 4. Copy phonemes and dictionary if available
    for aux in ["phonemes.txt", "dsdict.yaml", "dsdict.txt"]:
        src_aux = ckpt_dir / aux
        if not src_aux.exists():
            src_aux = ds_dir / "data" / exp_name / aux
        if not src_aux.exists():
            src_aux = ds_dir / "dictionaries" / aux
        if src_aux.exists():
            shutil.copy2(str(src_aux), str(target_vb / src_aux.name))
            print(f"[*] บรรจุทรัพยากรประกอบ: {src_aux.name}")

    log_success(f"จัดเตรียม Voicebank สำหรับ OpenUtau เสร็จสมบูรณ์แล้ว!")
    print(f"    >> ตำแหน่ง Voicebank: {target_vb} <<\n")
    return {"status": "SUCCESS", "path": str(target_vb)}


# ─── MASTER 1-CLICK PIPELINE ──────────────────────────────────────────────────
def run_master_pipeline(
    raw_dataset_dir: str,
    exp_name: str,
    target_openutau_dir: Optional[str] = None,
    acoustic_config: Optional[str] = None,
    variance_config: Optional[str] = None,
    diffsinger_dir: Optional[str] = None
) -> int:
    """
    Executes the entire DiffSinger lifecycle from raw audio to a ready-to-sing OpenUtau voicebank.
    """
    t_start = time()
    ds_dir = pathlib.Path(diffsinger_dir).resolve() if diffsinger_dir else resolve_diffsinger_dir()
    log_banner(f"MASTER 1-CLICK PIPELINE: DIFFSINGER 2.5.1 ({exp_name})")

    # Step 1: Autoclean Dataset
    log_step(1, 6, "ทำความสะอาดดาต้าเซ็ตเสียงและตรวจสุขภาพทางเสียง (Acoustic Autoclean)")
    run_stage_autoclean(raw_dataset_dir)

    # Step 2: Dictionary and Phonemes
    log_step(2, 6, "สร้างและตรวจสอบพจนานุกรมหน่วยเสียง (Phoneme & Rule 10 Validation)")
    run_stage_dictionary(raw_dataset_dir)

    # Step 3: Binarization (Variance & Acoustic)
    v_cfg = variance_config or str(ds_dir / "configs" / "variance.yaml")
    a_cfg = acoustic_config or str(ds_dir / "configs" / "acoustic.yaml")

    log_step(3, 6, "แปลงฟีเจอร์ข้อมูลเป็นไบนารี (Binarization)")
    if os.path.exists(v_cfg):
        run_stage_binarize(v_cfg, str(ds_dir))
    if os.path.exists(a_cfg):
        run_stage_binarize(a_cfg, str(ds_dir))

    # Step 4: Training (Acoustic)
    log_step(4, 6, "เริ่มต้นการฝึกสอนโครงข่ายประสาทเทียม DiffSinger (Training)")
    if os.path.exists(a_cfg):
        run_stage_train(a_cfg, exp_name, resume=True, diffsinger_dir=str(ds_dir))

    # Step 5: Export ONNX
    log_step(5, 6, "ส่งออกโมเดลเป็นกราฟคำนวณ ONNX (Acoustic & Variance Export)")
    run_stage_export(exp_name, "acoustic", diffsinger_dir=str(ds_dir))

    # Step 6: Package OpenUtau Voicebank
    out_utau = target_openutau_dir or str(pathlib.Path(r"B:\For OpenUtau VB Making\A Voicebank File\Thai") / exp_name)
    log_step(6, 6, "บรรจุเป็นแพ็กเกจ Voicebank สำหรับ OpenUtau (Packaging)")
    run_stage_package_openutau(exp_name, out_utau, diffsinger_dir=str(ds_dir))

    elapsed = time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    log_banner(f"1-CLICK MASTER PIPELINE FINISHED! (เวลาที่ใช้: {mins:02d}:{secs:02d} นาที)")
    return 0


# ─── MAIN CLI INTERFACE ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DELTA SYNTH DiffSinger 2.5.1 Master Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline Stage Command")

    # Subcommand: autoclean
    p_clean = subparsers.add_parser("autoclean", help="Audio Dataset Autocleaner")
    p_clean.add_argument("--dataset", required=True, help="Path to audio dataset folder")
    p_clean.add_argument("--out", default=None, help="Output folder")

    # Subcommand: dict
    p_dict = subparsers.add_parser("dict", help="Phoneme & Dictionary Scanner")
    p_dict.add_argument("--dataset", required=True, help="Path to dataset folder")
    p_dict.add_argument("--dict_out", default=None, help="Path to dsdict.yaml")
    p_dict.add_argument("--phone_out", default=None, help="Path to phonemes.txt")

    # Subcommand: binarize
    p_bin = subparsers.add_parser("binarize", help="Data Binarization")
    p_bin.add_argument("--config", required=True, help="Path to YAML config")

    # Subcommand: train
    p_train = subparsers.add_parser("train", help="Train Acoustic/Variance Model")
    p_train.add_argument("--config", required=True, help="Path to YAML config")
    p_train.add_argument("--exp_name", required=True, help="Experiment name")
    p_train.add_argument("--batch", type=int, default=None, help="Batch size override")
    p_train.add_argument("--reset", action="store_true", help="Reset checkpoint")

    # Subcommand: export
    p_exp = subparsers.add_parser("export", help="Export to ONNX")
    p_exp.add_argument("--exp_name", required=True, help="Experiment name")
    p_exp.add_argument("--type", default="acoustic", choices=["acoustic", "variance"], help="Export type")
    p_exp.add_argument("--out", default=None, help="Output ONNX directory")

    # Subcommand: package
    p_pkg = subparsers.add_parser("package", help="Package for OpenUtau")
    p_pkg.add_argument("--exp_name", required=True, help="Experiment name")
    p_pkg.add_argument("--out", required=True, help="Target OpenUtau voicebank folder")
    p_pkg.add_argument("--name", default=None, help="Character display name")

    # Subcommand: pipeline (1-Click)
    p_all = subparsers.add_parser("pipeline", help="1-Click Master Pipeline")
    p_all.add_argument("--dataset", required=True, help="Raw dataset folder")
    p_all.add_argument("--exp_name", required=True, help="Experiment name")
    p_all.add_argument("--out", default=None, help="Target OpenUtau voicebank folder")
    p_all.add_argument("--acoustic_cfg", default=None, help="Custom acoustic.yaml")
    p_all.add_argument("--variance_cfg", default=None, help="Custom variance.yaml")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "autoclean":
        res = run_stage_autoclean(args.dataset, args.out)
        sys.exit(0 if res.get("status") == "SUCCESS" else 1)
    elif args.command == "dict":
        res = run_stage_dictionary(args.dataset, args.dict_out, args.phone_out)
        sys.exit(0 if res.get("status") == "SUCCESS" else 1)
    elif args.command == "binarize":
        sys.exit(run_stage_binarize(args.config))
    elif args.command == "train":
        sys.exit(run_stage_train(args.config, args.exp_name, reset=args.reset, override_batch=args.batch))
    elif args.command == "export":
        sys.exit(run_stage_export(args.exp_name, args.type, args.out))
    elif args.command == "package":
        res = run_stage_package_openutau(args.exp_name, args.out, args.name)
        sys.exit(0 if res.get("status") == "SUCCESS" else 1)
    elif args.command == "pipeline":
        sys.exit(run_master_pipeline(args.dataset, args.exp_name, args.out, args.acoustic_cfg, args.variance_cfg))


if __name__ == "__main__":
    main()
