# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี (Date)       : 02/11/2569 (02 November 2026)
# 2. ชื่อไฟล์ที่ทำ (File Name)   : extract_feature_print.py
# 3. ลำดับเวอร์ชัน (Version)   : 4.3 (Identity & Soul Capture Edition)
# 4. เครดิตผู้สร้าง (Credit)    : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#                              Original by RVC Project
# 5. ตำแหน่งที่วางของไฟล์ (Path) : infer/modules/train/extract_feature_print.py
# 6. รายละเอียดการแก้ไขงาน    : เสริมการสกัดอัตลักษณ์เนื้อเสียงด้วย HuBERT ระดับ High-Fidelity 
#                              พร้อมลอจิกป้องกัน Memory Leak (Zero-Leak VRAM) อย่างรัดกุม
# ==============================================================================

import os
import sys
import gc
import time
import logging
import warnings
import traceback
from datetime import datetime

# บังคับใช้ UTF-8 ป้องกันปัญหาบน Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")
logging.getLogger("fairseq").setLevel(logging.ERROR)

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F

try:
    import fairseq
    import fairseq.data.dictionary
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
except Exception:
    pass

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

device_arg = sys.argv[1] if len(sys.argv) > 1 else "cpu"
n_part = min(2, max(1, int(sys.argv[2]))) if len(sys.argv) > 2 else 1
i_part = int(sys.argv[3]) if len(sys.argv) > 3 else 0

if len(sys.argv) == 6:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
elif len(sys.argv) >= 7:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
else:
    exp_dir = "logs/default"
    version = "v2"

device = device_arg
use_fp16 = False

if "privateuseone" not in device.lower() and "dml" not in device.lower():
    if device.lower() == "cpu":
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
        use_fp16 = True
    else:
        device = "cpu"
else:
    device = "cpu"
    use_fp16 = False

os.makedirs(exp_dir, exist_ok=True)
log_file_path = os.path.join(exp_dir, "extract_f0_feature.log")

def printt(strr: str) -> None:
    try:
        print(strr, flush=True)
    except Exception:
        pass
    try:
        with open(log_file_path, "a+", encoding="utf-8", errors="replace") as f:
            f.write(f"{strr}\n")
    except Exception:
        pass

def log_bilingual(th_msg: str, en_msg: str, use_sep: bool = False) -> None:
    if use_sep: printt("-" * 68)
    printt(f"[TH] {th_msg}\n[EN] {en_msg}")
    if use_sep: printt("-" * 68)

log_bilingual("🚀 เริ่มกระบวนการสกัดจิตวิญญาณและเอกลักษณ์เสียงด้วย HuBERT (v4.3)", 
              "Start HuBERT Identity & Soul Extraction (v4.3)", use_sep=True)

def readwave(wav_path: str):
    """อ่านคลื่นเสียงด้วยความละเอียดสูงสุด (High-Fidelity) เพื่อไม่ให้เสียบุคลิกเสียง"""
    try:
        wav, sr = sf.read(wav_path)
    except Exception:
        wav, sr = librosa.load(wav_path, sr=16000, mono=True)
        
    if len(wav) == 0: return None
    if wav.ndim == 2: wav = wav.mean(-1)
    
    if sr != 16000:
        # ใช้ High-Quality Resampler เพื่อรักษา Frequency Response
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000, res_type="soxr_vhq")
        
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2: feats = feats.mean(-1)
    if feats.numel() < 400: return None
    return feats.view(1, -1)

model_path = "assets/hubert/hubert_base.pt"
if not os.path.isfile(model_path):
    log_bilingual(f"❌ ไม่พบโมเดล HuBERT ที่ {model_path}", f"Error: HuBERT model not found at {model_path}", use_sep=True)
    sys.exit(1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path], suffix="")
    
model = models[0].to(device)
if use_fp16: model = model.half()
else: model = model.float()
model.eval()

wavPath = os.path.join(exp_dir, "1_16k_wavs")
outPath = os.path.join(exp_dir, "3_feature256" if version.lower() == "v1" else "3_feature768")
os.makedirs(outPath, exist_ok=True)

start_time = time.time()
try:
    if not os.path.exists(wavPath):
        sys.exit(1)
    all_files = sorted([x for x in os.listdir(wavPath) if x.lower().endswith(".wav")])
    todo = all_files[i_part::n_part]
    processed_count, skipped_count, error_count = 0, 0, 0

    for idx, file in enumerate(todo):
        try:
            wav_path = os.path.join(wavPath, file)
            base_name = os.path.splitext(file)[0]
            out_path = os.path.join(outPath, f"{base_name}.npy")

            if os.path.exists(out_path) and os.path.getsize(out_path) > 128:
                skipped_count += 1
                continue

            feats = readwave(wav_path)
            if feats is None:
                error_count += 1
                continue

            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
            inputs = {
                "source": feats.half().to(device) if use_fp16 else feats.float().to(device),
                "padding_mask": padding_mask.to(device),
                "output_layer": 9 if version.lower() == "v1" else 12,
            }

            with torch.no_grad():
                logits = model.extract_features(**inputs)
                extracted_feats = model.final_proj(logits[0]) if version.lower() == "v1" else logits[0]

            extracted_feats = extracted_feats.squeeze(0).float().cpu().numpy()
            
            if np.isnan(extracted_feats).sum() == 0 and not np.isinf(extracted_feats).any() and extracted_feats.size > 0:
                np.save(out_path, extracted_feats, allow_pickle=False)
                processed_count += 1
            else:
                error_count += 1

        except Exception:
            error_count += 1
            
        finally:
            # 🔴 กฎข้อที่ 8: การคืนหน่วยความจำ VRAM อย่างเข้มงวด (VRAM Safety Protocol)
            try:
                del feats
            except Exception: pass
            try:
                del padding_mask
            except Exception: pass
            try:
                del inputs
            except Exception: pass
            try:
                del logits
            except Exception: pass
            try:
                del extracted_feats
            except Exception: pass
            
            if idx % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    log_bilingual(f"🎉 สกัดเอกลักษณ์เสียงเสร็จสมบูรณ์ (สำเร็จ {processed_count}, ข้าม {skipped_count}, ขัดข้อง {error_count})", 
                  f"Identity extraction completed (Processed {processed_count}, Skipped {skipped_count}, Error {error_count})", use_sep=True)
finally:
    try:
        del model
    except Exception: pass
    try:
        del models
    except Exception: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()