# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : extract_feature_print.py
# เวอร์ชั่นโค้ด (Version)      : 3.6 (Smart Resume, Auto-Resample & VRAM Guard Edition)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/train/extract_feature_print.py
# หน้าที่ (Function/Role)     : สกัดคุณลักษณะเนื้อเสียงด้วยโมเดล HuBERT (256-dim สำหรับ v1 / 768-dim สำหรับ v2)
#                             พร้อมระบบตรวจสอบไฟล์ .npy เสียหาย, Auto-Resample 16kHz และซิงค์ Log แบบ Real-time
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Agentic AI
#                             Original by Patiphat Wongyai (Delta) & RVC-Project
# ==============================================================================

import os
import sys
import logging
import warnings
import traceback
import time
from datetime import datetime

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
logging.getLogger("fairseq").setLevel(logging.ERROR)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
n_part = max(1, int(sys.argv[2])) if len(sys.argv) > 2 else 1
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

import fairseq
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

try:
    import fairseq.data.dictionary
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
except Exception:
    pass

if "privateuseone" not in device.lower() and "dml" not in device.lower():
    if device.lower() == "cpu":
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
else:
    try:
        import torch_directml
        device = torch_directml.device(torch_directml.default_device())
        def forward_dml(ctx, x, scale):
            ctx.scale = scale
            res = x.clone().detach()
            return res
        fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
    except Exception:
        device = "cpu"

os.makedirs(exp_dir, exist_ok=True)
log_file_path = os.path.join(exp_dir, "extract_f0_feature.log")


def printt(strr: str) -> None:
    try:
        print(strr, flush=True)
    except Exception:
        try:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(str(strr).encode(enc, errors="replace").decode(enc), flush=True)
        except Exception:
            pass
    try:
        with open(log_file_path, "a+", encoding="utf-8", errors="replace") as f:
            f.write(f"{strr}\n")
    except Exception:
        pass


def log_bilingual(th_msg: str, en_msg: str, use_sep: bool = False) -> None:
    if use_sep:
        printt("-" * 65)
    printt(f"{th_msg}\n({en_msg})")
    if use_sep:
        printt("-" * 65)


log_bilingual("🚀 เริ่มกระบวนการสกัดคุณลักษณะเนื้อเสียงด้วย HuBERT (v3.6)", "Start HuBERT Feature Extraction (v3.6)", use_sep=True)
log_bilingual(f"📁 เป้าหมาย (Target): {exp_dir} | เวอร์ชัน (Version): {version.upper()}", f"Target: {exp_dir} | Version: {version.upper()}")

hubert_candidates = [
    os.getenv("hubert_path", ""),
    "assets/hubert/hubert_base.pt",
    "hubert_base.pt",
]
model_path = "assets/hubert/hubert_base.pt"
for cand in hubert_candidates:
    if cand and os.path.isfile(cand):
        model_path = cand
        break

wavPath = os.path.join(exp_dir, "1_16k_wavs")
outPath = os.path.join(exp_dir, "3_feature256" if version.lower() == "v1" else "3_feature768")
os.makedirs(outPath, exist_ok=True)


def readwave(wav_path: str, normalize: bool = False):
    try:
        wav, sr = sf.read(wav_path)
    except Exception:
        wav, sr = librosa.load(wav_path, sr=16000, mono=True)
    if len(wav) == 0:
        return None
    if wav.ndim == 2:
        wav = wav.mean(-1)
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:
        feats = feats.mean(-1)
    if feats.numel() < 400:
        return None
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats.view(1, -1)


log_bilingual(f"🎙️ โหลดโมเดลตัวสกัด (Loading Model): {model_path}", f"Loading HuBERT: {model_path}")
if not os.path.isfile(model_path):
    log_bilingual(f"❌ ข้อผิดพลาด: ไม่พบโมเดล HuBERT ที่ {model_path}", f"Error: HuBERT model not found at {model_path}", use_sep=True)
    sys.exit(1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path], suffix="")
model = models[0].to(device)
use_fp16 = str(device) not in ["mps", "cpu"] and "privateuseone" not in str(device).lower()
if use_fp16:
    model = model.half()
else:
    model = model.float()
model.eval()
log_bilingual(f"✅ โหลดโมเดลขึ้นหน่วยประมวลผลสำเร็จ (Device Ready)", f"Model loaded onto device: {device} (FP16: {use_fp16})")

start_time = time.time()
start_time_str = datetime.now().strftime("%H:%M:%S")
log_bilingual(f"⏱️ เวลาเริ่มต้น (Start Time): {start_time_str}", f"Start Time: {start_time_str}")

try:
    if not os.path.exists(wavPath):
        log_bilingual(f"❌ ไม่พบแฟ้มข้อมูลต้นฉบับ: {wavPath}", f"Source directory not found: {wavPath}", use_sep=True)
        sys.exit(1)

    all_files = sorted([x for x in os.listdir(wavPath) if x.lower().endswith(".wav")])
    total_files = len(all_files)

    if total_files == 0:
        log_bilingual("⚠️ ไม่มีไฟล์เสียงให้ดำเนินการ (No audio files found)", "No audio files found for extraction", use_sep=True)
        sys.exit(0)

    existing_files = [
        x for x in os.listdir(outPath)
        if x.lower().endswith(".npy") and os.path.getsize(os.path.join(outPath, x)) > 128
    ]
    log_bilingual(
        f"📊 ตรวจพบไฟล์ทั้งหมด {total_files} รายการ | ดำเนินการแล้ว {len(existing_files)} รายการ",
        f"Total files: {total_files} | Already processed: {len(existing_files)}",
        use_sep=True
    )

    todo = all_files[i_part::n_part]
    n = max(1, len(todo) // 10)
    processed_count = 0
    skipped_count = 0
    error_count = 0

    log_bilingual(f"🚀 เวิร์กเกอร์ {i_part} ได้รับมอบหมายงาน: {len(todo)} รายการ", f"Worker {i_part} assigned tasks: {len(todo)}")

    for idx, file in enumerate(todo):
        try:
            wav_path = os.path.join(wavPath, file)
            base_name = os.path.splitext(file)[0]
            out_path = os.path.join(outPath, f"{base_name}.npy")

            # ข้ามเฉพาะไฟล์ .npy ที่สมบูรณ์และมีขนาดมากกว่า 128 ไบต์
            if os.path.exists(out_path) and os.path.getsize(out_path) > 128:
                skipped_count += 1
                continue

            feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
            if feats is None:
                log_bilingual(f"⚠️ ข้ามไฟล์เปล่าหรือสั้นเกินไป (Skip empty/short): {file}", f"Skipped empty/too short file: {file}")
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
                log_bilingual(f"❌ ไฟล์ {file} มีค่าผิดปกติ (NaN/Inf) ถูกข้าม", f"File {file} contains NaN/Inf, skipped")
                error_count += 1

            if idx % n == 0 or idx == len(todo) - 1:
                progress_pct = ((idx + 1) / len(todo)) * 100
                log_bilingual(
                    f"⏳ ความคืบหน้า (Progress): [{idx + 1}/{len(todo)}] ({progress_pct:.1f}%) | กำลังประมวลผล: {file}",
                    f"Processing: {file}"
                )

            if idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            log_bilingual("🛑 หยุดการทำงานตามคำสั่งผู้ใช้งาน (Keyboard Interrupt)", "Interrupted by user", use_sep=True)
            break
        except Exception as e:
            log_bilingual(f"❌ ขัดข้องที่ไฟล์ (Failed at) {file}: {str(e)}", f"Failed at {file}: {str(e)}")
            error_count += 1
            logging.debug(traceback.format_exc())

    end_time_str = datetime.now().strftime("%H:%M:%S")
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)

    log_bilingual("🎉 สกัดคุณลักษณะเสร็จสิ้นสมบูรณ์ (Feature extraction completed)", "Feature extraction completed successfully", use_sep=True)
    log_bilingual(
        f"✅ สำเร็จ (Success): {processed_count} | ⏭️ ข้าม (Skipped): {skipped_count} | ❌ ผิดพลาด (Errors): {error_count}",
        f"Success: {processed_count} | Skipped: {skipped_count} | Errors: {error_count}"
    )
    log_bilingual(
        f"⏱️ เวลาสิ้นสุด: {end_time_str} | ระยะเวลาที่ใช้ไป: {mins:02d}:{secs:02d} (MM:SS)",
        f"End Time: {end_time_str} | Elapsed: {mins:02d}:{secs:02d} (MM:SS)",
        use_sep=True
    )

except Exception as e:
    log_bilingual(f"❌ เกิดข้อผิดพลาดร้ายแรง (Fatal Error): {str(e)}", f"Fatal Error: {str(e)}", use_sep=True)
    printt(traceback.format_exc())