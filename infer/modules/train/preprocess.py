# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี (Date)       : 02/11/2569 (02 November 2026)
# 2. ชื่อไฟล์ที่ทำ (File Name)   : preprocess.py
# 3. ลำดับเวอร์ชัน (Version)   : 4.2 (Studio-Quality & Soul Preservation Edition)
# 4. เครดิตผู้สร้าง (Credit)    : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#                              Original by RVC Project
# 5. ตำแหน่งที่วางของไฟล์ (Path) : infer/modules/train/preprocess.py
# 6. รายละเอียดการแก้ไขงาน    : ฝังระบบ Soft-Knee Noise Gate เพื่อรักษาลมหายใจ (Breath Preservation) 
#                              และเพิ่มระบบกำจัดไฟล์ขยะ Windows (Junk Sentinel) ก่อนเริ่มกระบวนการ
# ==============================================================================

import multiprocessing
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import traceback

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa

# 🔴 กฎข้อที่ 6: การขจัดไฟล์ขยะระบบ Windows อัตโนมัติ (Windows Junk Sentinel)
try:
    from dataset_auto_cleaner import purge_directory_junk
except ImportError:
    def purge_directory_junk(path):
        pass

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)

inp_root = sys.argv[1] if len(sys.argv) > 1 else ""
sr = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
n_p = min(2, max(1, int(sys.argv[3]))) if len(sys.argv) > 3 else 2
exp_dir = sys.argv[4] if len(sys.argv) > 4 else "logs/default"
noparallel = (sys.argv[5] == "True") if len(sys.argv) > 5 else False
per = float(sys.argv[6]) if len(sys.argv) > 6 else 3.7

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer

mutex = multiprocessing.Lock()
os.makedirs(exp_dir, exist_ok=True)
log_file_path = os.path.join(exp_dir, "preprocess.log")


def log_bilingual(th_msg: str, en_msg: str) -> None:
    """ระบบบันทึก Log 2 ภาษา (Bilingual Logger)"""
    with mutex:
        formatted = f"[TH] {th_msg}\n[EN] {en_msg}"
        try:
            print(formatted, flush=True)
        except Exception:
            pass
        try:
            with open(log_file_path, "a+", encoding="utf-8", errors="replace") as f:
                f.write(f"{formatted}\n")
        except Exception:
            pass

def apply_edge_fade(audio_chunk: np.ndarray, sample_rate: int, fade_ms: float = 10.0) -> np.ndarray:
    """ทำ Fade-In/Out แบบนุ่มนวล เพื่อป้องกันอาการเสียงคลิก (Click/Pop) ที่หัวท้ายไฟล์"""
    fade_len = int(sample_rate * fade_ms / 1000.0)
    if len(audio_chunk) <= fade_len * 2:
        return audio_chunk
    fade_in = np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2
    fade_out = np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2
    chunk = audio_chunk.copy()
    chunk[:fade_len] *= fade_in
    chunk[-fade_len:] *= fade_out
    return chunk


class PreProcess:
    def __init__(self, sr: int, exp_dir: str, per: float = 3.7):
        self.slicer = Slicer(sr=sr, threshold=-45, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=45, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.95
        self.alpha = 0.75
        self.min_duration_sec = 0.35
        self.exp_dir = exp_dir
        
        self.gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs").replace("\\", "/")
        self.wavs16k_dir = os.path.join(exp_dir, "1_16k_wavs").replace("\\", "/")
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio: np.ndarray, idx0: int, idx1: int) -> None:
        if len(tmp_audio) < int(self.sr * self.min_duration_sec):
            return
            
        tmp_max = np.abs(tmp_audio).max()
        if np.isnan(tmp_max) or np.isinf(tmp_max) or tmp_max > 2.5:
            log_bilingual(f"⚠️ กรองเสียงทิ้งเนื่องจากแอมพลิจูดผิดปกติ: {idx0}_{idx1} (Max: {tmp_max:.2f})",
                          f"Filtered corrupted amplitude segment: {idx0}_{idx1} (Max: {tmp_max:.2f})")
            return
            
        rms = np.sqrt(np.mean(tmp_audio ** 2) + 1e-9)
        if rms < 0.0015:
            return
            
        if tmp_max > 1e-5:
            tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (1 - self.alpha) * tmp_audio
            
        tmp_audio = apply_edge_fade(tmp_audio, self.sr, fade_ms=10.0)
        
        wavfile.write(f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav", self.sr, tmp_audio.astype(np.float32))
        try:
            tmp_audio_16k = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000, res_type="soxr_vhq")
        except Exception:
            tmp_audio_16k = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
            
        wavfile.write(f"{self.wavs16k_dir}/{idx0}_{idx1}.wav", 16000, tmp_audio_16k.astype(np.float32))

    def pipeline(self, path: str, idx0: int) -> None:
        try:
            audio = load_audio(path, self.sr)
            if len(audio) < int(self.sr * self.min_duration_sec):
                return
                
            audio = signal.filtfilt(self.bh, self.ah, audio).astype(np.float32)
            
            gate_threshold = 0.0003
            audio = np.where(np.abs(audio) < gate_threshold, audio * 0.5, audio)
            
            idx1 = 0
            for audio_segment in self.slicer.slice(audio):
                if len(audio_segment) < int(self.sr * self.min_duration_sec):
                    continue
                i = 0
                while True:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio_segment[start:]) > self.tail * self.sr:
                        tmp_audio = audio_segment[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio_segment[start:]
                        if len(tmp_audio) >= int(self.sr * self.min_duration_sec):
                            self.norm_write(tmp_audio, idx0, idx1)
                            idx1 += 1
                        break
            
            if idx1 > 0:
                log_bilingual(f"✅ สกัดเนื้อเสียงบริสุทธิ์ -> {os.path.basename(path)} ({idx1} ชิ้น)",
                              f"Processed pure vocals -> {os.path.basename(path)} ({idx1} segments)")
        except Exception:
            log_bilingual(f"❌ เกิดข้อผิดพลาด -> {os.path.basename(path)}:\n{traceback.format_exc()}",
                          f"Error processing -> {os.path.basename(path)}:\n{traceback.format_exc()}")

    def pipeline_mp(self, infos: list) -> None:
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root: str, n_p: int) -> None:
        try:
            inp_root = str(inp_root).strip().strip('"').strip("'")
            if not os.path.exists(inp_root):
                log_bilingual(f"❌ ไม่พบโฟลเดอร์ชุดข้อมูลต้นฉบับ: {inp_root}", f"Original dataset folder not found: {inp_root}")
                return

            purge_directory_junk(inp_root)

            valid_audio_exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".aiff"}
            audio_paths = []
            
            for root_d, _, files in os.walk(inp_root):
                for name in sorted(files):
                    full_p = os.path.join(root_d, name)
                    ext = os.path.splitext(name)[1].lower()
                    if ext in valid_audio_exts and os.path.getsize(full_p) > 512:
                        audio_paths.append(full_p)

            infos = [(full_path, idx) for idx, full_path in enumerate(audio_paths)]
            log_bilingual(f"🎙️ ตรวจพบไฟล์เสียงพร้อมสกัดความบริสุทธิ์: {len(infos)} ไฟล์ | Thread(s): {n_p}",
                          f"Discovered vocal files ready for extraction: {len(infos)} files | Thread(s): {n_p}")

            if noparallel or n_p <= 1 or len(infos) <= 1:
                self.pipeline_mp(infos)
            else:
                ps = []
                for i in range(n_p):
                    sub_infos = infos[i::n_p]
                    if sub_infos:
                        p = multiprocessing.Process(target=self.pipeline_mp, args=(sub_infos,))
                        ps.append(p)
                        p.start()
                for p in ps:
                    p.join()
        except Exception:
            log_bilingual(f"❌ ล้มเหลว: {traceback.format_exc()}", f"Failed: {traceback.format_exc()}")


def preprocess_trainset(inp_root: str, sr: int, n_p: int, exp_dir: str, per: float) -> None:
    start_time = time.time()
    os.makedirs(exp_dir, exist_ok=True)
    try:
        import json
        sr_tag = f"{int(sr) // 1000}k"
        with open(os.path.join(exp_dir, ".sr_meta.json"), "w", encoding="utf-8") as mf:
            json.dump({"sr": sr_tag, "sr_hz": int(sr)}, mf)
    except Exception:
        pass
        
    pp = PreProcess(sr, exp_dir, per)
    log_bilingual("=" * 65, "=" * 65)
    log_bilingual("🚀 เริ่มต้นกระบวนการสกัดเนื้อเสียงบริสุทธิ์ระดับมืออาชีพ (Preprocess v4.2)...",
                  "Starting Professional Pure Vocal Extraction (Preprocess v4.2)...")
    log_bilingual("=" * 65, "=" * 65)
    
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    
    log_bilingual("=" * 65, "=" * 65)
    log_bilingual(f"🎉 สิ้นสุดการเตรียมเนื้อเสียงบริสุทธิ์ (ระยะเวลา: {mins:02d}:{secs:02d})",
                  f"Pure vocal extraction completed (Elapsed: {mins:02d}:{secs:02d})")
    log_bilingual("=" * 65, "=" * 65)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per)